# src/rag_job_runner.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime

import aiohttp
from qdrant_client import QdrantClient

# --- 專案內部 import ----------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from rag_core.application.rag_engine import RAGEngine
from rag_core.domain.scenario import Scenario
from rag_core.domain.schema_checker import DataStructureChecker
from rag_core.infrastructure.embedding import EmbeddingManager
from rag_core.infrastructure.vector_store import VectorIndex
from rag_core.infrastructure.llm.llm_manager import LLMManager
from rag_core.infrastructure.llm.openai_adapter import OpenAIAdapter
from rag_core.infrastructure.llm.local_llama_adapter import LocalLlamaAdapter
from rag_core.utils.text_preprocessor import TextPreprocessor
from config.settings import config_manager
from utils import log_wrapper

logger = logging.getLogger(__name__)

class RAGJobRunner:
    def __init__(self, vector_index: VectorIndex, rag_engine: RAGEngine):
        self.vector_index = vector_index
        self.rag_engine = rag_engine
        self.settings = config_manager.settings
        self._semaphore = asyncio.Semaphore(10)  # 限制並發數為3

    async def _validate_data(self, data: Dict[str, Any], mode: str) -> None:
        """驗證數據結構"""
        checker = DataStructureChecker()
        try:
            checker.validate(data, mode=mode)
        except Exception as e:
            raise ValueError(f"Data validation failed for {mode}: {str(e)}")

    async def _process_documents(
        self,
        docs: List[Dict[str, Any]],
        scenario: Scenario,
        collection: str,
        direction: str
    ) -> List[Dict[str, Any]]:
        """處理文檔並執行 RAG"""
        results: List[Dict[str, Any]] = []

        async def _handle_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
            async with self._semaphore:
                # 根據方向選擇正確的 rag_k
                if direction == "forward":
                    rag_k = scenario.rag_k_forward if scenario.rag_k_forward is not None else scenario.rag_k
                    side = "reference"
                elif direction == "reverse":
                    rag_k = scenario.rag_k_reverse if scenario.rag_k_reverse is not None else scenario.rag_k
                    side = "input"
                else:  # both
                    rag_k = scenario.rag_k
                    side = "reference"  # 在 both 模式下，先搜尋 reference

                # 確保 rag_k 至少為 1
                rag_k = max(1, rag_k)

                idx_info = {
                    "collection_name": collection,
                    "filters": {"side": side},
                    "rag_k": rag_k,
                }
                return await self.rag_engine.generate_answer(
                    user_query=doc["text"],
                    root_uid=doc["group_uid"],
                    scenario=scenario,
                    index_info=idx_info
                )

        # 建立所有任務
        tasks = [_handle_doc(d) for d in docs]
        # 並發執行
        results = await asyncio.gather(*tasks)
        return results

    async def _send_callback(self, callback_url: str, result: Dict[str, Any]) -> None:
        """發送回調通知"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=result) as response:
                    if response.status != 200:
                        logger.error(f"Callback failed with status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send callback: {str(e)}")

    async def run_job(self, job_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        執行 RAG 任務
        
        Args:
            job_payload: 包含任務信息的字典
                - job_id: 任務ID
                - project_id: 專案ID
                - scenario: 場景設定
                - input_data: 輸入數據
                - reference_data: 參考數據
                - callback_url: 回調URL（可選）
        
        Returns:
            Dict[str, Any]: 任務結果
        """
        try:
            job_id = job_payload["job_id"]
            project_id = job_payload["project_id"]
            scenario_data = job_payload["scenario"]
            input_data = job_payload["input_data"]
            reference_data = job_payload["reference_data"]
            callback_url = job_payload.get("callback_url")

            # 1. 驗證數據
            await self._validate_data(reference_data, mode="reference")
            await self._validate_data(input_data, mode="input")

            # 2. 創建場景
            # 合併預設設定和用戶提供的設定
            default_scenario = self.settings.scenario.dict()
            scenario_data = {**default_scenario, **scenario_data}  # 用戶設定覆蓋預設設定
            scenario = Scenario(**scenario_data)

            # 3. 處理文檔
            ref_depth = scenario.reference_depth
            inp_depth = scenario.input_depth

            ref_docs = TextPreprocessor.flatten_levels(reference_data, ref_depth, side="reference")
            inp_docs = TextPreprocessor.flatten_levels(input_data, inp_depth, side="input")

            # 4. 分塊處理（如果需要）
            if scenario.chunk_size > 0:
                chunk_size = scenario.chunk_size

                # 處理參考文檔
                new_ref_docs: List[Dict[str, Any]] = []
                for d in ref_docs:
                    for i, c in enumerate(TextPreprocessor.chunk_text(d["text"], chunk_size)):
                        cuid = f"{d['group_uid']}_c{i}"
                        new_ref_docs.append({
                            "orig_sid": d["orig_sid"],
                            "group_uid": d["group_uid"],
                            "uid": cuid,
                            "sid": cuid,
                            "text": c,
                        })
                ref_docs = new_ref_docs

                # 處理輸入文檔
                new_inp_docs: List[Dict[str, Any]] = []
                for d in inp_docs:
                    for i, c in enumerate(TextPreprocessor.chunk_text(d["text"], chunk_size)):
                        cuid = f"{d['group_uid']}_c{i}"
                        new_inp_docs.append({
                            "orig_sid": d["orig_sid"],
                            "group_uid": d["group_uid"],
                            "uid": cuid,
                            "sid": cuid,
                            "text": c,
                        })
                inp_docs = new_inp_docs

            # 5. 設置集合名稱
            collection = f"{self.settings.vector_db.collection}_{project_id}"

            # 6. 根據方向處理文檔
            direction = scenario.direction.lower()
            if direction == "both":
                await self.vector_index.ingest_json(collection, ref_docs, mode="reference")
                await self.vector_index.ingest_json(collection, inp_docs, mode="input")
            elif direction == "forward":
                await self.vector_index.ingest_json(collection, ref_docs, mode="reference")
            elif direction == "reverse":
                await self.vector_index.ingest_json(collection, inp_docs, mode="input")

            # 7. 執行 RAG
            results = await self._process_documents(
                inp_docs,
                scenario,
                collection,
                direction
            )

            # 8. 準備結果
            result = {
                "job_id": job_id,
                "project_id": project_id,
                "status": "completed",
                "results": results,
                "completed_at": datetime.utcnow().isoformat()
            }

            # 9. 發送回調（如果有）
            if callback_url:
                await self._send_callback(callback_url, result)

            return result

        except Exception as e:
            error_result = {
                "job_id": job_payload["job_id"],
                "project_id": job_payload["project_id"],
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }

            # 發送錯誤回調（如果有）
            if callback_url := job_payload.get("callback_url"):
                await self._send_callback(callback_url, error_result)

            raise
