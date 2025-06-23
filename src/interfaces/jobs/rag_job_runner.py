"""
===============================================================================
    Module Name: rag_job_runner.py
    Description: Asynchronous RAG job runner for processing and callback.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
# src/rag_job_runner.py
from __future__ import annotations

import asyncio
import json
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
from utils.logging import log_wrapper

class RAGJobRunner:
    def __init__(self, vector_index: VectorIndex, rag_engine: RAGEngine):
        self.vector_index = vector_index
        self.rag_engine = rag_engine
        self.settings = config_manager.settings
        self._semaphore = asyncio.Semaphore(1)  # 限制文檔處理並發數
        log_wrapper.info("RAGJobRunner", "__init__", "RAGJobRunner 初始化完成")

    async def _validate_data(self, data: Dict[str, Any], mode: str) -> None:
        """驗證數據結構"""
        log_wrapper.info("RAGJobRunner", "_validate_data", f"開始驗證 {mode} 數據")
        checker = DataStructureChecker()
        try:
            checker.validate(data, mode=mode)
            log_wrapper.info("RAGJobRunner", "_validate_data", f"{mode} 數據驗證成功")
        except Exception as e:
            log_wrapper.error("RAGJobRunner", "_validate_data", f"{mode} 數據驗證失敗: {str(e)}")
            raise ValueError(f"Data validation failed for {mode}: {str(e)}")

    async def _process_documents(
        self,
        docs: List[Dict[str, Any]],
        scenario: Scenario,
        collection: str,
        direction: str
    ) -> List[Dict[str, Any]]:
        """處理文檔並執行 RAG"""
        log_wrapper.info("RAGJobRunner", "_process_documents", f"開始處理 {len(docs)} 個文檔，方向: {direction}")
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
                
                log_wrapper.debug("RAGJobRunner", "_handle_doc", f"處理文檔 {doc.get('uid', 'unknown')}，使用 rag_k={rag_k}")
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
        log_wrapper.info("RAGJobRunner", "_process_documents", f"文檔處理完成，共處理 {len(results)} 個文檔")
        return results

    async def _send_callback(self, callback_url: str, result: Dict[str, Any]) -> None:
        """發送回調通知"""
        try:
            log_wrapper.info("RAGJobRunner", "_send_callback", f"發送回調通知到 {callback_url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=result) as response:
                    if response.status != 200:
                        log_wrapper.error("RAGJobRunner", "_send_callback", f"回調失敗，狀態碼: {response.status}")
                    else:
                        log_wrapper.info("RAGJobRunner", "_send_callback", "回調通知發送成功")
        except Exception as e:
            log_wrapper.error("RAGJobRunner", "_send_callback", f"發送回調失敗: {str(e)}")

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
            # 確保 job_payload 是字典
            if isinstance(job_payload, str):
                job_payload = json.loads(job_payload)
                
            job_id = job_payload.get('job_id')
            log_wrapper.info("RAGJobRunner", "run_job", f"開始執行任務 {job_id}")
            
            project_id = job_payload.get("project_id")
            scenario_data = job_payload.get("scenario", {})
            input_data = job_payload.get("input_data", {})
            reference_data = job_payload.get("reference_data", {})
            callback_url = job_payload.get("callback_url")

            if not all([job_id, project_id, scenario_data, input_data, reference_data]):
                log_wrapper.error("RAGJobRunner", "run_job", "任務參數不完整")
                raise ValueError("Missing required fields in job_payload")

            log_wrapper.info("RAGJobRunner", "run_job", "開始驗證數據")
            # 1. 驗證數據
            await self._validate_data(reference_data, mode="reference")
            await self._validate_data(input_data, mode="input")

            log_wrapper.info("RAGJobRunner", "run_job", "創建場景")
            # 2. 創建場景
            # 合併預設設定和用戶提供的設定
            default_scenario = self.settings.scenario.dict()
            scenario_data = {**default_scenario, **scenario_data}  # 用戶設定覆蓋預設設定
            scenario = Scenario(**scenario_data)

            log_wrapper.info("RAGJobRunner", "run_job", "處理文檔")
            # 3. 處理文檔
            ref_depth = scenario.reference_depth
            inp_depth = scenario.input_depth

            ref_docs = TextPreprocessor.flatten_levels(reference_data, ref_depth, side="reference")
            inp_docs = TextPreprocessor.flatten_levels(input_data, inp_depth, side="input")
            
            log_wrapper.info("RAGJobRunner", "run_job", f"文檔處理完成，參考文檔: {len(ref_docs)} 個，輸入文檔: {len(inp_docs)} 個")

            # 4. 分塊處理（如果需要）
            if scenario.chunk_size > 0:
                log_wrapper.info("RAGJobRunner", "run_job", f"執行分塊處理，分塊大小: {scenario.chunk_size}")
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
                
                log_wrapper.info("RAGJobRunner", "run_job", f"分塊處理完成，參考文檔: {len(ref_docs)} 個，輸入文檔: {len(inp_docs)} 個")

            log_wrapper.info("RAGJobRunner", "run_job", "設置集合名稱")
            # 5. 設置集合名稱
            collection = f"{self.settings.vector_db.collection}_{project_id}"

            log_wrapper.info("RAGJobRunner", "run_job", f"處理文檔方向: {scenario.direction}")
            # 6. 根據方向處理文檔
            direction = scenario.direction.lower()
            if direction == "both":
                log_wrapper.info("RAGJobRunner", "run_job", "雙向模式：處理參考和輸入文檔")
                await self.vector_index.ingest_json(collection, ref_docs, mode="reference")
                await self.vector_index.ingest_json(collection, inp_docs, mode="input")
            elif direction == "forward":
                log_wrapper.info("RAGJobRunner", "run_job", "正向模式：處理參考文檔")
                await self.vector_index.ingest_json(collection, ref_docs, mode="reference")
            elif direction == "reverse":
                log_wrapper.info("RAGJobRunner", "run_job", "反向模式：處理輸入文檔")
                await self.vector_index.ingest_json(collection, inp_docs, mode="input")

            log_wrapper.info("RAGJobRunner", "run_job", "執行 RAG 處理")
            # 7. 執行 RAG
            results = await self._process_documents(
                inp_docs,
                scenario,
                collection,
                direction
            )

            log_wrapper.info("RAGJobRunner", "run_job", "準備結果")
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
                log_wrapper.info("RAGJobRunner", "run_job", "發送回調通知")
                await self._send_callback(callback_url, result)

            log_wrapper.info("RAGJobRunner", "run_job", f"任務 {job_id} 完成")
            return result

        except Exception as e:
            log_wrapper.error("RAGJobRunner", "run_job", f"任務執行失敗: {str(e)}")
            error_result = {
                "job_id": job_payload.get("job_id"),
                "project_id": job_payload.get("project_id"),
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }

            # 發送錯誤回調（如果有）
            if callback_url := job_payload.get("callback_url"):
                log_wrapper.info("RAGJobRunner", "run_job", "發送錯誤回調通知")
                await self._send_callback(callback_url, error_result)

            raise

    def run_rag_job(self, project_id: str, scenario: Dict[str, Any], input_data: Dict[str, Any], reference_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        同步版本的 RAG 任務執行器
        
        Args:
            project_id: 專案ID
            scenario: 場景設定
            input_data: 輸入數據
            reference_data: 參考數據
            
        Returns:
            List[Dict[str, Any]]: 處理結果
        """
        log_wrapper.info("RAGJobRunner", "run_rag_job", f"開始同步執行 RAG 任務，專案: {project_id}")
        
        try:
            # 創建事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 準備任務參數
            job_payload = {
                "job_id": f"sync_{project_id}_{datetime.utcnow().timestamp()}",
                "project_id": project_id,
                "scenario": scenario,
                "input_data": input_data,
                "reference_data": reference_data
            }
            
            # 執行任務
            result = loop.run_until_complete(self.run_job(job_payload))
            
            log_wrapper.info("RAGJobRunner", "run_rag_job", f"同步 RAG 任務完成，專案: {project_id}")
            return result.get("results", [])
            
        except Exception as e:
            log_wrapper.error("RAGJobRunner", "run_rag_job", f"同步 RAG 任務失敗: {str(e)}")
            raise
        finally:
            loop.close()
