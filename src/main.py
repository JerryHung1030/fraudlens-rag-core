# src/main.py
"""
主程式入口

‣ 依 .env 與情境設定檔 (scenario.yml) 來：
    1. 初始化 Embedding / VectorIndex / LLM
    2. 載入 **階層式 JSON** reference 與 input
    3. flatten → ingest → 逐筆 RAG 產生結果
    4. 依需求輸出 direction / rag_k / cof_threshold … 等欄位之 JSON
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
from typing import Any, Dict, List
from datetime import datetime

import argparse
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import re

# --- 專案內部 import ----------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# --- 新/修正 import ---
from services.rag_engine import RAGEngine
from services.scenario import Scenario

from managers.embedding_manager import EmbeddingManager
from managers.data_structure_checker import DataStructureChecker
from managers.vector_index import VectorIndex
from managers.llm_manager import LLMManager
from adapters.openai_adapter import OpenAIAdapter
from adapters.local_llama_adapter import LocalLlamaAdapter
from utils.text_preprocessor import TextPreprocessor
from src.utils.log_wrapper import log_wrapper

load_dotenv(override=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Helper
# ═════════════════════════════════════════════════════════════════════════════
def load_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log_wrapper.error(
            "main",
            "load_json_file",
            f"JSON file {path} not found!"
        )
        raise
    except json.JSONDecodeError as e:
        log_wrapper.error(
            "main",
            "load_json_file",
            f"JSON decode error in {path}: {e}"
        )
        raise


# ═════════════════════════════════════════════════════════════════════════════
#  初始化核心元件
# ═════════════════════════════════════════════════════════════════════════════
def setup_core() -> tuple[EmbeddingManager, VectorIndex, LLMManager]:
    """
    初始化 EmbeddingManager / VectorIndex / LLMManager
    """
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    vector_size = 1536
    embed_model_name = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

    # Embedding
    embed_mgr = EmbeddingManager(
        openai_api_key=openai_api_key,
        embedding_model_name=embed_model_name
    )
    
    # Data schema checker
    checker = DataStructureChecker()

    # Qdrant client
    qdrant = QdrantClient(url=qdrant_url)

    # VectorIndex
    vec_index = VectorIndex(
        embedding_manager=embed_mgr,
        data_checker=checker,
        qdrant_client=qdrant,
        default_collection_name=os.getenv("QDRANT_COLLECTION", "my_rag_collection"),
        vector_size=vector_size,
    )

    # LLM
    llm_mgr = LLMManager()
    llm_mgr.register_adapter(
        "openai",
        OpenAIAdapter(openai_api_key=openai_api_key, temperature=0.0, max_tokens=2048),
    )
    llm_mgr.register_adapter(
        "llama",
        LocalLlamaAdapter(model_path="models/llama.bin", temperature=0.0, max_tokens=2048),
    )
    llm_mgr.set_default_adapter("openai")

    return embed_mgr, vec_index, llm_mgr


# ═════════════════════════════════════════════════════════════════════════════
#  主流程
# ═════════════════════════════════════════════════════════════════════════════
async def main() -> None:
    # ---------- 1. 初始化 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="append to collection name for isolation",
        metavar="RUN_ID",
        # 限制 run-id 格式：1-32 字元，只允許英數字、底線、連字符
        choices=[x for x in [""] + [f"run_{i}" for i in range(1, 33)] if re.fullmatch(r"[A-Za-z0-9_\-]{1,32}", x)]
    )
    args = parser.parse_args()


    embedding_mgr, vector_index, llm_mgr = setup_core()

    # ---------- 2. 載入 scenario 設定 ----------
    scenario_path = os.getenv("SCENARIO_FILE", "scenario.yml")
    if not pathlib.Path(scenario_path).exists():
        # 若找不到，用最小可執行預設
        default_scenario = {
            "role_desc": "你是 RAG 比對助手",
            "reference_desc": "以下為待比對參考資料",
            "input_desc": "以下為使用者輸入",
            "direction": "forward",
            "rag_k": 5,
            "cof_threshold": 0.5,
            "reference_depth": 2,
            "input_depth": 1,
        }
        with open(scenario_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(default_scenario, fp, allow_unicode=True)
        log_wrapper.info(
            "main",
            "create_default_scenario",
            f"Default scenario.yml 已建立，路徑: {scenario_path}"
        )

    scenario_cfg: Dict[str, Any] = yaml.safe_load(open(scenario_path, "r", encoding="utf-8"))
    scenario = Scenario(**scenario_cfg)  # pydantic 物件

    # ---------- 3. 載入 reference / input JSON ----------
    ref_path = scenario_cfg.get("reference_json") or os.getenv("REFERENCE_JSON")
    inp_path = scenario_cfg.get("input_json") or os.getenv("INPUT_JSON")

    if not ref_path or not inp_path:
        log_wrapper.error(
            "main",
            "main",
            "reference_json 或 input_json 路徑未設定，請於 scenario.yml 或環境變數提供"
        )
        return

    ref_raw = load_json_file(ref_path)
    inp_raw = load_json_file(inp_path)

    # ---------- 4. 檢查 & flatten ----------
    checker = DataStructureChecker()
    try:
        checker.validate(ref_raw, mode="reference")
        checker.validate(inp_raw, mode="input")
    except Exception as e:
        log_wrapper.error(
            "main",
            "main",
            f"資料結構驗證失敗：{e}"
        )
        return

    ref_depth = scenario.reference_depth
    inp_depth = scenario.input_depth

    ref_docs = TextPreprocessor.flatten_levels(ref_raw, ref_depth, side="reference")
    inp_docs = TextPreprocessor.flatten_levels(inp_raw, inp_depth, side="input")

    # chunk
    if scenario.chunk_size > 0:
        chunk_size = scenario.chunk_size

        # chunk reference
        new_ref_docs: List[Dict[str, Any]] = []
        for d in ref_docs:
            for i, c in enumerate(TextPreprocessor.chunk_text(d["text"], chunk_size)):
                cuid = f"{d['group_uid']}_c{i}"
                new_ref_docs.append({
                    "orig_sid": d["orig_sid"],
                    "group_uid": d["group_uid"],
                    "uid": cuid,
                    "sid": cuid,  # 保險
                    "text": c,
                })
        ref_docs = new_ref_docs

        # chunk input
        new_inp_docs: List[Dict[str, Any]] = []
        for d in inp_docs:
            for i, c in enumerate(TextPreprocessor.chunk_text(d["text"], chunk_size)):
                cuid = f"{d['group_uid']}_c{i}"
                new_inp_docs.append({
                    "orig_sid": d["orig_sid"],
                    "group_uid": d["group_uid"],
                    "uid": cuid,
                    "sid": cuid,  # 保險
                    "text": c,
                })
        inp_docs = new_inp_docs

    direction = scenario.direction.lower()

    # ---------- 5. Ingest ----------
    base_collection = os.getenv("QDRANT_COLLECTION", "my_rag_collection")
    collection = base_collection
    if args.run_id:
        collection += f"_{args.run_id}"

    if direction == "both":
        vector_index.ingest_json(collection, ref_docs, mode="reference")
        vector_index.ingest_json(collection, inp_docs, mode="input")
    elif direction == "forward":
        vector_index.ingest_json(collection, ref_docs, mode="reference")
    elif direction == "reverse":
        vector_index.ingest_json(collection, inp_docs, mode="input")

    # ---------- 6. 準備 RAGEngine ----------
    rag_engine = RAGEngine(embedding_mgr, vector_index, llm_mgr)

    # ---------- 7. 執行 RAG ----------
    results: List[Dict[str, Any]] = []

    if direction == "forward":
        sem = asyncio.Semaphore(3)  # 可調整
        error_handler = lambda e: log_wrapper.error(
            "main",
            "main",
            f"Task failed: {str(e)}"
        )
        async def _handle_doc(doc):
            async with sem:
                idx_info = {
                    "collection_name": collection,
                    "filters": {"side": "reference"},
                    "rag_k": scenario.rag_k_forward or scenario.rag_k,
                }
                return await rag_engine.generate_answer(
                    user_query=doc["text"],
                    root_uid=doc["group_uid"],
                    scenario=scenario,
                    index_info=idx_info
                )

        # 建立所有任務
        tasks = [_handle_doc(d) for d in inp_docs]
        # gather
        partial_results = await asyncio.gather(*tasks)
        for pres in partial_results:
            if pres:  # 確保 pres 不為空
                results.extend(pres)

    elif direction == "reverse":
        # reference -> input
        for doc in ref_docs:
            index_info = {
                "collection_name": collection,
                "filters": {"side": "input"},
                "rag_k": scenario.rag_k_reverse or scenario.rag_k
            }
            partial_res = await rag_engine.generate_answer(
                user_query=doc["text"], 
                root_uid=doc["group_uid"], 
                scenario=scenario, 
                index_info=index_info
            )
            if partial_res:  # 確保 partial_res 不為空
                results.extend(partial_res)

    elif direction == "both":
        # 可自定義 forward / reverse 皆執行一次
        # 下列為示範
        # 1) forward
        for doc in inp_docs:
            index_info = {
                "collection_name": collection,
                "filters": {"side": "reference"},
                "rag_k": scenario.rag_k_forward or scenario.rag_k
            }
            forward_res = await rag_engine.generate_answer(
                user_query=doc["text"], 
                root_uid=doc["group_uid"], 
                scenario=scenario, 
                index_info=index_info
            )
            if forward_res:  # 確保 forward_res 不為空
                results.extend(forward_res)

        # 2) reverse
        for doc in ref_docs:
            index_info = {
                "collection_name": collection,
                "filters": {"side": "input"},
                "rag_k": scenario.rag_k_reverse or scenario.rag_k
            }
            rev_res = await rag_engine.generate_answer(
                user_query=doc["text"],
                root_uid=doc["group_uid"],
                scenario=scenario,
                index_info=index_info
            )
            if rev_res:  # 確保 rev_res 不為空
                results.extend(rev_res)

    else:
        log_wrapper.warning(
            "main",
            "main",
            "Unknown direction, skip RAG."
        )
        return

    # ---------- 8. 輸出 ----------
    out_path = scenario_cfg.get("output_json", "rag_result.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    log_wrapper.info(
        "main",
        "main",
        f"RAG 結果已寫入 {out_path}，共 {len(results)} 筆"
    )


if __name__ == "__main__":
    asyncio.run(main())
