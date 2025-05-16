# src/rag_job_runner.py
import json
import requests
from rag_core.domain.scenario import Scenario
from typing import Dict, Any, List

from rag_core.infrastructure.vector_store import VectorIndex
from rag_core.application.rag_engine import RAGEngine
from rag_core.utils.text_preprocessor import TextPreprocessor
from utils import log_wrapper


class RAGJobRunner:
    def __init__(self, vec_index: VectorIndex, rag_engine: RAGEngine):
        self.vec_index = vec_index
        self.rag_engine = rag_engine

    def run_job(self, job_payload: Dict[str, Any]):
        """
        RQ job 執行函式。
        job_payload 可能包含：
        {
          "job_id": "xxx",
          "project_id": "xxx",
          "scenario": {...},  # scenario dict
          "input_json": "...",
          "reference_json": "...",
          "callback_url": "https://...."
        }
        """
        job_id = job_payload["job_id"]
        log_wrapper.info(
            "RAGJobRunner",
            "run_job",
            f"[Job {job_id}] Starting RAG job..."
        )

        scenario_cfg = job_payload["scenario"]
        # 轉為 Pydantic model
        scenario = Scenario(**scenario_cfg)

        inp_path = job_payload.get("input_json")
        ref_path = job_payload.get("reference_json")

        inputs = _load_json(inp_path)
        refs = _load_json(ref_path)

        inp_depth = scenario.input_depth
        ref_depth = scenario.reference_depth

        inp_docs = TextPreprocessor.flatten_levels(inputs, inp_depth, side="input")
        ref_docs = TextPreprocessor.flatten_levels(refs, ref_depth, side="reference")

        # chunk
        chunk_size = scenario.chunk_size
        if chunk_size > 0:
            inp_docs = _chunk_docs(inp_docs, chunk_size)
            ref_docs = _chunk_docs(ref_docs, chunk_size)

        # ingest：示範做 per-project collection
        collection_name = f"rag_collection_{job_payload['project_id']}"
        self.vec_index.create_collection(collection_name)
        self.vec_index.ingest_json(collection_name, inp_docs, mode="input")
        self.vec_index.ingest_json(collection_name, ref_docs, mode="reference")

        # ---------- 2. 逐 input or reference 執行 RAG ----------
        results = []
        direction = scenario.direction.lower()

        if direction == "forward":
            for item in inp_docs:
                index_info = {
                    "collection_name": collection_name,
                    "filters": {"side": "reference"},
                    "rag_k": scenario.rag_k_forward or scenario.rag_k
                }
                # 同步呼叫
                res = self.rag_engine.generate_answer_sync(
                    user_query=item["text"],
                    root_uid=item["group_uid"],
                    scenario=scenario,
                    index_info=index_info
                )
                if res:  # 確保 res 不為空
                    results.extend(res)
        else:
            # reverse
            for item in ref_docs:
                index_info = {
                    "collection_name": collection_name,
                    "filters": {"side": "input"},
                    "rag_k": scenario.rag_k_reverse or scenario.rag_k
                }
                res = self.rag_engine.generate_answer_sync(
                    user_query=item["text"],
                    root_uid=item["group_uid"],
                    scenario=scenario,
                    index_info=index_info
                )
                if res:  # 確保 res 不為空
                    results.extend(res)

        # ---------- 3. 存檔 + callback ----------
        out_path = f"/tmp/{job_id}_result.json"
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        log_wrapper.info(
            "RAGJobRunner",
            "run_job",
            f"[Job {job_id}] RAG finished, total={len(results)}. Save to {out_path}."
        )

        cb_url = job_payload["callback_url"]
        try:
            resp = requests.post(cb_url, json={"job_id": job_id, "result_path": out_path})
            log_wrapper.info(
                "RAGJobRunner",
                "run_job",
                f"[Job {job_id}] Callback {cb_url} => status {resp.status_code}"
            )
        except Exception as e:
            log_wrapper.error(
                "RAGJobRunner",
                "run_job",
                f"[Job {job_id}] Callback error: {e}"
            )


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _chunk_docs(docs: List[Dict[str, Any]], chunk_size: int) -> List[Dict[str, Any]]:
    from rag_core.utils.text_preprocessor import TextPreprocessor
    new_docs = []
    for d in docs:
        # d 含 { "orig_sid", "group_uid", "uid", "text" }
        chunks = TextPreprocessor.chunk_text(d["text"], chunk_size)
        for i, c in enumerate(chunks):
            # chunk_uid = 原 group_uid + _c{i}
            chunk_uid = f"{d['group_uid']}_c{i}"

            new_docs.append({
                "orig_sid": d["orig_sid"],
                "group_uid": d["group_uid"],
                "uid": chunk_uid,     # 這就是 Qdrant 的 point.id
                "text": c,
                "sid": chunk_uid
            })
    return new_docs
