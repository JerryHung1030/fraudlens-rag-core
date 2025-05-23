# fastapi_app.py - FastAPI interface for queuing RAG jobs

import uuid
from typing import Dict, Any
from rq import Queue
from redis import Redis

from interfaces.jobs.rag_job_runner import RAGJobRunner

# 初始化 Redis connection & RQ Queue（可改成 config 參數）
redis_conn = Redis(host="127.0.0.1", port=6379, db=0)
rag_queue = Queue("rag_jobs", connection=redis_conn)


def start_rag_job(
    job_runner: RAGJobRunner,
    project_id: str,
    scenario: Dict[str, Any],
    input_json_path: str,
    reference_json_path: str,
    callback_url: str,
) -> str:
    """
    將 job_payload 丟到 RQ queue。回傳 job_id 讓後端追蹤。
    """
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "project_id": project_id,
        "scenario": scenario,
        "input_json": input_json_path,
        "reference_json": reference_json_path,
        "callback_url": callback_url,
    }

    # 使用 RQ enqueue
    job = rag_queue.enqueue(job_runner.run_job, payload)
    return job.get_id()
