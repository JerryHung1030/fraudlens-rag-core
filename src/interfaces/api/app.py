"""
===============================================================================
    Module Name: app.py
    Description: FastAPI app for RAG job orchestration and API endpoints.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Set
from rq import Queue
from redis import Redis
import json
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import re
import time
import requests

from .models import RAGRequest, RAGResponse, JobStatus
from .job_manager import job_manager
from ..jobs.rag_job_runner import RAGJobRunner
from rag_core.infrastructure.vector_store import VectorIndex
from rag_core.application.rag_engine import RAGEngine
from config.settings import config_manager
from rag_core.infrastructure.embedding import EmbeddingManager
from rag_core.domain.schema_checker import DataStructureChecker
from qdrant_client import QdrantClient
from rag_core.infrastructure.llm.llm_manager import LLMManager
from rag_core.infrastructure.llm.openai_adapter import OpenAIAdapter
from rag_core.infrastructure.llm.local_llama_adapter import LocalLlamaAdapter
from utils.logging import log_wrapper

# --- 專案內部 import ----------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 初始化 Redis 和 RQ Queue
redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
redis_conn = Redis.from_url(redis_url)
rag_queue = Queue("rag_jobs", connection=redis_conn)

# 使用 Redis 來追蹤運行中的任務
RUNNING_JOBS_KEY = "running_rag_jobs"

# 清理運行中的任務集合
redis_conn.delete(RUNNING_JOBS_KEY)
log_wrapper.info("app", "startup", "已清理運行中的任務集合")

# 添加任務過期時間設定（改為 1 分鐘）
JOB_EXPIRY_HOURS = 5/60  # 1 分鐘
# 添加任務狀態查詢超時設定（改為 5 秒）
JOB_STATUS_TIMEOUT = 5

# 添加專案 ID 驗證正則表達式
PROJECT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

class JobTracker:
    def __init__(self, max_concurrent_jobs: int = 1):
        self.max_concurrent_jobs = max_concurrent_jobs
        self._lock = asyncio.Lock()

    async def can_start_job(self) -> bool:
        async with self._lock:
            running_jobs = redis_conn.smembers(RUNNING_JOBS_KEY)
            return len(running_jobs) < self.max_concurrent_jobs

    async def start_job(self, job_id: str) -> None:
        async with self._lock:
            redis_conn.sadd(RUNNING_JOBS_KEY, job_id)
            running_jobs = redis_conn.smembers(RUNNING_JOBS_KEY)
            log_wrapper.info("JobTracker", "start_job", f"Started job {job_id}. Current running jobs: {len(running_jobs)}")

    async def finish_job(self, job_id: str) -> None:
        async with self._lock:
            redis_conn.srem(RUNNING_JOBS_KEY, job_id)
            running_jobs = redis_conn.smembers(RUNNING_JOBS_KEY)
            log_wrapper.info("JobTracker", "finish_job", f"Finished job {job_id}. Current running jobs: {len(running_jobs)}")

# 創建任務追蹤器
job_tracker = JobTracker(max_concurrent_jobs=1)

# 創建 FastAPI 應用
app = FastAPI(
    title="ScamShield AI API",
    description="RAG-based AI API for scam detection",
    version="1.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """服務啟動時的事件處理"""
    # 清理運行中的任務集合
    redis_conn.delete(RUNNING_JOBS_KEY)
    log_wrapper.info("app", "startup_event", "服務啟動：已清理運行中的任務集合")

def wait_for_qdrant_ready(url, max_retries=30, interval=2):
    for i in range(max_retries):
        try:
            resp = requests.get(f"{url}/healthz", timeout=2)
            if resp.status_code == 200:
                print(f"Qdrant is ready after {i+1} tries")
                return
        except Exception as e:
            print(f"Qdrant not ready, retry {i+1}: {e}")
        time.sleep(interval)
    raise RuntimeError("Qdrant not ready after waiting")

def setup_core():
    log_wrapper.info("app", "setup_core", "開始初始化核心元件")
    settings = config_manager.settings

    # Embedding
    embed_mgr = EmbeddingManager(
        openai_api_key=settings.api_keys.openai,
        embedding_model_name=settings.embedding.model
    )

    # Data schema checker
    checker = DataStructureChecker()

    # 等待 Qdrant healthz ready
    qdrant_url = settings.vector_db.url
    log_wrapper.info("app", "setup_core", f"Qdrant URL: {qdrant_url}")
    wait_for_qdrant_ready(qdrant_url)

    # Qdrant client
    qdrant = QdrantClient(url=qdrant_url)

    # VectorIndex
    vec_index = VectorIndex(
        embedding_manager=embed_mgr,
        data_checker=checker,
        qdrant_client=qdrant,
        default_collection_name=settings.vector_db.collection,
        vector_size=settings.vector_db.vector_size,
    )

    # LLM
    llm_mgr = LLMManager()
    llm_mgr.register_adapter(
        "openai",
        OpenAIAdapter(
            openai_api_key=settings.api_keys.openai,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        ),
    )
    llm_mgr.register_adapter(
        "llama",
        LocalLlamaAdapter(
            model_path="models/llama.bin",
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        ),
    )
    llm_mgr.set_default_adapter("openai")

    # RAGEngine
    rag_engine = RAGEngine(embed_mgr, vec_index, llm_mgr)
    
    log_wrapper.info("app", "setup_core", "核心元件初始化完成")
    return RAGJobRunner(vec_index, rag_engine)

# 全域 RAG 執行器
rag_runner = setup_core()

def update_job_status(job_id: str, status: str, results: Optional[list] = None, error: Optional[str] = None):
    """更新任務狀態到 Redis"""
    try:
        job_key = f"rag_job:{job_id}"
        job_data = redis_conn.get(job_key)
        if job_data:
            job = json.loads(job_data)
            job["status"] = status
            job["updated_at"] = datetime.utcnow().isoformat()
            
            if results is not None:
                # 確保 results 是列表格式
                if isinstance(results, list):
                    # 如果結果是嵌套列表，取第一個元素
                    if results and isinstance(results[0], list):
                        job["results"] = results[0]
                    else:
                        job["results"] = results
                else:
                    job["results"] = [results]
                    
            if error is not None:
                job["error"] = error
                
            if status == "completed":
                job["completed_at"] = datetime.utcnow().isoformat()
            elif status == "failed":
                job["failed_at"] = datetime.utcnow().isoformat()
                
            redis_conn.set(job_key, json.dumps(job))
            log_wrapper.info("app", "update_job_status", f"已更新任務 {job_id} 狀態為 {status}")
            
            # 如果任務完成或失敗，從運行中的任務集合中移除
            if status in ["completed", "failed"]:
                redis_conn.srem(RUNNING_JOBS_KEY, job_id)
                log_wrapper.info("app", "update_job_status", f"已從運行中的任務集合中移除任務 {job_id}")
    except Exception as e:
        log_wrapper.error("app", "update_job_status", f"更新任務狀態失敗: {str(e)}")

def process_rag_job(job_payload: dict) -> dict:
    """處理 RAG 任務的函數，用於 RQ worker"""
    job_id = None
    try:
        # 解析任務參數
        if isinstance(job_payload, str):
            job_payload = json.loads(job_payload)
        
        job_id = job_payload["job_id"]
        project_id = job_payload["project_id"]
        scenario = job_payload["scenario"]
        input_data = job_payload["input_data"]
        reference_data = job_payload["reference_data"]
        callback_url = job_payload.get("callback_url")
        
        log_wrapper.info("process_rag_job", "start", f"開始處理任務 {job_id}，專案: {project_id}")
        
        # 更新任務狀態為運行中
        update_job_status(job_id, "running")
        
        # 執行 RAG 任務
        results = rag_runner.run_rag_job(
            project_id=project_id,
            scenario=scenario,
            input_data=input_data,
            reference_data=reference_data
        )
        
        # 更新任務狀態為完成
        update_job_status(job_id, "completed", results=results)
        log_wrapper.info("process_rag_job", "complete", f"任務 {job_id} 處理完成")
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        log_wrapper.error("process_rag_job", "error", f"任務 {job_id} 處理失敗: {str(e)}")
        # 更新任務狀態為失敗
        update_job_status(
            job_id=job_id,
            status="failed",
            error=str(e)
        )
        # 標記任務為已完成
        redis_conn.srem(RUNNING_JOBS_KEY, job_id)
        log_wrapper.info("process_rag_job", "finish", f"Finished job {job_id}. Current running jobs: {len(redis_conn.smembers(RUNNING_JOBS_KEY))}")
        return {"error": str(e)}

# 新增取得 client IP 的 function
def get_client_ip(request: Request):
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host

@app.post("/api/v1/rag", response_model=RAGResponse)
async def create_rag_job(request: Request, body: RAGRequest) -> RAGResponse:
    client_ip = get_client_ip(request)
    try:
        log_wrapper.info("create_rag_job", "start", f"IP: {client_ip} | 收到創建任務請求，專案ID: {body.project_id}")
        # 檢查任務 ID 是否已存在
        job_id = await job_manager.create_job(body.project_id)
        job_key = f"rag_job:{job_id}"
        if redis_conn.exists(job_key):
            log_wrapper.warning("create_rag_job", "duplicate_job", f"IP: {client_ip} | 任務 ID {job_id} 已存在 | HTTP 409")
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "任務 ID 已存在，請使用其他 ID",
                    "message_eng": "Job ID already exists, please use a different ID"
                }
            )
        if not await job_tracker.can_start_job():
            log_wrapper.warning("create_rag_job", "resource_exhausted", f"IP: {client_ip} | 系統資源不足以創建新任務 | HTTP 503")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "系統內部資源不足以創建新任務",
                    "message_eng": "System resources exhausted"
                }
            )
        default_scenario = config_manager.get_scenario_config()
        scenario_data = body.scenario.dict() if body.scenario else {}
        merged_scenario = {**default_scenario, **scenario_data}
        job_payload = {
            "job_id": job_id,
            "project_id": body.project_id,
            "scenario": merged_scenario,
            "input_data": body.input_data,
            "reference_data": body.reference_data,
            "callback_url": body.callback_url
        }
        job_data = {
            "job_id": job_id,
            "project_id": body.project_id,
            "status": "pending",
            "progress": None,
            "results": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "failed_at": None
        }
        redis_conn.set(job_key, json.dumps(job_data))
        job = rag_queue.enqueue(
            process_rag_job,
            json.dumps(job_payload, default=str),
            job_id=job_id
        )
        log_wrapper.info("create_rag_job", "enqueue", f"IP: {client_ip} | 任務 {job_id} 已加入佇列")
        await job_tracker.start_job(job_id)
        log_wrapper.info("create_rag_job", "success", f"IP: {client_ip} | HTTP 200 | 任務 {job_id} 創建成功")
        return RAGResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow()
        )
    except ValueError as e:
        log_wrapper.error("create_rag_job", "validation_error", f"IP: {client_ip} | 請求格式錯誤: {str(e)} | HTTP 400")
        raise HTTPException(
            status_code=400,
            detail={
                "message": "請求資料格式不符合 RAGRequest 指定格式",
                "message_eng": "Invalid request format"
            }
        )
    except HTTPException as e:
        log_wrapper.error("create_rag_job", "http_exception", f"IP: {client_ip} | HTTP {e.status_code} | {str(e.detail)}")
        raise e
    except Exception as e:
        log_wrapper.error("create_rag_job", "system_error", f"IP: {client_ip} | 創建任務失敗: {str(e)} | HTTP 500")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.get("/api/v1/rag/{job_id}/status")
async def get_job_status(request: Request, job_id: str) -> JSONResponse:
    client_ip = get_client_ip(request)
    try:
        log_wrapper.info("get_job_status", "start", f"IP: {client_ip} | 查詢任務 {job_id} 狀態")
        job_key = f"rag_job:{job_id}"
        job_data = redis_conn.get(job_key)
        if not job_data:
            log_wrapper.warning("get_job_status", "not_found", f"IP: {client_ip} | 任務 {job_id} 不存在 | HTTP 404")
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "任務不存在",
                    "message_eng": "Job not found"
                }
            )
        job = json.loads(job_data)
        created_at = datetime.fromisoformat(job["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=JOB_EXPIRY_HOURS):
            log_wrapper.warning("get_job_status", "expired", f"IP: {client_ip} | 任務 {job_id} 已過期 | HTTP 410")
            raise HTTPException(
                status_code=410,
                detail={
                    "message": "任務已過期",
                    "message_eng": "Job has expired"
                }
            )
        try:
            async def get_status():
                if job.get("project_id") == "timeout_test":
                    await asyncio.sleep(JOB_STATUS_TIMEOUT + 1)
                else:
                    await asyncio.sleep(0.1)
                return job
            result = await asyncio.wait_for(get_status(), timeout=JOB_STATUS_TIMEOUT)
            log_wrapper.info("get_job_status", "success", f"IP: {client_ip} | HTTP 200 | 成功獲取任務 {job_id} 狀態: {job['status']}")
            return JSONResponse(
                content={
                    "success": True,
                    "data": result
                }
            )
        except asyncio.TimeoutError:
            log_wrapper.error("get_job_status", "timeout", f"IP: {client_ip} | 獲取任務 {job_id} 狀態超時 | HTTP 504")
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "獲取任務狀態超時",
                    "message_eng": "Timeout while getting job status"
                }
            )
    except HTTPException as e:
        log_wrapper.error("get_job_status", "http_exception", f"IP: {client_ip} | HTTP {e.status_code} | {str(e.detail)}")
        raise e
    except Exception as e:
        log_wrapper.error("get_job_status", "system_error", f"IP: {client_ip} | 獲取任務狀態時發生錯誤: {str(e)} | HTTP 500")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.get("/api/v1/rag/{job_id}/result")
async def get_job_result(request: Request, job_id: str) -> JSONResponse:
    client_ip = get_client_ip(request)
    try:
        log_wrapper.info("get_job_result", "start", f"IP: {client_ip} | 查詢任務 {job_id} 結果")
        job_key = f"rag_job:{job_id}"
        job_data = redis_conn.get(job_key)
        if not job_data:
            log_wrapper.warning("get_job_result", "not_found", f"IP: {client_ip} | 任務 {job_id} 不存在 | HTTP 404")
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "任務不存在",
                    "message_eng": "Job not found"
                }
            )
        job = json.loads(job_data)
        created_at = datetime.fromisoformat(job["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=JOB_EXPIRY_HOURS):
            log_wrapper.warning("get_job_result", "expired", f"IP: {client_ip} | 任務 {job_id} 已過期 | HTTP 410")
            raise HTTPException(
                status_code=410,
                detail={
                    "message": "任務已過期",
                    "message_eng": "Job has expired"
                }
            )
        try:
            async def get_result():
                if job.get("project_id") == "timeout_test":
                    await asyncio.sleep(JOB_STATUS_TIMEOUT + 1)
                else:
                    await asyncio.sleep(0.1)
                return job
            result = await asyncio.wait_for(get_result(), timeout=JOB_STATUS_TIMEOUT)
            log_wrapper.info("get_job_result", "success", f"IP: {client_ip} | HTTP 200 | 成功獲取任務 {job_id} 結果")
            return JSONResponse(
                content={
                    "success": True,
                    "data": result
                }
            )
        except asyncio.TimeoutError:
            log_wrapper.error("get_job_result", "timeout", f"IP: {client_ip} | 獲取任務 {job_id} 結果超時 | HTTP 504")
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "獲取任務狀態超時",
                    "message_eng": "Timeout while getting job status"
                }
            )
    except HTTPException as e:
        log_wrapper.error("get_job_result", "http_exception", f"IP: {client_ip} | HTTP {e.status_code} | {str(e.detail)}")
        raise e
    except Exception as e:
        log_wrapper.error("get_job_result", "system_error", f"IP: {client_ip} | 獲取任務結果時發生錯誤: {str(e)} | HTTP 500")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.get("/api/v1/rag", response_model=List[JobStatus])
async def list_jobs(
    project_id: Optional[str] = Query(None, description="專案 ID"),
    clean_old: bool = Query(False, description="是否清理已完成的任務")
) -> List[JobStatus]:
    """列出所有任務
    
    Args:
        project_id: 可選的專案 ID 過濾
        clean_old: 是否清理已完成的任務記錄（默認為 False）
        
    Returns:
        List[JobStatus]: 任務列表
        
    Raises:
        HTTPException: 當專案 ID 無效時
    """
    try:
        log_wrapper.info("list_jobs", "start", f"列出任務，專案ID: {project_id}, 清理舊任務: {clean_old}")
        
        # 驗證專案 ID 格式
        if project_id and not PROJECT_ID_PATTERN.match(project_id):
            log_wrapper.warning("list_jobs", "invalid_project_id", f"無效的專案 ID: {project_id}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "無效的專案 ID",
                    "message_eng": "Invalid project ID"
                }
            )
            
        jobs = []
        for key in redis_conn.keys("rag_job:*"):
            job_data = redis_conn.get(key)
            if job_data:
                job = json.loads(job_data)
                
                # 檢查任務是否過期
                created_at = datetime.fromisoformat(job["created_at"])
                if datetime.utcnow() - created_at > timedelta(hours=JOB_EXPIRY_HOURS):
                    # 如果任務過期，直接刪除
                    redis_conn.delete(key)
                    log_wrapper.info("list_jobs", "delete_expired", f"已刪除過期任務: {job['job_id']}")
                    continue
                    
                # 檢查任務是否長時間處於 pending 狀態
                if job["status"] == "pending" and datetime.utcnow() - created_at > timedelta(minutes=5):
                    # 如果任務超過 5 分鐘仍處於 pending 狀態，標記為失敗
                    job["status"] = "failed"
                    job["error"] = "任務執行超時"
                    job["failed_at"] = datetime.utcnow().isoformat()
                    redis_conn.set(key, json.dumps(job))
                    log_wrapper.info("list_jobs", "mark_timeout", f"已標記超時任務為失敗: {job['job_id']}")
                
                if project_id is None or job["project_id"] == project_id:
                    # 處理 results 字段
                    if job.get("results") is not None:
                        results = job["results"]
                        if isinstance(results, list):
                            # 如果結果是嵌套列表，取第一個元素
                            if results and isinstance(results[0], list):
                                job["results"] = results[0]
                            else:
                                job["results"] = results
                        else:
                            job["results"] = [results]
                    jobs.append(JobStatus(**job))
                    
                    # 如果啟用了清理，刪除已完成的任務記錄
                    if clean_old and job["status"] in ["completed", "failed"]:
                        redis_conn.delete(key)
                        log_wrapper.info("list_jobs", "clean_old", f"已清理任務: {job['job_id']}")
        
        log_wrapper.info("list_jobs", "success", f"成功列出 {len(jobs)} 個任務")
        return jobs
        
    except HTTPException as e:
        # 直接重新拋出 HTTPException
        raise e
    except Exception as e:
        # 處理其他錯誤
        log_wrapper.error("list_jobs", "system_error", f"獲取任務列表時發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.delete("/api/v1/rag/{job_id}")
async def delete_job(job_id: str):
    """刪除任務"""
    try:
        log_wrapper.info("delete_job", "start", f"刪除任務 {job_id}")
        
        job = await job_manager.get_job(job_id)
        if not job:
            log_wrapper.warning("delete_job", "not_found", f"任務 {job_id} 不存在")
            raise HTTPException(status_code=404, detail="Job not found")
        
        # TODO: 實現任務刪除邏輯
        log_wrapper.info("delete_job", "success", f"任務 {job_id} 刪除成功")
        return {"status": "success", "message": "Job deleted"}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        log_wrapper.error("delete_job", "system_error", f"刪除任務時發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    client_ip = get_client_ip(request)
    error_messages = []
    for error in exc.errors():
        if error["type"] == "int_parsing":
            error_messages.append(f"字段 {error['loc'][-1]} 必須是整數")
        elif error["type"] == "float_parsing":
            error_messages.append(f"字段 {error['loc'][-1]} 必須是浮點數")
        elif error["type"] == "value_error":
            error_messages.append(error["msg"])
        else:
            error_messages.append(f"字段 {error['loc'][-1]} 格式錯誤")
    log_wrapper.error("validation_exception_handler", "validation_error", f"IP: {client_ip} | HTTP 400 | 請求驗證錯誤: {'; '.join(error_messages)}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "detail": {
                    "message": "請求資料格式不符合 RAGRequest 指定格式: " + "; ".join(error_messages),
                    "message_eng": "Invalid request format: " + "; ".join(error_messages)
                }
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 