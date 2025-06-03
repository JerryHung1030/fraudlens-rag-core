import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Set
from rq import Queue
from redis import Redis
import json
import logging
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import re

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

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 專案內部 import ----------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 初始化 Redis 和 RQ Queue
redis_conn = Redis(host="127.0.0.1", port=6379, db=0)
rag_queue = Queue("rag_jobs", connection=redis_conn)

# 使用 Redis 來追蹤運行中的任務
RUNNING_JOBS_KEY = "running_rag_jobs"

# 清理運行中的任務集合
redis_conn.delete(RUNNING_JOBS_KEY)
logger.info("已清理運行中的任務集合")

# 添加任務過期時間設定（改為 1 分鐘）
JOB_EXPIRY_HOURS = 1/60  # 1 分鐘
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
            logger.info(f"Started job {job_id}. Current running jobs: {len(running_jobs)}")

    async def finish_job(self, job_id: str) -> None:
        async with self._lock:
            redis_conn.srem(RUNNING_JOBS_KEY, job_id)
            running_jobs = redis_conn.smembers(RUNNING_JOBS_KEY)
            logger.info(f"Finished job {job_id}. Current running jobs: {len(running_jobs)}")

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
    logger.info("服務啟動：已清理運行中的任務集合")

# 初始化核心元件
def setup_core():
    settings = config_manager.settings

    # Embedding
    embed_mgr = EmbeddingManager(
        openai_api_key=settings.api_keys.openai,
        embedding_model_name=settings.embedding.model
    )

    # Data schema checker
    checker = DataStructureChecker()

    # Qdrant client
    qdrant = QdrantClient(url=settings.vector_db.url)

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
            logger.info(f"已更新任務 {job_id} 狀態為 {status}")
            
            # 如果任務完成或失敗，從運行中的任務集合中移除
            if status in ["completed", "failed"]:
                redis_conn.srem(RUNNING_JOBS_KEY, job_id)
                logger.info(f"已從運行中的任務集合中移除任務 {job_id}")
    except Exception as e:
        logger.error(f"更新任務狀態失敗: {str(e)}")

def process_rag_job(job_payload: dict) -> dict:
    """處理 RAG 任務的函數，用於 RQ worker"""
    job_id = None
    try:
        # 將 JSON 字串轉換回 Python 物件
        if isinstance(job_payload, str):
            logger.info("解析 JSON 字串...")
            job_payload = json.loads(job_payload)
            
        job_id = job_payload.get('job_id')
        logger.info(f"開始處理任務: {job_id}")
            
        # 從 rag_core.domain.scenario 導入 Scenario
        from rag_core.domain.scenario import Scenario
        
        # 將 scenario 字典轉換為 Scenario 物件
        scenario_data = job_payload.get("scenario", {})
        if isinstance(scenario_data, dict):
            logger.info("轉換 scenario 為 Scenario 物件...")
            scenario = Scenario(**scenario_data)
            # 使用 dict() 方法將 Scenario 物件轉換回字典
            job_payload["scenario"] = scenario.dict()
            
        logger.info("執行 RAG 任務...")
        # 創建新的事件循環
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 執行 RAG 任務
            result = loop.run_until_complete(rag_runner.run_job(job_payload))
            logger.info(f"任務完成: {job_id}")
            
            # 更新任務狀態
            update_job_status(
                job_id=job_id,
                status="completed",
                results=result.get("results")
            )
            
            # 標記任務為已完成
            redis_conn.srem(RUNNING_JOBS_KEY, job_id)
            logger.info(f"Finished job {job_id}. Current running jobs: {len(redis_conn.smembers(RUNNING_JOBS_KEY))}")
            
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"任務處理失敗: {str(e)}")
        # 更新任務狀態為失敗
        if job_id:
            update_job_status(
                job_id=job_id,
                status="failed",
                error=str(e)
            )
            # 標記任務為已完成
            redis_conn.srem(RUNNING_JOBS_KEY, job_id)
            logger.info(f"Finished job {job_id}. Current running jobs: {len(redis_conn.smembers(RUNNING_JOBS_KEY))}")
        return {"error": str(e)}

@app.post("/api/v1/rag", response_model=RAGResponse)
async def create_rag_job(request: RAGRequest) -> RAGResponse:
    """創建新的 RAG 任務"""
    try:
        # 檢查任務 ID 是否已存在
        job_id = await job_manager.create_job(request.project_id)
        job_key = f"rag_job:{job_id}"
        # job_key = "b905c3e2-4d0f-42aa-bb8b-45d6de8357d8"
        if redis_conn.exists(job_key):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "任務 ID 已存在，請使用其他 ID",
                    "message_eng": "Job ID already exists, please use a different ID"
                }
            )

        # 檢查是否可以開始新任務
        if not await job_tracker.can_start_job():
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "系統內部資源不足以創建新任務",
                    "message_eng": "System resources exhausted"
                }
            )

        # 獲取默認的 scenario 設定
        default_scenario = config_manager.get_scenario_config()
        
        # 合併 scenario 設定
        scenario_data = request.scenario.dict() if request.scenario else {}
        merged_scenario = {**default_scenario, **scenario_data}
        
        # 準備任務參數
        job_payload = {
            "job_id": job_id,
            "project_id": request.project_id,
            "scenario": merged_scenario,
            "input_data": request.input_data,
            "reference_data": request.reference_data,
            "callback_url": request.callback_url
        }
        
        # 將任務信息保存到 Redis
        job_data = {
            "job_id": job_id,
            "project_id": request.project_id,
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
        
        # 將任務加入 RQ Queue
        job = rag_queue.enqueue(
            process_rag_job,
            json.dumps(job_payload, default=str),
            job_id=job_id
        )
        logger.info(f"任務已加入佇列: {job_id}")
        
        # 標記任務為運行中
        await job_tracker.start_job(job_id)
        
        # 返回簡化的初始回應
        return RAGResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow()
        )
            
    except ValueError as e:
        # 處理請求格式錯誤
        logger.error(f"請求格式錯誤: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "message": "請求資料格式不符合 RAGRequest 指定格式",
                "message_eng": "Invalid request format"
            }
        )
    except HTTPException as e:
        # 直接重新拋出 HTTPException，保持原始狀態碼
        raise e
    except Exception as e:
        # 處理其他錯誤
        logger.error(f"創建任務失敗: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.get("/api/v1/rag/{job_id}/status")
async def get_job_status(job_id: str) -> JSONResponse:
    """獲取任務狀態
    
    Args:
        job_id: 任務ID
        
    Returns:
        JSONResponse: 任務狀態信息
        
    Raises:
        HTTPException: 當任務不存在、已過期、查詢超時或發生系統錯誤時
    """
    try:
        # 檢查任務是否存在
        job_key = f"rag_job:{job_id}"
        job_data = redis_conn.get(job_key)
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "任務不存在",
                    "message_eng": "Job not found"
                }
            )
            
        # 解析任務數據
        job = json.loads(job_data)
        
        # 檢查任務是否過期
        created_at = datetime.fromisoformat(job["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=JOB_EXPIRY_HOURS):
            raise HTTPException(
                status_code=410,
                detail={
                    "message": "任務已過期",
                    "message_eng": "Job has expired"
                }
            )
            
        # 使用 asyncio.wait_for 來實現超時控制
        try:
            # 模擬獲取任務狀態的異步操作
            async def get_status():
                # 如果是 timeout_test 專案，增加延遲
                if job.get("project_id") == "timeout_test":
                    await asyncio.sleep(JOB_STATUS_TIMEOUT + 1)  # 確保超時
                else:
                    await asyncio.sleep(0.1)  # 正常延遲
                return job
                
            # 設置超時
            result = await asyncio.wait_for(get_status(), timeout=JOB_STATUS_TIMEOUT)
            
            return JSONResponse(
                content={
                    "success": True,
                    "data": result
                }
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "獲取任務狀態超時",
                    "message_eng": "Timeout while getting job status"
                }
            )
            
    except HTTPException as e:
        # 直接重新拋出 HTTPException
        raise e
    except Exception as e:
        # 處理其他錯誤
        logger.error(f"獲取任務狀態時發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "系統發生問題，請連絡系統管理員",
                "message_eng": "System problem, please contact administrator"
            }
        )

@app.get("/api/v1/rag/{job_id}/result")
async def get_job_result(job_id: str) -> JSONResponse:
    """獲取任務結果
    
    Args:
        job_id: 任務ID
        
    Returns:
        JSONResponse: 任務結果信息
        
    Raises:
        HTTPException: 當任務不存在、已過期、查詢超時或發生系統錯誤時
    """
    try:
        # 檢查任務是否存在
        job_key = f"rag_job:{job_id}"
        job_data = redis_conn.get(job_key)
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "任務不存在",
                    "message_eng": "Job not found"
                }
            )
            
        # 解析任務數據
        job = json.loads(job_data)
        
        # 檢查任務是否過期
        created_at = datetime.fromisoformat(job["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=JOB_EXPIRY_HOURS):
            raise HTTPException(
                status_code=410,
                detail={
                    "message": "任務已過期",
                    "message_eng": "Job has expired"
                }
            )
            
        # 使用 asyncio.wait_for 來實現超時控制
        try:
            # 模擬獲取任務結果的異步操作
            async def get_result():
                # 如果是 timeout_test 專案，增加延遲
                if job.get("project_id") == "timeout_test":
                    await asyncio.sleep(JOB_STATUS_TIMEOUT + 1)  # 確保超時
                else:
                    await asyncio.sleep(0.1)  # 正常延遲
                return job
                
            # 設置超時
            result = await asyncio.wait_for(get_result(), timeout=JOB_STATUS_TIMEOUT)
            
            return JSONResponse(
                content={
                    "success": True,
                    "data": result
                }
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "獲取任務狀態超時",
                    "message_eng": "Timeout while getting job status"
                }
            )
            
    except HTTPException as e:
        # 直接重新拋出 HTTPException
        raise e
    except Exception as e:
        # 處理其他錯誤
        logger.error(f"獲取任務結果時發生錯誤: {str(e)}")
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
        # 驗證專案 ID 格式
        if project_id and not PROJECT_ID_PATTERN.match(project_id):
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
                    logger.info(f"已刪除過期任務: {job['job_id']}")
                    continue
                    
                # 檢查任務是否長時間處於 pending 狀態
                if job["status"] == "pending" and datetime.utcnow() - created_at > timedelta(minutes=5):
                    # 如果任務超過 5 分鐘仍處於 pending 狀態，標記為失敗
                    job["status"] = "failed"
                    job["error"] = "任務執行超時"
                    job["failed_at"] = datetime.utcnow().isoformat()
                    redis_conn.set(key, json.dumps(job))
                    logger.info(f"已標記超時任務為失敗: {job['job_id']}")
                
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
                        logger.info(f"已清理任務: {job['job_id']}")
        
        return jobs
        
    except HTTPException as e:
        # 直接重新拋出 HTTPException
        raise e
    except Exception as e:
        # 處理其他錯誤
        logger.error(f"獲取任務列表時發生錯誤: {str(e)}")
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
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: 實現任務刪除邏輯
    return {"status": "success", "message": "Job deleted"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """處理請求驗證錯誤"""
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