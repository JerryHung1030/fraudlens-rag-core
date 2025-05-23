import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from rq import Queue
from redis import Redis
import json
import logging
from datetime import datetime

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
    except Exception as e:
        logger.error(f"更新任務狀態失敗: {str(e)}")

def process_rag_job(job_payload: dict) -> dict:
    """處理 RAG 任務的函數，用於 RQ worker"""
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
        return {"error": str(e)}

@app.post("/api/v1/rag", response_model=RAGResponse)
async def create_rag_job(request: RAGRequest) -> RAGResponse:
    """創建新的 RAG 任務"""
    try:
        # 創建新任務
        job_id = await job_manager.create_job(request.project_id)
        logger.info(f"創建新任務: {job_id}")
        
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
        job_key = f"rag_job:{job_id}"
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
        
        # 返回初始回應
        return RAGResponse(
            job_id=job_id,
            project_id=request.project_id,
            status="pending"
        )
        
    except Exception as e:
        logger.error(f"創建任務失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/rag/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """獲取任務狀態"""
    job_key = f"rag_job:{job_id}"
    job_data = redis_conn.get(job_key)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**json.loads(job_data))

@app.get("/api/v1/rag", response_model=List[JobStatus])
async def list_jobs(project_id: Optional[str] = None, clean_old: bool = False) -> List[JobStatus]:
    """列出所有任務
    
    Args:
        project_id: 可選的專案 ID 過濾
        clean_old: 是否清理已完成的任務記錄（默認為 False）
    """
    jobs = []
    for key in redis_conn.keys("rag_job:*"):
        job_data = redis_conn.get(key)
        if job_data:
            job = json.loads(job_data)
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

@app.delete("/api/v1/rag/{job_id}")
async def delete_job(job_id: str):
    """刪除任務"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: 實現任務刪除邏輯
    return {"status": "success", "message": "Job deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 