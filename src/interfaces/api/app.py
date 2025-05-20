import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

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

# --- 專案內部 import ----------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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

async def process_rag_job(job_id: str, request: RAGRequest):
    """背景處理 RAG 任務"""
    try:
        # 更新任務狀態為處理中
        await job_manager.update_job(job_id, "processing")
        
        # 準備任務參數
        job_payload = {
            "job_id": job_id,
            "project_id": request.project_id,
            "scenario": request.scenario,
            "input_data": request.input_data,
            "reference_data": request.reference_data,
            "callback_url": request.callback_url
        }
        
        # 執行 RAG 任務
        result = await rag_runner.run_job(job_payload)
        
        # 更新任務狀態為完成
        await job_manager.update_job(
            job_id,
            "completed",
            results=result.get("results"),
            progress=100.0
        )
        
    except Exception as e:
        # 更新任務狀態為失敗
        await job_manager.update_job(
            job_id,
            "failed",
            error=str(e)
        )

@app.post("/api/v1/rag", response_model=RAGResponse)
async def create_rag_job(
    request: RAGRequest,
    background_tasks: BackgroundTasks
) -> RAGResponse:
    """創建新的 RAG 任務"""
    try:
        # 創建新任務
        job_id = await job_manager.create_job(request.project_id)
        
        # 加入背景任務
        background_tasks.add_task(process_rag_job, job_id, request)
        
        # 返回初始回應
        return RAGResponse(
            job_id=job_id,
            project_id=request.project_id,
            status="pending"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/rag/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """獲取任務狀態"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/v1/rag", response_model=List[JobStatus])
async def list_jobs(project_id: Optional[str] = None) -> List[JobStatus]:
    """列出所有任務"""
    return await job_manager.list_jobs(project_id)

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