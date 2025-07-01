"""
===============================================================================
    Module Name: models.py
    Description: Pydantic models for RAG API requests, responses, and job status.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, validator
from datetime import datetime


class Scenario(BaseModel):
    direction: Optional[str] = "forward"
    role_desc: Optional[str] = None
    reference_desc: Optional[str] = None
    input_desc: Optional[str] = None
    rag_k: Optional[int] = 3
    rag_k_forward: Optional[int] = None
    rag_k_reverse: Optional[int] = None
    cof_threshold: Optional[float] = 0.5
    scoring_rule: Optional[str] = None
    llm_name: Optional[str] = "openai"
    reference_depth: Optional[int] = 1
    input_depth: Optional[int] = 1
    chunk_size: Optional[int] = 0
    max_prompt_tokens: Optional[int] = 8000

    @validator('direction')
    def validate_direction(cls, v):
        if v not in ["forward", "reverse", "both"]:
            raise ValueError("direction must be one of: forward, reverse, both")
        return v

    @validator('rag_k', 'rag_k_forward', 'rag_k_reverse')
    def validate_rag_k(cls, v):
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError("rag_k must be a positive integer")
        return v

    @validator('cof_threshold')
    def validate_cof_threshold(cls, v):
        if v is not None and (not isinstance(v, float) or v < 0 or v > 1):
            raise ValueError("cof_threshold must be a float between 0 and 1")
        return v


class RAGRequest(BaseModel):
    """RAG 任務請求模型"""
    project_id: str
    scenario: Optional[Scenario] = None
    input_data: Dict[str, Any]
    reference_data: Dict[str, Any]
    callback_url: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class RAGResponse(BaseModel):
    """RAG 任務回應模型"""
    job_id: str
    status: str
    created_at: datetime


class JobStatus(BaseModel):
    """任務狀態模型"""
    job_id: str
    project_id: str
    status: str
    progress: Optional[float] = None
    results: Optional[List[Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
