from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class RAGRequest(BaseModel):
    """RAG 任務請求模型"""
    project_id: str = Field(..., description="專案ID")
    scenario: Dict[str, Any] = Field(..., description="場景設定")
    input_data: Dict[str, Any] = Field(..., description="輸入數據")
    reference_data: Dict[str, Any] = Field(..., description="參考數據")
    callback_url: Optional[str] = Field(None, description="回調URL")

class RAGResponse(BaseModel):
    """RAG 任務回應模型"""
    job_id: str = Field(..., description="任務ID")
    project_id: str = Field(..., description="專案ID")
    status: str = Field(..., description="任務狀態")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="處理結果")
    error: Optional[str] = Field(None, description="錯誤訊息")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    failed_at: Optional[datetime] = Field(None, description="失敗時間")

class JobStatus(BaseModel):
    """任務狀態模型"""
    job_id: str = Field(..., description="任務ID")
    project_id: str = Field(..., description="專案ID")
    status: str = Field(..., description="任務狀態")
    progress: Optional[float] = Field(None, description="進度百分比")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="處理結果")
    error: Optional[str] = Field(None, description="錯誤訊息")
    created_at: datetime = Field(..., description="建立時間")
    updated_at: datetime = Field(..., description="最後更新時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    failed_at: Optional[datetime] = Field(None, description="失敗時間") 