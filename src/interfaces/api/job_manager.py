import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from .models import JobStatus

class JobManager:
    """RAG 任務管理器"""
    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, project_id: str) -> str:
        """創建新任務"""
        job_id = str(uuid.uuid4())
        async with self._lock:
            self._jobs[job_id] = JobStatus(
                job_id=job_id,
                project_id=project_id,
                status="pending",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        return job_id

    async def update_job(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        results: Optional[list] = None,
        error: Optional[str] = None
    ) -> None:
        """更新任務狀態"""
        async with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = status
                job.updated_at = datetime.utcnow()
                
                if progress is not None:
                    job.progress = progress
                if results is not None:
                    job.results = results
                if error is not None:
                    job.error = error
                
                if status == "completed":
                    job.completed_at = datetime.utcnow()
                elif status == "failed":
                    job.failed_at = datetime.utcnow()

    async def get_job(self, job_id: str) -> Optional[JobStatus]:
        """獲取任務狀態"""
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_jobs(self, project_id: Optional[str] = None) -> list[JobStatus]:
        """列出所有任務"""
        async with self._lock:
            if project_id:
                return [job for job in self._jobs.values() if job.project_id == project_id]
            return list(self._jobs.values())

# 全域任務管理器實例
job_manager = JobManager() 