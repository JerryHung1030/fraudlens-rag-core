import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from .models import JobStatus
from utils.logging import log_wrapper

class JobManager:
    """RAG 任務管理器"""
    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()
        log_wrapper.info("JobManager", "__init__", "JobManager 初始化完成")

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
        log_wrapper.info("JobManager", "create_job", f"創建新任務 {job_id}，專案ID: {project_id}")
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
                old_status = job.status
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
                
                log_wrapper.info("JobManager", "update_job", f"更新任務 {job_id} 狀態: {old_status} -> {status}")
            else:
                log_wrapper.warning("JobManager", "update_job", f"嘗試更新不存在的任務 {job_id}")

    async def get_job(self, job_id: str) -> Optional[JobStatus]:
        """獲取任務狀態"""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                log_wrapper.debug("JobManager", "get_job", f"獲取任務 {job_id} 狀態: {job.status}")
            else:
                log_wrapper.warning("JobManager", "get_job", f"嘗試獲取不存在的任務 {job_id}")
            return job

    async def list_jobs(self, project_id: Optional[str] = None) -> list[JobStatus]:
        """列出所有任務"""
        async with self._lock:
            if project_id:
                jobs = [job for job in self._jobs.values() if job.project_id == project_id]
                log_wrapper.info("JobManager", "list_jobs", f"列出專案 {project_id} 的任務，共 {len(jobs)} 個")
            else:
                jobs = list(self._jobs.values())
                log_wrapper.info("JobManager", "list_jobs", f"列出所有任務，共 {len(jobs)} 個")
            return jobs

# 全域任務管理器實例
job_manager = JobManager() 