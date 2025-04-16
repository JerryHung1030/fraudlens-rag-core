# src/managers/job_queue_manager.py
import uuid


class JobQueueManager:
    """
    可選: 若要做背景執行 (e.g. Celery, RQ)
    這裡做PoC，暫時不實作
    """
    def __init__(self):
        self.tasks = {}

    def enqueue_task(self, task_name: str, payload: any) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {"status": "PENDING", "payload": payload}
        return task_id

    def check_task_status(self, task_id: str) -> str:
        return self.tasks.get(task_id, {}).get("status", "NOT_FOUND")

    def fetch_result(self, task_id: str):
        return self.tasks.get(task_id, {}).get("result", None)
