"""
===============================================================================
    Module Name: run_api.py
    Description: Entrypoint for starting FastAPI server and RQ worker.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
import os
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 設置環境變數
os.environ["PYTHONPATH"] = str(project_root) + ":" + str(project_root / "src")
os.environ["RAG_CONFIG_PATH"] = str(project_root / "config")

import uvicorn
import subprocess
import signal
import time
from typing import Optional
import threading
from utils.logging import log_wrapper

def print_output(pipe, prefix):
    """打印進程輸出"""
    for line in iter(pipe.readline, ''):
        print(f"{prefix}: {line.strip()}")
    pipe.close()

def start_rq_worker():
    """啟動 RQ worker"""
    log_wrapper.info("run_api", "start_rq_worker", "開始啟動 RQ worker")
    worker_path = str(project_root / "src" / "interfaces" / "api")
    
    # 設置環境變數
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + ":" + str(project_root / "src")
    env["RAG_CONFIG_PATH"] = str(project_root / "config")
    
    worker_cmd = [
        "rq", "worker",
        "--path", worker_path,
        "--name", "rag_worker",
        "--verbose",
        "--url", "redis://localhost:6379/0",
        "rag_jobs"
    ]
    
    try:
        # 啟動 worker 進程
        worker_process = subprocess.Popen(
            worker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # 啟動輸出監控線程
        threading.Thread(target=print_output, args=(worker_process.stdout, "RQ Worker"), daemon=True).start()
        threading.Thread(target=print_output, args=(worker_process.stderr, "RQ Worker Error"), daemon=True).start()
        
        # 檢查 worker 是否成功啟動
        if worker_process.poll() is not None:
            log_wrapper.error("run_api", "start_rq_worker", "Worker 啟動失敗")
            raise RuntimeError("Worker failed to start")
            
        log_wrapper.info("run_api", "start_rq_worker", "RQ worker 啟動成功")
        print("RQ worker started successfully")
        return worker_process
    except Exception as e:
        log_wrapper.error("run_api", "start_rq_worker", f"啟動 RQ worker 失敗: {str(e)}")
        print(f"Failed to start RQ worker: {e}")
        raise

def check_redis_connection():
    """檢查 Redis 連接"""
    try:
        import redis
        redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
        redis_client.ping()
        log_wrapper.info("run_api", "check_redis_connection", "Redis 連接檢查成功")
        return True
    except Exception as e:
        log_wrapper.error("run_api", "check_redis_connection", f"Redis 連接失敗: {str(e)}")
        print(f"Redis 連接失敗: {str(e)}")
        return False

def cleanup(worker_process: Optional[subprocess.Popen]):
    """清理進程"""
    if worker_process:
        log_wrapper.info("run_api", "cleanup", "開始清理 worker 進程")
        worker_process.terminate()
        worker_process.wait()
        log_wrapper.info("run_api", "cleanup", "worker 進程清理完成")

def main():
    log_wrapper.info("run_api", "main", "開始啟動 API 服務")
    
    # 檢查 Redis 連接
    if not check_redis_connection():
        log_wrapper.error("run_api", "main", "Redis 服務未啟動，無法繼續")
        print("請確保 Redis 服務已啟動")
        sys.exit(1)

    # 啟動 RQ worker
    worker_process = start_rq_worker()
    
    try:
        # 啟動 FastAPI 應用
        log_wrapper.info("run_api", "main", "開始啟動 FastAPI 應用")
        uvicorn.run(
            "src.interfaces.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except KeyboardInterrupt:
        log_wrapper.info("run_api", "main", "收到中斷信號，正在關閉服務")
        print("\n正在關閉服務...")
    finally:
        # 清理進程
        cleanup(worker_process)
        log_wrapper.info("run_api", "main", "API 服務已關閉")
        print("服務已關閉")

if __name__ == "__main__":
    main() 