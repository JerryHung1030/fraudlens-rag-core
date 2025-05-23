import os
import sys
import uvicorn
import subprocess
import signal
import time
from typing import Optional
import threading
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 設置環境變數
os.environ["PYTHONPATH"] = str(project_root)
os.environ["RAG_CONFIG_PATH"] = str(project_root / "config")

def print_output(pipe, prefix):
    """打印進程輸出"""
    for line in iter(pipe.readline, ''):
        print(f"{prefix}: {line.strip()}")
    pipe.close()

def start_rq_worker():
    """啟動 RQ worker"""
    worker_path = str(project_root / "src" / "interfaces" / "api")
    
    # 設置環境變數
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["RAG_CONFIG_PATH"] = str(project_root / "config")
    
    worker_cmd = [
        "rq", "worker",
        "--path", worker_path,
        "--name", "rag_worker",
        "--verbose",
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
            raise RuntimeError("Worker failed to start")
            
        print("RQ worker started successfully")
        return worker_process
    except Exception as e:
        print(f"Failed to start RQ worker: {e}")
        raise

def check_redis_connection():
    """檢查 Redis 連接"""
    try:
        import redis
        redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
        redis_client.ping()
        return True
    except Exception as e:
        print(f"Redis 連接失敗: {str(e)}")
        return False

def cleanup(worker_process: Optional[subprocess.Popen]):
    """清理進程"""
    if worker_process:
        worker_process.terminate()
        worker_process.wait()

def main():
    # 檢查 Redis 連接
    if not check_redis_connection():
        print("請確保 Redis 服務已啟動")
        sys.exit(1)

    # 啟動 RQ worker
    worker_process = start_rq_worker()
    
    try:
        # 啟動 FastAPI 應用
        uvicorn.run(
            "src.interfaces.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except KeyboardInterrupt:
        print("\n正在關閉服務...")
    finally:
        # 清理進程
        cleanup(worker_process)
        print("服務已關閉")

if __name__ == "__main__":
    main() 