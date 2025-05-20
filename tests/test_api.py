import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"

async def create_rag_job(session, project_id, input_text, ref_text):
    """創建單個 RAG 任務"""
    test_data = {
        "project_id": project_id,
        "scenario": {
            "direction": "both",
            "rag_k": 3,
            "reference_depth": 1,
            "input_depth": 1,
            "chunk_size": 0
        },
        "input_data": {
            "level1": [
                {
                    "sid": f"input_{project_id}",
                    "text": input_text,
                    "metadata": {"type": "test", "timestamp": datetime.now().isoformat()}
                }
            ]
        },
        "reference_data": {
            "level1": [
                {
                    "sid": f"ref_{project_id}",
                    "text": ref_text,
                    "metadata": {"type": "test", "timestamp": datetime.now().isoformat()}
                }
            ]
        }
    }
    
    async with session.post(f"{BASE_URL}/rag", json=test_data) as response:
        return await response.json()

async def check_job_status(session, job_id):
    """檢查任務狀態"""
    async with session.get(f"{BASE_URL}/rag/{job_id}") as response:
        return await response.json()

async def wait_for_job_completion(session, job_id, max_retries=10):
    """等待任務完成"""
    for _ in range(max_retries):
        status = await check_job_status(session, job_id)
        if status["status"] in ["completed", "failed"]:
            return status
        await asyncio.sleep(2)
    return None

async def test_concurrent_rag():
    """測試並發 RAG 請求"""
    # 準備多個測試案例
    test_cases = [
        {
            "project_id": "project_1",
            "input_text": "這是第一個測試輸入文本",
            "ref_text": "這是第一個測試參考文本"
        },
        {
            "project_id": "project_2",
            "input_text": "這是第二個測試輸入文本",
            "ref_text": "這是第二個測試參考文本"
        },
        {
            "project_id": "project_3",
            "input_text": "這是第三個測試輸入文本",
            "ref_text": "這是第三個測試參考文本"
        }
    ]
    
    print("\n開始並發測試...")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # 1. 並發創建多個任務
        print("\n1. 創建多個 RAG 任務:")
        create_tasks = [
            create_rag_job(session, tc["project_id"], tc["input_text"], tc["ref_text"])
            for tc in test_cases
        ]
        job_results = await asyncio.gather(*create_tasks)
        
        for result in job_results:
            print(f"創建任務回應: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 2. 並發等待所有任務完成
        print("\n2. 等待所有任務完成:")
        job_ids = [result["job_id"] for result in job_results]
        wait_tasks = [wait_for_job_completion(session, job_id) for job_id in job_ids]
        completed_results = await asyncio.gather(*wait_tasks)
        
        for result in completed_results:
            print(f"任務完成狀態: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 3. 列出所有任務
        print("\n3. 列出所有任務:")
        async with session.get(f"{BASE_URL}/rag") as response:
            all_jobs = await response.json()
            print(json.dumps(all_jobs, indent=2, ensure_ascii=False))
    
    end_time = time.time()
    print(f"\n總執行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    asyncio.run(test_concurrent_rag()) 