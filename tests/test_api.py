import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from config.settings import config_manager

BASE_URL = "http://localhost:8000/api/v1"

async def create_rag_job(session, project_id, input_text, ref_text, scenario=None):
    """創建單個 RAG 任務"""
    # 從 config_manager 獲取默認設定
    default_scenario = config_manager.get_scenario_config()
    
    test_data = {
        "project_id": project_id,
        "scenario": scenario or default_scenario,
        "input_data": {
            "level1": [
                {
                    "sid": f"input_{project_id}",
                    "text": input_text,
                    # "metadata": {"type": "test", "project": project_id}
                }
            ]
        },
        "reference_data": {
            "level1": [
                {
                    "sid": f"ref_{project_id}",
                    "text": ref_text,
                    # "metadata": {"type": "test", "project": project_id}
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

async def wait_for_job_completion(session, job_id, max_retries=30, retry_interval=2):
    """等待任務完成"""
    for i in range(max_retries):
        try:
            status = await check_job_status(session, job_id)
            print(f"檢查任務 {job_id} 狀態 (嘗試 {i+1}/{max_retries}): {json.dumps(status, indent=2, ensure_ascii=False)}")
            
            if status and status.get("status") in ["completed", "failed"]:
                return status
                
            await asyncio.sleep(retry_interval)
        except Exception as e:
            print(f"檢查任務狀態時發生錯誤: {str(e)}")
            await asyncio.sleep(retry_interval)
            
    print(f"任務 {job_id} 在 {max_retries} 次嘗試後仍未完成")
    return None

async def test_concurrent_rag():
    """測試並發 RAG 請求"""
    # 準備多個測試案例
    test_cases = [
        {
            "project_id": "project_1",
            "input_text": "這是第一個測試輸入文本",
            "ref_text": "這是第一個測試參考文本",
            "use_default_scenario": True,  # 使用默認設定
            "scenario": None
        },
        {
            "project_id": "project_2",
            "input_text": "這是第二個測試輸入文本",
            "ref_text": "這是第二個測試參考文本",
            "use_default_scenario": False,  # 使用部分自定義設定
            "scenario": {
                "direction": "both",
                "rag_k": 3,
                "cof_threshold": 0.7,
                "role_desc": "你是RAG助手，負責比較文本相似度",
                "reference_desc": "Reference 為參考文本，用於比對",
                "input_desc": "Input 為輸入文本，需要與參考文本進行比對",
                "scoring_rule": "請根據文本相似度給出信心分數，並標記出相似的文本片段"
            }
        },
        {
            "project_id": "project_3",
            "input_text": "這是第三個測試輸入文本",
            "ref_text": "這是第三個測試參考文本",
            "use_default_scenario": False,  # 使用完整自定義設定
            "scenario": {
                "direction": "both",
                "role_desc": "你是RAG助手，負責比較文本相似度",
                "reference_desc": "Reference 為參考文本，用於比對",
                "input_desc": "Input 為輸入文本，需要與參考文本進行比對",
                "rag_k": 3,
                "rag_k_forward": 3,
                "rag_k_reverse": 3,
                "cof_threshold": 0.5,
                "scoring_rule": "請根據文本相似度給出信心分數，並標記出相似的文本片段",
                "llm_name": "openai",
                "reference_depth": 1,
                "input_depth": 1,
                "chunk_size": 0,
                "max_prompt_tokens": 8000
            }
        }
    ]
    
    print("\n開始並發測試...")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # 0. 清理所有已完成的任務
        print("\n0. 清理所有已完成的任務:")
        async with session.get(f"{BASE_URL}/rag?clean_old=true") as response:
            await response.json()
        
        # 1. 並發創建多個任務
        print("\n1. 創建多個 RAG 任務:")
        create_tasks = [
            create_rag_job(session, tc["project_id"], tc["input_text"], tc["ref_text"], 
                          scenario=tc.get("scenario") if not tc.get("use_default_scenario") else None)
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
            if result:
                print(f"任務完成狀態: {json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                print("任務狀態檢查失敗")
        
        # 3. 列出所有任務
        print("\n3. 列出所有任務:")
        async with session.get(f"{BASE_URL}/rag") as response:
            all_jobs = await response.json()
            print(json.dumps(all_jobs, indent=2, ensure_ascii=False))
    
    end_time = time.time()
    print(f"\n總執行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    asyncio.run(test_concurrent_rag()) 