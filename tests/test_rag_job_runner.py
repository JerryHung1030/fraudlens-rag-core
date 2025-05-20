import asyncio
import json
import os
import sys
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interfaces.jobs.rag_job_runner import RAGJobRunner
from src.rag_core.domain.scenario import Scenario
from src.rag_core.infrastructure.vector_store import VectorIndex
from src.rag_core.application.rag_engine import RAGEngine

# 測試數據
MOCK_REFERENCE_DATA = {
    "level1": [
        {
            "sid": "ref1",
            "text": "This is a reference document about AI.",
            "metadata": {"category": "technology"}
        }
    ]
}

MOCK_INPUT_DATA = {
    "level1": [
        {
            "sid": "input1",
            "text": "Tell me about AI.",
            "metadata": {"category": "question"}
        }
    ]
}

MOCK_SCENARIO = {
    "name": "test_scenario",
    "description": "Test scenario for RAG",
    "parameters": {
        "direction": "forward",
        "reference_depth": 1,
        "input_depth": 1,
        "chunk_size": 0,
        "rag_k": 3
    }
}

@pytest.fixture
def mock_vector_index():
    """創建模擬的 VectorIndex"""
    mock = Mock(spec=VectorIndex)
    mock.ingest_json = AsyncMock()
    return mock

@pytest.fixture
def mock_rag_engine():
    """創建模擬的 RAGEngine"""
    mock = Mock(spec=RAGEngine)
    mock.generate_answer = AsyncMock(return_value={
        "answer": "AI is a field of computer science.",
        "sources": ["ref1"]
    })
    return mock

@pytest.fixture
def rag_job_runner(mock_vector_index, mock_rag_engine):
    """創建 RAGJobRunner 實例"""
    return RAGJobRunner(mock_vector_index, mock_rag_engine)

@pytest.mark.asyncio
async def test_successful_rag_job(rag_job_runner, mock_vector_index, mock_rag_engine):
    """測試成功的 RAG 任務執行"""
    # 準備測試數據
    job_payload = {
        "job_id": "test_job_1",
        "project_id": "test_project",
        "scenario": MOCK_SCENARIO,
        "input_data": MOCK_INPUT_DATA,
        "reference_data": MOCK_REFERENCE_DATA,
        "callback_url": "http://test-callback.com"
    }

    # 執行任務
    result = await rag_job_runner.run_job(job_payload)

    # 驗證結果
    assert result["job_id"] == "test_job_1"
    assert result["status"] == "completed"
    assert "results" in result
    assert "completed_at" in result

    # 驗證 VectorIndex 被正確調用
    mock_vector_index.ingest_json.assert_called()

    # 驗證 RAGEngine 被正確調用
    mock_rag_engine.generate_answer.assert_called()

@pytest.mark.asyncio
async def test_failed_rag_job(rag_job_runner):
    """測試失敗的 RAG 任務執行"""
    # 準備無效的測試數據
    job_payload = {
        "job_id": "test_job_2",
        "project_id": "test_project",
        "scenario": MOCK_SCENARIO,
        "input_data": {},  # 無效的輸入數據
        "reference_data": MOCK_REFERENCE_DATA,
        "callback_url": "http://test-callback.com"
    }

    # 執行任務並驗證異常
    with pytest.raises(ValueError):
        await rag_job_runner.run_job(job_payload)

@pytest.mark.asyncio
async def test_callback_handling(rag_job_runner):
    """測試回調處理"""
    # 模擬 aiohttp.ClientSession
    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

        # 準備測試數據
        job_payload = {
            "job_id": "test_job_3",
            "project_id": "test_project",
            "scenario": MOCK_SCENARIO,
            "input_data": MOCK_INPUT_DATA,
            "reference_data": MOCK_REFERENCE_DATA,
            "callback_url": "http://test-callback.com"
        }

        # 執行任務
        result = await rag_job_runner.run_job(job_payload)

        # 驗證回調被調用
        mock_session.return_value.__aenter__.return_value.post.assert_called_once()

@pytest.mark.asyncio
async def test_concurrent_processing(rag_job_runner, mock_rag_engine):
    """測試並發處理"""
    # 準備多個文檔的測試數據
    large_input_data = {
        "level1": [
            {
                "sid": f"input{i}",
                "text": f"Question {i}",
                "metadata": {"category": "question"}
            }
            for i in range(10)
        ]
    }

    job_payload = {
        "job_id": "test_job_4",
        "project_id": "test_project",
        "scenario": MOCK_SCENARIO,
        "input_data": large_input_data,
        "reference_data": MOCK_REFERENCE_DATA
    }

    # 執行任務
    result = await rag_job_runner.run_job(job_payload)

    # 驗證結果
    assert result["status"] == "completed"
    assert len(result["results"]) == 10

    # 驗證並發限制
    assert mock_rag_engine.generate_answer.call_count == 10

@pytest.mark.asyncio
async def test_different_directions(rag_job_runner, mock_vector_index):
    """測試不同的處理方向"""
    directions = ["forward", "reverse", "both"]
    for direction in directions:
        mock_vector_index.reset_mock()
        scenario = MOCK_SCENARIO.copy()
        scenario["parameters"]["direction"] = direction
        job_payload = {
            "job_id": f"test_job_{direction}",
            "project_id": "test_project",
            "scenario": scenario,
            "input_data": MOCK_INPUT_DATA,
            "reference_data": MOCK_REFERENCE_DATA
        }
        # Patch flatten_levels to return different lists for ref_docs and inp_docs
        with patch('src.rag_core.utils.text_preprocessor.TextPreprocessor.flatten_levels', side_effect=[
            [{"text": "ref", "group_uid": "g1", "orig_sid": "s1"}],  # ref_docs
            [{"text": "inp", "group_uid": "g2", "orig_sid": "s2"}],  # inp_docs
        ]):
            result = await rag_job_runner.run_job(job_payload)
        assert result["status"] == "completed"
        if direction == "both":
            assert mock_vector_index.ingest_json.call_count == 2
        else:
            assert mock_vector_index.ingest_json.call_count == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 