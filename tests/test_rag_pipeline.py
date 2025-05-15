# tests/test_rag_pipeline.py
import pytest
from unittest.mock import MagicMock, AsyncMock

from services.rag_engine import RAGEngine
from services.scenario import Scenario
from managers.llm_manager import LLMManager
from managers.embedding_manager import EmbeddingManager
from managers.vector_index import VectorIndex
from managers.data_structure_checker import DataStructureChecker


@pytest.mark.asyncio
async def test_rag_pipeline_mocked():
    # 1. 準備 mock embedding（同步方法用 MagicMock）
    mock_embed_mgr = EmbeddingManager(openai_api_key="fake_key")
    mock_embed_mgr.generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    # 2. 準備 mock LLM（直接 AsyncMock）
    mock_llm_mgr = LLMManager()
    mock_adapter = AsyncMock()
    # 模擬 async_generate_response 回傳固定 JSON 字串
    mock_adapter.async_generate_response.return_value = (
        '[{"input_uid":"u1","ref_uid":"r1","confidence":0.9}]'
    )
    mock_llm_mgr.register_adapter("mock_llm", mock_adapter)
    mock_llm_mgr.set_default_adapter("mock_llm")

    # 3. 準備 mock vector_index
    checker = DataStructureChecker()
    mock_vec_index = VectorIndex(
        embedding_manager=mock_embed_mgr,
        data_checker=checker,
        qdrant_client=MagicMock(),
        default_collection_name="test_collection",
        vector_size=3,
    )
    mock_vec_index.search = MagicMock(return_value=[
        {"uid": "r1", "payload": {}, "text": "some text", "score": 0.98}
    ])

    # 4. 準備 scenario：新版 Scenario 要傳入 rag_k、rag_k_forward、rag_k_reverse
    scenario = Scenario(
        direction="forward",
        cof_threshold=0.5,
        rag_k=1,
        rag_k_forward=1,
        rag_k_reverse=1,
        llm_name="mock_llm",
    )

    # 5. 執行 RAGEngine
    rag_engine = RAGEngine(mock_embed_mgr, mock_vec_index, mock_llm_mgr)
    results = await rag_engine.generate_answer(
        user_query="test query",
        root_uid="u1",
        scenario=scenario,
        index_info={
            "collection_name": "test_collection",
            "filters": {},
            # 也可以只給 rag_k，因為 scenario.rag_k_forward/ reverse 也有值
            "rag_k": 1,
        },
    )

    # 6. 驗證
    assert isinstance(results, list)
    assert len(results) == 1

    out = results[0]
    assert out["direction"] == "forward"
    assert out["root_uid"] == "u1"
    assert out["rag_k"] == 1  # 驗證實際使用的 candidates 數量

    preds = out["predictions"]
    assert isinstance(preds, list)
    assert len(preds) == 1

    # VectorIndex.search 回傳的 score → similarity_score
    assert preds[0]["similarity_score"] == 0.98
    # LLM 模擬結果的 confidence
    assert preds[0]["confidence"] == 0.9
