import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from rag_core.application.rag_engine import RAGEngine
from rag_core.domain.scenario import Scenario
from rag_core.infrastructure.llm.llm_manager import LLMManager
from rag_core.infrastructure.embedding import EmbeddingManager
from rag_core.infrastructure.vector_store import VectorIndex
from rag_core.domain.schema_checker import DataStructureChecker


def setup_rag_engine():
    mock_embed_mgr = EmbeddingManager(openai_api_key="fake")
    mock_embed_mgr.generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    mock_llm_mgr = LLMManager()
    mock_adapter = AsyncMock()
    mock_adapter.async_generate_response.return_value = (
        '[{"input_uid":"u1","ref_uid":"r1","confidence":0.9}]'
    )
    mock_llm_mgr.register_adapter("mock_llm", mock_adapter)
    mock_llm_mgr.set_default_adapter("mock_llm")

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

    engine = RAGEngine(mock_embed_mgr, mock_vec_index, mock_llm_mgr)
    scenario = Scenario(
        direction="forward",
        cof_threshold=0.5,
        rag_k=1,
        rag_k_forward=1,
        rag_k_reverse=1,
        llm_name="mock_llm",
    )
    index_info = {"collection_name": "test_collection", "filters": {}, "rag_k": 1}
    return engine, scenario, index_info


def test_generate_answer_sync_outside_loop():
    engine, scenario, index_info = setup_rag_engine()
    results = engine.generate_answer_sync(
        user_query="test query",
        root_uid="u1",
        scenario=scenario,
        index_info=index_info,
    )
    assert isinstance(results, list)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_generate_answer_sync_inside_loop():
    engine, scenario, index_info = setup_rag_engine()
    results = await asyncio.to_thread(
        engine.generate_answer_sync,
        "test query",
        "u1",
        scenario,
        index_info,
    )
    assert isinstance(results, list)
    assert len(results) == 1

