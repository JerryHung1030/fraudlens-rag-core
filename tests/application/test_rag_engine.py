import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from rag_core.application.rag_engine import RAGEngine
from rag_core.exceptions import EmbeddingError, VectorSearchError, PromptTooLongError, LLMError

# A simple mock for the Scenario pydantic model
class MockScenario:
    def __init__(self,
                 direction="forward",
                 rag_k_forward=5,
                 rag_k_reverse=3,
                 rag_k=4, # Fallback
                 llm_name="default_llm", # Can be overridden by test
                 max_prompt_tokens=1000, # For PromptBuilder, though not directly used by RAGEngine tests often
                 role_desc="Test Role",
                 reference_desc="Test Reference",
                 input_desc="Test Input",
                 scoring_rule="Test Scoring Rule",
                 cof_threshold=0.1):
        self.direction = direction
        self.rag_k_forward = rag_k_forward
        self.rag_k_reverse = rag_k_reverse
        self.rag_k = rag_k
        self.llm_name = llm_name
        self.max_prompt_tokens = max_prompt_tokens
        self.role_desc = role_desc
        self.reference_desc = reference_desc
        self.input_desc = input_desc
        self.scoring_rule = scoring_rule
        self.cof_threshold = cof_threshold

@pytest.fixture
def mock_embedding_manager():
    manager = AsyncMock()
    manager.generate_embedding_async = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return manager

@pytest.fixture
def mock_vector_index():
    index = AsyncMock()
    index.search_async = AsyncMock(return_value=[
        {'uid': 'doc1', 'text': 'Document 1 text', 'score': 0.9},
        {'uid': 'doc2', 'text': 'Document 2 text', 'score': 0.8},
    ])
    return index

@pytest.fixture
def mock_llm_manager():
    manager = MagicMock() # Use MagicMock to allow attribute assignment like default_adapter_name
    mock_adapter = AsyncMock()
    mock_adapter.async_generate_response = AsyncMock(return_value="LLM generated answer.")
    manager.get_adapter = MagicMock(return_value=mock_adapter)
    manager.default_adapter_name = "default_llm_adapter"
    return manager

@pytest.fixture
def mock_prompt_builder(mocker):
    # Patch PromptBuilder where it's used by RAGEngine
    mock_pb_instance = MagicMock()
    mock_pb_instance.build_prompt = MagicMock(return_value=("Formatted prompt string", 2)) # prompt, num_candidates
    
    # Mock the class constructor to return our instance
    # The path should be where PromptBuilder is imported/used by RAGEngine
    mocker.patch('rag_core.application.rag_engine.PromptBuilder', return_value=mock_pb_instance)
    return mock_pb_instance


@pytest.fixture
def mock_result_formatter(mocker):
    mock_rf_instance = MagicMock()
    mock_rf_instance.parse_and_format = MagicMock(return_value=[{"formatted": "result"}])

    # Mock the class constructor to return our instance
    # The path should be where ResultFormatter is imported/used by RAGEngine
    mocker.patch('rag_core.application.rag_engine.ResultFormatter', return_value=mock_rf_instance)
    return mock_rf_instance


@pytest.mark.asyncio
async def test_generate_answer_successful_path(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager,
    mock_prompt_builder, # Fixture to patch PromptBuilder
    mock_result_formatter # Fixture to patch ResultFormatter
):
    """
    Scenario 1: Successful path for generate_answer.
    """
    # Arrange
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "What is RAG?"
    scenario = MockScenario(llm_name="test_llm_from_scenario") # Override default
    index_info = {"rag_k": 3} # Use specific rag_k from index_info

    # Act
    result = await engine.generate_answer(user_query, scenario, index_info)

    # Assert
    # 1. Embedding manager called
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)
    
    # 2. Vector index called
    expected_k = index_info["rag_k"]
    mock_vector_index.search_async.assert_awaited_once_with(
        query_vector=await mock_embedding_manager.generate_embedding_async(user_query),
        k=expected_k,
        filter_expr=None # Assuming no filter for this test
    )
    
    # 3. PromptBuilder called
    # PromptBuilder instance is now mock_prompt_builder directly
    mock_prompt_builder.build_prompt.assert_called_once()
    # Check specific arguments of build_prompt call
    args, kwargs = mock_prompt_builder.build_prompt.call_args
    assert kwargs['user_query'] == user_query
    assert kwargs['context_docs'] == await mock_vector_index.search_async() # Using the return_value
    assert kwargs['scenario'] == scenario

    # 4. LLM Manager and Adapter called
    # get_adapter should be called with scenario.llm_name if provided, else default
    mock_llm_manager.get_adapter.assert_called_once_with(scenario.llm_name)
    
    mock_adapter_instance = mock_llm_manager.get_adapter()
    mock_adapter_instance.async_generate_response.assert_awaited_once_with(
        "Formatted prompt string" # from mock_prompt_builder
    )

    # 5. ResultFormatter called
    # ResultFormatter instance is now mock_result_formatter directly
    mock_result_formatter.parse_and_format.assert_called_once_with(
        llm_output="LLM generated answer.", # from mock_adapter
        context_docs=await mock_vector_index.search_async(),
        num_used_candidates=2, # from mock_prompt_builder
        scenario=scenario
    )

    # 6. Final result
    assert result == [{"formatted": "result"}] # from mock_result_formatter

@pytest.mark.asyncio
async def test_generate_answer_embedding_error(
    mocker,
    mock_embedding_manager,
    mock_vector_index, # Still needed for RAGEngine instantiation
    mock_llm_manager   # Still needed for RAGEngine instantiation
):
    """
    Scenario 2: Embedding error.
    - Configure embedding_manager.generate_embedding_async to raise an Exception.
    - Assert that generate_answer raises EmbeddingError.
    """
    # Arrange
    mock_embedding_manager.generate_embedding_async.side_effect = Exception("Test embedding generation failed")
    
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Query that will cause embedding error"
    scenario = MockScenario()
    index_info = {}

    # Act & Assert
    with pytest.raises(EmbeddingError) as excinfo:
        await engine.generate_answer(user_query, scenario, index_info)
    
    assert "Embedding generation failed" in str(excinfo.value)
    assert "Test embedding generation failed" in str(excinfo.value.original_exception)

@pytest.mark.asyncio
async def test_generate_answer_vector_search_error(
    mocker,
    mock_embedding_manager, # Still needed
    mock_vector_index,
    mock_llm_manager # Still needed
):
    """
    Scenario 3: Vector search error.
    - Configure vector_index.search_async to raise an Exception.
    - Assert that generate_answer raises VectorSearchError.
    """
    # Arrange
    mock_vector_index.search_async.side_effect = Exception("Test vector search failed")
    
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Query that will cause vector search error"
    scenario = MockScenario()
    index_info = {} # rag_k will default from scenario.rag_k

    # Act & Assert
    with pytest.raises(VectorSearchError) as excinfo:
        await engine.generate_answer(user_query, scenario, index_info)
    
    assert "Vector search failed" in str(excinfo.value)
    assert "Test vector search failed" in str(excinfo.value.original_exception)
    # Ensure embedding was called before search
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)

@pytest.mark.asyncio
async def test_generate_answer_no_search_hits(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager,
    mock_prompt_builder, # Patched PromptBuilder fixture
    mock_result_formatter # Patched ResultFormatter fixture
):
    """
    Scenario 4: No search hits.
    - Configure vector_index.search_async to return an empty list.
    - Assert that generate_answer returns an empty list.
    - Assert that PromptBuilder.build_prompt and LLM methods were NOT called.
    """
    # Arrange
    mock_vector_index.search_async.return_value = [] # No search hits
    
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Query that returns no search hits"
    scenario = MockScenario()
    index_info = {}

    # Act
    result = await engine.generate_answer(user_query, scenario, index_info)

    # Assert
    assert result == []
    
    # Ensure embedding was called
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)
    # Ensure vector search was called
    mock_vector_index.search_async.assert_awaited_once()
    
    # Ensure PromptBuilder and LLM/ResultFormatter were NOT called
    mock_prompt_builder.build_prompt.assert_not_called()
    mock_llm_manager.get_adapter().async_generate_response.assert_not_called()
    mock_result_formatter.parse_and_format.assert_not_called()

@pytest.mark.asyncio
async def test_generate_answer_prompt_too_long_error(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager,
    mock_prompt_builder # Patched PromptBuilder fixture
):
    """
    Scenario 5: PromptTooLongError during prompt building.
    - Mock PromptBuilder.build_prompt to raise PromptTooLongError.
    - Assert that generate_answer re-raises this error.
    """
    # Arrange
    # mock_prompt_builder is the instance of the mocked PromptBuilder
    mock_prompt_builder.build_prompt.side_effect = PromptTooLongError("Test: Prompt is too long")
    
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Query that leads to a long prompt"
    scenario = MockScenario()
    index_info = {}

    # Act & Assert
    with pytest.raises(PromptTooLongError) as excinfo:
        await engine.generate_answer(user_query, scenario, index_info)
    
    assert "Test: Prompt is too long" in str(excinfo.value)
    
    # Ensure previous steps were called
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)
    mock_vector_index.search_async.assert_awaited_once()
    mock_prompt_builder.build_prompt.assert_called_once() # build_prompt was called and raised error

@pytest.mark.asyncio
async def test_generate_answer_llm_adapter_not_found(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager,
    mock_prompt_builder # Patched PromptBuilder fixture
):
    """
    Scenario 6: LLM adapter not found.
    - Configure llm_manager.get_adapter to return None.
    - Assert that generate_answer raises LLMError with the message "找不到 LLM 適配器: ...".
    """
    # Arrange
    mock_llm_manager.get_adapter.return_value = None # Adapter not found
    
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Query for which LLM adapter is not found"
    scenario = MockScenario(llm_name="non_existent_adapter")
    index_info = {}

    # Act & Assert
    with pytest.raises(LLMError) as excinfo:
        await engine.generate_answer(user_query, scenario, index_info)
    
    assert "找不到 LLM 適配器" in str(excinfo.value)
    assert scenario.llm_name in str(excinfo.value)
    
    # Ensure previous steps were called
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)
    mock_vector_index.search_async.assert_awaited_once()
    mock_prompt_builder.build_prompt.assert_called_once()
    mock_llm_manager.get_adapter.assert_called_once_with(scenario.llm_name)

@pytest.mark.asyncio
async def test_generate_answer_llm_generation_error(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager,
    mock_prompt_builder # Patched PromptBuilder fixture
):
    """
    Scenario 7: LLM generation error.
    - Configure llm_manager.get_adapter().async_generate_response to raise an Exception.
    - Assert that generate_answer raises LLMError with the message "LLM 生成失敗: ...".
    """
    # Arrange
    mock_adapter_instance = mock_llm_manager.get_adapter() # Get the mocked adapter
    mock_adapter_instance.async_generate_response.side_effect = Exception("Test LLM generation failed")
    
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Query that causes LLM generation error"
    scenario = MockScenario(llm_name="test_llm")
    index_info = {}

    # Act & Assert
    with pytest.raises(LLMError) as excinfo:
        await engine.generate_answer(user_query, scenario, index_info)
    
    assert "LLM 生成失敗" in str(excinfo.value)
    assert "Test LLM generation failed" in str(excinfo.value.original_exception)
    
    # Ensure previous steps were called
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)
    mock_vector_index.search_async.assert_awaited_once()
    mock_prompt_builder.build_prompt.assert_called_once()
    mock_llm_manager.get_adapter.assert_called_once_with(scenario.llm_name)
    mock_adapter_instance.async_generate_response.assert_awaited_once() # Was called and raised error

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "index_info_rag_k, scenario_direction, scenario_rag_k_forward, scenario_rag_k_reverse, scenario_rag_k_fallback, expected_k",
    [
        (7, "forward", 5, 3, 4, 7),  # Case 1: index_info["rag_k"] takes precedence
        (None, "forward", 5, 3, 4, 5), # Case 2: scenario.rag_k_forward used
        (None, "reverse", 5, 3, 4, 3), # Case 3: scenario.rag_k_reverse used
        (None, "forward", None, 3, 4, 4), # Case 4a: scenario.rag_k (fallback) used when forward is None
        (None, "reverse", 5, None, 4, 4), # Case 4b: scenario.rag_k (fallback) used when reverse is None
        ({}, "forward", 5, 3, 4, 5),      # Case 5: index_info is empty dict, use scenario.rag_k_forward
        ({"other_key": 10}, "reverse", 5, 3, 4, 3) # Case 6: index_info has other keys, use scenario.rag_k_reverse
    ],
    ids=[
        "index_info_rag_k",
        "scenario_rag_k_forward",
        "scenario_rag_k_reverse",
        "fallback_rag_k_from_forward_none",
        "fallback_rag_k_from_reverse_none",
        "empty_index_info_dict",
        "other_keys_in_index_info"
    ]
)
async def test_generate_answer_rag_k_selection(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager,
    mock_prompt_builder,
    mock_result_formatter,
    index_info_rag_k, scenario_direction, scenario_rag_k_forward, scenario_rag_k_reverse, scenario_rag_k_fallback, expected_k
):
    """
    Scenario 8: Different rag_k sources.
    Tests various conditions for selecting the 'k' value for vector search.
    """
    # Arrange
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Test query for rag_k selection"
    
    scenario = MockScenario(
        direction=scenario_direction,
        rag_k_forward=scenario_rag_k_forward,
        rag_k_reverse=scenario_rag_k_reverse,
        rag_k=scenario_rag_k_fallback,
        llm_name="test_llm" # Ensure an LLM name is set
    )
    
    if isinstance(index_info_rag_k, dict): # For cases like empty_index_info_dict
        index_info = index_info_rag_k
    elif index_info_rag_k is None:
        index_info = {} # Default to empty if None for rag_k, to simulate absence
    else:
        index_info = {"rag_k": index_info_rag_k}


    # Act
    await engine.generate_answer(user_query, scenario, index_info)

    # Assert
    # Main assertion: vector_index.search_async called with the correct k
    mock_vector_index.search_async.assert_awaited_once_with(
        query_vector=await mock_embedding_manager.generate_embedding_async(user_query),
        k=expected_k,
        filter_expr=None # Assuming no filter for these test cases
    )
    
    # Ensure other parts of the pipeline were called to confirm it's a successful path otherwise
    mock_embedding_manager.generate_embedding_async.assert_awaited_once_with(user_query)
    mock_prompt_builder.build_prompt.assert_called_once()
    mock_llm_manager.get_adapter().async_generate_response.assert_awaited_once()
    mock_result_formatter.parse_and_format.assert_called_once()

# Tests for generate_answer_sync

def test_generate_answer_sync_successful_path(
    mocker,
    mock_embedding_manager, # Needed for engine instantiation
    mock_vector_index,      # Needed for engine instantiation
    mock_llm_manager        # Needed for engine instantiation
):
    """
    Scenario 1 (sync): Successful path for generate_answer_sync.
    - Mock RAGEngine.generate_answer (async) to return a specific result.
    - Call generate_answer_sync and assert it returns the mocked result.
    - Assert generate_answer (async) was called with correct arguments.
    """
    # Arrange
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Sync test query"
    scenario = MockScenario()
    index_info = {"sync_test_key": "value"}
    expected_result = [{"sync_result": "success"}]

    # Mock the async generate_answer method ON THE INSTANCE
    mocked_async_generate = AsyncMock(return_value=expected_result)
    mocker.patch.object(engine, 'generate_answer', new=mocked_async_generate)

    # Act
    result = engine.generate_answer_sync(user_query, scenario, index_info)

    # Assert
    assert result == expected_result
    mocked_async_generate.assert_awaited_once_with(user_query, scenario, index_info)


def test_generate_answer_sync_exception_propagation(
    mocker,
    mock_embedding_manager,
    mock_vector_index,
    mock_llm_manager
):
    """
    Scenario 2 (sync): Exception during async execution.
    - Mock RAGEngine.generate_answer (async) to raise an EmbeddingError.
    - Call generate_answer_sync and assert it re-raises the EmbeddingError.
    """
    # Arrange
    engine = RAGEngine(
        embedding_manager=mock_embedding_manager,
        vector_index=mock_vector_index,
        llm_manager=mock_llm_manager
    )
    user_query = "Sync query causing error"
    scenario = MockScenario()
    index_info = {}
    
    # Mock the async generate_answer method to raise an error
    mocked_async_generate = AsyncMock(side_effect=EmbeddingError("Async embedding failed"))
    mocker.patch.object(engine, 'generate_answer', new=mocked_async_generate)

    # Act & Assert
    with pytest.raises(EmbeddingError) as excinfo:
        engine.generate_answer_sync(user_query, scenario, index_info)
    
    assert "Async embedding failed" in str(excinfo.value)
    mocked_async_generate.assert_awaited_once_with(user_query, scenario, index_info)

# End of tests for RAGEngine
