import pytest
import asyncio
from unittest.mock import MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor

from src.rag_core.infrastructure.embedding import EmbeddingManager
from langchain_openai import OpenAIEmbeddings # Used for type hinting if needed, but will be mocked

# Sample data
SAMPLE_API_KEY = "test_api_key_explicit"
SAMPLE_MODEL_NAME = "test_model_explicit"
SAMPLE_EMBEDDING_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5]
SAMPLE_TEXT_TO_EMBED = "This is a test text."

@pytest.fixture
def mock_config_manager_settings(mocker):
    mock_settings = MagicMock()
    mock_settings.api_keys.openai = "test_api_key_from_config"
    mock_settings.embedding.model = "test_model_from_config"
    mock_settings.thread_pool.embed_pool = 5 # Example value for max_workers
    
    mocker.patch('src.rag_core.infrastructure.embedding.config_manager', mock_settings)
    return mock_settings

@pytest.fixture
def mock_openai_embeddings_class(mocker):
    mock_instance = MagicMock(spec=OpenAIEmbeddings) # Use spec for better mocking
    mock_instance.embed_query = MagicMock(return_value=SAMPLE_EMBEDDING_VECTOR)
    
    mock_class = MagicMock(return_value=mock_instance)
    mocker.patch('src.rag_core.infrastructure.embedding.OpenAIEmbeddings', mock_class)
    return mock_class, mock_instance


class TestEmbeddingManagerInit:
    def test_init_with_explicit_parameters(self, mock_openai_embeddings_class, mock_config_manager_settings):
        """
        Test EmbeddingManager instantiation with explicit API key and model name.
        Verifies OpenAIEmbeddings is called with these explicit parameters.
        """
        mock_class, _ = mock_openai_embeddings_class
        
        manager = EmbeddingManager(
            openai_api_key=SAMPLE_API_KEY,
            embedding_model_name=SAMPLE_MODEL_NAME
        )
        
        mock_class.assert_called_once_with(
            api_key=SAMPLE_API_KEY,
            model=SAMPLE_MODEL_NAME
        )
        assert manager._executor is not None # Check executor is initialized

    def test_init_with_config_values(self, mock_openai_embeddings_class, mock_config_manager_settings):
        """
        Test EmbeddingManager instantiation using values from config_manager.settings.
        Verifies OpenAIEmbeddings is called with values from the mocked config.
        """
        mock_class, _ = mock_openai_embeddings_class
        settings = mock_config_manager_settings # Ensure fixture is used to patch

        manager = EmbeddingManager() # No explicit params
        
        mock_class.assert_called_once_with(
            api_key=settings.api_keys.openai,
            model=settings.embedding.model
        )
        assert manager._executor is not None


class TestEmbeddingManagerGenerateEmbedding:
    def test_generate_embedding_successful(self, mock_openai_embeddings_class, mock_config_manager_settings):
        """
        Test successful embedding generation.
        """
        _, mock_instance = mock_openai_embeddings_class
        manager = EmbeddingManager()
        
        result = manager.generate_embedding(SAMPLE_TEXT_TO_EMBED)
        
        mock_instance.embed_query.assert_called_once_with(SAMPLE_TEXT_TO_EMBED)
        assert result == SAMPLE_EMBEDDING_VECTOR

    @pytest.mark.parametrize("empty_return_value", [None, []])
    def test_generate_embedding_empty_result_from_model(self, mock_openai_embeddings_class, mock_config_manager_settings, empty_return_value):
        """
        Test ValueError is raised when OpenAIEmbeddings().embed_query returns None or [].
        """
        _, mock_instance = mock_openai_embeddings_class
        mock_instance.embed_query.return_value = empty_return_value
        
        manager = EmbeddingManager()
        
        with pytest.raises(ValueError) as excinfo:
            manager.generate_embedding(SAMPLE_TEXT_TO_EMBED)
        assert "Embedding generation failed: No embedding vector returned." in str(excinfo.value)

    def test_generate_embedding_exception_from_model(self, mock_openai_embeddings_class, mock_config_manager_settings):
        """
        Test that an exception from OpenAIEmbeddings().embed_query is re-raised.
        """
        _, mock_instance = mock_openai_embeddings_class
        original_exception = ConnectionError("Simulated API connection error")
        mock_instance.embed_query.side_effect = original_exception
        
        manager = EmbeddingManager()
        
        with pytest.raises(ConnectionError) as excinfo:
            manager.generate_embedding(SAMPLE_TEXT_TO_EMBED)
        assert excinfo.value == original_exception


@pytest.mark.asyncio
class TestEmbeddingManagerGenerateEmbeddingAsync:
    async def test_generate_embedding_async_successful(self, mocker, mock_config_manager_settings):
        """
        Test successful async embedding generation.
        Mocks the synchronous generate_embedding method.
        """
        manager = EmbeddingManager() # Real manager, but its sync method will be mocked

        # Mock the synchronous method on the instance
        mocked_sync_generate = MagicMock(return_value=SAMPLE_EMBEDDING_VECTOR)
        mocker.patch.object(manager, 'generate_embedding', new=mocked_sync_generate)
        
        result = await manager.generate_embedding_async(SAMPLE_TEXT_TO_EMBED)
        
        mocked_sync_generate.assert_called_once_with(SAMPLE_TEXT_TO_EMBED)
        assert result == SAMPLE_EMBEDDING_VECTOR
        # Ensure the executor was used by checking for loop.run_in_executor's call
        # This is a bit more involved, might need to mock asyncio.get_event_loop() if direct assertion is hard

    async def test_generate_embedding_async_exception_in_sync_call(self, mocker, mock_config_manager_settings):
        """
        Test exception propagation from the synchronous call in async version.
        """
        manager = EmbeddingManager()
        original_exception = ValueError("Simulated error in sync generate_embedding")

        mocked_sync_generate = MagicMock(side_effect=original_exception)
        mocker.patch.object(manager, 'generate_embedding', new=mocked_sync_generate)
        
        with pytest.raises(ValueError) as excinfo:
            await manager.generate_embedding_async(SAMPLE_TEXT_TO_EMBED)
        
        mocked_sync_generate.assert_called_once_with(SAMPLE_TEXT_TO_EMBED)
        assert excinfo.value == original_exception


class TestEmbeddingManagerShutdown:
    def test_shutdown_calls_executor_shutdown(self, mock_config_manager_settings, mocker):
        """
        Test that manager.shutdown() calls _executor.shutdown(wait=False).
        """
        manager = EmbeddingManager() # Initializes a real ThreadPoolExecutor

        # Mock the _executor attribute on this specific instance AFTER it's created
        mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
        manager._executor = mock_executor_instance # Replace the real executor with a mock

        manager.shutdown()
        
        mock_executor_instance.shutdown.assert_called_once_with(wait=False)

    def test_shutdown_handles_no_executor(self, mock_config_manager_settings):
        """ Test that shutdown doesn't fail if _executor is None (e.g., if init failed before executor creation). """
        manager = EmbeddingManager()
        manager._executor = None # Simulate a scenario where executor wasn't created or was nulled
        try:
            manager.shutdown() # Should not raise an error
        except Exception as e:
            pytest.fail(f"manager.shutdown() raised an exception unexpectedly: {e}")

# End of test file
