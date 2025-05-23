import pytest
import asyncio
from unittest.mock import MagicMock, patch

from rag_core.infrastructure.llm.local_llama_adapter import LocalLlamaAdapter
from rag_core.infrastructure.llm.base_adapter import LLMAdapter # For patching super().handle_error

# --- Fixtures ---

@pytest.fixture
def adapter_instance():
    """Provides a LocalLlamaAdapter instance for tests."""
    return LocalLlamaAdapter(model_path="path/to/model", temperature=0.5, max_tokens=512)

# --- Test Classes ---

class TestLocalLlamaAdapterInit:
    def test_init_parameters_and_model_none(self):
        """
        Scenario 1: Test __init__.
        Instantiate LocalLlamaAdapter and assert attributes.
        Assert adapter.model is None (as per current super().__init__(model=None)).
        """
        model_path = "path/to/model"
        temperature = 0.5
        max_tokens = 512
        
        adapter = LocalLlamaAdapter(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        assert adapter.model_path == model_path
        assert adapter.temperature == temperature
        assert adapter.max_tokens == max_tokens
        assert adapter.model is None # As super().__init__(model=None) is called

class TestLocalLlamaAdapterGenerateResponse:
    def test_successful_placeholder_response(self, adapter_instance):
        """
        Scenario 1 (sync generate): Successful placeholder response.
        """
        result = adapter_instance.generate_response("test prompt")
        assert result == "Local model sync result"

    # As per pragmatic approach in instructions, skipping direct test of except block for now.

class TestLocalLlamaAdapterStreamResponse:
    def test_successful_placeholder_stream(self, adapter_instance):
        """
        Scenario 1 (sync stream): Successful placeholder stream.
        """
        result = list(adapter_instance.stream_response("test prompt"))
        assert result == ["Local model streaming part1", "Local model streaming part2"]

    # As per pragmatic approach, skipping direct test of except block for now.

@pytest.mark.asyncio
class TestLocalLlamaAdapterAsyncGenerateResponse:
    async def test_async_generate_successful_execution(self, adapter_instance, mocker):
        """
        Scenario 1 (async generate): Successful execution.
        Mocks the synchronous adapter.generate_response.
        """
        custom_sync_output = "custom sync output for async test"
        mock_sync_generate = mocker.patch.object(
            adapter_instance, 
            'generate_response', 
            return_value=custom_sync_output
        )
        
        prompt = "test async prompt"
        result = await adapter_instance.async_generate_response(prompt)
        
        mock_sync_generate.assert_called_once_with(prompt)
        assert result == custom_sync_output

    async def test_async_generate_exception_from_sync_method(self, adapter_instance, mocker):
        """
        Scenario 2 (async generate): Exception from sync method.
        Mocks adapter.generate_response to raise RuntimeError.
        """
        error_message = "Sync method failed during async call"
        mocker.patch.object(
            adapter_instance,
            'generate_response',
            side_effect=RuntimeError(error_message)
        )
        
        prompt = "test async prompt causing error"
        with pytest.raises(RuntimeError, match=error_message):
            await adapter_instance.async_generate_response(prompt)

@pytest.mark.asyncio
class TestLocalLlamaAdapterAsyncStreamResponse:
    async def test_async_stream_successful_stream(self, adapter_instance, mocker):
        """
        Scenario 1 (async stream): Successful stream.
        Mocks adapter.stream_response.
        """
        custom_sync_parts = ["custom stream part 1", "custom stream part 2"]
        mock_sync_stream = mocker.patch.object(
            adapter_instance,
            'stream_response',
            return_value=iter(custom_sync_parts) # Ensure it's an iterator
        )
        
        prompt = "test async stream prompt"
        result_chunks = [chunk async for chunk in adapter_instance.async_stream_response(prompt)]
        
        mock_sync_stream.assert_called_once_with(prompt)
        assert result_chunks == custom_sync_parts

    async def test_async_stream_exception_from_sync_method_during_iteration(self, adapter_instance, mocker):
        """
        Scenario 2 (async stream): Exception from sync stream method (raised during iteration).
        """
        error_message = "Sync stream failed during yield in async call"

        def mock_sync_stream_gen_with_error(prompt_text):
            yield "part1 from sync error gen"
            raise RuntimeError(error_message)

        mocker.patch.object(
            adapter_instance,
            'stream_response',
            side_effect=mock_sync_stream_gen_with_error # Use side_effect for generators
        )
        
        prompt = "test async stream error prompt"
        with pytest.raises(RuntimeError, match=error_message):
            # Consume the async generator to trigger the exception
            async_results = []
            async for chunk in adapter_instance.async_stream_response(prompt):
                async_results.append(chunk)


class TestLocalLlamaAdapterHandleError:
    def test_handle_error_calls_super(self, adapter_instance, mocker):
        """
        Test overridden handle_error method calls super().handle_error.
        """
        # Path to LLMAdapter's handle_error within the base_adapter module
        # This needs to be the location where LLMAdapter (the class itself) is defined
        # or where it's imported if you're patching an instance's inherited method.
        # For patching a method on a parent class that will be called via super(),
        # you patch it on the parent class directly.
        mock_super_handle_error = mocker.patch(
            'rag_core.infrastructure.llm.base_adapter.LLMAdapter.handle_error'
        )
        
        test_exception = Exception("LocalLlama specific error details")
        adapter_instance.handle_error(test_exception)
        
        mock_super_handle_error.assert_called_once_with(adapter_instance, test_exception)
        # Note: The first argument to a method mocked on a class (when called from an instance)
        # is the instance itself. If you mock `LLMAdapter.handle_error`,
        # when `adapter_instance.handle_error(test_exception)` calls `super().handle_error(e)`,
        # the mocked `LLMAdapter.handle_error` receives `adapter_instance` as `self`.

# End of test file
