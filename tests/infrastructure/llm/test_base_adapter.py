import pytest
from unittest.mock import MagicMock, patch
from rag_core.infrastructure.llm.base_adapter import LLMAdapter

class TestLLMAdapterInit:
    def test_init_sets_model(self, mocker):
        """Test that LLMAdapter.__init__ correctly sets the model."""
        mock_model = mocker.Mock()
        adapter = LLMAdapter(model=mock_model)
        assert adapter.model == mock_model

class TestLLMAdapterAbstractMethods:
    @pytest.fixture
    def adapter_instance(self):
        # LLMAdapter can be instantiated without a model for these tests,
        # as it's not used by the abstract methods before they raise NotImplementedError.
        # If __init__ required a model strictly, we'd pass mocker.Mock() here.
        return LLMAdapter(model=MagicMock())


    def test_generate_response_raises_not_implemented(self, adapter_instance):
        """Test that generate_response raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            adapter_instance.generate_response("test prompt")

    def test_stream_response_raises_not_implemented(self, adapter_instance):
        """Test that stream_response raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            # Attempt to get an iterator (which stream_response should be)
            # and then call next() or list() on it.
            # Directly calling might not execute up to the yield if it's a generator.
            # However, if it's truly abstract, the call itself should fail.
            list(adapter_instance.stream_response("test prompt"))


    @pytest.mark.asyncio
    async def test_async_generate_response_raises_not_implemented(self, adapter_instance):
        """Test that async_generate_response raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await adapter_instance.async_generate_response("test prompt")

    @pytest.mark.asyncio
    async def test_async_stream_response_raises_not_implemented(self, adapter_instance):
        """Test that async_stream_response raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            # Similar to stream_response, attempt to iterate
            async for _ in adapter_instance.async_stream_response("test prompt"):
                pass # Should not reach here


class TestLLMAdapterHandleError:
    def test_handle_error_logs_correctly(self, mocker):
        """Test that handle_error logs the error with the correct details."""
        mock_log_error = mocker.patch('rag_core.infrastructure.llm.base_adapter.log_wrapper.error')
        adapter = LLMAdapter(model=MagicMock())
        test_exception = Exception("Test error details")
        
        adapter.handle_error(test_exception)
        
        mock_log_error.assert_called_once_with(
            "BaseAdapter",  # Class name from the method
            "handle_error", # Method name
            "LLMAdapter Error: Test error details" # Formatted message
        )


class TestLLMAdapterGenerateResponseSync:
    @pytest.fixture
    def adapter_instance(self, mocker):
        # For these tests, the model itself is not directly used by generate_response_sync,
        # but the generate_response method (which would use it) is mocked.
        return LLMAdapter(model=mocker.Mock())

    def test_generate_response_sync_successful_path(self, mocker, adapter_instance):
        """Scenario 4a: generate_response succeeds."""
        expected_response = "successful response"
        mocker.patch.object(adapter_instance, 'generate_response', return_value=expected_response)
        
        result = adapter_instance.generate_response_sync("test prompt")
        
        assert result == expected_response
        adapter_instance.generate_response.assert_called_once_with("test prompt")

    def test_generate_response_sync_exception_path(self, mocker, adapter_instance):
        """Scenario 4b: generate_response raises an exception."""
        original_exception = ValueError("LLM failure")
        mocker.patch.object(adapter_instance, 'generate_response', side_effect=original_exception)
        mock_log_error = mocker.patch('rag_core.infrastructure.llm.base_adapter.log_wrapper.error')
        
        with pytest.raises(ValueError, match="LLM failure") as excinfo:
            adapter_instance.generate_response_sync("test prompt")
        
        assert excinfo.value == original_exception
        adapter_instance.generate_response.assert_called_once_with("test prompt")
        mock_log_error.assert_called_once_with(
            "BaseAdapter",
            "generate_response_sync",
            "LLMAdapter Error: LLM failure"
        )

# End of test file
