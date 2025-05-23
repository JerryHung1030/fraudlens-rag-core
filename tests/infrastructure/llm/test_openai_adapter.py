import pytest
import asyncio
from unittest.mock import MagicMock, patch, call
import time # For mocking time.sleep

# Assuming openai is installed in the environment, otherwise this might fail at collection time
# For testing, we only need the exception type. If actual openai lib is not present,
# we can create a dummy class for OpenAIError for the tests to run.
try:
    from openai import OpenAIError, APIError # APIError is a common one, or RateLimitError, etc.
except ImportError:
    # Create a dummy OpenAIError if the library is not installed in the test environment
    class OpenAIError(Exception):
        pass
    class APIError(OpenAIError):
        pass


from rag_core.infrastructure.llm.openai_adapter import OpenAIAdapter
from rag_core.infrastructure.llm.base_adapter import LLMAdapter # For patching super().handle_error

# --- Fixtures ---

@pytest.fixture
def mock_openai_client_instance():
    """Mocks the instance of openai.OpenAI()."""
    mock_client = MagicMock()
    # Mock for non-streaming response
    mock_response_non_stream = MagicMock()
    mock_response_non_stream.choices = [MagicMock()]
    mock_response_non_stream.choices[0].message = MagicMock()
    mock_response_non_stream.choices[0].message.content = "Mocked LLM response"
    
    # Mock for streaming response
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock()]
    mock_chunk_1.choices[0].delta = MagicMock()
    mock_chunk_1.choices[0].delta.content = "Hello"
    
    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock()]
    mock_chunk_2.choices[0].delta = MagicMock()
    mock_chunk_2.choices[0].delta.content = " world"

    mock_chunk_3 = MagicMock() # Represents end of stream or None content
    mock_chunk_3.choices = [MagicMock()]
    mock_chunk_3.choices[0].delta = MagicMock()
    mock_chunk_3.choices[0].delta.content = None
    
    mock_client.chat.completions.create.side_effect = lambda *args, **kwargs: \
        iter([mock_chunk_1, mock_chunk_2, mock_chunk_3]) if kwargs.get("stream") else mock_response_non_stream
        
    return mock_client

@pytest.fixture
def mock_openai_class(mocker, mock_openai_client_instance):
    """Mocks the openai.OpenAI class."""
    mock_class = mocker.patch('rag_core.infrastructure.llm.openai_adapter.OpenAI', return_value=mock_openai_client_instance)
    return mock_class

@pytest.fixture
def mock_log_wrapper(mocker):
    """Mocks the log_wrapper module."""
    return mocker.patch('rag_core.infrastructure.llm.openai_adapter.log_wrapper')

@pytest.fixture
def mock_time_sleep(mocker):
    """Mocks time.sleep."""
    return mocker.patch('rag_core.infrastructure.llm.openai_adapter.time.sleep')


# --- Test Classes ---

class TestOpenAIAdapterInit:
    def test_init_valid_parameters(self, mock_openai_class, mock_log_wrapper):
        """Scenario 1: Valid initialization."""
        api_key = "sk-testkey123"
        model_name = "gpt-3.5-turbo"
        temperature = 0.8
        max_tokens = 1500
        
        adapter = OpenAIAdapter(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        mock_openai_class.assert_called_once_with(api_key=api_key)
        assert adapter.model_name == model_name
        assert adapter.temperature == temperature
        assert adapter.max_tokens == max_tokens
        mock_log_wrapper.warning.assert_not_called()

    def test_init_invalid_api_key_empty(self, mock_openai_class, mock_log_wrapper):
        """Scenario 2: Invalid API key (empty)."""
        with pytest.raises(ValueError, match="OpenAI API key cannot be empty."):
            OpenAIAdapter(api_key="", model_name="gpt-3.5-turbo")
        mock_openai_class.assert_not_called()

    def test_init_api_key_format_warning(self, mock_openai_class, mock_log_wrapper):
        """Scenario 3: API key format warning (no 'sk-')."""
        api_key_no_prefix = "testkey123"
        OpenAIAdapter(api_key=api_key_no_prefix, model_name="gpt-3.5-turbo")
        
        mock_openai_class.assert_called_once_with(api_key=api_key_no_prefix)
        mock_log_wrapper.warning.assert_called_once_with(
            "OpenAIAdapter",
            "__init__",
            f"OpenAI API key '{api_key_no_prefix}' does not start with 'sk-'. This might be an invalid key."
        )

class TestOpenAIAdapterGenerateResponse:
    @pytest.fixture
    def adapter(self, mock_openai_class, mock_log_wrapper): # Ensure mocks are active
        return OpenAIAdapter(api_key="sk-testkey")

    def test_successful_first_try(self, adapter, mock_openai_client_instance, mock_time_sleep):
        """Scenario 1: Successful (1st try)."""
        prompt = "Translate 'hello' to French."
        response = adapter.generate_response(prompt)
        
        mock_openai_client_instance.chat.completions.create.assert_called_once_with(
            model=adapter.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=adapter.temperature,
            max_tokens=adapter.max_tokens,
            stream=False
        )
        assert response == "Mocked LLM response"
        mock_time_sleep.assert_not_called()

    def test_successful_after_retries(self, adapter, mock_openai_client_instance, mock_log_wrapper, mock_time_sleep):
        """Scenario 2: Successful (after retries)."""
        prompt = "Why is the sky blue?"
        
        # Setup side effects for chat.completions.create
        # First two calls raise OpenAIError, third one returns a valid response
        mock_valid_response = MagicMock()
        mock_valid_response.choices = [MagicMock()]
        mock_valid_response.choices[0].message = MagicMock()
        mock_valid_response.choices[0].message.content = "Finally succeeded"

        mock_openai_client_instance.chat.completions.create.side_effect = [
            OpenAIError("Simulated error 1"),
            APIError("Simulated API error 2"), # APIError is a subclass of OpenAIError
            mock_valid_response 
        ]
        
        response = adapter.generate_response(prompt)
        
        assert response == "Finally succeeded"
        assert mock_openai_client_instance.chat.completions.create.call_count == 3
        assert mock_time_sleep.call_count == 2
        # Check log calls for retries
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter", "generate_response", "OpenAI API error: Simulated error 1. Retrying (1/2)..."
        )
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter", "generate_response", "OpenAI API error: Simulated API error 2. Retrying (2/2)..."
        )


    def test_failure_all_retries(self, adapter, mock_openai_client_instance, mock_log_wrapper, mock_time_sleep):
        """Scenario 3: Failure (all retries)."""
        prompt = "Explain quantum physics simply."
        mock_openai_client_instance.chat.completions.create.side_effect = OpenAIError("Persistent failure")
        
        with pytest.raises(OpenAIError, match="Persistent failure"):
            adapter.generate_response(prompt)
            
        assert mock_openai_client_instance.chat.completions.create.call_count == 3
        assert mock_time_sleep.call_count == 2
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter", "generate_response", "OpenAI API error: Persistent failure. Retrying (1/2)..."
        )
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter", "generate_response", "OpenAI API error: Persistent failure. Retrying (2/2)..."
        )
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter", "generate_response", "OpenAI API error after 3 retries: Persistent failure"
        )


    @pytest.mark.parametrize("error_message, specific_log_message", [
        ("Connection error details", "OpenAI API error: Connection error details"),
        ("Invalid API key provided.", "OpenAI API error: Invalid API key provided.")
    ])
    def test_specific_openai_error_logging(self, adapter, mock_openai_client_instance, mock_log_wrapper, error_message, specific_log_message):
        """Scenario 4: Specific OpenAIError logging (final failure)."""
        prompt = "Test specific error."
        mock_openai_client_instance.chat.completions.create.side_effect = OpenAIError(error_message)

        with pytest.raises(OpenAIError):
            adapter.generate_response(prompt)
        
        # Check the final error log after all retries
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter", "generate_response", f"{specific_log_message} after 3 retries" # Adjusted to match the new format
        )


class TestOpenAIAdapterStreamResponse:
    @pytest.fixture
    def adapter(self, mock_openai_class, mock_log_wrapper):
        return OpenAIAdapter(api_key="sk-testkey")

    def test_successful_stream(self, adapter, mock_openai_client_instance):
        """Scenario 1: Successful stream."""
        prompt = "Stream a short story."
        
        # The default side_effect of mock_openai_client_instance.chat.completions.create
        # already handles streaming mocks.
        
        response_chunks = list(adapter.stream_response(prompt))
        
        mock_openai_client_instance.chat.completions.create.assert_called_once_with(
            model=adapter.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=adapter.temperature,
            max_tokens=adapter.max_tokens,
            stream=True  # Crucial for streaming
        )
        
        collected_content = "".join(response_chunks)
        assert collected_content == "Hello world" # Based on mock_openai_client_instance setup

    def test_openai_error_during_stream(self, adapter, mock_openai_client_instance, mocker):
        """Scenario 2: OpenAIError during stream."""
        prompt = "This stream will fail."
        mock_openai_client_instance.chat.completions.create.side_effect = OpenAIError("Stream connection broken")
        
        # Mock the adapter's own handle_error method for this specific test
        mock_handle_error = mocker.patch.object(adapter, 'handle_error')
        
        # Consume the generator to trigger the exception
        with pytest.raises(OpenAIError, match="Stream connection broken"):
            list(adapter.stream_response(prompt))
            
        mock_handle_error.assert_called_once()
        # Check that the exception passed to handle_error is the one raised
        args, _ = mock_handle_error.call_args
        assert isinstance(args[0], OpenAIError)
        assert "Stream connection broken" in str(args[0])


@pytest.mark.asyncio
class TestOpenAIAdapterAsyncGenerateResponse:
    @pytest.fixture
    def adapter(self, mock_openai_class, mock_log_wrapper):
        return OpenAIAdapter(api_key="sk-testkey")

    async def test_async_generate_response_successful(self, adapter, mocker, mock_log_wrapper):
        """Scenario 1 (async): Success."""
        prompt = "Async prompt"
        expected_sync_response = "Async success via sync mock"
        
        # Mock the synchronous generate_response method on the instance
        mock_sync_generate = mocker.patch.object(adapter, 'generate_response', return_value=expected_sync_response)
        
        result = await adapter.async_generate_response(prompt)
        
        assert result == expected_sync_response
        mock_sync_generate.assert_called_once_with(prompt)
        # Ensure the adapter's own error logging within async_generate_response is not called
        # by filtering out handle_error calls made by the sync generate_response
        # For this test, we only care that async_generate_response itself doesn't log an *additional* error.
        # If generate_response were to fail, it would log via its own handle_error.
        
        # Get all calls to log_wrapper.error
        error_log_calls = mock_log_wrapper.error.call_args_list
        # Filter out calls that might originate from the (mocked) synchronous generate_response's internal error handling
        async_method_specific_logs = [
            c for c in error_log_calls 
            if c.args[0] == "OpenAIAdapter" and c.args[1] == "async_generate_response"
        ]
        assert not async_method_specific_logs


    async def test_async_generate_response_exception(self, adapter, mocker, mock_log_wrapper):
        """Scenario 2 (async): Exception from generate_response."""
        prompt = "Async prompt causing error"
        original_exception = OpenAIError("Error from sync generate_response")
        
        mocker.patch.object(adapter, 'generate_response', side_effect=original_exception)
        
        with pytest.raises(OpenAIError, match="Error from sync generate_response"):
            await adapter.async_generate_response(prompt)
            
        # Check that the async method's own error logging was called
        mock_log_wrapper.error.assert_any_call(
            "OpenAIAdapter",
            "async_generate_response",
            f"Error in async_generate_response: {str(original_exception)}"
        )


@pytest.mark.asyncio
class TestOpenAIAdapterAsyncStreamResponse:
    @pytest.fixture
    def adapter(self, mock_openai_class, mock_log_wrapper):
        return OpenAIAdapter(api_key="sk-testkey")

    async def test_async_stream_response_successful(self, adapter, mocker):
        """Scenario 1 (async stream): Success."""
        prompt = "Async stream prompt"
        expected_chunks = ["chunk1 ", "chunk2"]
        
        # Mock the synchronous stream_response method on the instance
        mock_sync_stream = mocker.patch.object(adapter, 'stream_response', return_value=iter(expected_chunks))
        
        collected_chunks = []
        async for chunk in adapter.async_stream_response(prompt):
            collected_chunks.append(chunk)
            
        assert "".join(collected_chunks) == "".join(expected_chunks)
        mock_sync_stream.assert_called_once_with(prompt)

    async def test_async_stream_response_exception(self, adapter, mocker, mock_log_wrapper):
        """ Test exception propagation from sync stream_response in async version. """
        prompt = "Async stream failing"
        original_exception = OpenAIError("Error from sync stream_response")

        mocker.patch.object(adapter, 'stream_response', side_effect=original_exception)

        with pytest.raises(OpenAIError, match="Error from sync stream_response"):
            async for _ in adapter.async_stream_response(prompt):
                pass
        
        # Assert that the specific log in async_stream_response is called
        mock_log_wrapper.error.assert_any_call(
             "OpenAIAdapter",
             "async_stream_response",
             f"Error in async_stream_response: {str(original_exception)}"
        )


class TestOpenAIAdapterHandleErrorOverridden:
    @pytest.fixture
    def adapter(self, mock_openai_class): # log_wrapper is not directly used by this test's handle_error
        return OpenAIAdapter(api_key="sk-testkey")

    def test_handle_error_calls_super_and_logs_openai_specific(self, adapter, mocker, mock_log_wrapper): # mock_log_wrapper needed here
        """Test overridden handle_error method."""
        mock_super_handle_error = mocker.patch.object(LLMAdapter, 'handle_error') # Patch on the parent
        
        test_exception = Exception("OpenAI detail error")
        adapter.handle_error(test_exception)
        
        mock_super_handle_error.assert_called_once_with(test_exception)
        mock_log_wrapper.error.assert_any_call( # Use any_call if super also logs
            "OpenAIAdapter", # Class name from the overridden method
            "handle_error",  # Method name
            f"OpenAI Specific Error: {str(test_exception)}" # OpenAIAdapter's specific message
        )

# End of test file
