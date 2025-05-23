import pytest
from unittest.mock import MagicMock, patch

from src.rag_core.utils.token_counter import TokenCounter

# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_enc_cache():
    """Automatically clears TokenCounter._enc_cache before each test run."""
    TokenCounter._enc_cache.clear()
    yield # Ensure cleanup happens after test if needed, though clear() is usually enough

@pytest.fixture
def mock_tiktoken_encoding_for_model(mocker):
    """Mocks tiktoken.encoding_for_model."""
    return mocker.patch('src.rag_core.utils.token_counter.encoding_for_model')

@pytest.fixture
def mock_tiktoken_get_encoding(mocker):
    """Mocks tiktoken.get_encoding."""
    return mocker.patch('src.rag_core.utils.token_counter.get_encoding')

@pytest.fixture
def mock_encoder(mocker):
    """Creates a generic mock encoder object."""
    encoder = MagicMock()
    encoder.encode = MagicMock()
    encoder.decode = MagicMock()
    return encoder

# --- Test Classes ---

class TestTokenCounterGetEncoder: # Implicitly tested via public methods
    def test_get_encoder_model_found_by_encoding_for_model(
        self, mock_tiktoken_encoding_for_model, mock_tiktoken_get_encoding, mock_encoder
    ):
        """
        _get_encoder Scenario 1: Model found by encoding_for_model.
        - mock_encoding_for_model returns a specific mock encoder for "gpt-4".
        - Call TokenCounter.count("text", "gpt-4"). mock_encoding_for_model called.
        - Call TokenCounter.count("text2", "gpt-4") again. mock_encoding_for_model NOT called (cache hit).
        """
        model_name = "gpt-4"
        mock_tiktoken_encoding_for_model.return_value = mock_encoder
        mock_encoder.encode.return_value = [1, 2, 3] # Dummy encode output

        # First call
        TokenCounter.count("text", model_name)
        mock_tiktoken_encoding_for_model.assert_called_once_with(model_name)
        mock_tiktoken_get_encoding.assert_not_called()
        mock_encoder.encode.assert_called_once_with("text")

        # Reset mock call counts for the next assertion
        mock_tiktoken_encoding_for_model.reset_mock()
        mock_encoder.encode.reset_mock()
        
        # Second call (cache hit)
        TokenCounter.count("text2", model_name)
        mock_tiktoken_encoding_for_model.assert_not_called()
        mock_tiktoken_get_encoding.assert_not_called()
        mock_encoder.encode.assert_called_once_with("text2") # Encoder from cache is used

    def test_get_encoder_model_not_found_fallback_to_get_encoding(
        self, mock_tiktoken_encoding_for_model, mock_tiktoken_get_encoding, mock_encoder
    ):
        """
        _get_encoder Scenario 2: Model not found, fallback to get_encoding("cl100k_base").
        - mock_encoding_for_model raises KeyError for "unknown_model".
        - mock_get_encoding returns "fallback_encoder" for "cl100k_base".
        - Call TokenCounter.count for "unknown_model". Both mocks called.
        - Call TokenCounter.count again for "unknown_model". Neither mock called (cache hit).
        """
        unknown_model_name = "unknown_model"
        fallback_encoder_name = "cl100k_base"
        
        mock_tiktoken_encoding_for_model.side_effect = KeyError("Model not found by encoding_for_model")
        
        # Create a different mock encoder for the fallback to distinguish
        fallback_mock_encoder = MagicMock()
        fallback_mock_encoder.encode.return_value = [10, 20]
        mock_tiktoken_get_encoding.return_value = fallback_mock_encoder

        # First call
        TokenCounter.count("text", unknown_model_name)
        mock_tiktoken_encoding_for_model.assert_called_once_with(unknown_model_name)
        mock_tiktoken_get_encoding.assert_called_once_with(fallback_encoder_name)
        fallback_mock_encoder.encode.assert_called_once_with("text")

        # Reset mock call counts
        mock_tiktoken_encoding_for_model.reset_mock()
        mock_tiktoken_get_encoding.reset_mock()
        fallback_mock_encoder.encode.reset_mock()

        # Second call (cache hit for fallback)
        TokenCounter.count("text2", unknown_model_name)
        mock_tiktoken_encoding_for_model.assert_not_called()
        mock_tiktoken_get_encoding.assert_not_called()
        fallback_mock_encoder.encode.assert_called_once_with("text2")


class TestTokenCounterCount:
    def test_count_basic(self, mock_tiktoken_encoding_for_model, mock_encoder):
        """
        count Scenario 1: Basic count.
        - Mock mock_encoder.encode("test text") to return [10, 20, 30].
        - mock_encoding_for_model.return_value = mock_encoder.
        - Assert TokenCounter.count("test text", "gpt-4o") == 3.
        """
        test_text = "test text"
        model_name = "gpt-4o"
        expected_tokens = [10, 20, 30]
        
        mock_encoder.encode.return_value = expected_tokens
        mock_tiktoken_encoding_for_model.return_value = mock_encoder
        
        count = TokenCounter.count(test_text, model_name)
        
        assert count == len(expected_tokens)
        mock_encoder.encode.assert_called_once_with(test_text)
        mock_tiktoken_encoding_for_model.assert_called_once_with(model_name)

    # Model usage verification is implicitly covered by TestTokenCounterGetEncoder tests.

class TestTokenCounterTruncate:
    def test_truncate_occurs(self, mock_tiktoken_encoding_for_model, mock_encoder):
        """
        truncate Scenario 1: Truncation occurs.
        - mock_encoder.encode("long example text") returns [1,2,3,4,5].
        - mock_encoder.decode([1,2,3]) returns "long exa".
        - Assert TokenCounter.truncate("long example text", 3, "gpt-4o") == "long exa".
        """
        full_text = "long example text"
        truncated_text = "long exa"
        max_tokens = 3
        model_name = "gpt-4o"
        
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoder.decode.return_value = truncated_text
        mock_tiktoken_encoding_for_model.return_value = mock_encoder
        
        result = TokenCounter.truncate(full_text, max_tokens, model_name)
        
        assert result == truncated_text
        mock_encoder.encode.assert_called_once_with(full_text)
        mock_encoder.decode.assert_called_once_with([1, 2, 3]) # Tokens up to max_tokens
        mock_tiktoken_encoding_for_model.assert_called_once_with(model_name)

    def test_truncate_no_truncation_needed(self, mock_tiktoken_encoding_for_model, mock_encoder):
        """
        truncate Scenario 2: No truncation (max_tokens > actual tokens).
        - mock_encoder.encode("short text") returns [1,2].
        - Assert TokenCounter.truncate("short text", 5, "gpt-4o") == "short text".
        - mock_encoder.decode should NOT be called.
        """
        short_text = "short text"
        max_tokens = 5
        model_name = "gpt-4o"
        
        mock_encoder.encode.return_value = [1, 2]
        mock_tiktoken_encoding_for_model.return_value = mock_encoder
        
        result = TokenCounter.truncate(short_text, max_tokens, model_name)
        
        assert result == short_text
        mock_encoder.encode.assert_called_once_with(short_text)
        mock_encoder.decode.assert_not_called() # Crucial: decode not called if no truncation
        mock_tiktoken_encoding_for_model.assert_called_once_with(model_name)

    @pytest.mark.parametrize("max_tokens_val", [0, -1, -10])
    def test_truncate_max_tokens_less_than_or_equal_to_zero(
        self, mock_tiktoken_encoding_for_model, mock_tiktoken_get_encoding, max_tokens_val
    ):
        """
        truncate Scenario 3: max_tokens <= 0.
        - Assert TokenCounter.truncate("any text", 0, "gpt-4o") == "any text".
        - Assert TokenCounter.truncate("any text", -1, "gpt-4o") == "any text".
        - Ensure tiktoken functions were NOT called.
        """
        text_to_truncate = "any text"
        model_name = "gpt-4o" # Model name doesn't matter here as it shouldn't be used
        
        result = TokenCounter.truncate(text_to_truncate, max_tokens_val, model_name)
        
        assert result == text_to_truncate
        mock_tiktoken_encoding_for_model.assert_not_called()
        mock_tiktoken_get_encoding.assert_not_called()

    # Model usage verification is implicitly covered by TestTokenCounterGetEncoder tests.

# End of test file
