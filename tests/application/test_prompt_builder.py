import pytest
from unittest.mock import patch, MagicMock
import types

from rag_core.application.prompt_builder import PromptBuilder
from rag_core.exceptions import PromptTooLongError

# A simple mock for the Scenario pydantic model
class MockScenario:
    def __init__(self,
                 llm_name="test_llm",
                 max_prompt_tokens=1000,
                 direction="forward",
                 role_desc="Test Role",
                 reference_desc="Test Reference",
                 input_desc="Test Input",
                 scoring_rule="Test Scoring Rule",
                 cof_threshold=0.1):
        self.llm_name = llm_name
        self.max_prompt_tokens = max_prompt_tokens
        self.direction = direction
        self.role_desc = role_desc
        self.reference_desc = reference_desc
        self.input_desc = input_desc
        self.scoring_rule = scoring_rule
        self.cof_threshold = cof_threshold

@pytest.fixture
def mock_token_counter(mocker):
    """Fixture to mock TokenCounter.count."""
    return mocker.patch('rag_core.application.prompt_builder.TokenCounter.count')

def test_build_prompt_basic_generation(mock_token_counter):
    """
    Scenario 1: Basic prompt generation.
    - Mock TokenCounter.count to return a fixed value.
    - Provide a simple user_query, context_docs, and a scenario.
    - Assert that the generated prompt string contains the user_query and content from context_docs.
    - Assert that the returned number of used candidates is correct.
    """
    # Arrange
    mock_token_counter.return_value = 10  # Each string costs 10 tokens

    prompt_builder = PromptBuilder()
    user_query = "What is the capital of France?"
    context_docs = [
        {'uid': 'doc1', 'text': 'Paris is the capital of France.', 'score': 0.9},
        {'uid': 'doc2', 'text': 'France is a country in Europe.', 'score': 0.8},
    ]
    scenario = MockScenario(max_prompt_tokens=200) # Header (10) + Query (10) + NL (10) +  Ref (10) + Cand1 (10) + NL (10) + Cand2 (10) = 70 tokens, well within limits

    # Act
    prompt, num_used_candidates = prompt_builder.build_prompt(
        user_query=user_query,
        context_docs=context_docs,
        scenario=scenario
    )

    # Assert
    assert user_query in prompt
    assert "Paris is the capital of France." in prompt
    assert "France is a country in Europe." in prompt
    assert num_used_candidates == 2

    # Verify TokenCounter.count was called for header, query, and each candidate text
    # Header parts: role_desc, direction, reference_desc, input_desc, scoring_rule
    # Plus the query, newlines, and each document's text.
    # The exact number of calls can be tricky due to internal formatting,
    # so we'll check key calls for now.
    
    # Expected calls:
    # 1. Header string (constructed internally)
    # 2. User query
    # 3. Newline after query
    # 4. Reference section header
    # 5. Candidate 1 text
    # 6. Newline after candidate 1
    # 7. Candidate 2 text
    # (No newline after last candidate in current implementation)

    # A more robust check would be to verify calls with specific arguments if needed,
    # but for now, let's check if it was called multiple times.
    assert mock_token_counter.call_count > 5 
    mock_token_counter.assert_any_call(scenario.llm_name, user_query)
    mock_token_counter.assert_any_call(scenario.llm_name, context_docs[0]['text'])
    mock_token_counter.assert_any_call(scenario.llm_name, context_docs[1]['text'])

    # Check that the prompt structure is roughly as expected
    assert f"【Role】\n{scenario.role_desc}" in prompt
    assert f"【模式】\n{scenario.direction}" in prompt
    assert f"【參考資料】\n{scenario.reference_desc}" in prompt
    assert f"【任務】\n{scenario.input_desc}" in prompt
    assert f"【評分規則】\n{scenario.scoring_rule}" in prompt
    assert f"【Query】\n{user_query}" in prompt
    assert f"【Candidate 1】(uid: doc1, score: 0.9)\n{context_docs[0]['text']}" in prompt
    assert f"【Candidate 2】(uid: doc2, score: 0.8)\n{context_docs[1]['text']}" in prompt

def test_build_prompt_too_long_header_query_exceeds_max(mock_token_counter):
    """
    Scenario 2: Prompt too long (header + query exceeds max_tokens).
    - Mock TokenCounter.count such that the initial count for header and query
      exceeds scenario.max_prompt_tokens.
    - Use pytest.raises to assert that PromptTooLongError is raised.
    """
    # Arrange
    # Mock TokenCounter.count to return a value that makes header + query too long
    # Let header cost 60 tokens and query cost 50 tokens. Total = 110 tokens.
    def side_effect_count(llm_name, text_to_count):
        if text_to_count == "What is the capital of France?":
            return 50  # Query tokens
        # For any other text (assume it's part of the header construction)
        return 10 # Fixed value for other parts of header, let's say header is constructed of 6 such parts.

    mock_token_counter.side_effect = side_effect_count

    prompt_builder = PromptBuilder()
    user_query = "What is the capital of France?"
    context_docs = [
        {'uid': 'doc1', 'text': 'Paris is the capital of France.', 'score': 0.9}
    ]
    # max_prompt_tokens is 100, but header (6*10=60) + query (50) = 110
    scenario = MockScenario(max_prompt_tokens=100)

    # Act & Assert
    with pytest.raises(PromptTooLongError) as excinfo:
        prompt_builder.build_prompt(
            user_query=user_query,
            context_docs=context_docs,
            scenario=scenario
        )
    assert "Header 和 Query 的總長度" in str(excinfo.value)
    assert "已經超過 max_prompt_tokens" in str(excinfo.value)

def test_build_prompt_too_long_no_candidates_fit(mock_token_counter):
    """
    Scenario 3: Prompt too long (no candidates can fit).
    - Mock TokenCounter.count such that header and query fit, but adding
      the token count of the first candidate (and the newline after it)
      exceeds scenario.max_prompt_tokens.
    - Use pytest.raises to assert that PromptTooLongError is raised with a
      message indicating no candidates could be added.
    """
    # Arrange
    # Header (e.g. 6 parts * 5 tokens/part = 30) + Query (20) + NL (5) = 55 tokens.
    # First candidate (50) + NL (5) = 55 tokens.
    # Total with first candidate = 55 + 55 = 110 tokens.
    # Max tokens = 100.
    def side_effect_count(llm_name, text_to_count):
        if text_to_count == "What is the capital of France?":
            return 20  # Query tokens
        elif "Paris is the capital of France." in text_to_count:
            return 50  # First candidate tokens
        elif text_to_count == "\n":
            return 5 # Newline tokens
        # For any other text (assume it's part of the header construction or reference header)
        return 5

    mock_token_counter.side_effect = side_effect_count

    prompt_builder = PromptBuilder()
    user_query = "What is the capital of France?"
    context_docs = [
        {'uid': 'doc1', 'text': 'Paris is the capital of France.', 'score': 0.9},
        {'uid': 'doc2', 'text': 'France is a country in Europe.', 'score': 0.8},
    ]
    scenario = MockScenario(max_prompt_tokens=100) # Header (30) + Query (20) + NL(5) + Ref_Header(5) = 60. Remaining 40. First cand (50) too big.

    # Act & Assert
    with pytest.raises(PromptTooLongError) as excinfo:
        prompt_builder.build_prompt(
            user_query=user_query,
            context_docs=context_docs,
            scenario=scenario
        )
    assert "沒有任何 Candidate 能放進 prompt" in str(excinfo.value)
    assert "請調整 max_prompt_tokens 或減少 context_docs 數量" in str(excinfo.value)


def test_build_prompt_candidates_truncation(mock_token_counter):
    """
    Scenario 4: Candidates truncation.
    - Provide multiple context_docs (e.g., 3 docs).
    - Mock TokenCounter.count such that header + query + candidate1 + candidate2 fits,
      but header + query + candidate1 + candidate2 + candidate3 exceeds scenario.max_prompt_tokens.
    - Assert that the prompt includes only the first two candidates.
    - Assert that the returned number of used candidates is 2.
    """
    # Arrange
    # Header (30) + Query (20) + NL (5) + Ref_Header (5) = 60 tokens.
    # Cand1 (10) + NL (5) = 15
    # Cand2 (10) + NL (5) = 15
    # Cand3 (10)
    # Total with 2 cands = 60 + 15 + 15 = 90 tokens.
    # Total with 3 cands = 60 + 15 + 15 + 10 = 100 tokens. (If no NL after last)
    # Let max_tokens = 95. So Cand3 should be excluded.
    # If NL is added after last candidate, then Total with 2 cands = 60 + 15 + 15 = 90 tokens.
    # Total with 3 cands = 60 + 15 + 15 + 15 = 105 tokens.
    # Let's assume NL is not added after the last candidate based on current prompt structure.

    def side_effect_count(llm_name, text_to_count):
        if text_to_count == "What is the capital of France?":
            return 20  # Query tokens
        elif "Candidate 1 text" in text_to_count: # context_docs[0]
            return 10
        elif "Candidate 2 text" in text_to_count: # context_docs[1]
            return 10
        elif "Candidate 3 text" in text_to_count: # context_docs[2]
            return 10 # This one should make it exceed
        elif text_to_count == "\n":
            return 5 # Newline tokens
        # For any other text (assume it's part of the header construction or reference header)
        return 5 # e.g. role_desc, direction, reference_desc, input_desc, scoring_rule, ref_header_text. (6 * 5 = 30 for header)

    mock_token_counter.side_effect = side_effect_count

    prompt_builder = PromptBuilder()
    user_query = "What is the capital of France?"
    context_docs = [
        # These will be sorted by score by PromptBuilder
        {'uid': 'doc3', 'text': 'Candidate 3 text', 'score': 0.7},
        {'uid': 'doc1', 'text': 'Candidate 1 text (High score)', 'score': 0.9},
        {'uid': 'doc2', 'text': 'Candidate 2 text (Mid score)', 'score': 0.8},
    ]
    # Header(30) + Query(20) + NL(5) + Ref_Header(5) = 60
    # Cand1_text(10) + NL(5) = 15. Total = 75
    # Cand2_text(10) (no NL after last) = 10. Total = 85. This fits.
    # If Cand2 has NL: Cand2_text(10) + NL(5) = 15. Total = 90. This fits.
    # Cand3_text(10). Total = 90 + 10 = 100. This exceeds.
    # Let's set max_prompt_tokens to 95.
    scenario = MockScenario(max_prompt_tokens=95)


    # Act
    prompt, num_used_candidates = prompt_builder.build_prompt(
        user_query=user_query,
        context_docs=context_docs,
        scenario=scenario
    )

    # Assert
    assert "Candidate 1 text (High score)" in prompt
    assert "Candidate 2 text (Mid score)" in prompt
    assert "Candidate 3 text" not in prompt
    assert num_used_candidates == 2

    # Check order due to sorting
    assert prompt.find("Candidate 1 text (High score)") < prompt.find("Candidate 2 text (Mid score)")

    # Verify token counts.
    # Header (6*5=30) + Query (20) + NL (5) + Ref Header (5) = 60
    # Cand1 (uid: doc1, score: 0.9) + text (10) + NL (5) = 15 (text is 'Candidate 1 text (High score)')
    # Cand2 (uid: doc2, score: 0.8) + text (10) = 10 (text is 'Candidate 2 text (Mid score)')
    # Expected total tokens = 60 + 15 + 10 = 85. This is <= 95.
    # If we add Cand3 (text: 10 tokens) + NL (5), total would be 85 + 10 + 5 = 100. (If NL after each)
    # If no NL after last: 85 + 10 = 95. This would fit exactly.
    # The prompt format is "【Candidate X】(...)\n{text}\n" for all but last, and "【Candidate X】(...)\n{text}" for last.
    # So, after Cand1: 60 + 10(text) + 5(NL) = 75. (Assuming 【Candidate X】(...) part has 0 token cost based on mock)
    # After Cand2: 75 + 10(text) = 85. (No NL after last one)
    # Adding Cand3: 85 (current_prompt) + 5 (NL for previous cand2) + 10 (cand3 text) = 100. This exceeds 95.
    # So, yes, only 2 candidates should be used.

    # Let's refine the side_effect to be more explicit about what text gets what token count
    # to make the test less fragile.
    # Header parts: role_desc, direction, reference_desc, input_desc, scoring_rule (5 * 5 = 25)
    # Query (20)
    # NL_after_query (5)
    # Ref_header_text (5)
    # Total fixed part = 25 + 20 + 5 + 5 = 55
    #
    # Candidate 1 (doc1, score 0.9): Text="Candidate 1 text (High score)" (10 tokens)
    #   - Cand_Header_String: "【Candidate 1】(uid: doc1, score: 0.9)\n" (let's say 5 tokens for this wrapper)
    #   - Cand_Text: 10 tokens
    #   - NL_after_cand_text: 5 tokens
    #   Total for Cand1 block = 5 + 10 + 5 = 20 tokens.
    #   Prompt total = 55 + 20 = 75 tokens.
    #
    # Candidate 2 (doc2, score 0.8): Text="Candidate 2 text (Mid score)" (10 tokens)
    #   - Cand_Header_String: "【Candidate 2】(uid: doc2, score: 0.8)\n" (5 tokens)
    #   - Cand_Text: 10 tokens
    #   (No NL after last candidate's text in the current prompt structure)
    #   Total for Cand2 block = 5 + 10 = 15 tokens.
    #   Prompt total = 75 + 15 = 90 tokens. This fits within 95.
    #
    # Candidate 3 (doc3, score 0.7): Text="Candidate 3 text" (10 tokens)
    #   - Add NL for previous candidate (Cand2): 5 tokens. Current total = 90 + 5 = 95.
    #   - Cand_Header_String: "【Candidate 3】(uid: doc3, score: 0.7)\n" (5 tokens)
    #   - Cand_Text: 10 tokens
    #   Total for Cand3 block = 5 + 10 = 15 tokens.
    #   Hypothetical prompt total = 95 (prompt_after_NL_for_cand2) + 5 (cand3_header) + 10 (cand3_text) = 110. This exceeds 95.
    #
    # So the logic holds.

def test_build_prompt_different_directions(mock_token_counter):
    """
    Scenario 5: Different directions (forward/reverse).
    - Create two scenario objects, one with direction="forward" and another with direction="reverse".
    - Mock TokenCounter.count to allow all content to fit.
    - Assert that the direction-specific text (e.g., "【模式】\nforward" or "【模式】\nreverse")
      appears in the respective prompts.
    """
    # Arrange
    mock_token_counter.return_value = 1 # Make token counts negligible

    prompt_builder = PromptBuilder()
    user_query = "Test query"
    context_docs = [{'uid': 'doc1', 'text': 'Test content', 'score': 0.9}]

    scenario_forward = MockScenario(direction="forward", max_prompt_tokens=200)
    scenario_reverse = MockScenario(direction="reverse", max_prompt_tokens=200)

    # Act
    prompt_forward, _ = prompt_builder.build_prompt(
        user_query=user_query,
        context_docs=context_docs,
        scenario=scenario_forward
    )
    prompt_reverse, _ = prompt_builder.build_prompt(
        user_query=user_query,
        context_docs=context_docs,
        scenario=scenario_reverse
    )

    # Assert
    assert "【模式】\nforward" in prompt_forward
    assert "【模式】\nreverse" in prompt_reverse
    assert "【模式】\nreverse" not in prompt_forward
    assert "【模式】\nforward" not in prompt_reverse

def test_build_prompt_empty_context_docs(mock_token_counter):
    """
    Scenario 6: Empty context_docs.
    - Call build_prompt with an empty context_docs list.
    - Mock TokenCounter.count as needed.
    - Use pytest.raises to assert that PromptTooLongError is raised.
    """
    # Arrange
    mock_token_counter.return_value = 1 # Token counts don't matter much here, just need to avoid other errors.
    prompt_builder = PromptBuilder()
    user_query = "Test query"
    scenario = MockScenario(max_prompt_tokens=200)

    # Act & Assert
    with pytest.raises(PromptTooLongError) as excinfo:
        prompt_builder.build_prompt(
            user_query=user_query,
            context_docs=[], # Empty list
            scenario=scenario
        )
    # Based on current implementation, it raises error if _candidates_str is empty.
    assert "沒有任何 Candidate 能放進 prompt" in str(excinfo.value)

def test_build_prompt_custom_llm_name_in_scenario(mock_token_counter):
    """
    Scenario 7: Custom llm_name in scenario.
    - Provide a scenario with a specific llm_name (e.g., "custom_model_test").
    - Mock TokenCounter.count and ensure it's called with this llm_name
      for token counting.
    """
    # Arrange
    mock_token_counter.return_value = 1 # Token counts don't matter, just checking calls
    custom_llm_name = "custom_model_test_001"

    prompt_builder = PromptBuilder()
    user_query = "Test query for custom LLM."
    context_docs = [
        {'uid': 'doc1', 'text': 'Content for custom LLM.', 'score': 0.9},
    ]
    scenario = MockScenario(llm_name=custom_llm_name, max_prompt_tokens=500)

    # Act
    prompt_builder.build_prompt(
        user_query=user_query,
        context_docs=context_docs,
        scenario=scenario
    )

    # Assert
    # Check that all calls to TokenCounter.count used the custom_llm_name
    # Important texts that should be counted:
    # - scenario.role_desc
    # - scenario.direction
    # - scenario.reference_desc
    # - scenario.input_desc
    # - scenario.scoring_rule
    # - user_query
    # - "\n" (newlines)
    # - "【參考資料】" (reference header text)
    # - context_docs[0]['text']
    # - "【Query】" (query header text)
    # - f"【Candidate 1】(uid: {context_docs[0]['uid']}, score: {context_docs[0]['score']})\n" (candidate header)

    # We can check a few key ones explicitly
    mock_token_counter.assert_any_call(custom_llm_name, scenario.role_desc)
    mock_token_counter.assert_any_call(custom_llm_name, user_query)
    mock_token_counter.assert_any_call(custom_llm_name, context_docs[0]['text'])

    # More general check: iterate through all calls
    for call_args in mock_token_counter.call_args_list:
        args, _ = call_args
        assert args[0] == custom_llm_name, f"TokenCounter.count called with incorrect llm_name: {args[0]}"

# No more tests for now. End of file.
