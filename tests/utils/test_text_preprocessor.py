import pytest
from unittest.mock import patch
from src.rag_core.utils.text_preprocessor import TextPreprocessor

# --- Test Cases for flatten_levels ---

FLATTEN_LEVELS_TEST_CASES = [
    # Scenario 1: depth == 1
    (
        {"level1": [{"sid": "s1", "text": "t1"}, {"sid": "s2", "text": "t2"}]},
        1, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1", "uid": "input-s1", "text": "t1", "side": "input"},
            {"orig_sid": "s2", "group_uid": "input-s2", "uid": "input-s2", "text": "t2", "side": "input"}
        ],
        "depth_1_basic"
    ),
    # Scenario 2: depth == 2, perfect match
    (
        {"level1": [{"sid": "s1", "text": "t1_l1", "level2": [{"sid": "s1_1", "text": "t1_l2"}]}]},
        2, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1_s1_1", "uid": "input-s1_s1_1", "text": "t1_l1\nt1_l2", "side": "input"}
        ],
        "depth_2_perfect_match"
    ),
    # Scenario 3: depth == 3, data is shallower (ends at level 2)
    (
        {"level1": [{"sid": "s1", "text": "t1_l1", "level2": [{"sid": "s1_1", "text": "t1_l2"}]}]},
        3, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1_s1_1", "uid": "input-s1_s1_1", "text": "t1_l1\nt1_l2", "side": "input"}
        ],
        "depth_3_data_shallower"
    ),
    # Scenario 4: depth == 2, data is deeper (has level3, but only processes up to level2 text/uid)
    (
        {"level1": [{"sid": "s1", "text": "t1_l1", "level2": [{"sid": "s1_1", "text": "t1_l2", "level3": [{"sid": "s1_1_1", "text": "t1_l3"}]}]}]},
        2, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1_s1_1", "uid": "input-s1_s1_1", "text": "t1_l1\nt1_l2", "side": "input"}
        ],
        "depth_2_data_deeper"
    ),
    # Scenario 5: Missing levelX key (e.g., level2 missing when depth >= 2)
    (
        {"level1": [{"sid": "s1", "text": "t1_l1"}]}, # No level2 key
        2, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1", "uid": "input-s1", "text": "t1_l1", "side": "input"}
        ],
        "depth_2_missing_level2_key"
    ),
    # Scenario 6: Missing sid or text in items
    (
        {"level1": [{"text": "t_no_sid"}, {"sid": "s_no_text"}]},
        1, "input",
        [
            {"orig_sid": "", "group_uid": "input-", "uid": "input-", "text": "t_no_sid", "side": "input"},
            {"orig_sid": "s_no_text", "group_uid": "input-s_no_text", "uid": "input-s_no_text", "text": "", "side": "input"}
        ],
        "depth_1_missing_sid_or_text"
    ),
    # Scenario 7: Empty levelX list (e.g., level2: [] when depth >= 2)
    (
        {"level1": [{"sid": "s1", "text": "t1", "level2": []}]},
        2, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1", "uid": "input-s1", "text": "t1", "side": "input"}
        ],
        "depth_2_empty_level2_list"
    ),
    # Scenario 8a: Empty input data
    (
        {}, 2, "input", [], "empty_input_data"
    ),
    # Scenario 8b: No level1 (or level1 is empty list)
    (
        {"level1": []}, 2, "input", [], "empty_level1_list"
    ),
    # Additional test: multiple items at level 1, some with deeper levels, some not
    (
        {"level1": [
            {"sid": "s1", "text": "t1_l1", "level2": [{"sid": "s1_1", "text": "t1_l2"}]},
            {"sid": "s2", "text": "t2_l1"}
        ]},
        2, "ref",
        [
            {"orig_sid": "s1", "group_uid": "ref-s1_s1_1", "uid": "ref-s1_s1_1", "text": "t1_l1\nt1_l2", "side": "ref"},
            {"orig_sid": "s2", "group_uid": "ref-s2", "uid": "ref-s2", "text": "t2_l1", "side": "ref"}
        ],
        "depth_2_mixed_items"
    ),
    # Additional test: depth 3, data goes to level 3
    (
        {"level1": [{"sid": "s1", "text": "t1_l1", "level2": [{"sid": "s1_1", "text": "t1_l2", "level3": [{"sid": "s1_1_1", "text": "t1_l3"}]}]}]},
        3, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1_s1_1_s1_1_1", "uid": "input-s1_s1_1_s1_1_1", "text": "t1_l1\nt1_l2\nt1_l3", "side": "input"}
        ],
        "depth_3_full_match"
    ),
    # Additional test: item with missing text at deeper level
     (
        {"level1": [{"sid": "s1", "text": "t1_l1", "level2": [{"sid": "s1_1"}]}]}, # s1_1 has no text
        2, "input",
        [
            {"orig_sid": "s1", "group_uid": "input-s1_s1_1", "uid": "input-s1_s1_1", "text": "t1_l1\n", "side": "input"} # Note the trailing newline
        ],
        "depth_2_missing_text_deeper"
    ),
]

@pytest.mark.parametrize("data, depth, side, expected_output, test_id", FLATTEN_LEVELS_TEST_CASES)
def test_flatten_levels(data, depth, side, expected_output, test_id):
    assert TextPreprocessor.flatten_levels(data, depth, side) == expected_output, f"Test ID: {test_id}"


# --- Test Cases for chunk_text ---

CHUNK_TEXT_TEST_CASES = [
    # Scenario 1: max_len <= 0
    ("test", 0, ["test"], "max_len_zero"),
    ("test", -5, ["test"], "max_len_negative"),
    # Scenario 2: Text length <= max_len
    ("short", 10, ["short"], "text_shorter_than_max_len"),
    ("exact", 5, ["exact"], "text_exact_max_len"),
    # Scenario 3: Text length > max_len
    ("longexampletext", 5, ["longe", "xampl", "etext"], "text_longer_than_max_len_no_padding"),
    ("another example", 7, ["another", " exampl", "e"], "text_longer_with_spaces"),
    ("onemore", 3, ["one", "mor", "e"], "text_longer_odd_len"),
    ("", 5, [""], "empty_string_input"), # Edge case: empty string
]

@pytest.mark.parametrize("text, max_len, expected_chunks, test_id", CHUNK_TEXT_TEST_CASES)
def test_chunk_text_various_scenarios(text, max_len, expected_chunks, test_id):
    assert TextPreprocessor.chunk_text(text, max_len) == expected_chunks, f"Test ID: {test_id}"

def test_chunk_text_utf8_warning(mocker):
    """
    Scenario 4: Warning for UTF-8 length.
    Text: "你好世界" (UTF-8 bytes: 12 chars: 4). max_len=3 (chars).
    Use mocker.patch('builtins.print').
    Assert print called and result is correct.
    """
    mocked_print = mocker.patch('builtins.print')
    text_chinese = "你好世界" # Each char is 3 bytes in UTF-8
    max_len_chars = 3
    
    # Expected behavior: chunks by characters, but warns if byte length of original text > max_len
    # The warning condition in the code is `len(text.encode('utf-8')) > max_len`
    # This seems like a potential misunderstanding in the original code's warning logic,
    # as max_len is for characters, not bytes.
    # However, we test the code as written.
    # If max_len is 3 (chars), len(text.encode('utf-8')) which is 12 > 3, so warning should appear.
    
    expected_chunks = ["你好世", "界"]
    result = TextPreprocessor.chunk_text(text_chinese, max_len_chars)
    
    assert result == expected_chunks
    mocked_print.assert_called_once_with(f"[chunk_text] Warning: text utf8-len={len(text_chinese.encode('utf-8'))} > {max_len_chars}, might chunk incorrectly.")

# End of test file
