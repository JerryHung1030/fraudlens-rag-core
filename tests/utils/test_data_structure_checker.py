import pytest
from rag_core.utils.data_structure_checker import DataStructureChecker
from rag_core.exceptions import DataStructureError

# Test cases for check_structure
# Each tuple: (data_to_check, expected_structure, expected_to_pass, error_message_substring_if_fail)

VALID_LIST_OF_DICTS = [
    {"uid": "a", "text": "text_a", "score": 0.1},
    {"uid": "b", "text": "text_b", "score": 0.2}
]
INVALID_TYPE_IN_LIST = [
    {"uid": "a", "text": "text_a", "score": 0.1},
    "not_a_dict"
]
MISSING_KEY_IN_DICT = [
    {"uid": "a", "text": "text_a", "score": 0.1},
    {"uid": "b", "text_content": "text_b", "score": 0.2} # 'text' key is missing
]
WRONG_VALUE_TYPE_IN_DICT = [
    {"uid": "a", "text": "text_a", "score": "not_a_float"}, # score should be float
    {"uid": "b", "text": "text_b", "score": 0.2}
]
EMPTY_LIST_VALID = []

# Expected structures
EXPECTED_DOC_STRUCTURE = {
    "uid": str,
    "text": str,
    "score": float
}

test_cases_check_structure = [
    # Positive cases
    (VALID_LIST_OF_DICTS, [EXPECTED_DOC_STRUCTURE], True, None),
    (EMPTY_LIST_VALID, [EXPECTED_DOC_STRUCTURE], True, None), # Empty list is valid against list of dicts
    ({"name": "test", "version": 1.0}, {"name": str, "version": float}, True, None), # Single dict
    (VALID_LIST_OF_DICTS[0], EXPECTED_DOC_STRUCTURE, True, None), # Single dict from list

    # Negative cases: List checks
    ("not_a_list", [EXPECTED_DOC_STRUCTURE], False, "Data is not a list"),
    (INVALID_TYPE_IN_LIST, [EXPECTED_DOC_STRUCTURE], False, "Item at index 1 is not a dictionary"),
    (MISSING_KEY_IN_DICT, [EXPECTED_DOC_STRUCTURE], False, "Missing key 'text' in dictionary at index 1"),
    (WRONG_VALUE_TYPE_IN_DICT, [EXPECTED_DOC_STRUCTURE], False, "Value for key 'score' in dictionary at index 0 is not of type float"),
    
    # Negative cases: Dict checks
    ({"name": "test", "version": "1"}, {"name": str, "version": float}, False, "Value for key 'version' is not of type float"),
    ({"name": "test"}, {"name": str, "version": float}, False, "Missing key 'version'"),
    ("not_a_dict", {"name": str}, False, "Data is not a dictionary"),
    (None, {"name": str}, False, "Data is not a dictionary"),
    (None, [EXPECTED_DOC_STRUCTURE], False, "Data is not a list"),

    # Nested Structures (Optional, can add if checker supports it, for now assume flat)
    # Example: ({"user": {"name": "John", "age": 30}}, {"user": {"name": str, "age": int}}, True, None),
    # Example: ({"user": {"name": "John", "age": "30"}}, {"user": {"name": str, "age": int}}, False, "Value for key 'age' in nested dict 'user' is not of type int"),
]

@pytest.mark.parametrize("data, structure, should_pass, error_msg_substr", test_cases_check_structure)
def test_check_structure(data, structure, should_pass, error_msg_substr):
    if should_pass:
        try:
            DataStructureChecker.check_structure(data, structure)
        except DataStructureError as e:
            pytest.fail(f"check_structure raised DataStructureError unexpectedly: {e}")
    else:
        with pytest.raises(DataStructureError) as excinfo:
            DataStructureChecker.check_structure(data, structure)
        if error_msg_substr:
            assert error_msg_substr in str(excinfo.value), \
                f"Expected error message to contain '{error_msg_substr}', but got '{str(excinfo.value)}'"

# Test cases for check_list_of_dicts (specific helper, if it has unique logic not covered by check_structure)
# For now, check_structure with [EXPECTED_DOC_STRUCTURE] covers most list of dicts scenarios.
# If check_list_of_dicts has additional specific checks (e.g., non-empty list, uniform dicts beyond types), add them here.

# Example of how one might test a specific method if it existed and had unique logic:
# def test_check_list_of_dicts_custom_scenario():
#     with pytest.raises(DataStructureError, match="some specific message"):
#         DataStructureChecker.check_list_of_dicts([{"key": 1}, {"key": "string"}], {"key": int}, ensure_uniform_keys=True)


# Test cases for specific error messages or complex scenarios can be added as individual test functions if needed.
def test_check_structure_detailed_error_for_wrong_type_in_list():
    data = [{"key": "value"}, "string_element", {"key": "value2"}]
    structure = [{"key": str}]
    with pytest.raises(DataStructureError) as excinfo:
        DataStructureChecker.check_structure(data, structure)
    assert "Item at index 1 is not a dictionary" in str(excinfo.value)
    assert "Expected a dictionary, but got <class 'str'>" in str(excinfo.value)

def test_check_structure_detailed_error_for_missing_key():
    data = [{"key1": "value1"}, {"key2": "value2"}] # key1 is missing in second dict
    structure = [{"key1": str}]
    with pytest.raises(DataStructureError) as excinfo:
        DataStructureChecker.check_structure(data, structure)
    assert "Missing key 'key1' in dictionary at index 1" in str(excinfo.value)

def test_check_structure_detailed_error_for_wrong_value_type():
    data = [{"key1": "value1", "key2": 123}] # key2 should be str
    structure = {"key1": str, "key2": str}
    with pytest.raises(DataStructureError) as excinfo:
        DataStructureChecker.check_structure(data, structure)
    assert "Value for key 'key2' is not of type str" in str(excinfo.value)
    assert "Expected type <class 'str'>, but got <class 'int'>" in str(excinfo.value)

def test_check_structure_with_none_data_and_list_structure():
    """Test case for when data is None and structure expects a list."""
    data = None
    structure = [{"key1": str}]  # Expects a list of dicts
    with pytest.raises(DataStructureError) as excinfo:
        DataStructureChecker.check_structure(data, structure)
    assert "Data is not a list. Expected a list, but got <class 'NoneType'>" in str(excinfo.value)

def test_check_structure_with_none_data_and_dict_structure():
    """Test case for when data is None and structure expects a dict."""
    data = None
    structure = {"key1": str}  # Expects a dict
    with pytest.raises(DataStructureError) as excinfo:
        DataStructureChecker.check_structure(data, structure)
    assert "Data is not a dictionary. Expected a dictionary, but got <class 'NoneType'>" in str(excinfo.value)

# End of test file.
