import pytest
from src.rag_core.utils.blacklist import BlacklistManager

class TestBlacklistManagerInit:
    def test_init_no_blacklist_db_provided(self):
        """Scenario 1: No blacklist_db provided."""
        manager = BlacklistManager()
        assert manager.blacklist_db == []

    def test_init_blacklist_db_provided(self):
        """Scenario 2: blacklist_db provided."""
        my_list = ["item1", "item2"]
        manager = BlacklistManager(blacklist_db=my_list)
        assert manager.blacklist_db is my_list # Check for identity

# --- Test Cases for check_urls ---
CHECK_URLS_TEST_CASES = [
    # Scenario 1: Empty blacklist
    ([], "http://example.com", [], "empty_blacklist"),
    # Scenario 2: No matches
    (["http://blocked.com"], "http://allowed.com", [], "no_matches"),
    # Scenario 3: One match
    (["http://blocked.com"], "visit http://blocked.com now", ["http://blocked.com"], "one_match"),
    # Scenario 4: Multiple matches
    (["url1.com", "site2.org"], "check url1.com and site2.org", ["url1.com", "site2.org"], "multiple_matches"),
    # Scenario 5: Partial match (substring)
    (["example.com"], "goto http://example.com/path", ["example.com"], "partial_match_substring"),
    # Scenario 6: Empty text input
    (["http://blocked.com"], "", [], "empty_text_input"),
    # Scenario 7: Case sensitivity - no match
    (["Domain.com"], "check domain.com", [], "case_sensitive_no_match"),
    # Scenario 7: Case sensitivity - match
    (["Domain.com"], "check Domain.com", ["Domain.com"], "case_sensitive_match"),
    # Additional: URL in blacklist, no http prefix in text
    (["example.com"], "Visit example.com today", ["example.com"], "no_http_prefix_in_text"),
    # Additional: Multiple occurrences of the same blacklisted URL
    (["bad.com"], "bad.com is bad.com", ["bad.com", "bad.com"], "multiple_occurrences_same_url"),
]

@pytest.mark.parametrize("blacklist, text, expected, test_id", CHECK_URLS_TEST_CASES)
def test_check_urls(blacklist, text, expected, test_id):
    manager = BlacklistManager(blacklist_db=blacklist)
    result = manager.check_urls(text)
    if test_id == "multiple_matches" or test_id == "multiple_occurrences_same_url": # Order might not be guaranteed by findall
        assert set(result) == set(expected), f"Test ID: {test_id}"
    else:
        assert result == expected, f"Test ID: {test_id}"

# --- Test Cases for check_line_ids ---
CHECK_LINE_IDS_TEST_CASES = [
    # Scenario 1: Empty blacklist
    ([], "id: user123", [], "empty_blacklist"),
    # Scenario 2: No matches
    (["official_id"], "my id is test_id", [], "no_matches"),
    # Scenario 3: One match
    (["user456"], "contact user456 for info", ["user456"], "one_match"),
    # Scenario 4: Multiple matches
    (["id_A", "id_B"], "users id_A and id_B are here", ["id_A", "id_B"], "multiple_matches"),
    # Scenario 5: Partial match (substring) - current implementation is exact word match due to \b
    # If BlacklistManager uses regex without word boundaries for line IDs, this test would change.
    # Assuming current implementation (from problem description) implies exact word match for IDs.
    # Let's test based on the idea that it finds the ID if it's present as a "word".
    (["admin"], "line id: admin_user", ["admin"], "partial_match_substring_as_word_boundary"), # "admin" is found in "admin_user"
    (["admin"], "line id: admin", ["admin"], "exact_match_as_word"),
    (["admin"], "line id: myadmin", ["admin"], "prefix_match_as_word_boundary"), # "admin" is found in "myadmin"
    (["admin"], "line id: superadmin", ["admin"], "suffix_match_as_word_boundary"), # "admin" is found in "superadmin"
    (["admin"], "administrator", ["admin"], "substring_in_longer_word"), # "admin" is found in "administrator"
    # Scenario 6: Empty text input
    (["user789"], "", [], "empty_text_input"),
    # Scenario 7: Case sensitivity - no match
    (["UserID"], "check userid", [], "case_sensitive_no_match"),
    # Scenario 7: Case sensitivity - match
    (["UserID"], "check UserID", ["UserID"], "case_sensitive_match"),
    # Additional: Multiple occurrences of the same blacklisted ID
    (["idXYZ"], "idXYZ is repeated idXYZ", ["idXYZ", "idXYZ"], "multiple_occurrences_same_id"),
]

@pytest.mark.parametrize("blacklist, text, expected, test_id", CHECK_LINE_IDS_TEST_CASES)
def test_check_line_ids(blacklist, text, expected, test_id):
    manager = BlacklistManager(blacklist_db=blacklist)
    result = manager.check_line_ids(text)
    # For line IDs, re.findall might return them in order, but using set for safety if order is not guaranteed.
    if "multiple_matches" in test_id or "multiple_occurrences" in test_id:
        assert set(result) == set(expected), f"Test ID: {test_id}"
    else:
        assert result == expected, f"Test ID: {test_id}"

# End of test file
