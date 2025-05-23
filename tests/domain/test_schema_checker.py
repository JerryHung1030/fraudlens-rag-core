import pytest
from rag_core.domain.schema_checker import DataStructureChecker, DataSchemaError

@pytest.fixture
def checker():
    return DataStructureChecker()

# Helper function to create valid nested data
def create_valid_data(levels=5, sid_prefix="s", text_prefix="text", add_extra_keys=False):
    data = {}
    current_level_list = [{"sid": f"{sid_prefix}_root", "text": f"{text_prefix}_root"}]
    if add_extra_keys:
        current_level_list[0]["extra_key_root"] = "extra_value_root"

    if levels == 0: # For is_valid_level_structure, which might get just the list part
        return current_level_list

    data["level1"] = current_level_list
    
    for i in range(1, levels): # Starts from level 2 effectively
        next_level_list = [{"sid": f"{sid_prefix}_l{i+1}_{j}", "text": f"{text_prefix}_l{i+1}_{j}"} for j in range(1)]
        if add_extra_keys:
            next_level_list[0][f"extra_key_l{i+1}"] = f"extra_value_l{i+1}"
            
        # Attach to all items in current_level_list
        for item in current_level_list:
            item[f"level{i+1}"] = next_level_list # Next level key, e.g. level2, level3
        
        current_level_list = next_level_list # Move to the newly created list for the next iteration
        if i + 1 == levels: # If the next level to be populated is the last specified level
            break
    return data

class TestDataStructureCheckerValidate:

    def test_validate_valid_structure_deepest_level(self, checker):
        """Scenario 1: Valid structure (deepest level up to level5)."""
        data = create_valid_data(levels=5)
        try:
            checker.validate(data, mode="default", max_depth=5) 
        except DataSchemaError as e:
            pytest.fail(f"Validation failed unexpectedly: {e.details}")

    def test_validate_valid_structure_shallower_level(self, checker):
        """Scenario 2: Valid structure (shallower level, e.g., up to level2)."""
        data = create_valid_data(levels=2)
        try:
            checker.validate(data, mode="default", max_depth=2)
        except DataSchemaError as e:
            pytest.fail(f"Validation failed unexpectedly: {e.details}")
            
    def test_validate_valid_structure_with_extra_keys(self, checker):
        """Test valid structure with extra keys at various levels."""
        data = create_valid_data(levels=3, add_extra_keys=True)
        try:
            checker.validate(data, mode="default", max_depth=3)
        except DataSchemaError as e:
            pytest.fail(f"Validation with extra keys failed unexpectedly: {e.details}")


    def test_validate_invalid_top_level_not_a_dict(self, checker):
        """Scenario 3: Invalid: Top level not a dict."""
        data = "not a dictionary"
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default")
        assert "Top-level data is not a dict at path 'root'" in excinfo.value.details[0]

    def test_validate_invalid_missing_level1_key(self, checker):
        """Scenario 4: Invalid: Missing level1 key."""
        data = {"level2": []} 
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default")
        assert "Missing key 'level1' at path 'root'" in excinfo.value.details[0]
        
    @pytest.mark.parametrize("level_key_to_make_invalid, path_segment", [
        ("level1", "root"),
        ("level2", "root/level1[0]"),
        ("level3", "root/level1[0]/level2[0]")
    ])
    def test_validate_invalid_level_x_not_a_list(self, checker, level_key_to_make_invalid, path_segment):
        """Scenario 5: Invalid: levelX is not a list."""
        data = create_valid_data(levels=5) # Create a valid structure first
        
        # Introduce the error
        if level_key_to_make_invalid == "level1":
            data["level1"] = "not a list"
        elif level_key_to_make_invalid == "level2":
            data["level1"][0]["level2"] = "not a list"
        elif level_key_to_make_invalid == "level3":
            data["level1"][0]["level2"][0]["level3"] = "not a list"
            
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default")
        
        assert f"'{level_key_to_make_invalid}' should be a list at path '{path_segment}'" in excinfo.value.details[0]


    @pytest.mark.parametrize("level_to_corrupt, expected_path_segment", [
        (1, "root/level1[0]"),
        (2, "root/level1[0]/level2[0]"),
        (3, "root/level1[0]/level2[0]/level3[0]")
    ])
    def test_validate_invalid_item_in_level_x_list_not_a_dict(self, checker, level_to_corrupt, expected_path_segment):
        """Scenario 6: Invalid: Item in levelX list is not a dict."""
        data = create_valid_data(levels=level_to_corrupt) # Create data up to the point of corruption
        
        # Corrupt the data
        if level_to_corrupt == 1:
            data["level1"] = ["not a dict"]
        elif level_to_corrupt == 2:
            data["level1"][0]["level2"] = ["not a dict"]
        elif level_to_corrupt == 3:
            data["level1"][0]["level2"][0]["level3"] = ["not a dict"]
            
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default", max_depth=level_to_corrupt)
        assert f"Item at '{expected_path_segment}' is not a dict" in excinfo.value.details[0]


    @pytest.mark.parametrize("item_data_modifier, expected_msg_part, path_segment_suffix", [
        (lambda item: item.pop("sid"), "Missing or invalid 'sid'", ""), # Missing sid
        (lambda item: item.pop("text"), "Missing or invalid 'text'", ""), # Missing text
        (lambda item: item.update({"sid": 123}), "Missing or invalid 'sid'", ""), # sid not a string
        (lambda item: item.update({"text": 123}), "Missing or invalid 'text'", ""), # text not a string
    ])
    def test_validate_invalid_missing_or_invalid_sid_text_at_level1(self, checker, item_data_modifier, expected_msg_part, path_segment_suffix):
        """Scenario 7a: Invalid sid/text at level1."""
        data = {"level1": [{"sid": "s1", "text": "t1"}]}
        item_data_modifier(data["level1"][0]) # Modify the item

        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default")
        assert f"{expected_msg_part} at 'root/level1[0]{path_segment_suffix}'" in excinfo.value.details[0]

    def test_validate_invalid_sid_text_at_nested_level2(self, checker):
        """Scenario 7b: Invalid sid/text at level2."""
        data = create_valid_data(levels=2)
        data["level1"][0]["level2"][0]["sid"] = 123 # Invalid sid type at level2 item 0
        
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default", max_depth=2)
        assert "Missing or invalid 'sid' at 'root/level1[0]/level2[0]'" in excinfo.value.details[0]


    def test_validate_valid_structure_with_max_depth_param(self, checker):
        """Scenario 8a: Data valid to level3. Call validate with max_depth=2. Assert no error."""
        data = create_valid_data(levels=3)
        try:
            checker.validate(data, mode="default", max_depth=2)
        except DataSchemaError as e:
            pytest.fail(f"Validation failed unexpectedly with max_depth=2: {e.details}")

    def test_validate_error_at_level3_max_depth_2_no_error(self, checker):
        """Scenario 8b: Data has error at level3. Call validate with max_depth=2. Assert no error."""
        data = create_valid_data(levels=2) # Valid up to L2 items
        data["level1"][0]["level2"][0]["level3"] = "not a list" # Error at level 3 structure
        try:
            checker.validate(data, mode="default", max_depth=2) # max_depth stops before L3
        except DataSchemaError as e:
            pytest.fail(f"Validation failed unexpectedly with max_depth=2 and error at level3: {e.details}")

    def test_validate_error_at_level3_max_depth_3_raises_error(self, checker):
        """Scenario 8c: Data has error at level3. Call validate with max_depth=3. Assert DataSchemaError."""
        data = create_valid_data(levels=2) 
        data["level1"][0]["level2"][0]["level3"] = "this is not a list" 
        
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default", max_depth=3)
        assert "'level3' should be a list at path 'root/level1[0]/level2[0]'" in excinfo.value.details[0]


    def test_validate_multiple_errors(self, checker):
        """Scenario 9: Multiple errors. Assert DataSchemaError and check details."""
        data = {
            "level1": [
                {"sid": 123, "text": "text1"},      # Error 1: sid not string
                {"text": "text2"},                  # Error 2: missing sid
                {"sid": "s3", "level2": "notalist"} # Error 3: missing text, Error 4: level2 not list
            ]
        }
        with pytest.raises(DataSchemaError) as excinfo:
            checker.validate(data, mode="default", max_depth=2) # max_depth=2 to catch level2 error
        
        details_str = "; ".join(excinfo.value.details)
        assert "Missing or invalid 'sid' at 'root/level1[0]'" in details_str
        assert "Missing or invalid 'sid' at 'root/level1[1]'" in details_str
        assert "Missing or invalid 'text' at 'root/level1[2]'" in details_str # This error
        assert "'level2' should be a list at path 'root/level1[2]'" in details_str # This error path for level2
        # The number of errors might be tricky due to cascading, let's check for key messages
        assert len(excinfo.value.details) == 4


class TestDataStructureCheckerIsValidLevelStructure:
    # is_valid_level_structure uses self.validate internally.
    # The behavior of validate is already extensively tested above.
    # These tests will focus on the boolean return value of is_valid_level_structure.

    def test_is_valid_level_structure_true_for_valid_data(self, checker):
        data = create_valid_data(levels=2)
        assert checker.is_valid_level_structure(data) == True

    def test_is_valid_level_structure_false_for_invalid_data(self, checker):
        data = {"level1": "not a list"}
        assert checker.is_valid_level_structure(data) == False
        
    def test_is_valid_level_structure_empty_dict(self, checker):
        # validate will raise "Missing key 'level1'"
        assert checker.is_valid_level_structure({}) == False

    def test_is_valid_level_structure_top_level_not_dict(self, checker):
        assert checker.is_valid_level_structure("string") == False
        
    def test_is_valid_level_structure_respects_max_depth_via_validate(self, checker):
        """Test that is_valid_level_structure implicitly uses default max_depth of validate (5)."""
        # This test assumes `validate` uses its default `max_depth` if not specified by `is_valid_level_structure`
        # The current `is_valid_level_structure` does not pass `max_depth`.
        # It calls `self.validate(data, mode="")` which implies `max_depth` defaults to 5.
        data_valid_to_l2_error_at_l3 = create_valid_data(levels=2)
        data_valid_to_l2_error_at_l3["level1"][0]["level2"][0]["level3"] = "error here"

        # If validate is called with max_depth=2, it should pass.
        # If validate is called with max_depth=3 (or default 5), it should fail.
        # Since is_valid_level_structure doesn't set max_depth, validate's default is used.
        
        # To test this properly, we'd need to know if `validate`'s default `max_depth` is used.
        # The current `validate` signature `max_depth: int = 5`.
        # So, errors beyond level 5 would be ignored, but errors at level 3 (like above) would be caught.
        assert checker.is_valid_level_structure(data_valid_to_l2_error_at_l3) == False


class TestDataSchemaError:
    def test_data_schema_error_instantiation_and_str(self):
        """Test DataSchemaError instantiation and string representation."""
        details = ["Error detail 1", "Another error here"]
        error = DataSchemaError(details=details)
        
        error_str = str(error)
        # The base class __init__ is "Data schema validation failed"
        # The __str__ method is "DataSchemaError: {'; '.join(self.details)}"
        assert "DataSchemaError: Error detail 1; Another error here" == error_str
        assert error.details == details

    def test_data_schema_error_single_detail(self):
        details = ["A single specific error occurred"]
        error = DataSchemaError(details=details)
        assert "DataSchemaError: A single specific error occurred" == str(error)
        assert error.details == details

    def test_data_schema_error_empty_details(self):
        # While the type hint is List[str], it could technically be called with an empty list.
        details = []
        error = DataSchemaError(details=details)
        assert "DataSchemaError: " == str(error) # Joins to empty string
        assert error.details == []

    def test_data_schema_error_is_exception(self):
        error = DataSchemaError(details=["test"])
        assert isinstance(error, Exception)

