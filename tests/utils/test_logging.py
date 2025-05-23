import pytest
import importlib
import os
from unittest.mock import MagicMock, patch

# Initial import of the module to be tested. This will be reloaded in tests.
import src.utils.logging

# --- Fixtures ---

@pytest.fixture
def mock_logger(mocker):
    """Mocks src.utils.logging.logger and its methods."""
    logger_instance = MagicMock()
    logger_instance.add = MagicMock()
    logger_instance.critical = MagicMock()
    logger_instance.error = MagicMock()
    logger_instance.warning = MagicMock()
    logger_instance.info = MagicMock()
    logger_instance.debug = MagicMock()
    mocker.patch('src.utils.logging.logger', logger_instance)
    return logger_instance

@pytest.fixture
def mock_system_settings(mocker):
    """Mocks src.utils.logging.config_manager.settings.system."""
    # This mock will be configured per test for is_debug
    system_settings_mock = MagicMock() 
    mocker.patch('src.utils.logging.config_manager.settings.system', system_settings_mock)
    return system_settings_mock

@pytest.fixture
def reloaded_logging_module(mock_logger, mock_system_settings):
    """
    Fixture to reload the logging module after mocks for logger and system_settings are in place.
    This ensures that class variables like IS_DEBUG are set based on the mocked values.
    It returns the reloaded module.
    """
    # Mocks (mock_logger, mock_system_settings) are already active due to pytest fixture order.
    # This fixture's job is to ensure the module is reloaded using those mocks.
    reloaded_module = importlib.reload(src.utils.logging)
    return reloaded_module


# --- Test get_caller ---

@pytest.mark.parametrize("input_path, expected_filename", [
    ("/path/to/some_file.py", "some_file.py"),
    ("another_file.py", "another_file.py"),
    ("no_extension", "no_extension"),
    ("/path/to/deep/nested/module.py", "module.py"),
    ("", "") # Edge case: empty path
])
def test_get_caller(input_path, expected_filename, reloaded_logging_module):
    # get_caller is a static method on log_wrapper in the reloaded module
    result = reloaded_logging_module.log_wrapper.get_caller(input_path)
    assert result == expected_filename

# --- Test compose_msg ---

@pytest.mark.parametrize("module, func, msg, expected_composed_msg", [
    ("ModuleA", "FunctionB", "Log C", "module: [ModuleA] func: [FunctionB] | Log C"),
    ("Main", "run", "Process started", "module: [Main] func: [run] | Process started"),
    ("", "", "", "module: [] func: [] | "), # Edge case: empty inputs
])
def test_compose_msg(module, func, msg, expected_composed_msg, reloaded_logging_module):
    # compose_msg is a static method on log_wrapper in the reloaded module
    result = reloaded_logging_module.log_wrapper.compose_msg(module, func, msg)
    assert result == expected_composed_msg

# --- Test log_wrapper standard logging methods ---

@pytest.mark.parametrize("log_method_name, logger_method_name", [
    ("critical", "critical"),
    ("error", "error"),
    ("warning", "warning"),
    ("info", "info"),
])
def test_log_wrapper_standard_methods(
    log_method_name, logger_method_name, mock_logger, reloaded_logging_module
):
    """
    Tests critical, error, warning, info methods of log_wrapper.
    """
    module_name = "TestModule"
    func_name = "test_function"
    message = f"{log_method_name} message"
    expected_composed_message = f"module: [{module_name}] func: [{func_name}] | {message}"

    # Get the static method from the reloaded log_wrapper class
    log_method_to_call = getattr(reloaded_logging_module.log_wrapper, log_method_name)
    log_method_to_call(module_name, func_name, message)

    # Get the corresponding mock method from the globally patched mock_logger
    mocked_logger_method = getattr(mock_logger, logger_method_name)
    mocked_logger_method.assert_called_once_with(expected_composed_message)

    # Ensure other log levels were not called
    for other_method_name in ["critical", "error", "warning", "info", "debug"]:
        if other_method_name != logger_method_name:
            other_mocked_method = getattr(mock_logger, other_method_name)
            other_mocked_method.assert_not_called()


# --- Test log_wrapper.debug (conditional) ---

def test_log_wrapper_debug_when_is_debug_true(mock_logger, mock_system_settings, reloaded_logging_module):
    """
    Test log_wrapper.debug when IS_DEBUG is True.
    """
    mock_system_settings.is_debug = True # Configure the mock
    
    # Reload the module to pick up the new mock_system_settings.is_debug value
    # The reloaded_logging_module fixture already does this, but if we change mock_system_settings
    # *after* it has run, we might need to reload again.
    # The fixture order should handle this: mock_system_settings runs, then reloaded_logging_module runs.
    # So, this explicit reload might be redundant if IS_DEBUG is set before reloaded_logging_module is resolved.
    # For safety and clarity, especially if is_debug was changed after initial reload:
    reloaded_module_after_set = importlib.reload(src.utils.logging)


    module_name = "DebugModule"
    func_name = "debug_func"
    message = "This is a debug message"
    expected_composed_message = f"module: [{module_name}] func: [{func_name}] | {message}"

    reloaded_module_after_set.log_wrapper.debug(module_name, func_name, message)
    
    mock_logger.debug.assert_called_once_with(expected_composed_message)

def test_log_wrapper_debug_when_is_debug_false(mock_logger, mock_system_settings, reloaded_logging_module):
    """
    Test log_wrapper.debug when IS_DEBUG is False.
    """
    mock_system_settings.is_debug = False # Configure the mock

    # Reload the module to pick up the new mock_system_settings.is_debug value.
    # Similar to the true case, reloaded_logging_module should handle initial setup.
    # For safety if this test runs after the True case and state needs to be reset based on new mock value:
    reloaded_module_after_set = importlib.reload(src.utils.logging)

    module_name = "DebugModule"
    func_name = "debug_func"
    message = "This is a debug message (should not appear)"

    reloaded_module_after_set.log_wrapper.debug(module_name, func_name, message)
    
    mock_logger.debug.assert_not_called()


# --- (Optional) Test module-level logger.add calls ---
# These tests are more complex due to when logger.add is called (at module import time).
# They require patching before the module is loaded or very careful reloading.

@patch('src.utils.logging.os.makedirs')
@patch('src.utils.logging.os.path.exists')
def test_logger_add_calls_on_module_load(mock_path_exists, mock_makedirs, mocker):
    """
    Tests that logger.add is called correctly when the logging module is imported/reloaded.
    This requires careful management of mocks *before* the module is (re)loaded.
    """
    # Ensure mocks are set up *before* the critical import/reload
    mock_path_exists.return_value = False # Simulate log directory does not exist

    # Mock the logger that will be used by the module when it's reloaded
    # This needs to be the same mock object that the module `src.utils.logging` will see.
    # The `mock_logger` fixture patches 'src.utils.logging.logger'.
    # When we reload src.utils.logging, it will pick up this patched logger.
    
    # We need to ensure the logger is mocked *before* the reload operation that triggers `logger.add`.
    # The `mock_logger` fixture handles patching `src.utils.logging.logger`.
    # We will call `importlib.reload` within the test.
    
    local_mock_logger = MagicMock()
    local_mock_logger.add = MagicMock() # Crucial part for this test
    mocker.patch('src.utils.logging.logger', local_mock_logger)


    # Configure system.is_debug for this specific reload context if necessary for logger.add calls
    # Assuming default behavior of logger.add isn't affected by IS_DEBUG for this test
    mock_system_settings_for_add = MagicMock()
    mock_system_settings_for_add.is_debug = False # Or True, depending on what logic logger.add might have
    mocker.patch('src.utils.logging.config_manager.settings.system', mock_system_settings_for_add)


    # Reload the module to trigger the logger.add calls
    importlib.reload(src.utils.logging)

    # Assertions
    mock_makedirs.assert_called_once_with(os.path.join(src.utils.logging.PROJECT_ROOT, "logs"), exist_ok=True)
    
    # Expected calls to logger.add
    # Call 1: Error log file
    log_file_error = os.path.join(src.utils.logging.PROJECT_ROOT, "logs", "error.log")
    local_mock_logger.add.assert_any_call(
        log_file_error, 
        level="ERROR", 
        rotation="10 MB", 
        retention="14 days",
        format=src.utils.logging.LOG_FORMAT, # Use the actual format string from the module
        encoding="utf-8"
    )
    
    # Call 2: Debug log file (assuming is_debug might affect this, but testing default path)
    log_file_debug = os.path.join(src.utils.logging.PROJECT_ROOT, "logs", "debug.log")
    local_mock_logger.add.assert_any_call(
        log_file_debug, 
        level="DEBUG", 
        rotation="10 MB",
        retention="7 days",
        format=src.utils.logging.LOG_FORMAT,
        encoding="utf-8"
    )
    assert local_mock_logger.add.call_count == 2

# End of test file
