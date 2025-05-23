import pytest
import json
import os
from unittest.mock import MagicMock, mock_open, patch

# Import the functions to be tested from cli_main
from src.interfaces.cli_main import load_json_file, setup_core

# Import classes that will be mocked (for type hinting and spec if needed)
from src.rag_core.infrastructure.embedding import EmbeddingManager
from src.rag_core.domain.schema_checker import DataStructureChecker
from qdrant_client import QdrantClient
from src.rag_core.infrastructure.vector_store import VectorIndex
from src.rag_core.infrastructure.llm.llm_manager import LLMManager
from src.rag_core.infrastructure.llm.openai_adapter import OpenAIAdapter
from src.rag_core.infrastructure.llm.local_llama_adapter import LocalLlamaAdapter


# --- Fixtures ---

@pytest.fixture
def mock_log_wrapper_cli(mocker):
    """Mocks log_wrapper used in cli_main.py."""
    return mocker.patch('src.interfaces.cli_main.log_wrapper')

# --- Tests for load_json_file ---

class TestLoadJsonFile:
    def test_load_json_file_successful(self, mocker, mock_log_wrapper_cli):
        """Scenario 1: Successful load."""
        file_path = "dummy.json"
        json_content_str = '{"key": "value"}'
        expected_data = {"key": "value"}

        m = mock_open(read_data=json_content_str)
        mocker.patch('builtins.open', m)
        
        # Mock json.load within the context of src.interfaces.cli_main
        mock_json_load = mocker.patch('src.interfaces.cli_main.json.load', return_value=expected_data)

        result = load_json_file(file_path)

        m.assert_called_once_with(file_path, 'r', encoding='utf-8')
        mock_json_load.assert_called_once_with(m()) # m() is the file handle
        assert result == expected_data
        mock_log_wrapper_cli.error.assert_not_called()

    def test_load_json_file_not_found_error(self, mocker, mock_log_wrapper_cli):
        """Scenario 2: FileNotFoundError."""
        file_path = "nonexistent.json"
        
        mocker.patch('builtins.open', side_effect=FileNotFoundError(f"No such file: {file_path}"))

        with pytest.raises(FileNotFoundError):
            load_json_file(file_path)
        
        mock_log_wrapper_cli.error.assert_called_once_with(
            "load_json_file", # module_name (derived from func name)
            "load_json_file", # func_name
            f"File not found: {file_path}"
        )

    def test_load_json_file_decode_error(self, mocker, mock_log_wrapper_cli):
        """Scenario 3: json.JSONDecodeError."""
        file_path = "bad.json"
        invalid_json_str = "invalid json"

        m = mock_open(read_data=invalid_json_str)
        mocker.patch('builtins.open', m)
        
        # Mock json.load to raise JSONDecodeError
        # The error instance needs 'msg', 'doc', 'pos' arguments for standard representation
        decode_error = json.JSONDecodeError("Expecting value", invalid_json_str, 0)
        mock_json_load = mocker.patch('src.interfaces.cli_main.json.load', side_effect=decode_error)

        with pytest.raises(json.JSONDecodeError):
            load_json_file(file_path)
        
        m.assert_called_once_with(file_path, 'r', encoding='utf-8')
        mock_json_load.assert_called_once_with(m())
        mock_log_wrapper_cli.error.assert_called_once_with(
            "load_json_file", # module_name
            "load_json_file", # func_name
            f"Error decoding JSON from file {file_path}: Expecting value: line 1 column 1 (char 0)"
        )

# --- Tests for setup_core ---

@pytest.fixture
def mock_config_manager_for_setup_core(mocker):
    """Mocks config_manager and its nested settings for setup_core."""
    mock_cfg_mgr = MagicMock()
    
    # APIKeysConfig
    mock_cfg_mgr.settings.api_keys.openai = "test_openai_key_from_config"
    
    # EmbeddingConfig
    mock_cfg_mgr.settings.embedding.model = "text-embedding-ada-002"
    
    # VectorDBConfig
    mock_cfg_mgr.settings.vector_db.collection = "default_collection"
    mock_cfg_mgr.settings.vector_db.vector_size = 1536
    mock_cfg_mgr.settings.vector_db.qdrant_host = "qdrant_test_host"
    mock_cfg_mgr.settings.vector_db.qdrant_port = 1234
    # For QdrantClient, we need url or host/port. The code uses host/port if url is default.
    # Let's assume default url and host/port are used.
    mock_cfg_mgr.settings.vector_db.url = "http://localhost:6333" # Default, so host/port will be used

    # LLMConfig (for OpenAIAdapter)
    mock_cfg_mgr.settings.llm.model = "gpt-3.5-turbo-config"
    mock_cfg_mgr.settings.llm.temperature = 0.75
    mock_cfg_mgr.settings.llm.max_tokens = 2048
    
    # LocalLlamaConfig (specific for LocalLlamaAdapter)
    # Assuming LocalLlamaAdapter takes these from config_manager.settings.llm as well,
    # or has its own section. The prompt implies it uses fixed "models/llama.bin".
    # The actual LocalLlamaAdapter in the codebase takes model_path, temperature, max_tokens.
    # Let's assume these map from a 'local_llama' section or similar if different from 'llm'.
    # For simplicity, let's assume it also uses settings.llm for temp/max_tokens and has a fixed path.
    # Or, if setup_core uses a dedicated config path for llama_model_path:
    mock_cfg_mgr.settings.local_llama.model_path = "config_path/to/llama.bin"
    mock_cfg_mgr.settings.local_llama.temperature = 0.6
    mock_cfg_mgr.settings.local_llama.max_tokens = 1024


    mocker.patch('src.interfaces.cli_main.config_manager', mock_cfg_mgr)
    return mock_cfg_mgr

@pytest.fixture
def mock_embedding_manager_class(mocker):
    instance = MagicMock(spec=EmbeddingManager)
    return mocker.patch('src.interfaces.cli_main.EmbeddingManager', return_value=instance), instance

@pytest.fixture
def mock_data_structure_checker_class(mocker):
    instance = MagicMock(spec=DataStructureChecker)
    return mocker.patch('src.interfaces.cli_main.DataStructureChecker', return_value=instance), instance

@pytest.fixture
def mock_qdrant_client_class_cli(mocker): # Renamed to avoid conflict if used elsewhere
    instance = MagicMock(spec=QdrantClient)
    return mocker.patch('src.interfaces.cli_main.QdrantClient', return_value=instance), instance

@pytest.fixture
def mock_vector_index_class(mocker):
    instance = MagicMock(spec=VectorIndex)
    return mocker.patch('src.interfaces.cli_main.VectorIndex', return_value=instance), instance

@pytest.fixture
def mock_llm_manager_class(mocker):
    instance = MagicMock(spec=LLMManager)
    instance.register_adapter = MagicMock()
    instance.set_default_adapter = MagicMock()
    return mocker.patch('src.interfaces.cli_main.LLMManager', return_value=instance), instance

@pytest.fixture
def mock_openai_adapter_class(mocker):
    instance = MagicMock(spec=OpenAIAdapter)
    return mocker.patch('src.interfaces.cli_main.OpenAIAdapter', return_value=instance), instance

@pytest.fixture
def mock_local_llama_adapter_class(mocker):
    instance = MagicMock(spec=LocalLlamaAdapter)
    return mocker.patch('src.interfaces.cli_main.LocalLlamaAdapter', return_value=instance), instance


class TestSetupCore:
    def test_setup_core_initialization_and_wiring(
        self,
        mock_config_manager_for_setup_core, # Provides settings
        mock_embedding_manager_class,
        mock_data_structure_checker_class,
        mock_qdrant_client_class_cli,
        mock_vector_index_class,
        mock_llm_manager_class,
        mock_openai_adapter_class,
        mock_local_llama_adapter_class,
        mock_log_wrapper_cli # To ensure it's available if setup_core logs
    ):
        cfg_settings = mock_config_manager_for_setup_core.settings
        
        # Get mocked classes and their instances from fixtures
        MockEmbeddingManager, embed_mgr_instance = mock_embedding_manager_class
        MockDataStructureChecker, checker_instance = mock_data_structure_checker_class
        MockQdrantClient, qdrant_client_instance = mock_qdrant_client_class_cli
        MockVectorIndex, vector_index_instance = mock_vector_index_class
        MockLLMManager, llm_mgr_instance = mock_llm_manager_class
        MockOpenAIAdapter, openai_adapter_instance = mock_openai_adapter_class
        MockLocalLlamaAdapter, local_llama_adapter_instance = mock_local_llama_adapter_class

        # Call the function to test
        result_embed_mgr, result_vector_index, result_llm_mgr = setup_core()

        # Assertions
        MockEmbeddingManager.assert_called_once_with(
            openai_api_key=cfg_settings.api_keys.openai,
            embedding_model_name=cfg_settings.embedding.model
        )
        MockDataStructureChecker.assert_called_once_with()
        
        # QdrantClient uses host/port if url is default
        # Assuming default URL scenario based on how QdrantClient is typically used in VectorIndex
        MockQdrantClient.assert_called_once_with(
            host=cfg_settings.vector_db.qdrant_host,
            port=cfg_settings.vector_db.qdrant_port
        )
        # Or if it used URL:
        # MockQdrantClient.assert_called_once_with(url=cfg_settings.vector_db.url)


        MockVectorIndex.assert_called_once_with(
            embedding_manager=embed_mgr_instance,
            data_checker=checker_instance,
            qdrant_client=qdrant_client_instance, # The instance returned by MockQdrantClient
            collection_name=cfg_settings.vector_db.collection,
            vector_size=cfg_settings.vector_db.vector_size
        )
        
        MockLLMManager.assert_called_once_with()
        
        MockOpenAIAdapter.assert_called_once_with(
            api_key=cfg_settings.api_keys.openai,
            model_name=cfg_settings.llm.model,
            temperature=cfg_settings.llm.temperature,
            max_tokens=cfg_settings.llm.max_tokens
        )
        
        # The prompt says LocalLlamaAdapter uses "models/llama.bin" fixed path
        # But the fixture mock_config_manager_for_setup_core provides a config path.
        # Let's assume setup_core uses the fixed path as per prompt for this test.
        # If setup_core is changed to use config, this assertion needs to change.
        fixed_llama_model_path = "models/llama.bin" 
        MockLocalLlamaAdapter.assert_called_once_with(
            # model_path=cfg_settings.local_llama.model_path, # If using config path
            model_path=fixed_llama_model_path, # As per prompt's specific instruction for setup_core
            temperature=cfg_settings.local_llama.temperature, # Assuming these come from a specific 'local_llama' config section
            max_tokens=cfg_settings.local_llama.max_tokens
        )

        llm_mgr_instance.register_adapter.assert_any_call("openai", openai_adapter_instance)
        llm_mgr_instance.register_adapter.assert_any_call("llama", local_llama_adapter_instance)
        assert llm_mgr_instance.register_adapter.call_count == 2
        
        llm_mgr_instance.set_default_adapter.assert_called_once_with("openai")
        
        assert result_embed_mgr is embed_mgr_instance
        assert result_vector_index is vector_index_instance
        assert result_llm_mgr is llm_mgr_instance

# End of test file
