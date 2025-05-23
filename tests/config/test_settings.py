import pytest
import os
import yaml
import importlib
from unittest.mock import mock_open, patch

# Import the specific classes and functions to be tested
from src.config.settings import (
    APIKeysConfig,
    ScenarioConfig,
    Settings,
    ConfigManager,
    get_config_manager,
    PROJECT_ROOT  # Used for path resolution tests
)
import src.config.settings as settings_module

# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_get_config_manager_cache():
    """Automatically clear lru_cache for get_config_manager before each test."""
    get_config_manager.cache_clear()
    yield

@pytest.fixture
def mock_project_root(mocker):
    """Mocks PROJECT_ROOT for consistent path testing."""
    test_root = "/test_project_root"
    mocker.patch('src.config.settings.PROJECT_ROOT', test_root)
    return test_root

# --- Test Pydantic Models Directly ---

class TestAPIKeysConfig:
    def test_api_keys_config_with_env_var(self, mocker):
        """With OPENAI_API_KEY="env_openai_key" mocked, assert cfg.openai."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "env_openai_key"}, clear=True)
        cfg = APIKeysConfig()
        assert cfg.openai == "env_openai_key"

    def test_api_keys_config_without_env_var(self, mocker):
        """Without env var, assert cfg.openai == "" (default)."""
        mocker.patch.dict(os.environ, {}, clear=True) # Ensure env var is not set
        cfg = APIKeysConfig()
        assert cfg.openai == ""

class TestScenarioConfigPathResolution:
    def test_scenario_config_path_resolution(self, mock_project_root):
        """Test path resolution with mocked PROJECT_ROOT."""
        scfg = ScenarioConfig(
            reference_json="data/ref.json",
            input_json="data/in.json",
            output_json="out/res.json"
        )
        expected_ref = os.path.join(mock_project_root, "data/ref.json")
        expected_in = os.path.join(mock_project_root, "data/in.json")
        expected_out = os.path.join(mock_project_root, "out/res.json")
        
        assert scfg.reference_json == expected_ref
        assert scfg.input_json == expected_in
        assert scfg.output_json == expected_out

    def test_scenario_config_with_none_paths(self, mock_project_root):
        """Test ScenarioConfig with None paths."""
        scfg = ScenarioConfig(reference_json=None, input_json=None, output_json=None)
        assert scfg.reference_json is None
        assert scfg.input_json is None
        assert scfg.output_json is None

# --- Test Settings Class (Pydantic BaseSettings behavior) ---

class TestSettingsClass:
    def test_settings_defaults_only(self, mocker):
        """Defaults only: Clear relevant env vars. Check a few default values."""
        mocker.patch.dict(os.environ, {}, clear=True)
        # Reload settings module to re-evaluate Settings with cleared env vars
        importlib.reload(settings_module)
        s = settings_module.Settings()
        
        assert s.llm.model == "gpt-4o"  # Example default
        assert s.vector_db.url == "http://localhost:6333" # Example default
        assert s.system.is_debug is False # Example default

    def test_settings_env_vars_override_defaults(self, mocker):
        """Env vars override defaults."""
        mock_env_vars = {
            "OPENAI_API_KEY": "env_key",
            "QDRANT_URL": "env_q_url",
            "LLM_MODEL": "env_llm",
            "SYSTEM_IS_DEBUG": "true", # Pydantic converts "true" to True for bool
            "SCENARIO_REFERENCE_JSON": "test_ref.json" # Example for nested model
        }
        mocker.patch.dict(os.environ, mock_env_vars, clear=True)
        
        # Reload settings module to ensure Settings picks up new env vars
        importlib.reload(settings_module)
        s = settings_module.Settings()
        
        assert s.api_keys.openai == "env_key"
        assert s.vector_db.url == "env_q_url"
        assert s.llm.model == "env_llm"
        assert s.system.is_debug is True
        # For nested models, check if the environment variable for a sub-field is picked up
        # Note: Pydantic v2 settings_nested_delimiter is '__' by default.
        # The prompt used SYSTEM__IS_DEBUG which is fine.
        # For SCENARIO_REFERENCE_JSON, it would be SCENARIO__REFERENCE_JSON if settings_nested_delimiter was used.
        # However, pydantic also allows direct env var names if they match field names.
        # Let's test a field that would be directly set by an env var if not nested.
        # The current Settings structure does not directly expose SCENARIO_REFERENCE_JSON at top level.
        # It's settings.scenario.reference_json. Pydantic handles this via nested model instantiation.
        # So, we test if the ScenarioConfig within Settings got its value.
        # This requires SCENARIO_REFERENCE_JSON to be correctly interpreted by ScenarioConfig.
        # For BaseSettings, env vars for sub-models are typically prefixed, e.g., RAG_CORE_SCENARIO_REFERENCE_JSON
        # if RAG_CORE_ is the prefix for Settings.
        # Let's assume no prefix for this test as per prompt structure.
        # The prompt seems to imply that SYSTEM_IS_DEBUG works, so direct field names for submodels might be expected.
        # This part of pydantic behavior can be tricky. Let's assume direct mapping for sub-fields if unprefixed.
        # Actually, for nested models, Pydantic v1 used `parent_field_name__child_field_name`.
        # Pydantic v2 uses `model_config = SettingsConfigDict(env_nested_delimiter='__')` by default.
        # So, SCENARIO__REFERENCE_JSON would be the way.
        # The example `SYSTEM_IS_DEBUG` implies the structure `system: SystemConfig`.
        # The prompt's `LLM_MODEL` implies `llm: LLMConfig`.
        # Let's add an env var for a ScenarioConfig field to test this.
        mocker.patch.dict(os.environ, {"SCENARIO_REFERENCE_JSON": "env_scenario_ref.json"}, clear=False) # Add to existing
        importlib.reload(settings_module)  # Reload again after adding new var
        s_reloaded = settings_module.Settings()
        assert s_reloaded.scenario.reference_json.endswith("env_scenario_ref.json") #endswith due to path resolution


# --- Test ConfigManager Instance and _load_settings ---

class TestConfigManager:

    @pytest.fixture(autouse=True)
    def setup_mocks_for_config_manager(self, mocker):
        """Setup common mocks for ConfigManager tests."""
        self.mock_os_path_exists = mocker.patch('src.config.settings.os.path.exists')
        self.mock_open_file = mocker.patch('builtins.open', new_callable=mock_open)
        self.mock_yaml_safe_load = mocker.patch('src.config.settings.yaml.safe_load')

    def test_no_settings_base_yml(self, mocker):
        """No settings.base.yml: Load from env vars and defaults."""
        self.mock_os_path_exists.return_value = False # settings.base.yml does not exist
        mocker.patch.dict(os.environ, {"LLM_MODEL": "env_only_model"}, clear=True)
        
        importlib.reload(settings_module)  # Reload to re-evaluate ConfigManager global state potentially
        manager = settings_module.ConfigManager()  # Test with a fresh instance
        
        assert manager.settings.llm.model == "env_only_model"
        assert manager.settings.vector_db.vector_size == 768 # Default value

    def test_settings_base_yml_loads_and_overrides(self, mocker, mock_project_root):
        """settings.base.yml loads; env vars take precedence for non-YAML fields or if YAML values are Pydantic defaults."""
        self.mock_os_path_exists.return_value = True # settings.base.yml exists
        yaml_content = {
            "llm": {"model": "yaml_model"}, 
            "vector_db": {"collection": "yaml_coll"}
            # vector_db.url is NOT in YAML, so env var should take effect
        }
        self.mock_yaml_safe_load.return_value = yaml_content
        self.mock_open_file.return_value.read.return_value = yaml.dump(yaml_content)


        # Env vars: LLM_MODEL should be overridden by YAML if YAML is not a default.
        # QDRANT_URL should take effect as it's not in YAML.
        # Pydantic's behavior: Env vars have higher precedence than values from YAML file IF the YAML value is a default.
        # If YAML specifies a non-default value, it takes precedence over env vars unless the env var is for a different field.
        # The prompt implies YAML takes precedence for fields it defines.
        # Let's test based on the prompt's interpretation: YAML defined fields override env vars.
        mocker.patch.dict(os.environ, {
            "LLM_MODEL": "env_model_overridden_by_yaml", 
            "QDRANT_URL": "env_q_url_takes_effect"
        }, clear=True)
        
        importlib.reload(settings_module)
        manager = settings_module.ConfigManager()

        assert manager.settings.llm.model == "yaml_model" # YAML value for 'model'
        assert manager.settings.vector_db.collection == "yaml_coll" # YAML value for 'collection'
        assert manager.settings.vector_db.url == "env_q_url_takes_effect" # Env var, as 'url' not in YAML


    def test_settings_base_yml_is_empty(self, mocker):
        """settings.base.yml is empty: Load from env vars and defaults."""
        self.mock_os_path_exists.return_value = True
        self.mock_yaml_safe_load.return_value = {} # Empty YAML
        self.mock_open_file.return_value.read.return_value = yaml.dump({})

        mocker.patch.dict(os.environ, {"LLM_MODEL": "env_model_takes_effect_empty_yaml"}, clear=True)
        
        importlib.reload(settings_module)
        manager = settings_module.ConfigManager()
        
        assert manager.settings.llm.model == "env_model_takes_effect_empty_yaml"
        assert manager.settings.vector_db.port == 6333 # Default

    def test_get_scenario_config(self, mocker):
        """Test ConfigManager.get_scenario_config."""
        # Setup ConfigManager with some scenario data in its settings
        self.mock_os_path_exists.return_value = False # No YAML file
        mocker.patch.dict(os.environ, {
            "SCENARIO_REFERENCE_JSON": "env_ref.json",
            "SCENARIO_INPUT_JSON": "env_in.json",
            "SCENARIO_OUTPUT_JSON": "env_out.json"
        }, clear=True)
        
        importlib.reload(settings_module)
        manager = settings_module.ConfigManager()
        
        # Ensure paths are resolved in the scenario config loaded into settings
        expected_scenario_dict = settings_module.ScenarioConfig(
            reference_json="env_ref.json",
            input_json="env_in.json",
            output_json="env_out.json"
        ).model_dump() # Use model_dump for Pydantic v2

        sc_dict_from_manager = manager.get_scenario_config()
        
        # We need to compare based on the values after path resolution
        # The ScenarioConfig in manager.settings will have resolved paths.
        assert sc_dict_from_manager['reference_json'].endswith("env_ref.json")
        assert sc_dict_from_manager['input_json'].endswith("env_in.json")
        assert sc_dict_from_manager['output_json'].endswith("env_out.json")
        
        # Check against manager.settings.scenario directly
        assert sc_dict_from_manager == manager.settings.scenario.model_dump()


# --- Test get_config_manager Singleton Behavior ---

def test_get_config_manager_singleton(mocker):
    """Test get_config_manager singleton behavior."""
    # Mock os.path.exists to False to simplify ConfigManager instantiation
    mocker.patch('src.config.settings.os.path.exists', return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)

    # Must reload the module if we want to test the global config_manager instance behavior
    # or the caching of get_config_manager across different mock setups IF get_config_manager
    # itself depends on module-level state that mocks would change.
    # For lru_cache, clearing it (done by fixture) is key.
    
    importlib.reload(settings_module)  # Ensure a fresh start for the module context
    manager1 = settings_module.get_config_manager()
    
    importlib.reload(settings_module)  # Simulate a different part of code importing/calling
    manager2 = settings_module.get_config_manager()  # Should get from cache if not reloaded
                                                        # but reload + cache_clear ensures clean test

    # The key is that get_config_manager is cached. Reloading the module
    # itself gives a new function object if we don't re-import carefully.
    # The autouse fixture clear_get_config_manager_cache handles the cache clearing.
    
    # Re-import get_config_manager after reload if module was reloaded
    # For this test, we want to see if two calls to THE SAME get_config_manager function object
    # return the same ConfigManager instance.
    
    # Let's ensure we're calling the same function object for cache to work as expected in test
    current_get_config_manager = settings_module.get_config_manager
    current_get_config_manager.cache_clear() # Clear again just in case

    manager1 = current_get_config_manager()
    manager2 = current_get_config_manager()
    
    assert manager1 is manager2

# End of test file
