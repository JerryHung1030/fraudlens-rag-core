import pytest
from unittest.mock import MagicMock # For spec if needed, but mocker.Mock is fine

from rag_core.infrastructure.llm.llm_manager import LLMManager
from rag_core.infrastructure.llm.base_adapter import LLMAdapter # For spec

class TestLLMManagerInit:
    def test_init_default_state(self):
        """
        Test LLMManager instantiation.
        Asserts that manager.adapters is an empty dictionary and 
        manager.default_adapter_name is an empty string.
        """
        manager = LLMManager()
        assert manager.adapters == {}
        assert manager.default_adapter_name == ""

class TestLLMManagerRegisterAdapter:
    def test_register_adapter_adds_to_adapters_dict(self, mocker):
        """
        Test register_adapter.
        Asserts that the adapter is correctly added to the manager.adapters dictionary.
        """
        manager = LLMManager()
        mock_adapter = mocker.Mock(spec=LLMAdapter)
        
        manager.register_adapter("test_adapter", mock_adapter)
        
        assert "test_adapter" in manager.adapters
        assert manager.adapters["test_adapter"] is mock_adapter

class TestLLMManagerSetDefaultAdapter:
    def test_set_default_adapter_updates_name(self):
        """
        Test set_default_adapter.
        Asserts that manager.default_adapter_name is updated.
        """
        manager = LLMManager()
        manager.set_default_adapter("my_default")
        assert manager.default_adapter_name == "my_default"

class TestLLMManagerGetAdapter:
    def test_get_adapter_by_specific_name_exists(self, mocker):
        """Scenario 4a: Get by specific name (adapter exists)."""
        manager = LLMManager()
        mock_adapter1 = mocker.Mock(spec=LLMAdapter)
        manager.register_adapter("adapter1", mock_adapter1)
        
        result = manager.get_adapter("adapter1")
        assert result is mock_adapter1

    def test_get_adapter_by_specific_name_does_not_exist(self, mocker):
        """Scenario 4b: Get by specific name (adapter does not exist)."""
        manager = LLMManager()
        # Register a different adapter to ensure it's not just an empty dict issue
        mock_adapter_other = mocker.Mock(spec=LLMAdapter)
        manager.register_adapter("other_adapter", mock_adapter_other)

        result = manager.get_adapter("non_existent_adapter")
        assert result is None

    def test_get_adapter_default_when_name_is_none_default_set_and_exists(self, mocker):
        """Scenario 4c: Get default adapter when name is None (default is set and adapter exists)."""
        manager = LLMManager()
        mock_adapter_default = mocker.Mock(spec=LLMAdapter)
        manager.register_adapter("default_one", mock_adapter_default)
        manager.set_default_adapter("default_one")
        
        result = manager.get_adapter() # No name, should use default
        assert result is mock_adapter_default

    def test_get_adapter_default_when_name_is_none_default_not_set(self, mocker):
        """Scenario 4d: Get default adapter when name is None (default is not set)."""
        manager = LLMManager()
        mock_adapter1 = mocker.Mock(spec=LLMAdapter)
        manager.register_adapter("adapter1", mock_adapter1)
        # default_adapter_name is "" by default
        
        result = manager.get_adapter()
        assert result is None

    def test_get_adapter_default_when_name_is_none_default_set_but_adapter_not_registered(self, mocker):
        """Scenario 4e: Get default adapter when name is None (default is set, but adapter with that name is not registered)."""
        manager = LLMManager()
        # Register some other adapter to make sure adapters dict isn't empty
        mock_adapter_other = mocker.Mock(spec=LLMAdapter)
        manager.register_adapter("other_adapter", mock_adapter_other)

        manager.set_default_adapter("unregistered_default")
        
        result = manager.get_adapter()
        assert result is None

# End of test file
