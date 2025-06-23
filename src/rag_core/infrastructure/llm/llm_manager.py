"""
===============================================================================
    Module Name: llm_manager.py
    Description: Manager for registering and retrieving LLM adapters.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
# src/managers/llm_manager.py
from typing import Dict
from rag_core.infrastructure.llm.base_adapter import LLMAdapter


class LLMManager:
    """
    管理多個 LLMAdapter
    """
    def __init__(self):
        self.adapters: Dict[str, LLMAdapter] = {}
        self.default_adapter_name: str = ""

    def register_adapter(self, name: str, adapter: LLMAdapter):
        self.adapters[name] = adapter

    def get_adapter(self, name: str = None) -> LLMAdapter:
        if not name:
            name = self.default_adapter_name
        return self.adapters.get(name, None)

    def set_default_adapter(self, name: str):
        self.default_adapter_name = name