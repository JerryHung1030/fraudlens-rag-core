# src/services/compliance_rag_service.py
from typing import Dict, Any
from .base_rag_service import BaseRAGService
from managers.regulations_manager import RegulationsManager


class ComplianceRAGService(BaseRAGService):
    def __init__(
        self,
        embedding_manager,
        vector_store_manager,
        llm_manager,
        regulations_manager: RegulationsManager,
        domain_key="COMPLIANCE",
        prompt_template="Compliance checking prompt template",
        selected_llm_name=None
    ):
        super().__init__(embedding_manager, vector_store_manager, llm_manager, domain_key, prompt_template, selected_llm_name)
        self.regulations_manager = regulations_manager

    def map_regulations(self):
        # PoC: 可做法規比對，或載入
        self.regulations_manager.load_law_documents()
        # ...
        pass

    async def generate_answer(self, user_query: str, filters: Dict[str, Any] = None) -> str:
        # 如果需要先更新法規，再執行
        # self.map_regulations()
        return await super().generate_answer(user_query, filters)
