# src/services/fraud_rag_service.py
from typing import List
from .base_rag_service import BaseRAGService
from managers.blacklist_manager import BlacklistManager


class FraudRAGService(BaseRAGService):
    def __init__(
        self,
        embedding_manager,
        vector_store_manager,
        llm_manager,
        blacklist_manager: BlacklistManager,
        domain_key="FRAUD",
        prompt_template="Fraud detection prompt template",
        selected_llm_name=None
    ):
        super().__init__(
            embedding_manager,
            vector_store_manager,
            llm_manager,
            domain_key,
            prompt_template,
            selected_llm_name
        )
        self.blacklist_manager = blacklist_manager

    def check_blacklist(self, text: str) -> List[str]:
        suspicious_urls = self.blacklist_manager.check_urls(text)
        suspicious_line_ids = self.blacklist_manager.check_line_ids(text)
        return suspicious_urls + suspicious_line_ids

    async def generate_answer(self, user_query: str, filters=None) -> str:
        # 先做黑名單檢查
        hits = self.check_blacklist(user_query)
        if hits:
            # 例如: 若 hits 不為空，就顯示 JSON
            # 產生一個 JSON array, 可能表示 "blacklist match"
            return '[{"label":"blacklist","evidence":"","confidence":1.0,"start_idx":-1,"end_idx":-1}]'

        # 否則再做 RAG
        return await super().generate_answer(user_query, filters)

    def build_prompt(self, user_query: str, context: List[str]):
        # or override if you want different instructions, or keep parent
        return super().build_prompt(user_query, context)
