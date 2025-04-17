# src/services/fraud_rag_service.py
import logging
from typing import List, Dict, Any
from .base_rag_service import BaseRAGService
from managers.blacklist_manager import BlacklistManager

logger = logging.getLogger(__name__)


class FraudRAGService(BaseRAGService):
    """
    詐騙偵測：
    - prompt 要求 output 欄位：code、label、evidence、confidence、start_idx、end_idx
    - label = "<code><category>:<desc>"（必須跟檢索 Doc 完全一致）
    - 另外: 若發現黑名單(網址/line id...)，直接回傳 [{"label":"blacklist"...}]
    """

    def __init__(
        self,
        embedding_manager,
        vector_store_manager,
        llm_manager,
        blacklist_manager: BlacklistManager,
        domain_key="FRAUD",
        selected_llm_name=None,
    ):
        super().__init__(
            embedding_manager,
            vector_store_manager,
            llm_manager,
            domain_key,
            selected_llm_name
        )
        self.blacklist_manager = blacklist_manager
        self.prompt_header = (
            "你是詐騙樣式偵測器，根據下方候選樣式，判斷user貼文是否命中詐騙樣式。\n"
            "請直接輸出JSON array, 格式：\n"
            "[\n"
            "  {\n"
            "    \"code\": \"...\",\n"
            "    \"label\": \"...\",\n"
            "    \"evidence\": \"...\",\n"
            "    \"confidence\": 0.95,\n"
            "    \"start_idx\": 10,\n"
            "    \"end_idx\": 20\n"
            "  }\n"
            "]\n"
            "如未命中, 請輸出 []。\n"
        )

    def _hit_blacklist(self, text: str) -> bool:
        return bool(
            self.blacklist_manager.check_urls(text) + self.blacklist_manager.check_line_ids(text)
        )

    def build_prompt(self, user_query: str, context_docs: List[dict]) -> str:
        # context_docs = [{"doc_id":..., "text":"...", "score":...}, ...]
        lines = []
        for i, doc in enumerate(context_docs, start=1):
            lines.append(f"[Doc#{i}] doc_id={doc['doc_id']} sim={doc['score']:.3f}\n{doc['text']}\n")
        docs_txt = "\n".join(lines)

        guide = """
        {
        "code": "7-16",
        "label": "7-16假求職詐騙:高薪可預支薪水",
        "evidence": "貼文中的欺詐關鍵字",
        "confidence": 0.9,
        "doc_id": "7-16",  
        "start_idx": 10,
        "end_idx": 15
        }
        """

        return (
            f"{self.prompt_header}\n"
            f"User Query: {user_query}\n"
            f"---所有候選樣式---\n{docs_txt}\n"
            f"範例: {guide}\n"
            "請務必回傳JSON array，每個物件必須包含 doc_id, code, label, evidence, confidence, start_idx, end_idx"
        )

    async def generate_answer(self, user_query: str, filters: Dict[str, Any] | None = None) -> str:
        if self._hit_blacklist(user_query):
            return '[{"label":"blacklist","evidence":"","confidence":1,"start_idx":-1,"end_idx":-1}]'
        return await super().generate_answer(user_query, filters)
