# src/services/base_rag_service.py
import logging
import json
from typing import Dict, Any, List
from managers.embedding_manager import EmbeddingManager
from managers.vector_store_manager import VectorStoreManager
from managers.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class BaseRAGService:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store_manager: VectorStoreManager,
        llm_manager: LLMManager,
        domain_key: str,
        prompt_template: str,
        selected_llm_name: str = None
    ):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.domain_key = domain_key
        self.prompt_template = prompt_template
        self.selected_llm_name = selected_llm_name

    def retrieve_context(
        self,
        user_query: str,
        filters: Dict[str, Any] = None,
        k: int = 10
    ) -> List[str]:
        """
        回傳 List[str]，只包含文檔內容，不再帶相似度
        """
        logger.debug(f"retrieve_context: domain={self.domain_key}, user_query={user_query[:50]}..., filters={filters}, k={k}")

        try:
            query_vector = self.embedding_manager.generate_embedding(user_query)
            logger.debug(f"Embedding dimension={len(query_vector)} for user_query(前50): {user_query[:50]}")
            docs = self.vector_store_manager.search_similar(
                domain=self.domain_key,
                query_vector=query_vector,
                k=k,
                filters=filters
            )
            logger.info(f"retrieve_context got {len(docs)} docs for domain={self.domain_key}")
            # 可debug docs content
            if docs:
                logger.debug(f"Sample doc[0]: {docs[0][:60]}...")

            return docs
        except Exception as e:
            logger.error(f"retrieve_context error: {str(e)}")
            return []

    def build_prompt(
        self,
        user_query: str,
        context: List[str]
    ) -> str:
        """
        context: List[str]，每個元素為檢索到的文檔內容，如 '7-16假求職詐騙:高薪可預支薪水'。
        LLM 需輸出 JSON，包含 confidence 由 LLM 自行估計。
        """

        # 每個 Doc 內容通常就是"編號+詐騙類型:子項描述"
        # e.g. '7-16假求職詐騙:高薪可預支薪水'

        context_lines = []
        for idx, doc_text in enumerate(context):
            context_lines.append(f"[Doc #{idx + 1}]\n{doc_text}\n")

        combined_context = "\n".join(context_lines)

        # 要求 LLM 產生 JSON
        instructions = """
你是一個資訊抽取器，請依照以下格式輸出 JSON array，每個物件包含：
{
  "label": "必須與檢索到的Doc文字完全一致(例如 '7-16假求職詐騙:高薪可預支薪水')",
  "evidence": "需要在User Query(貼文)中找到的原始片段(需完全對應)",
  "confidence": <0~1之間數值, 由你判斷>,
  "start_idx": <evidence 在 User Query 中的開始位置>,
  "end_idx": <evidence 在 User Query 中的結束位置>
}

若沒有結果，請輸出 [] (空陣列即可)，不要輸出其他多餘文字。
"""
        prompt = (
            f"{self.prompt_template}\n"
            f"{instructions}\n"
            f"User Query: {user_query}\n"
            f"---Retrieved Documents---\n{combined_context}\n"
        )
        return prompt

    async def generate_answer(self, user_query: str, filters: Dict[str, Any] = None) -> str:
        """
        回傳 JSON string；若無檢索到內容，將回傳空的 JSON array ("[]")。
        置信度 confidence 由 LLM 產生。
        """
        logger.debug(f"generate_answer called with user_query(前50)={user_query[:50]}..., filters={filters}")

        try:
            # 1) 取得檢索結果(純文檔)
            docs = self.retrieve_context(user_query, filters)
            if not docs:
                logger.info("No context retrieved; returning empty JSON.")
                return "[]"

            # 2) 建Prompt
            prompt = self.build_prompt(user_query, docs)

            # 3) 選擇 LLM adapter
            adapter_name = self.selected_llm_name or self.llm_manager.default_adapter_name
            llm_adapter = self.llm_manager.get_adapter(adapter_name)
            if not llm_adapter:
                logger.error(f"No LLM adapter found for name={adapter_name}")
                return "[]"

            # 4) 呼叫 LLM, 要求 JSON
            logger.debug(f"Sending prompt to LLM adapter={adapter_name}, prompt length={len(prompt)}")
            raw_answer = await llm_adapter.async_generate_response(prompt)
            logger.debug(f"LLM raw answer(前80)={raw_answer[:80]}...")

            # 4.1) 移除可能的三引號 code fence (```json 與 ```)，以避免 json.loads 解析錯誤
            cleaned_answer = raw_answer.strip()

            # 若開頭含有 ``` (可能是 ```json 或 ```)，刪除該段直到換行
            if cleaned_answer.startswith("```"):
                idx = cleaned_answer.find("\n")
                if idx != -1:
                    cleaned_answer = cleaned_answer[idx:].strip()
                else:
                    # 若找不到換行，就直接將 ``` 砍掉
                    cleaned_answer = cleaned_answer[3:].strip()

            # 若末尾含有 ```，則去除
            if cleaned_answer.endswith("```"):
                cleaned_answer = cleaned_answer[: cleaned_answer.rfind("```")].strip()

            # 5) 嘗試解析 LLM 回傳
            try:
                results = json.loads(cleaned_answer)
                if not isinstance(results, list):
                    logger.warning("LLM returned JSON but not a list, returning empty.")
                    return "[]"

                updated_list = []
                for item in results:
                    evidence = item.get("evidence", "")
                    # 在 user_query 裡面找 evidence substring
                    start_idx = user_query.find(evidence)
                    if start_idx >= 0:
                        end_idx = start_idx + len(evidence)
                    else:
                        start_idx = -1
                        end_idx = -1

                    # item["confidence"] 由 LLM 提供
                    item["start_idx"] = start_idx
                    item["end_idx"] = end_idx
                    updated_list.append(item)

                final_json = json.dumps(updated_list, ensure_ascii=False)
                logger.info(f"Final JSON result length={len(final_json)}")
                return final_json
            except Exception as parse_e:
                logger.error(f"LLM JSON parse error: {parse_e}, raw_answer={raw_answer}")
                return "[]"

        except Exception as e:
            logger.error(f"generate_answer error: {str(e)}")
            return "[]"
