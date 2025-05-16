# services/rag_engine.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, List

from rag_core.domain.scenario import Scenario
from rag_core.application.prompt_builder import PromptBuilder
from rag_core.application.result_formatter import ResultFormatter
from rag_core.exceptions import (
    PromptTooLongError,
    EmbeddingError,
    VectorSearchError,
    LLMError
)
from utils import log_wrapper


class RAGEngine:
    """
    只做： 根據 scenario & index_info 做 検索 + LLM 推論 + 後處理
    不讀檔案，不做 flatten/ingest。
    """

    def __init__(self, embedding_manager, vector_index, llm_manager):
        self.embedding_manager = embedding_manager
        self.vector_index = vector_index
        self.llm_manager = llm_manager
        # self.semaphore = asyncio.Semaphore(3)  # <- 移除 or 交由 Worker process

    async def generate_answer(
        self,
        user_query: str,
        root_uid: str,
        scenario: Scenario,
        index_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        非同步函式： scenario 為 pydantic 模型
                    index_info 包含 { "collection_name":..., "filters":..., "rag_k":... }
        """
        # 1) 參數
        c_name = index_info["collection_name"]
        filters = index_info.get("filters", {})
        # scenario 內可取 direction / rag_k_xxx
        direction = scenario.direction.lower()

        # 依 direction 選擇 rag_k
        if "rag_k" in index_info:
            rag_k = index_info["rag_k"]
        else:
            rag_k = scenario.rag_k_forward if direction == "forward" else scenario.rag_k_reverse
            rag_k = rag_k or scenario.rag_k

        # 2) embedding
        try:
            vec = await self.embedding_manager.generate_embedding_async(user_query)
        except Exception as e:
            log_wrapper.error(
                "RAGEngine",
                "generate_answer",
                f"Embedding error: {e}"
            )
            raise EmbeddingError(f"向量轉換失敗: {str(e)}")

        # 3) search
        try:
            hits = await self.vector_index.search_async(
                collection_name=c_name,
                query_vector=vec,
                k=rag_k,
                filters=filters
            )
        except Exception as e:
            log_wrapper.error(
                "RAGEngine",
                "generate_answer",
                f"Vector search error: {e}"
            )
            raise VectorSearchError(f"向量搜尋失敗: {str(e)}")

        if not hits: 
            return []  # 真的沒有找到結果

        # 4) build prompt
        try:
            prompt, used_k = PromptBuilder.build_prompt(user_query, hits, scenario)
        except PromptTooLongError as e:
            log_wrapper.error(
                "RAGEngine",
                "generate_answer",
                f"Prompt oversize: {e}"
            )
            raise PromptTooLongError(f"提示詞過長: {str(e)}")

        # 5) LLM
        llm_name = scenario.llm_name or self.llm_manager.default_adapter_name
        adapter = self.llm_manager.get_adapter(llm_name)
        if not adapter:
            raise LLMError(f"找不到 LLM 適配器: {llm_name}")

        try:
            raw_llm_output = await adapter.async_generate_response(prompt)
        except Exception as e:
            log_wrapper.error(
                "RAGEngine",
                "generate_answer",
                f"LLM error: {e}"
            )
            raise LLMError(f"LLM 生成失敗: {str(e)}")

        # 6) format
        cof_threshold = scenario.cof_threshold
        results = ResultFormatter.parse_and_format(
            raw_llm_output=raw_llm_output,
            hits=hits,
            root_uid=root_uid,
            direction=direction,
            llm_name=llm_name,
            rag_k_used=used_k,
            cof_threshold=cof_threshold
        )
        return results

    def generate_answer_sync(
        self,
        user_query: str,
        root_uid: str,
        scenario: Scenario,
        index_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        同步 wrapper，使用 asyncio.run() 執行非同步程式碼。
        建議：直接使用 generate_answer() 的非同步版本。
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在事件迴圈中運行，直接使用 create_task
                return asyncio.create_task(
                    self.generate_answer(user_query, root_uid, scenario, index_info)
                )
            else:
                # 有事件迴圈但未運行，使用 run_until_complete
                return loop.run_until_complete(
                    self.generate_answer(user_query, root_uid, scenario, index_info)
                )
        except RuntimeError:
            # 沒有事件迴圈：使用 asyncio.run()
            try:
                return asyncio.run(
                    asyncio.wait_for(
                        self.generate_answer(user_query, root_uid, scenario, index_info),
                        timeout=30
                    )
                )
            except asyncio.TimeoutError:
                log_wrapper.error(
                    "RAGEngine",
                    "generate_answer_sync",
                    "generate_answer_sync 執行超時"
                )
                raise RuntimeError("操作超時，請稍後重試")
            except Exception as e:
                log_wrapper.error(
                    "RAGEngine",
                    "generate_answer_sync",
                    f"generate_answer_sync 執行錯誤: {e}"
                )
                raise
