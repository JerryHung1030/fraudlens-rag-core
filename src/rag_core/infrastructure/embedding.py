"""
===============================================================================
    Module Name: embedding.py
    Description: Embedding manager for text-to-vector conversion using LangChain.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
# src/managers/embedding_manager.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

from utils.logging import log_wrapper
from langchain_openai import OpenAIEmbeddings
from config.settings import config_manager


class EmbeddingManager:
    """
    負責將文字或段落轉為向量 (利用 LangChain Embeddings).
    注意：此 Embeddings 與 LLM 模型可分開設定
    """
    def __init__(
        self,
        openai_api_key: str = None,
        embedding_model_name: str = None
    ):
        """
        :param openai_api_key: OpenAI API 金鑰，若未提供則使用設定檔中的值
        :param embedding_model_name: 嵌入模型名稱，若未提供則使用設定檔中的值
        """
        settings = config_manager.settings
        self.openai_api_key = openai_api_key or settings.api_keys.openai
        self.embedding_model_name = embedding_model_name or settings.embedding.model

        # 如果想用其他embedding,可在此切換
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.openai_api_key
        )
        self._executor = ThreadPoolExecutor(max_workers=settings.thread_pool.embed_pool)

    def generate_embedding(self, text: str) -> List[float]:
        """
        回傳向量 (List[float])
        若發生錯誤則拋出例外，避免回傳空向量導致搜尋結果不可預期
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            if not embedding:
                raise ValueError("Embedding model returned empty result")
            return embedding
        except Exception as e:
            log_wrapper.error(
                "EmbeddingManager",
                "generate_embedding",
                f"Error generating embedding: {e}"
            )
            raise  # 重新拋出例外，讓上層決定如何處理

    async def generate_embedding_async(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.generate_embedding,
            text
        )

    def shutdown(self):
        """顯式關閉執行緒池，避免 RQ fork 時洩漏執行緒"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
