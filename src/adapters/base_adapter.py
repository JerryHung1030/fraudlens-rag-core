# src/adapters/base_adapter.py
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    介面/基底類，用於各種 LLM 實作（OpenAI, LocalLlama...）
    """
    def __init__(self, model: any = None):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        """同步生成結果"""
        raise NotImplementedError

    def stream_response(self, prompt: str):
        """同步streaming生成"""
        raise NotImplementedError

    async def async_generate_response(self, prompt: str) -> str:
        """非同步生成結果"""
        raise NotImplementedError

    async def async_stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """非同步streaming生成"""
        raise NotImplementedError

    def handle_error(self, e: Exception) -> None:
        """錯誤處理機制，可自訂 Log/通知/回傳訊息等"""
        logger.error(f"LLMAdapter Error: {str(e)}")
        # 依需求可選擇回傳或 raise
        # raise e
