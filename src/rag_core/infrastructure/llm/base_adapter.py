# src/adapters/base_adapter.py
from typing import AsyncGenerator
from utils import log_wrapper


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
        log_wrapper.error(
            "BaseAdapter",
            "handle_error",
            f"LLMAdapter Error: {str(e)}"
        )

    def generate_response_sync(self, prompt: str) -> str:
        """同步生成結果，帶有錯誤處理"""
        try:
            return self.generate_response(prompt)
        except Exception as e:
            log_wrapper.error(
                "BaseAdapter",
                "generate_response_sync",
                f"LLMAdapter Error: {str(e)}"
            )
            raise
