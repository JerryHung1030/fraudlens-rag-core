# src/adapters/openai_adapter.py
import asyncio
from typing import AsyncGenerator
import time
import random

from openai import OpenAI, OpenAIError
from .base_adapter import LLMAdapter
from utils.logging import log_wrapper


class OpenAIAdapter(LLMAdapter):
    """
    使用新版 openai Python 套件的方式，並指定 model="gpt-4o"。
    適用於已配置的 'gpt-4o' 模型或代理 API。
    """
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        :param openai_api_key: OpenAI或代理API的金鑰
        :param temperature: 生成溫度
        :param max_tokens: 回應最大 token 數
        """
        # 驗證 API Key
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is not set or is empty")
        
        if not openai_api_key.startswith("sk-"):
            log_wrapper.warning(
                "OpenAIAdapter",
                "__init__",
                "API key format may be incorrect (should start with 'sk-')"
            )

        # 固定使用 "gpt-4o" 作為 model 名稱
        super().__init__(model=model_name)

        self.temperature = temperature
        self.max_tokens = max_tokens

        # 使用新版 openai.OpenAI(client) 初始化
        self.client = OpenAI(api_key=openai_api_key.strip())
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """
        同步 (blocking) 呼叫 gpt-4o API 取得回覆。
        """
        max_retry = 3
        for i in range(max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30  # 增加超時設定
                )
                return response.choices[0].message.content.strip()
            except OpenAIError as e:
                error_msg = str(e)
                if "Connection" in error_msg:
                    log_wrapper.error(
                        "OpenAIAdapter",
                        "generate_response",
                        f"Connection error: {error_msg}"
                    )
                elif "API key" in error_msg:
                    log_wrapper.error(
                        "OpenAIAdapter",
                        "generate_response",
                        "Invalid or expired API key"
                    )
                else:
                    log_wrapper.error(
                        "OpenAIAdapter",
                        "generate_response",
                        f"OpenAI API error: {error_msg}"
                    )
                
                if i == max_retry - 1:
                    raise  # 最後一次重試失敗時拋出例外
                
                sleep_sec = 2**i + random.random()
                log_wrapper.warning(
                    "OpenAIAdapter",
                    "generate_response",
                    f"[Retry {i + 1}/{max_retry}] Wait {sleep_sec:.1f}s then retry..."
                )
                time.sleep(sleep_sec)

    def stream_response(self, prompt: str):
        """
        同步(阻塞)streaming呼叫 gpt-4o API，yield 逐段回覆。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            # response為迭代器，每個chunk包含 delta
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIError as e:
            self.handle_error(e)

    async def async_generate_response(self, prompt: str) -> str:
        """
        非同步方式執行 generate_response。
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.generate_response, prompt)
        except Exception as e:
            log_wrapper.error(
                "OpenAIAdapter",
                "async_generate_response",
                f"Async generation error: {str(e)}"
            )
            raise

    async def async_stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        非同步方式streaming回傳回應內容。
        """
        loop = asyncio.get_event_loop()

        def _sync_stream():
            for chunk in self.stream_response(prompt):
                yield chunk

        gen = _sync_stream()
        while True:
            try:
                chunk = await loop.run_in_executor(None, next, gen, None)
                if chunk is None:
                    break
                yield chunk
            except StopIteration:
                break

    def handle_error(self, e: Exception) -> None:
        """
        錯誤處理: 記錄log, 並可自行擴充其他行為(通知/重試等)。
        """
        super().handle_error(e)
        log_wrapper.error(
            "OpenAIAdapter",
            "handle_error",
            f"OpenAIAdapter error: {str(e)}"
        )

    async def _generate_with_retry(self, prompt: str, max_retry: int = 3) -> str:
        """重試機制"""
        for i in range(max_retry):
            try:
                return await self._generate(prompt)
            except Exception:
                if i == max_retry - 1:
                    raise
                sleep_sec = (i + 1) * 2
                log_wrapper.warning(
                    "OpenAIAdapter",
                    "_generate_with_retry",
                    f"[Retry {i + 1}/{max_retry}] Wait {sleep_sec:.1f}s then retry..."
                )
                await asyncio.sleep(sleep_sec)
