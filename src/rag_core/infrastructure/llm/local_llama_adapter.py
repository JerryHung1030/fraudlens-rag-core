# src/adapters/local_llama_adapter.py
import asyncio
from typing import AsyncGenerator
from .base_adapter import LLMAdapter


class LocalLlamaAdapter(LLMAdapter):
    def __init__(self, model_path: str, temperature: float = 0.7, max_tokens: int = 1024):
        super().__init__(model=None)
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        # TODO: initialize local model, e.g. llama_cpp_python binding

    def generate_response(self, prompt: str) -> str:
        try:
            # 在本地模型上做推論 (同步)
            # response = self.model(prompt=prompt, temperature=self.temperature, ...)
            return "Local model sync result"
        except Exception as e:
            self.handle_error(e)
            return "Error in LocalLlama generate_response"

    def stream_response(self, prompt: str):
        try:
            yield "Local model streaming part1"
            yield "Local model streaming part2"
        except Exception as e:
            self.handle_error(e)

    async def async_generate_response(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_response, prompt)

    async def async_stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        
        def _sync_gen():
            for chunk in self.stream_response(prompt):
                yield chunk

        gen = _sync_gen()
        while True:
            chunk = await loop.run_in_executor(None, next, gen, None)
            if chunk is None:
                break
            yield chunk

    def handle_error(self, e: Exception) -> None:
        super().handle_error(e)
        # 可擴充更多錯誤處理
