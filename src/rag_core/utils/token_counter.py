"""
===============================================================================
    Module Name: token_counter.py
    Description: Token counting and truncation utilities for LLM prompts.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
# utils/token_counter.py
from tiktoken import encoding_for_model, get_encoding


class TokenCounter:
    _enc_cache = {}

    @staticmethod
    def _get_encoder(model: str):
        # 加入 fallback 機制
        if model not in TokenCounter._enc_cache:
            try:
                enc = encoding_for_model(model)
            except KeyError:
                # 非內建 model 時，降回 cl100k_base
                enc = get_encoding("cl100k_base")
            TokenCounter._enc_cache[model] = enc
        return TokenCounter._enc_cache[model]

    @staticmethod
    def count(text: str, model: str = "gpt-4o") -> int:
        enc = TokenCounter._get_encoder(model)
        return len(enc.encode(text))

    @staticmethod
    def truncate(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
        if max_tokens <= 0:
            return text
        enc = TokenCounter._get_encoder(model)
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])
