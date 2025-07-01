"""
===============================================================================
    Module Name: exceptions.py
    Description: Custom exceptions for prompt, embedding, vector search, and LLM errors.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""


class PromptTooLongError(Exception):
    """提示文字過長，超過模型可處理的最大長度限制"""
    def __init__(self, message: str = "提示文字過長"):
        self.message = message
        super().__init__(self.message)


class EmbeddingError(Exception):
    """嵌入模型錯誤，無法將文字轉換為向量表示"""
    def __init__(self, message: str = "嵌入模型錯誤"):
        self.message = message
        super().__init__(self.message)


class VectorSearchError(Exception):
    """向量搜尋錯誤，無法在向量資料庫中進行相似度搜尋"""
    def __init__(self, message: str = "向量搜尋錯誤"):
        self.message = message
        super().__init__(self.message)


class LLMError(Exception):
    """大型語言模型錯誤，無法生成回應"""
    def __init__(self, message: str = "LLM 錯誤"):
        self.message = message
        super().__init__(self.message)
