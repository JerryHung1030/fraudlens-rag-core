from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """應用程式設定"""
    # OpenAI 設定
    OPENAI_API_KEY: str = "sk-xxx"
    
    # 向量資料庫設定
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "my_rag_collection"
    
    # 執行緒池設定
    VECTOR_POOL: int = 8
    EMBED_POOL: int = 8
    
    # Embedding 模型設定
    EMBED_MODEL: str = "text-embedding-ada-002"
    
    # LLM 設定
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """取得設定單例"""
    return Settings()

# 全域設定實例
settings = get_settings() 