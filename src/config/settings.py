"""
===============================================================================
    Module Name: settings.py
    Description: Configuration management using Pydantic and YAML for the RAG system.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import yaml
import os
from functools import lru_cache

# 取得專案根目錄
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class APIKeysConfig(BaseModel):
    """API Keys 相關設定"""
    openai: str = Field(default="", env="OPENAI_API_KEY")
    # 可以加入其他 API keys

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class SystemConfig(BaseModel):
    """系統相關設定"""
    is_debug: bool = Field(default=False, alias="IS_DEBUG")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    log_file_path: str = Field(default="app.log", alias="LOG_FILE_PATH")
    error_log_file_path: str = Field(default="error.log", alias="ERROR_LOG_FILE_PATH")


class LLMConfig(BaseModel):
    """LLM 相關設定"""
    model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    max_prompt_tokens: int = Field(default=8000)


class VectorDBConfig(BaseModel):
    """向量資料庫相關設定"""
    url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    collection: str = Field(default="my_rag_collection", alias="QDRANT_COLLECTION")
    vector_size: int = Field(default=1536)


class ThreadPoolConfig(BaseModel):
    """執行緒池相關設定"""
    vector_pool: int = Field(default=8, alias="VECTOR_POOL")
    embed_pool: int = Field(default=8, alias="EMBED_POOL")


class EmbeddingConfig(BaseModel):
    """Embedding 相關設定"""
    model: str = Field(default="text-embedding-ada-002", alias="EMBED_MODEL")


class ScenarioConfig(BaseModel):
    role_desc: str = ""
    reference_desc: str = ""
    input_desc: str = ""
    direction: str = Field("forward", pattern="^(forward|reverse|both)$")
    rag_k: int = 5
    rag_k_forward: Optional[int] = None
    rag_k_reverse: Optional[int] = None
    cof_threshold: float = 0.5
    reference_depth: int = 1
    input_depth: int = 1
    chunk_size: int = 0
    scoring_rule: str = ""
    llm_name: Optional[str] = None
    reference_json: Optional[str] = None
    input_json: Optional[str] = None
    output_json: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # 轉換相對路徑為絕對路徑
        if self.reference_json:
            self.reference_json = os.path.join(PROJECT_ROOT, self.reference_json)
        if self.input_json:
            self.input_json = os.path.join(PROJECT_ROOT, self.input_json)
        if self.output_json:
            self.output_json = os.path.join(PROJECT_ROOT, self.output_json)

    class Config:
        extra = "allow"


class Settings(BaseSettings):
    """應用程式設定"""
    # API Keys 設定
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    
    # 子設定
    system: SystemConfig = Field(default_factory=SystemConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    thread_pool: ThreadPoolConfig = Field(default_factory=ThreadPoolConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


class ConfigManager:
    """設定管理器"""
    def __init__(self):
        self._settings = None
        self._load_settings()

    def _load_settings(self):
        """載入所有設定"""
        # 載入設定
        self._settings = Settings()

        # 載入 base 設定檔
        base_config_path = os.path.join(os.path.dirname(__file__), "settings.base.yml")
        if os.path.exists(base_config_path):
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}
                if "api_keys" in base_config:
                    self._settings.api_keys = APIKeysConfig(**base_config["api_keys"])
                if "scenario" in base_config:
                    self._settings.scenario = ScenarioConfig(**base_config["scenario"])
                if "system" in base_config:
                    self._settings.system = SystemConfig(**base_config["system"])
                if "llm" in base_config:
                    self._settings.llm = LLMConfig(**base_config["llm"])
                if "vector_db" in base_config:
                    self._settings.vector_db = VectorDBConfig(**base_config["vector_db"])
                if "thread_pool" in base_config:
                    self._settings.thread_pool = ThreadPoolConfig(**base_config["thread_pool"])
                if "embedding" in base_config:
                    self._settings.embedding = EmbeddingConfig(**base_config["embedding"])

    @property
    def settings(self) -> Settings:
        """取得設定單例"""
        return self._settings

    def get_scenario_config(self) -> Dict[str, Any]:
        """取得情境設定"""
        return self._settings.scenario.dict()


@lru_cache()
def get_config_manager() -> ConfigManager:
    """取得設定管理器單例"""
    return ConfigManager()


# 全域設定實例
config_manager = get_config_manager()
