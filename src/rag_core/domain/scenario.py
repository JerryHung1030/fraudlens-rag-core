"""
===============================================================================
    Module Name: scenario.py
    Description: Scenario configuration model for RAG tasks.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
# services/scenario.py
from pydantic import BaseModel, Field
from typing import Optional


class Scenario(BaseModel):
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
    max_prompt_tokens: int = 8000     # 預設給 gpt-4o，一半留給回覆

    # 你可以再加 scenario-specific 欄位

    class Config:
        validate_by_name = True
        extra = "allow"  # or "forbid" to更嚴格
