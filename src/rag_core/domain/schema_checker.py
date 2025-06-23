"""
===============================================================================
    Module Name: schema_checker.py
    Description: Validation for hierarchical JSON data structure in RAG pipeline.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
from utils.logging import log_wrapper
from typing import Any, Dict, List


class DataSchemaError(Exception):
    """
    自定義異常，用於表示資料結構驗證失敗，包含錯誤詳情列表。
    """
    def __init__(self, details: List[str]):
        super().__init__("Data schema validation failed")
        self.details = details

    def __str__(self):
        return f"DataSchemaError: {'; '.join(self.details)}"


class DataStructureChecker:
    """
    驗證巢狀 JSON 是否符合階層式結構 (level1~level5, sid, text) 的規範。

    方法:
      - validate(data: dict, mode: str) -> None
        驗證失敗時拋出 DataSchemaError, 成功不回傳。
      - is_valid_level_structure(data: dict) -> bool
        僅回傳驗證結果，不拋異常。
    """

    MAX_LEVEL = 5

    def validate(self, data: Dict[str, Any], mode: str, max_depth: int = 5) -> None:
        """
        驗證頂層及子層結構，若有任一錯誤則拋出 DataSchemaError。

        Args:
            data: 原始 JSON 資料
            mode: 字串標識模式 ("input" 或 "reference"), 目前未用

        max_depth: 2‑9. default=5, user can pass smaller if want.
        """
        errors: List[str] = []

        def _recurse(node: Any, level: int, path: str):
            if level > max_depth:
                return
            
            key = f"level{level}"
            if level == 1:
                # 頂層 node 應該包含 'level1'
                if not isinstance(node, dict):
                    errors.append(f"Top-level data is not a dict at path '{path}'")
                    return
                if key not in node:
                    errors.append(f"Missing key '{key}' at path '{path}'")
                    return
                children = node[key]
            else:
                # 子節點
                if not isinstance(node, dict):
                    errors.append(f"Expected dict at path '{path}', got {type(node).__name__}")
                    return
                if key not in node:
                    # 沒有此層，不必往下
                    return
                children = node[key]

            if not isinstance(children, list):
                errors.append(f"'{key}' should be a list at path '{path}'")
                return

            for idx, item in enumerate(children):
                item_path = f"{path}/{key}[{idx}]"
                if not isinstance(item, dict):
                    errors.append(f"Item at '{item_path}' is not a dict")
                    continue
                # 檢查 sid 和 text
                sid = item.get("sid")
                text = item.get("text")
                if not isinstance(sid, str):
                    errors.append(f"Missing or invalid 'sid' at '{item_path}'")
                if not isinstance(text, str):
                    errors.append(f"Missing or invalid 'text' at '{item_path}'")
                # 繼續下一層，直到 MAX_LEVEL
                if level < DataStructureChecker.MAX_LEVEL:
                    _recurse(item, level + 1, item_path)

        _recurse(data, 1, "root")

        if errors:
            log_wrapper.error(
                "DataStructureChecker",
                "validate",
                f"Data structure validation errors: {errors}"
            )
            raise DataSchemaError(details=errors)

    def is_valid_level_structure(self, data: Dict[str, Any]) -> bool:
        """
        回傳資料結構是否有效，不拋異常。
        """
        try:
            self.validate(data, mode="")
            return True
        except DataSchemaError:
            return False
