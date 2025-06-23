# src/utils/text_preprocessor.py
from typing import Any, Dict, List

"""
===============================================================================
    Module Name: text_preprocessor.py
    Description: Flatten and chunk hierarchical JSON for RAG input processing.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""

class TextPreprocessor:
    """
    提供階層式 JSON 資料的 Flatten 與文字 Chunk 工具。
    名稱對應：
      ・orig_sid  = 業務原始 ID (p001)
      ・group_uid = 代表整篇 (input-p001)，樹的根
      ・uid       = chunk 用 (input-p001_c0)
    """

    @staticmethod
    def flatten_levels(data: Dict[str, Any], depth: int, side: str) -> List[Dict[str, Any]]:
        """
        根據 depth，將階層結構攤平。
        - orig_sid: 直接取 node 最初的 sid (或每層相同)
        - group_uid: side-{接上所有 sid 路徑}，表示這整顆樹 / 這篇文章
        - uid: 預設等於 group_uid（若之後 chunk，就加 _c{i}）
        """
        results: List[Dict[str, Any]] = []

        if depth == 1:
            lvl1 = data.get("level1", [])
            for item in lvl1:
                # ① orig_sid
                orig_sid = item.get("sid", "")  # 使用 get 方法安全地獲取 sid
                # ② group_uid = side-orig_sid (若只有一層)
                group_uid = f"{side}-{orig_sid}"
                # ③ uid = group_uid (暫無 chunk)
                results.append({
                    "orig_sid": orig_sid,
                    "group_uid": group_uid,
                    "uid": group_uid,
                    "text": item.get("text", "")
                })
            return results

        def _dfs(current_data: Dict[str, Any], level: int, sid_acc: List[str], text_acc: List[str]):
            if level >= depth:
                # ② group_uid = side-拼接sid
                group_uid = f"{side}-{'_'.join(sid_acc)}"
                # ③ uid = group_uid (尚未 chunk)
                results.append({
                    "orig_sid": sid_acc[0],     # 第一層sid 當作業務主鍵
                    "group_uid": group_uid,
                    "uid": group_uid,
                    "text": "\n".join(text_acc)
                })
                return

            key = f"level{level}"
            children = current_data.get(key, [])

            if not isinstance(children, list) or not children:
                # 沒子層了，也算終點
                group_uid = f"{side}-{'_'.join(sid_acc)}"
                results.append({
                    "orig_sid": sid_acc[0],
                    "group_uid": group_uid,
                    "uid": group_uid,
                    "text": "\n".join(text_acc)
                })
                return

            # 繼續 DFS
            for child in children:
                next_sid = child.get("sid", "")
                next_text = child.get("text", "")
                _dfs(child, level + 1, sid_acc + [next_sid], text_acc + [next_text])

        # 從 level1 開始展開
        level1_items = data.get("level1", [])
        for item in level1_items:
            first_sid = item.get("sid", "")
            first_text = item.get("text", "")
            _dfs(
                current_data=item,
                level=2,
                sid_acc=[first_sid],
                text_acc=[first_text]
            )

        return results

    @staticmethod
    def chunk_text(text: str, max_len: int) -> List[str]:
        """
        2‑2: 簡易校正: 用 utf-8 編碼長度，若 too large 再切
        這裡示範一種做法: 
        - 先計算 text.encode("utf-8") 長度
        - 若 > max_len, 以 slicing = int
          注意: 可能有 multi-byte
          更準確做法建議 token-based chunk
        """
        # ❼ TODO: Use real token-based chunk (e.g. with tiktoken) 
        # or sentence-based splitting, to handle multi-byte chars
        # and ensure we don't exceed LLM token limit. 
        # For now, keep simple char-based slicing.

        encoded_len = len(text.encode("utf-8"))
        if max_len > 0 and encoded_len > max_len:
            print(f"[chunk_text] Warning: text utf8-len={encoded_len} > {max_len}, might chunk incorrectly.")

        if max_len <= 0 or len(text) <= max_len:
            return [text]

        chunks: List[str] = []
        start = 0
        total = len(text)
        while start < total:
            end = min(start + max_len, total)
            chunks.append(text[start:end])
            start = end
        return chunks
