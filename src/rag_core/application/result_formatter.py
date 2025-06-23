# services/result_formatter.py
import json
from utils.logging import log_wrapper
from typing import Any, Dict, List


class ResultFormatter:
    """
    負責解析 LLM 輸出、修正欄位 (e.g. forward/reverse 調整) 以及掛上相似分數。
    """

    @staticmethod
    def parse_and_format(
        raw_llm_output: str,
        hits: List[Dict[str, Any]],
        root_uid: str,
        direction: str,
        llm_name: str,
        rag_k_used: int,
        cof_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        回傳結構：
          [
            {
              "direction": "forward|reverse|both",
              "root_uid": "整篇 group_uid",
              "model": "llm名稱",
              "rag_k": "使用k值",
              "cof_threshold": "0~1之間",
              "predictions": [
                  {
                    "input_uid":...,
                    "ref_uid":...,
                    ...
                    "similarity_score":...
                  }, ...
              ]
            }
          ]
        """

        # 1) 去除可能的 ```fence
        output = raw_llm_output.strip()
        if output.startswith("```"):
            output = output.partition("\n")[-1].strip()
        if output.endswith("```"):
            output = output.rpartition("```")[0].strip()

        # 2) 嘗試解析 JSON
        try:
            predictions = json.loads(output)
        except Exception:
            # fallback: 把整段 output 包成一筆 {"plain_text":..., "confidence":0}
            log_wrapper.warning(
                "ResultFormatter",
                "parse_and_format",
                f"LLM JSON parse error, fallback to plain_text. raw={output[:200]}"
            )
            predictions = [{"plain_text": output, "confidence": 1.0}]  # 繞過 confidence 過濾

        if not isinstance(predictions, list):
            # 若解析出來不是 list，也同樣 fallback
            log_wrapper.warning(
                "ResultFormatter",
                "parse_and_format",
                "Parsed output is not a list. Force empty."
            )
            predictions = []

        # 構建 {uid -> similarity_score}
        score_map = {h["uid"]: h["score"] for h in hits}

        # 3) 依 direction "reverse" 時交換 input/ref
        for pred in predictions:
            pred.setdefault("evidences", [])
            pred.setdefault("start_end_idx", [])

            if direction == "reverse":
                (pred["input_uid"], pred["ref_uid"]) = (pred.get("ref_uid"), pred.get("input_uid"))
                (pred["input_text"], pred["ref_text"]) = (pred.get("ref_text"), pred.get("input_text"))

            ref_uid = pred.get("ref_uid")
            pred["similarity_score"] = score_map.get(ref_uid, 0.0)

        # 4) 過濾 confidence < cof_threshold
        filtered_preds = [p for p in predictions if p.get("confidence", 0.0) >= cof_threshold]

        return [{
            "direction": direction,
            "root_uid": root_uid,
            "model": llm_name,
            "rag_k": rag_k_used,
            "cof_threshold": cof_threshold,
            "predictions": filtered_preds
        }]
