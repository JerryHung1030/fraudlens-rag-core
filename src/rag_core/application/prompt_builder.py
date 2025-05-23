# services/prompt_builder.py
import json
from typing import List, Dict, Any, Tuple
from rag_core.utils.token_counter import TokenCounter
from rag_core.exceptions import PromptTooLongError
from utils import log_wrapper


class PromptBuilder:
    FIELD_DESC = """
必要欄位：
• input_uid, input_text, ref_uid, ref_text, evidences, start_end_idx, confidence
""".strip()

    @staticmethod
    def build_prompt(
        user_query: str,
        context_docs: List[Dict[str, Any]],
        scenario: Any  # pydantic Scenario
    ) -> Tuple[str, int]:  # 回傳 (prompt, 實際使用的 candidates 數量)

        s = scenario
        llm_model = s.llm_name or "gpt-4o"          # 給 TokenCounter 用
        max_tok = s.max_prompt_tokens             # 新增欄位，預設 8k

        # ───── header & user_query ─────
        example_json = [{
            "input_uid": "INPUT_UID", "input_text": "INPUT_TEXT",
            "ref_uid": "REF_UID", "ref_text": "REF_TEXT",
            "evidences": ["keyword1"], "start_end_idx": [[0, 3]],
            "confidence": 0.9
        }]

        header_parts = [
            f"【Role】\n{s.role_desc}\n",
            f"【模式】\n{s.direction}\n",
            f"【參考資料】\n{s.reference_desc}\n",
            f"【任務】\n{s.input_desc}\n\n",
            f"{PromptBuilder.FIELD_DESC}\n\n",
            f"{s.scoring_rule}\n\n",
            f"請比較 user_query 與下列 Candidates，若 confidence < {s.cof_threshold}，請忽略。\n",
            "範例：\n",
            f"{json.dumps(example_json, ensure_ascii=False, indent=2)}\n",
            "───────────────\n",
            f"【Query】\n{user_query}\n",
            "【Candidates】\n",
        ]

        used_tok = 0
        for part in header_parts:
            used_tok += TokenCounter.count(llm_model, part)
            if used_tok >= max_tok:
                raise PromptTooLongError(
                    f"Header+Query 已 {used_tok} tokens (> {max_tok})"
                )

        prompt = "".join(header_parts)

        # ───── append candidates until budget exhausted ─────
        cand_lines: List[str] = []
        # 先依相似度高到低，保留最有價值的
        sorted_docs = sorted(context_docs, key=lambda d: -d.get("score", 0.0))

        for i, doc in enumerate(sorted_docs, 1):
            cand_header = f"【Candidate {i}】(uid: {doc['uid']}, score: {doc.get('score', 0)})\n"
            cand_text = doc['text']
            token_cost = TokenCounter.count(llm_model, cand_header) + TokenCounter.count(llm_model, cand_text)
            newline_tok = TokenCounter.count(llm_model, "\n") if i < len(sorted_docs) else 0
            if used_tok + token_cost + newline_tok > max_tok:
                log_wrapper.info(
                    "PromptBuilder",
                    "build_prompt",
                    f"Prompt token budget({max_tok}) 用盡，僅保留 {i - 1} / {len(sorted_docs)} 個 Candidates"
                )
                break
            line = cand_header + cand_text
            if i < len(sorted_docs):
                line += "\n"
            cand_lines.append(line)
            used_tok += token_cost + newline_tok

        if not cand_lines:
            raise PromptTooLongError("沒有任何 Candidate 能放進 prompt，請調整 max_prompt_tokens 或減少 context_docs 數量")

        prompt = prompt + "".join(cand_lines)
        log_wrapper.debug(
            "PromptBuilder",
            "build_prompt",
            f"Prompt tokens={used_tok} (limit={max_tok}), used {len(cand_lines)} candidates"
        )
        return prompt, len(cand_lines)  # 回傳 prompt 和實際使用的 candidates 數量
