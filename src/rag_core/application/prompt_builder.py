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
        direction = s.direction.lower()
        dir_note = {
            "forward": "【模式】forward\n・user_query ＝ 貼文\n・Candidates＝標籤/外規\n",
            "reverse": "【模式】reverse\n・user_query ＝ 標籤\n・Candidates＝貼文\n"
        }.get(direction, "【模式】both (暫只示範 forward / reverse)\n")

        example_json = [{
            "input_uid": "INPUT_UID", "input_text": "INPUT_TEXT",
            "ref_uid": "REF_UID", "ref_text": "REF_TEXT",
            "evidences": ["keyword1"], "start_end_idx": [[0, 3]],
            "confidence": 0.9
        }]

        header = (
            f"{s.role_desc or '你是RAG助手'}\n"
            f"{dir_note}"
            f"【Reference 說明】{s.reference_desc}\n"
            f"【Input 說明】{s.input_desc}\n\n"
            f"{PromptBuilder.FIELD_DESC}\n\n"
            f"{s.scoring_rule}\n\n"
            f"請比較 user_query 與下列 Candidates，若 confidence < {s.cof_threshold}，請忽略。\n"
            "範例：\n"
            f"{json.dumps(example_json, ensure_ascii=False, indent=2)}\n"
            "─────────────────────────\n"
            "==== user_query ====\n"
        )

        used_tok = TokenCounter.count(llm_model, header)
        used_tok += TokenCounter.count(llm_model, user_query)
        used_tok += TokenCounter.count(llm_model, "\n")  # newline after query
        header += f"{user_query}\n==== Candidates ====\n"
        if used_tok >= max_tok:
            raise PromptTooLongError(
                f"Header+Query 已 {used_tok} tokens (> {max_tok})"
            )

        # ───── append candidates until budget exhausted ─────
        cand_lines: List[str] = []
        # 先依相似度高到低，保留最有價值的
        sorted_docs = sorted(context_docs, key=lambda d: -d.get("score", 0.0))

        for i, doc in enumerate(sorted_docs, 1):
            line = f"[Cand#{i}] uid={doc['uid']} score={doc.get('score', 0):.3f}\n{doc['text']}"
            line_tok = TokenCounter.count(llm_model, line)
            newline_tok = TokenCounter.count(llm_model, "\n")

            if used_tok + line_tok + newline_tok > max_tok:
                log_wrapper.info(
                    "PromptBuilder",
                    "build_prompt",
                    f"Prompt token budget({max_tok}) 用盡，僅保留 {i - 1} / {len(sorted_docs)} 個 Candidates"
                )
                break
            cand_lines.append(line + "\n")
            used_tok += line_tok + newline_tok

        if not cand_lines:
            raise PromptTooLongError("沒有任何 Candidate 能放進 prompt，請降低 rag_k 或放寬 token 限制")

        prompt = header + "".join(cand_lines)
        log_wrapper.debug(
            "PromptBuilder",
            "build_prompt",
            f"Prompt tokens={used_tok} (limit={max_tok}), used {len(cand_lines)} candidates"
        )
        return prompt, len(cand_lines)  # 回傳 prompt 和實際使用的 candidates 數量
