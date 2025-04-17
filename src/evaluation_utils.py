# src/evaluation_utils.py

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)


def save_results_to_json(forward_results: list, reverse_results: list, folder: str = "./evaluation_output") -> None:
    """
    將 forward (外對內) 與 reverse (內對外) 兩者結果分別存成 JSON, 方便後續比對.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    forward_path = os.path.join(folder, "forward_results.json")
    reverse_path = os.path.join(folder, "reverse_results.json")

    with open(forward_path, "w", encoding="utf-8") as f:
        json.dump(forward_results, f, ensure_ascii=False, indent=2)
    with open(reverse_path, "w", encoding="utf-8") as f:
        json.dump(reverse_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved forward_results to {forward_path}")
    logger.info(f"Saved reverse_results to {reverse_path}")


def build_matrix(forward_results: list, reverse_results: list):
    """
    建立一個「外部法規 × 內部規範」的二維矩陣.
    - forward_results: [{"law_clause_id":"CSMA-1","evidences":[...], ...}, ...]
      => 代表 外->內
    - reverse_results: [{"internal_clause_id":"CSTI-1-SEC-001-5","evidences":[...], ...}, ...]
      => 代表 內->外

    matrix的定義:
      rows = ext_keys (外部法規 ID)
      cols = int_keys (內部條文 ID)

    matrix 中的值:
      0 = 沒有連結
      1 = 只有外→內
      2 = 只有內→外
      3 = 外→內 & 內→外  (bitwise = 1|2)
    """
    # 1) 先收集「外部法規ID」ext_keys, 以及「內部條文ID」int_keys.
    ext_keys = []
    for item in forward_results:
        ext_id = item["law_clause_id"]
        if ext_id not in ext_keys:
            ext_keys.append(ext_id)

    int_keys = []
    for item in reverse_results:
        int_id = item["internal_clause_id"]
        if int_id not in int_keys:
            int_keys.append(int_id)

    # 2) 構建 2D 矩陣
    matrix = np.zeros((len(ext_keys), len(int_keys)), dtype=np.int8)

    # =========== 填入 外->內 =========== #
    # 只要 forward 的 evidences 裡帶有 doc_id (對應某個 internal), 就將 matrix[ext_id, int_id] 標記為 1 (或 1|=).
    for f_item in forward_results:
        ext_id = f_item["law_clause_id"]
        row_idx = ext_keys.index(ext_id)  # 取得外部法規 row index
        evidences = f_item.get("evidences", [])
        # 為了避免重複加多次(若同 doc_id 有多條 evidence),
        # 可以先收集本 iter 內 doc_id set
        doc_ids_hit = set()
        for ev in evidences:
            int_id = ev.get("doc_id")  # ex. "CSTI-1-SEC-001-5"
            if not int_id:
                continue
            if int_id not in int_keys:
                continue
            if int_id in doc_ids_hit:
                # 同一 doc_id已經標記過, 不要再+1
                continue
            doc_ids_hit.add(int_id)

            col_idx = int_keys.index(int_id)
            # 原本 matrix[row_idx, col_idx] 可能是0或2(代表之前內->外)
            # 現在要標記外->內 => +1 => 用 or 避免超過3
            prev_val = matrix[row_idx, col_idx]
            new_val = prev_val | 1  # bitwise 1
            matrix[row_idx, col_idx] = new_val

    # =========== 填入 內->外 =========== #
    for r_item in reverse_results:
        int_id = r_item["internal_clause_id"]
        col_idx = int_keys.index(int_id)  # 取得 internal row index
        evidences = r_item.get("evidences", [])
        doc_ids_hit = set()
        for ev in evidences:
            ext_id = ev.get("doc_id")  # ex. "CSMA-18"
            if not ext_id:
                continue
            if ext_id not in ext_keys:
                continue
            if ext_id in doc_ids_hit:
                # 同一 doc_id已經標記過, 不要再+2
                continue
            doc_ids_hit.add(ext_id)

            row_idx = ext_keys.index(ext_id)
            prev_val = matrix[row_idx, col_idx]
            new_val = prev_val | 2  # bitwise 2
            matrix[row_idx, col_idx] = new_val

    return matrix, ext_keys, int_keys


def visualize_matrix(
    matrix: np.ndarray,
    ext_keys: list,
    int_keys: list,
    out_png: str = "./evaluation_output/matrix.png"
):
    """
    用 matplotlib 產生彩色格子圖：
      0 = 沒有連結(白)
      1 = only 外→內(紅)
      2 = only 內→外(藍)
      3 = both (紫)

    並且加上明顯的網格線，方便對應.
    """

    fig, ax = plt.subplots(figsize=(max(8, len(int_keys) * 0.7), max(6, len(ext_keys) * 0.7)))

    # 自訂 color map
    cmap = ListedColormap(["white", "lightcoral", "lightblue", "violet"])
    cax = ax.imshow(matrix, cmap=cmap, aspect="equal", vmin=0, vmax=3)

    # 設置 ticks
    ax.set_xticks(np.arange(len(int_keys)))
    ax.set_yticks(np.arange(len(ext_keys)))
    ax.set_xticklabels(int_keys, rotation=45, ha="right")
    ax.set_yticklabels(ext_keys)

    # 加網格
    ax.set_xticks(np.arange(-0.5, len(int_keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ext_keys), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 填上數字(0,1,2,3 -> 可能可省, or 僅對>0)
    for i in range(len(ext_keys)):
        for j in range(len(int_keys)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="black")

    # colorbar
    cb = fig.colorbar(cax, ax=ax, ticks=[0, 1, 2, 3])
    cb.set_label("0=none,1=F,2=R,3=F+R")

    # Set title
    ax.set_title("External vs Internal Cross Matrix")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    logger.info(f"Matrix saved to {out_png}")
    # plt.show()  # 如需視覺化時可開
