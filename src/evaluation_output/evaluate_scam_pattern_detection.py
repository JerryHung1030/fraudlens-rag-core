import pandas as pd
import json


def robust_parse_json(json_str):
    """
    嘗試以多種方式去解析 JSON 字串，避免因多餘換行或空白導致解析失敗。
    若最終仍無法解析，則回傳 None。
    """
    if not isinstance(json_str, str):
        return None
    
    tmp = json_str.strip()
    if not tmp:
        return None
    
    # 第一階段：嘗試直接 parse
    try:
        return json.loads(tmp)
    except:
        pass
    
    # 第二階段：嘗試去除可能的 \\n 等雜訊後再 parse
    tmp_no_escaped_newline = tmp.replace('\\n', ' ').replace('\r', '').replace('\n', ' ')
    try:
        return json.loads(tmp_no_escaped_newline.strip())
    except:
        return None


def get_id_set_from_json(parsed_json):
    """
    輸入為已成功 parse 成 Python list/dict 的 JSON 物件，
    回傳其中所有 'id' 的 set (若沒有 'id' 則略過)。
    若輸入不是 list，就視需求看是否要強制轉成 list 或直接回傳空 set。
    """
    if not isinstance(parsed_json, list):
        return set()
    ids = set()
    for item in parsed_json:
        if isinstance(item, dict) and 'id' in item:
            ids.add(str(item['id']))
    return ids


def parse_codes_or_json(raw_str):
    """
    1) 先嘗試以 JSON 方式解析 raw_str，若成功且結果非空，
       回傳該 JSON 裡面的 id set (e.g. {'5-4', '5-6', ...})
    2) 若 JSON 解析不到任何 id，再嘗試當作 code 列表（以逗號分隔）處理：
       e.g. "'5-11,'5-12,'5-4" -> {'5-11','5-12','5-4'}
    3) 若兩者都失敗/空，就回傳空 set。
    """
    # 嘗試 JSON parse
    parsed_json = robust_parse_json(raw_str)
    if parsed_json is not None:
        json_id_set = get_id_set_from_json(parsed_json)
        if len(json_id_set) > 0:
            return json_id_set  # 若 JSON 中有解析出 id，直接用它
    
    # 若 JSON 沒結果，改用 code 列表邏輯
    raw_str = raw_str.strip()
    if not raw_str or raw_str == '[]':
        return set()
    
    # 以逗號分隔
    items = [x.strip() for x in raw_str.split(',') if x.strip()]
    code_set = set()
    for it in items:
        # 如果前面有 ' 字樣，就去掉
        if it.startswith("'"):
            it = it[1:]
        code_set.add(it)
    return code_set


def evaluate_performance(gpt4o_ids, final_ids):
    matched_ids = gpt4o_ids.intersection(final_ids)
    missing_ids = final_ids.difference(gpt4o_ids)
    extra_ids = gpt4o_ids.difference(final_ids)

    matched_count = len(matched_ids)
    missing_count = len(missing_ids)
    extra_count = len(extra_ids)

    # Precision
    if (matched_count + extra_count) == 0:
        precision = 0
    else:
        precision = matched_count / (matched_count + extra_count)

    # Recall
    if (matched_count + missing_count) == 0:
        recall = 0
    else:
        recall = matched_count / (matched_count + missing_count)

    # F1
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return matched_ids, missing_ids, extra_ids, precision, recall, f1


def add_single_quote_prefix(codes):
    """
    輸入一個 set 或 list 的 code (e.g. {"5-4", "8-3"})
    回傳以逗號分隔且每個 code 前面加上 '。
    e.g. -> "'5-4,'8-3"
    """
    sorted_codes = sorted(codes)
    return ",".join("'" + c for c in sorted_codes)


def main():
    xls = pd.ExcelFile('input_data.xlsx', engine='openpyxl')

    # 假設我們只處理 dif_between_4o_&_answer 這張表
    diff_df = pd.read_excel(xls, 'dif_between_4o_&_answer', engine='openpyxl')

    diff_df.rename(columns={
        '使用者輸入': 'user_input',
        'GPT4o的詐騙模式偵測結果': 'gpt4o_result',
        '最終結果': 'final_result'
    }, inplace=True)

    new_columns = [
        'row_id',
        'user_input',
        'gpt4o_raw_detection',
        'gpt4o_detection_json',
        'final_answer_json',
        'gpt4o_detected_codes',
        'final_answer_codes',
        'matched_codes',
        'missing_codes',
        'extra_codes',
        'precision',
        'recall',
        'f1_score'
    ]

    output_data = []

    for idx, row in diff_df.iterrows():
        row_id = idx + 2
        user_input = str(row['user_input']) if not pd.isna(row['user_input']) else ""
        gpt4o_raw = str(row['gpt4o_result']) if not pd.isna(row['gpt4o_result']) else ""
        final_raw = str(row['final_result']) if not pd.isna(row['final_result']) else ""

        # 1) 試圖解析成 "id" set，兼容 JSON 與 code 列表
        gpt4o_id_set = parse_codes_or_json(gpt4o_raw)
        final_id_set = parse_codes_or_json(final_raw)

        # 2) 用 robust_parse_json + get_id_set_from_json 產生對應的 JSON 供輸出（若是 code list就無法轉出來?）
        gpt4o_json_parsed = robust_parse_json(gpt4o_raw)
        final_json_parsed = robust_parse_json(final_raw)

        if gpt4o_json_parsed is None:
            gpt4o_json_parsed = []
        if final_json_parsed is None:
            final_json_parsed = []
        
        gpt4o_json_str = json.dumps(gpt4o_json_parsed, ensure_ascii=False)
        final_json_str = json.dumps(final_json_parsed, ensure_ascii=False)

        # 3) 計算評估指標
        matched_ids, missing_ids, extra_ids, precision, recall, f1 = evaluate_performance(
            gpt4o_id_set, final_id_set
        )

        # 4) 在輸出給 Excel 前，對 code 加上 `'` prefix，避免 Excel 誤判日期
        gpt4o_detected_codes_str = add_single_quote_prefix(gpt4o_id_set)
        final_answer_codes_str = add_single_quote_prefix(final_id_set)
        matched_codes_str = add_single_quote_prefix(matched_ids)
        missing_codes_str = add_single_quote_prefix(missing_ids)
        extra_codes_str = add_single_quote_prefix(extra_ids)

        row_dict = {
            'row_id': row_id,
            'user_input': user_input,
            'gpt4o_raw_detection': gpt4o_raw.replace('\n', '\\n'),
            'gpt4o_detection_json': gpt4o_json_str,
            'final_answer_json': final_json_str,
            'gpt4o_detected_codes': gpt4o_detected_codes_str,
            'final_answer_codes': final_answer_codes_str,
            'matched_codes': matched_codes_str,
            'missing_codes': missing_codes_str,
            'extra_codes': extra_codes_str,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        output_data.append(row_dict)

    evaluated_df = pd.DataFrame(output_data, columns=new_columns)

    # 將最終結果輸出為 Excel
    evaluated_df.to_excel('evaluation_result.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':
    main()
