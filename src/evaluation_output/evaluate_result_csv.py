#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

##############################
# 1. 讀取資料
##############################
# 假設檔名為 evaluation_result.xlsx，請依實際檔名調整
df = pd.read_excel('evaluation_result.xlsx', engine='openpyxl')


##############################
# 2. 工具函式
##############################
def parse_code_list(code_str):
    """
    將多標籤欄位(如 '2-13,'5-12,'5-4')轉成list, 去除多餘空白和引號
    """
    if pd.isnull(code_str):
        return []
    return [c.strip().strip("'") for c in code_str.split(',') if c.strip()]


def get_parent_label(code):
    """
    從 '5-12' 取出 '5'；若無 '-' 則直接回傳整個字串
    """
    return code.split('-')[0] if '-' in code else code


##############################
# 3. 資料前處理
##############################

# 3.1 解析多標籤欄位 (GPT預測 & 最終答案)
df['gpt_labels'] = df['gpt4o_detected_codes'].apply(parse_code_list)
df['final_labels'] = df['final_answer_codes'].apply(parse_code_list)

# 3.2 取得父類別 (explode 方便做統計)
exploded_final = df.explode('final_labels').dropna(subset=['final_labels'])
exploded_final['final_parent'] = exploded_final['final_labels'].apply(get_parent_label)

##############################
# 4. 繪製圖表
##############################

###############
# (圖1) 父類別分佈 (Bar Chart)
###############
parent_count = exploded_final['final_parent'].value_counts().reset_index()
parent_count.columns = ['parent_label', 'count']

plt.figure(figsize=(6, 4))
sns.barplot(x='parent_label', y='count', data=parent_count, color='skyblue')
plt.title('Distribution of Parent Labels in final_answer_codes')
plt.xlabel('Parent Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('parent_label_distribution.png')
plt.close()

###############
# (圖2) 各標籤平均 Precision/Recall/F1 (Grouped Bar) 
###############
# 假設每列都有 precision / recall / f1_score，這些是針對「該列整體」的多標籤結果。
# 這裡示範：計算整份資料的平均 P/R/F，並繪製成簡單長條圖做比較。
mean_precision = df['precision'].mean()
mean_recall = df['recall'].mean()
mean_f1 = df['f1_score'].mean()

metrics_df = pd.DataFrame({
    'metric': ['Precision', 'Recall', 'F1-score'],
    'value': [mean_precision, mean_recall, mean_f1]
})

plt.figure(figsize=(6, 4))
sns.barplot(x='metric', y='value', hue='metric', data=metrics_df, palette='viridis', legend=False)
for i, v in enumerate(metrics_df['value']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.ylim(0, 1)
plt.title('Average Precision / Recall / F1 across all samples')
plt.tight_layout()
plt.savefig('average_precision_recall_f1.png')
plt.close()


###############
# (圖3) Matched / Missing / Extra 總體比例 (Pie Chart)
###############
def count_codes_in_str(code_str):
    """計算逗號切分後的標籤數量"""
    if pd.isnull(code_str) or str(code_str).strip() == '':
        return 0
    return len(parse_code_list(str(code_str)))


matched_total = df['matched_codes'].apply(count_codes_in_str).sum()
missing_total = df['missing_codes'].apply(count_codes_in_str).sum()
extra_total = df['extra_codes'].apply(count_codes_in_str).sum()

pie_data = [matched_total, missing_total, extra_total]
pie_labels = ['Matched', 'Missing', 'Extra']

plt.figure(figsize=(5, 5))
plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=140)
plt.title('Matched vs Missing vs Extra (Total)')
plt.tight_layout()
plt.savefig('matched_missing_extra_pie.png')
plt.close()


###############
# (圖4) 混淆矩陣 (父類別) 
###############
# 只取「第一個」預測父類別 vs. 第一個「最終答案」父類別 (可能會忽略多標籤，但用來示意)
df['first_final_parent'] = df['final_labels'].apply(
    lambda x: get_parent_label(x[0]) if len(x) > 0 else 'None'
)
df['first_gpt_parent'] = df['gpt_labels'].apply(
    lambda x: get_parent_label(x[0]) if len(x) > 0 else 'None'
)

labels = sorted(list(set(df['first_final_parent']) | set(df['first_gpt_parent'])))
cm = confusion_matrix(df['first_final_parent'], df['first_gpt_parent'], labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted (GPT Parent)')
plt.ylabel('True (Final Parent)')
plt.title('Confusion Matrix of Parent Label (First Label Only)')
plt.tight_layout()
plt.savefig('confusion_matrix_parent_label.png')
plt.close()
