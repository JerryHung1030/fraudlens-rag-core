# 日誌配置說明

## 概述

RAGCore-X 現在使用增強的日誌配置，支持按日期命名文件和按行數自動分割。

## 新功能

### 1. 按日期命名的文件
- **一般日誌**: `app_YYYY-MM-DD.log`
- **錯誤日誌**: `error_YYYY-MM-DD.log`
- 每天自動創建新的日誌文件

### 2. 按行數自動分割
- **一般日誌**: 每 10,000 行自動分割
- **錯誤日誌**: 每 5,000 行自動分割
- 分割後的文件名格式: `app_YYYY-MM-DD_HH-mm-ss.log`

### 3. 文件管理
- **保留期限**: 30 天
- **壓縮**: 舊文件自動壓縮為 ZIP 格式
- **自動清理**: 超過保留期限的文件自動刪除

## 配置詳情

### 日誌文件位置
```
logs/
├── app_2024-01-15.log          # 當天的一般日誌
├── app_2024-01-15_14-30-25.log # 分割後的一般日誌
├── error_2024-01-15.log        # 當天的錯誤日誌
├── error_2024-01-15_14-30-25.log # 分割後的錯誤日誌
└── *.zip                       # 壓縮的舊日誌文件
```

### 日誌格式
```
2024-01-15 14:30:25.123 | INFO | module: [app] func: [startup] | 服務啟動成功
```

### 配置參數
- `rotation`: "10000 lines" (一般日誌) / "5000 lines" (錯誤日誌)
- `retention`: "30 days"
- `compression`: "zip"
- `enqueue`: true (線程安全)

## 使用方法

### 在代碼中使用
```python
from utils.logging import log_wrapper

# 記錄不同級別的日誌
log_wrapper.info("module_name", "function_name", "信息內容")
log_wrapper.warning("module_name", "function_name", "警告內容")
log_wrapper.error("module_name", "function_name", "錯誤內容")
log_wrapper.debug("module_name", "function_name", "調試內容")
```

### 日誌級別
- **DEBUG**: 調試信息 (僅在 debug 模式下輸出)
- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 錯誤信息
- **CRITICAL**: 嚴重錯誤

## 測試

運行測試腳本驗證配置：
```bash
python test_new_logging.py
```

## 注意事項

1. **文件分割**: 當文件達到指定行數時，會自動創建新文件
2. **日期變更**: 每天 00:00 會自動創建新的日誌文件
3. **磁盤空間**: 定期檢查 logs 目錄大小，避免佔用過多空間
4. **性能**: 使用 enqueue=True 確保線程安全，但可能略微影響性能

## 自定義配置

可以在 `src/config/settings.base.yml` 中修改以下參數：
```yaml
system:
  log_dir: logs                    # 日誌目錄
  log_file_path: app.log          # 一般日誌文件名
  error_log_file_path: error.log  # 錯誤日誌文件名
``` 