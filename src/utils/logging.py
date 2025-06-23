"""
===============================================================================
    Module Name: logging.py
    Description: Loguru-based logging wrapper for system-wide logging and rotation.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
import os
from datetime import datetime
from loguru import logger
from config.settings import config_manager


class log_wrapper:
    """
    Log Wrapper : 設定 loguru 的 logger，並提供呼叫介面
    """
    # 從設定管理器取得設定
    settings = config_manager.settings.system
    IS_DEBUG = settings.is_debug

    # 設定日誌路徑 - 使用絕對路徑
    # 獲取專案根目錄（假設 src/utils/logging.py 在專案結構中）
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    log_dir = os.path.join(project_root, settings.log_dir)
    
    # 建立目錄
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 生成包含日期的文件名
    today = datetime.now().strftime("%Y-%m-%d")
    base_name = settings.log_file_path.replace('.log', '')
    error_base_name = settings.error_log_file_path.replace('.log', '')
    
    log_file_path = os.path.join(log_dir, f"{base_name}_{today}.log")
    error_log_file_path = os.path.join(log_dir, f"{error_base_name}_{today}.log")

    # 設定一般 log
    logger.add(
        log_file_path,
        level='DEBUG',
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        enqueue=True,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    # 設定錯誤 log
    logger.add(
        error_log_file_path,
        level='ERROR',
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        enqueue=True,
        rotation="5 MB",
        retention="30 days",
        compression="zip"
    )

    @staticmethod
    def get_caller(file_name):
        """剖析呼叫端檔案名稱"""
        file_name = str(file_name)
        if file_name.endswith('.py'):
            if '/' in file_name:
                tmp_list = file_name.split('/')
                return tmp_list[-1]
            else:
                return file_name
        return file_name

    @staticmethod
    def compose_msg(module_name, func_name, msg):
        """組合欲輸出的訊息"""
        return f"module: [{module_name}] func: [{func_name}] | {msg}"

    @classmethod
    def critical(cls, module_name, func_name, msg):
        logger.critical(cls.compose_msg(module_name, func_name, msg))

    @classmethod
    def error(cls, module_name, func_name, msg):
        logger.error(cls.compose_msg(module_name, func_name, msg))

    @classmethod
    def warning(cls, module_name, func_name, msg):
        logger.warning(cls.compose_msg(module_name, func_name, msg))

    @classmethod
    def info(cls, module_name, func_name, msg):
        logger.info(cls.compose_msg(module_name, func_name, msg))

    @classmethod
    def debug(cls, module_name, func_name, msg):
        if cls.IS_DEBUG:
            logger.debug(cls.compose_msg(module_name, func_name, msg))


# 創建一個單例實例
log_wrapper = log_wrapper()
