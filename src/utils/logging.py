import os
from loguru import logger
from config.settings import config_manager


class log_wrapper:
    """
    Log Wrapper : 設定 loguru 的 logger，並提供呼叫介面
    """
    # 從設定管理器取得設定
    settings = config_manager.settings.system
    IS_DEBUG = settings.is_debug

    # 設定日誌路徑
    log_dir = os.sep.join(settings.log_dir.split('/'))
    LOG_FILE_PATH = os.path.join(log_dir, settings.log_file_path)
    ERROR_LOG_FILE_PATH = os.path.join(log_dir, settings.error_log_file_path)

    # 建立目錄
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 設定一般 log
    logger.add(
        LOG_FILE_PATH,
        level='DEBUG',
        rotation='00:00',  # 每日 00:00 自動 rotate
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
    )

    # 設定錯誤 log
    logger.add(
        ERROR_LOG_FILE_PATH,
        level='ERROR',
        rotation='00:00',
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
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
