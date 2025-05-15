import os
from loguru import logger
from src.utils import property_helper


class log_wrapper:
    """
    Log Wrapper : 設定 loguru 的 logger，並提供呼叫介面
    """
    if str(property_helper.get_property('LOG_INFO', 'IS_DEBUG')).upper() == 'TRUE':
        IS_DEBUG = True
    else:
        IS_DEBUG = False

    tmp_list = str(property_helper.get_property('LOG_INFO', 'LOG_DIR')).split('/')
    log_dir = os.sep.join(tmp_list[:])  # e.g. "app/logs"
    LOG_FILE_PATH = log_dir + os.sep + str(property_helper.get_property('LOG_INFO', 'LOG_FILE_PATH'))
    ERROR_LOG_FILE_PATH = log_dir + os.sep + str(property_helper.get_property('LOG_INFO', 'ERROR_LOG_FILE_PATH'))

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