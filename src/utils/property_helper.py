import configparser
import os

cfg = configparser.ConfigParser()
# 指定 ini 檔路徑
ini_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "system.ini")
cfg.read(ini_path, encoding='utf-8')

def get_property(section_val, key_val):
    """
    取得設定檔資訊
    Args:
        section_val (str): ini 裡的 [節]
        key_val (str): 鍵
    Returns:
        str or None
    """
    if section_val is not None and key_val is not None:
        try:
            return cfg.get(str(section_val), str(key_val))
        except Exception:
            return None
    else:
        return None