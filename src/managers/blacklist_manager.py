# src/managers/blacklist_manager.py
from typing import List


class BlacklistManager:
    def __init__(self, blacklist_db: any = None):
        self.blacklist_db = blacklist_db or []

    def check_urls(self, text: str) -> List[str]:
        # PoC: 假設黑名單是一堆惡意網域
        found = []
        for url in self.blacklist_db:
            if url in text:
                found.append(url)
        return found

    def check_line_ids(self, text: str) -> List[str]:
        # PoC: 假設黑名單是一堆 line ID
        found = []
        for line_id in self.blacklist_db:
            if line_id in text:
                found.append(line_id)
        return found
