from typing import Dict, Any, List, Optional
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, FieldCondition, MatchValue

from .data_structure_checker import DataStructureChecker
from .embedding_manager import EmbeddingManager
from config.settings import settings
from src.utils.log_wrapper import log_wrapper



class VectorIndex:
    """
    統一向量資料庫介面，負責 ingest 與 search。
    支援 Qdrant 作為後端。
    """
    # 全域共享一個 thread‐pool，避免大量併發將預設 executor 用完
    _executor = ThreadPoolExecutor(max_workers=settings.VECTOR_POOL)
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        data_checker: DataStructureChecker,
        qdrant_client: QdrantClient,
        default_collection_name: str = None,
        vector_size: int = 1536
    ):
        self.embedding_manager = embedding_manager
        self.data_checker = data_checker
        self.qdrant_client = qdrant_client
        self.default_collection_name = default_collection_name or settings.QDRANT_COLLECTION
        self.vector_size = vector_size

        # Ensure default collection exists
        self._ensure_collection(self.default_collection_name)

    def _ensure_collection(self, collection_name: str):
        """
        確保 collection 存在，若不存在則建立。
        避免重複建立而清空現有資料。
        """
        log_wrapper.info(
            "VectorIndex",
            "ensure_collection",
            f"確保 collection [{collection_name}] 存在..."
        )
        
        try:
            # 檢查 collection 是否存在
            self.qdrant_client.get_collection(collection_name)
            log_wrapper.info(
                "VectorIndex",
                "ensure_collection",
                f"Collection '{collection_name}' 已存在，跳過建立。"
            )
            return
            
        except Exception:
            log_wrapper.info(
                "VectorIndex",
                "ensure_collection",
                f"Collection '{collection_name}' 不存在，開始建立..."
            )
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                log_wrapper.info(
                    "VectorIndex",
                    "ensure_collection",
                    f"Collection '{collection_name}' 建立成功。"
                )
            except Exception as e:
                log_wrapper.error(
                    "VectorIndex",
                    "ensure_collection",
                    f"建立 collection {collection_name} 失敗: {e}"
                )
                raise

    def create_collection(self, collection_name: str):
        """明確建立或重建 collection"""
        self._ensure_collection(collection_name)
        log_wrapper.info(
            "VectorIndex",
            "ensure_collection",
            f"Collection {collection_name} is ready."
        )

    def ingest_json(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        mode: str  # "input" or "reference"
    ) -> None:
        """
        驗證並處理後，將 JSON 資料 ingest 到向量庫。
        每筆資料需有 'uid'、'text' 欄位，其他欄位作為 metadata。
        ❺ 若重複 ingest 同筆 uid，會覆蓋舊資料；
           下方示例：以時間戳後綴確保每筆都是獨立 ID
        """
        if not data:
            log_wrapper.warning(
                "VectorIndex",
                "ingest_json",
                "No documents to ingest."
            )
            return

        side_label = "reference" if mode == "reference" else "input"

        for item in data:
            # 確保三個欄位都在
            if "orig_sid" not in item:
                item["orig_sid"] = "undefined"  # or raise
            if "group_uid" not in item:
                item["group_uid"] = f"{side_label}-group_???"
            item["side"] = side_label
            item["sid"] = item.get("sid", item["uid"])

            if "uid" not in item or "text" not in item:
                log_wrapper.error(
                    "VectorIndex",
                    "ingest_json",
                    f"ingest_json: missing uid/text {item}"
                )
                return

        points: List[PointStruct] = []
        for item in data:
            chunk_uid = item["uid"]  # chunk id
            # 將業務 ID 轉換為合法的 Qdrant point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_uid))
            vector = self.embedding_manager.generate_embedding(item["text"])
            payload = dict(item)     # 包含 orig_sid, group_uid, side, text, ...
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self._ensure_collection(collection_name)
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            log_wrapper.info(
                "VectorIndex",
                "ingest_json",
                f"Ingested {len(points)} points into [{collection_name}]."
            )
        except Exception as e:
            log_wrapper.error(
                "VectorIndex",
                "ingest_json",
                f"Failed to ingest data into {collection_name}: {e}"
            )

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        以向量檢索，回傳 top-k 相似結果。
        回傳格式: List of {{'uid':..., 'payload':..., 'score':...}}
        """
        self._ensure_collection(collection_name)
        try:
            qfilter = None
            if filters:
                must = []
                for key, value in filters.items():
                    # 支援 orig_sid 過濾，讓前端可以用業務 ID 搜尋
                    if key == "orig_sid":
                        must.append(FieldCondition(key="orig_sid", match=MatchValue(value=value)))
                    elif key == "group_uid":
                        must.append(FieldCondition(key="group_uid", match=MatchValue(value=value)))
                    else:
                        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
                qfilter = models.Filter(must=must)

            # v1.7+ 用 query_filter=
            results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=qfilter,
            )

            output: List[Dict[str, Any]] = []
            for hit in results:
                # 用 payload["sid"] 當我們原始的 UID
                output.append({
                    "uid": hit.payload.get("sid", hit.id),
                    "payload": hit.payload,
                    "text": hit.payload.get("text", ""),
                    "score": hit.score
                })
            return output

        except Exception as e:
            log_wrapper.error(
                "VectorIndex",
                "search",
                f"Search error on {collection_name}: {e}"
            )
            return []

    async def search_async(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int,
        filters: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        # 指定我們自己的 executor
        return await loop.run_in_executor(
            VectorIndex._executor,
            self.search,
            collection_name, query_vector, k, filters
        )

    def remove_document(self, collection_name: str, uid: str) -> None:
        """依業務 ID (orig_sid) 刪除指定點"""
        try:
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[FieldCondition(key="orig_sid", match=MatchValue(value=uid))]
                )
            )
            log_wrapper.info(
                "VectorIndex",
                "remove_document",
                f"Removed points with orig_sid={uid} from {collection_name}."
            )
        except Exception as e:
            log_wrapper.error(
                "VectorIndex",
                "remove_document",
                f"Failed to remove points with orig_sid={uid}: {e}"
            )

    def update_document(
        self,
        collection_name: str,
        uid: str,
        new_text: str,
        new_meta: Dict[str, Any]
    ) -> None:
        """更新單一點：重生成 embedding 並 upsert"""
        try:
            vector = self.embedding_manager.generate_embedding(new_text)
            payload = dict(new_meta)
            payload["uid"] = uid
            payload.setdefault("sid", uid)
            payload.setdefault("group_uid", new_meta.get("group_uid", ""))
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=uid, vector=vector, payload=payload)]
            )
            log_wrapper.info(
                "VectorIndex",
                "update_document",
                f"Updated point {uid} in {collection_name}."
            )
        except Exception as e:
            log_wrapper.error(
                "VectorIndex",
                "update_document",
                f"Failed to update point {uid}: {e}"
            )

    def get_document_by_id(self, collection_name: str, uid: str) -> List[Dict[str, Any]]:
        """
        依 sid 檢索點的 metadata 和 score。Key 要對上 payload 裡的 `sid`。
        """
        try:
            # 使用 retrieve 直接依 ID 取得點
            points = self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[uid]
            )
            return [{
                "uid": point.payload.get("sid", point.id),
                "payload": point.payload,
                "score": 1.0  # retrieve 不計算相似度，固定為 1.0
            } for point in points]
        except Exception as e:
            log_wrapper.error(
                "VectorIndex",
                "get_document_by_id",
                f"Error fetching by sid={uid}: {e}"
            )
            return []
