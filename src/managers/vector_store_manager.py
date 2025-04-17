# src/managers/vector_store_manager.py
import logging
from typing import Dict, Any, List, Optional
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    使用 LangChain + Qdrant 來管理向量資料庫。
    重點：search_similar_with_score() 回傳 List[dict], 每個dict包含 doc_id, text, score
    """
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "my_collection",
        api_key: Optional[str] = None,
        vector_size: int = 1536  # 依 embeddings size
    ):
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key)

        # 先建立(或重建) collection
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

        # 用於檢索
        self.qdrant_store: Optional[Qdrant] = None

    def add_documents(self, domain: str, docs: List[Any], metadata: Dict):
        """
        以前用在詐騙模式, item = {"code","category","desc"}
        """
        logger.debug(f"add_documents domain={domain}, count={len(docs)}")
        doc_list = []
        for idx, item in enumerate(docs):
            label_str = f"{item['code']}{item['category']}:{item['desc']}"
            merged_meta = dict(metadata)
            merged_meta["domain"] = domain
            merged_meta["doc_id"] = item.get("code", f"{domain}-{idx}")
            merged_meta["code"] = item["code"]
            merged_meta["category"] = item["category"]
            merged_meta["desc"] = item["desc"]

            doc_list.append(Document(page_content=label_str, metadata=merged_meta))

        if not doc_list:
            return

        if self.qdrant_store is None:
            self.qdrant_store = Qdrant.from_documents(
                documents=doc_list,
                embedding=self.embedding_manager.embedding_model,
                collection_name=self.collection_name
            )
        else:
            self.qdrant_store.add_documents(doc_list)

    def add_jsonl_documents(
        self,
        domain: str,
        json_lines: list[dict],
        text_key: str,
        meta_keys: List[str] | None = None
    ):
        """
        將任意 jsonl 帶進 Qdrant.
        會給每段產生 doc_id, 也可從 item["clause_id"] 來設定 doc_id.
        """
        if not json_lines:
            return
        docs = []
        for idx, item in enumerate(json_lines):
            para = item[text_key]
            doc_id = item.get("clause_id", f"{domain}-{idx}")
            meta = {k: item.get(k, "") for k in (meta_keys or [])}
            meta["domain"] = domain
            meta["doc_id"] = doc_id
            docs.append(Document(page_content=para, metadata=meta))

        if not docs:
            return

        if self.qdrant_store is None:
            self.qdrant_store = Qdrant.from_documents(
                documents=docs,
                embedding=self.embedding_manager.embedding_model,
                collection_name=self.collection_name
            )
        else:
            self.qdrant_store.add_documents(docs)

    def search_similar_with_score(
        self,
        domain: str,
        query_vector: List[float],
        k: int,
        filters: Dict[str, Any] | None = None
    ) -> List[dict]:
        """
        回傳 List[dict], 每個 dict = {"doc_id":..., "text":..., "score":...}
        """
        if not self.qdrant_store:
            logger.error("Qdrant store not initialized.")
            return []

        hits = self.qdrant_store.similarity_search_with_score_by_vector(
            embedding=query_vector, k=k, filter=filters or None
        )  # List[(Document, float)]

        results = []
        for doc, score in hits:
            if doc.metadata.get("domain") == domain:
                doc_id = doc.metadata.get("doc_id", "")
                results.append({
                    "doc_id": doc_id,
                    "text": doc.page_content,
                    "score": score
                })

        return results

    # 以下為尚未實作，PoC:
    def update_document(self, domain: str, doc_id: str, new_content: str):
        logger.warning("update_document not implemented")

    def remove_document(self, domain: str, doc_id: str):
        logger.warning("remove_document not implemented")
