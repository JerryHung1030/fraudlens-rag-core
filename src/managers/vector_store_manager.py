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
    使用 LangChain + Qdrant 來管理向量資料庫，不再回傳相似度，由 LLM 自行評估置信度。
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

        # 預先建立(或重建) collection
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        # 用於檢索
        self.qdrant_store: Optional[Qdrant] = None

    def add_documents(self, domain: str, docs: List[Any], metadata: Dict):
        logger.debug(f"add_documents called with domain={domain}, doc_count={len(docs)}")

        doc_list = []
        for item in docs:
            # item = {"code":"7-16", "category":"假求職詐騙", "desc":"高薪可預支薪水"}
            # 組合 page_content: '7-16假求職詐騙:高薪可預支薪水'
            label_str = f"{item['code']}{item['category']}:{item['desc']}"
            
            merged_meta = dict(metadata)  # 從外部帶入 e.g. {"type":"scam_pattern"}
            merged_meta["domain"] = domain
            merged_meta["code"] = item["code"]
            merged_meta["category"] = item["category"]
            merged_meta["desc"] = item["desc"]

            doc_list.append(Document(page_content=label_str, metadata=merged_meta))

        if not doc_list:
            logger.warning("No documents to add.")
            return

        if self.qdrant_store is None:
            logger.debug(f"Creating new Qdrant store with {len(doc_list)} docs (collection={self.collection_name})")
            self.qdrant_store = Qdrant.from_documents(
                documents=doc_list,
                embedding=self.embedding_manager.embedding_model,
                collection_name=self.collection_name
            )
        else:
            logger.debug(f"Adding {len(doc_list)} docs to existing Qdrant store.")
            self.qdrant_store.add_documents(doc_list)

    def update_document(self, domain: str, doc_id: str, new_content: str):
        logger.warning("update_document not implemented in PoC Qdrant")

    def remove_document(self, domain: str, doc_id: str):
        logger.warning("remove_document not implemented in PoC Qdrant")

    def search_similar(
        self,
        domain: str,
        query_vector: List[float],
        k: int,
        filters: Dict[str, Any] = None
    ) -> List[str]:
        """
        回傳 List[str]，只包含文件內容，不再回傳相似度。
        以 Qdrant similarity_search_by_vector() 為基礎，無法取得實際分數。
        """
        if not self.qdrant_store:
            logger.error("Qdrant store not initialized.")
            return []

        logger.debug(f"search_similar called domain={domain}, k={k}, filters={filters}")
        # 單純呼叫 similarity_search_by_vector
        found_docs = self.qdrant_store.similarity_search_by_vector(
            embedding=query_vector, k=k
        )
        logger.debug(f"Found {len(found_docs)} docs from similarity_search_by_vector()")

        # 過濾 domain & filters
        results = []
        for idx, doc in enumerate(found_docs):
            # logging doc metadata
            logger.debug(f"Doc #{idx} metadata: {doc.metadata}")
            if doc.metadata.get("domain") == domain:
                if self._match_filters(doc.metadata, filters):
                    results.append(doc.page_content)
        logger.info(f"Final doc_count after domain/filters: {len(results)}")
        return results

    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        for key, val in filters.items():
            if metadata.get(key) != val:
                return False
        return True
