# tests/test_vector_store_manager.py

import pytest
pytest.skip(
    "Vector store manager tests require the old VectorStoreManager implementation",
    allow_module_level=True
)
import random
import string

from managers.embedding_manager import EmbeddingManager
from managers.vector_index import VectorStoreManager


@pytest.fixture(scope="session")
def embedding_manager():
    """
    建立一個測試用的 EmbeddingManager。
    可以視情況改成 Mock，避免實際呼叫 OpenAI Embeddings 產生費用。
    這裡僅示範，如果需要 Mock，請自行修改 embed_query() 回傳固定向量。
    """
    # 用假的 API key 或設定
    return EmbeddingManager(openai_api_key="DUMMY_KEY", embedding_model_name="text-embedding-ada-002")


@pytest.fixture(scope="session")
def vector_store_manager(embedding_manager):
    """
    建立一個 VectorStoreManager，連到本機預設http://localhost:6333的 Qdrant。
    預設 collection_name = "test_collection"，測試結束後可以不保留資料。
    """
    vsm = VectorStoreManager(
        embedding_manager=embedding_manager,
        qdrant_url="http://localhost:6333",
        default_collection_name="test_collection",
        api_key=None,  # 若 Qdrant 有開啟auth可放key
        vector_size=1536  # 依您的embedding大小
    )
    return vsm


def random_doc_id(prefix="DOC") -> str:
    """
    小工具：產生隨機 doc_id 以避免測試資料重複。
    """
    rand_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{prefix}-{rand_suffix}"


def test_add_standard_documents(vector_store_manager):
    """
    測試 add_standard_documents() 功能，包括:
      1) 建立一個新的 collection
      2) 新增多筆文件
      3) 確認是否能正確取得
    """
    collection_name = "test_add_standard_docs"
    vector_store_manager.create_collection(collection_name)

    docs = [
        {
            "doc_id": random_doc_id(),
            "text": "Hello world! This is a test document for embedding.",
            "context": "test_context",
            "category": "example"
        },
        {
            "doc_id": random_doc_id(),
            "text": "Another test doc, containing different contents.",
            "context": "test_context",
            "category": "example2"
        }
    ]

    vector_store_manager.add_standard_documents(collection_name, docs)

    # 使用 get_document_by_id() 驗證其中一筆
    test_doc_id = docs[0]["doc_id"]
    fetched_docs = vector_store_manager.get_document_by_id(collection_name, test_doc_id)
    assert len(fetched_docs) == 1, f"Should only find 1 doc by doc_id={test_doc_id}"
    assert fetched_docs[0]["doc_id"] == test_doc_id
    assert fetched_docs[0]["metadata"].get("context") == "test_context"
    assert fetched_docs[0]["metadata"].get("category") == "example"


def test_add_jsonl_documents(vector_store_manager):
    """
    測試 add_jsonl_documents()，原理同 add_standard_documents()。
    這裡直接傳入與 add_standard_documents 相同格式
    （加上 doc_id, text, context...），只要符合 JSONL 格式就能用。
    """
    collection_name = "test_add_jsonl_docs"
    vector_store_manager.create_collection(collection_name)

    json_lines = [
        {
            "doc_id": random_doc_id(),
            "text": "This is from add_jsonl_documents #1",
            "context": "jsonl_test",
            "extra_field": "foo"
        },
        {
            "doc_id": random_doc_id(),
            "text": "This is from add_jsonl_documents #2",
            "context": "jsonl_test",
            "extra_field": "bar"
        }
    ]

    vector_store_manager.add_jsonl_documents(collection_name, json_lines)
    # 簡單驗證
    test_doc_id = json_lines[1]["doc_id"]
    fetched_docs = vector_store_manager.get_document_by_id(collection_name, test_doc_id)
    assert len(fetched_docs) == 1
    assert fetched_docs[0]["doc_id"] == test_doc_id
    assert fetched_docs[0]["metadata"].get("context") == "jsonl_test"
    assert fetched_docs[0]["metadata"].get("extra_field") == "bar"


def test_remove_document(vector_store_manager):
    """
    測試 remove_document()。
    1) 先新增一筆
    2) remove_document()
    3) 再 get_document_by_id() 應該拿不到
    """
    collection_name = "test_remove_doc"
    vector_store_manager.create_collection(collection_name)

    doc_id = random_doc_id("REM")
    doc = [{
        "doc_id": doc_id,
        "text": "Document to be removed",
        "context": "remove_test"
    }]

    vector_store_manager.add_standard_documents(collection_name, doc)

    # 確認已加進去
    fetched_before = vector_store_manager.get_document_by_id(collection_name, doc_id)
    assert len(fetched_before) == 1, "Should find the doc before removal"

    # remove
    vector_store_manager.remove_document(collection_name, doc_id)

    # 確認已刪除
    fetched_after = vector_store_manager.get_document_by_id(collection_name, doc_id)
    assert len(fetched_after) == 0, "Should not find the doc after removal"


def test_update_document(vector_store_manager):
    """
    測試 update_document():
      1) 新增一筆 doc
      2) update_document() -> 內容(embedding)被覆蓋
      3) 重新 get_document_by_id()，驗證 metadata 是否更新成功
    """
    collection_name = "test_update_doc"
    vector_store_manager.create_collection(collection_name)

    doc_id = random_doc_id("UPD")
    initial_doc = [{
        "doc_id": doc_id,
        "text": "Initial text",
        "context": "update_test",
        "version": "1.0"
    }]

    vector_store_manager.add_standard_documents(collection_name, initial_doc)

    # Update
    new_text = "Updated text for embedding"
    new_meta = {
        "context": "update_test_modified",
        "version": "2.0",
        "another_field": "Hello"
    }
    vector_store_manager.update_document(
        collection_name=collection_name,
        doc_id=doc_id,
        new_text=new_text,
        new_meta=new_meta
    )

    # 驗證
    docs_after = vector_store_manager.get_document_by_id(collection_name, doc_id)
    assert len(docs_after) == 1, "Should still be 1 doc after update"
    updated_doc = docs_after[0]
    assert updated_doc["metadata"].get("version") == "2.0"
    assert updated_doc["metadata"].get("another_field") == "Hello"
    # text 不會存在 metadata，因為 add_standard_documents() 不存 text 到payload
    # 但embedding已更新。無法直接在這裡斷言embedding正確與否(除非Mock Embedding)。
    # 這裡僅確定 metadata 被更新了即可。


def test_search_similar_with_score(vector_store_manager):
    """
    測試 search_similar_with_score:
      1) 新增幾筆文件
      2) 自行生成 query 的embedding (可 Mock or 用真embedding)
      3) 看多筆文件中誰最接近 (可能都很小, 但至少能確認flow)
    """
    collection_name = "test_search_similar"
    vector_store_manager.create_collection(collection_name)

    test_docs = [
        {
            "doc_id": random_doc_id("SRCH"),
            "text": "Apple banana cat dog",
            "context": "search_test"
        },
        {
            "doc_id": random_doc_id("SRCH"),
            "text": "Banana tree house roof",
            "context": "search_test"
        },
        {
            "doc_id": random_doc_id("SRCH"),
            "text": "Pineapple apple orchard",
            "context": "search_test"
        },
    ]
    vector_store_manager.add_standard_documents(collection_name, test_docs)

    # 產生 Query 的 embedding：
    # 若您想要可預期結果，可 Mock EmbeddingManager，使 embed_query() 回傳固定向量。
    # 這邊示範直接用真embedding, 效果可能不明顯但可測流程。
    query_text = "apple fruit"
    embedding = vector_store_manager.embedding_manager.generate_embedding(query_text)
    results = vector_store_manager.search_similar_with_score(
        collection_name=collection_name,
        query_vector=embedding,
        k=2
    )

    assert len(results) <= 2, "Should return up to 2 hits"

    if len(results) > 0:
        print("Top search result =>", results[0])
        # 大致檢查 doc_id / text 有值
        assert "doc_id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]
