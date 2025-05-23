import pytest
import asyncio
import uuid
from unittest.mock import MagicMock, patch, call, ANY

# Skip entire module if qdrant_client is not available
pytest.importorskip("qdrant_client", reason="qdrant_client package required for vector store tests")

# Qdrant specific imports for mocking and type hints
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.models import (
        PointStruct,
        ScoredPoint,
        PointRecord,
        Distance,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )
except Exception:  # pragma: no cover - missing features
    pytest.skip("Required qdrant_client models are unavailable", allow_module_level=True)

# Project specific imports
from src.rag_core.infrastructure.vector_store import VectorIndex
from src.rag_core.infrastructure.embedding import EmbeddingManager # For spec
from src.rag_core.domain.schema_checker import DataStructureChecker # For spec

# --- Fixtures ---

@pytest.fixture
def mock_qdrant_client_instance(mocker):
    """Mocks the instance of QdrantClient."""
    mock_client = MagicMock(spec=QdrantClient)
    
    # Mock get_collection: by default, assume collection exists
    mock_client.get_collection = MagicMock()
    
    # Mock create_collection
    mock_client.create_collection = MagicMock()
    
    # Mock upsert
    mock_client.upsert = MagicMock()
    
    # Mock search - default to returning a list of ScoredPoint mocks
    mock_scored_point1 = MagicMock(spec=ScoredPoint)
    mock_scored_point1.id = str(uuid.uuid4())
    mock_scored_point1.payload = {"uid": "doc1", "text": "Document 1 text", "orig_sid":"orig_doc1_sid", "side":"default_side"}
    mock_scored_point1.score = 0.9
    mock_scored_point1.vector = [0.1] * 10 # Dummy vector

    mock_scored_point2 = MagicMock(spec=ScoredPoint)
    mock_scored_point2.id = str(uuid.uuid4())
    mock_scored_point2.payload = {"uid": "doc2", "text": "Document 2 text", "orig_sid":"orig_doc2_sid", "side":"default_side"}
    mock_scored_point2.score = 0.8
    mock_scored_point2.vector = [0.2] * 10 # Dummy vector
    
    mock_client.search = MagicMock(return_value=[mock_scored_point1, mock_scored_point2])
    
    # Mock delete
    mock_client.delete = MagicMock()
    
    # Mock retrieve - default to returning a list of PointRecord mocks
    mock_point_record = MagicMock(spec=PointRecord)
    mock_point_record.id = str(uuid.uuid4())
    mock_point_record.payload = {"uid": "retrieved_doc", "text": "Retrieved document text"}
    mock_point_record.vector = [0.3] * 10 # Dummy vector
    mock_client.retrieve = MagicMock(return_value=[mock_point_record])
    
    return mock_client

@pytest.fixture
def mock_qdrant_client_class(mocker, mock_qdrant_client_instance):
    """Mocks the QdrantClient class, returning the mocked instance."""
    mock_class = mocker.patch('src.rag_core.infrastructure.vector_store.QdrantClient', return_value=mock_qdrant_client_instance)
    return mock_class

@pytest.fixture
def mock_embedding_manager(mocker):
    """Mocks EmbeddingManager."""
    mock_em = mocker.Mock(spec=EmbeddingManager)
    mock_em.generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3]) # Sample embedding
    return mock_em

@pytest.fixture
def mock_data_structure_checker(mocker):
    """Mocks DataStructureChecker."""
    # By default, validate does nothing (passes)
    mock_dsc = mocker.Mock(spec=DataStructureChecker)
    mock_dsc.validate = MagicMock() 
    return mock_dsc

@pytest.fixture
def mock_config_manager_settings(mocker):
    """Mocks config_manager.settings."""
    mock_settings = MagicMock()
    mock_settings.vector_db.collection = "test_collection"
    mock_settings.vector_db.vector_size = 10 # Must match dummy vectors if used
    mock_settings.vector_db.qdrant_host = "localhost"
    mock_settings.vector_db.qdrant_port = 6333
    mock_settings.thread_pool.vector_pool = 4 # Example value
    
    mocker.patch('src.rag_core.infrastructure.vector_store.config_manager', mock_settings)
    return mock_settings

@pytest.fixture
def mock_log_wrapper(mocker):
    """Mocks log_wrapper."""
    return mocker.patch('src.rag_core.infrastructure.vector_store.log_wrapper')

@pytest.fixture
def mock_uuid5(mocker):
    """Mocks uuid.uuid5 to return predictable UUIDs based on input."""
    # This allows us to check point IDs if they are derived from UIDs
    def deterministic_uuid5(namespace, name):
        # Simple mock: return a string that includes the name for easy assertion
        return f"uuid_for_{name}" 
    
    return mocker.patch('src.rag_core.infrastructure.vector_store.uuid.uuid5', side_effect=deterministic_uuid5)


# --- Test Classes ---

class TestVectorIndexInit:
    def test_init_collection_exists(self, mock_qdrant_client_class, mock_qdrant_client_instance, 
                                    mock_embedding_manager, mock_data_structure_checker, 
                                    mock_config_manager_settings, mock_log_wrapper):
        """
        Scenario 1: Default collection exists.
        `qdrant_client.get_collection` succeeds.
        `create_collection` NOT called.
        """
        # get_collection is mocked to succeed by default in mock_qdrant_client_instance
        
        vector_index = VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )
        
        mock_qdrant_client_class.assert_called_once_with(
            host=mock_config_manager_settings.vector_db.qdrant_host,
            port=mock_config_manager_settings.vector_db.qdrant_port
        )
        mock_qdrant_client_instance.get_collection.assert_called_once_with(
            collection_name=mock_config_manager_settings.vector_db.collection
        )
        mock_qdrant_client_instance.create_collection.assert_not_called()
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "__init__", 
            f"Successfully connected to Qdrant. Collection '{mock_config_manager_settings.vector_db.collection}' exists."
        )
        assert vector_index.collection_name == mock_config_manager_settings.vector_db.collection

    def test_init_collection_does_not_exist_create_success(
            self, mock_qdrant_client_class, mock_qdrant_client_instance, 
            mock_embedding_manager, mock_data_structure_checker, 
            mock_config_manager_settings, mock_log_wrapper):
        """
        Scenario 2: Default collection does not exist, create success.
        `get_collection` raises an error (simulating collection not found).
        `create_collection` is called and succeeds.
        """
        # Simulate get_collection failing (e.g., collection not found)
        # The actual error from qdrant_client might be more specific, like ValueError or an unexpected RPC error.
        # For this test, a generic Exception is fine as the code catches `Exception`.
        mock_qdrant_client_instance.get_collection.side_effect = Exception("Collection not found")
        
        vector_index = VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )
        
        mock_qdrant_client_instance.get_collection.assert_called_once_with(
            collection_name=mock_config_manager_settings.vector_db.collection
        )
        mock_qdrant_client_instance.create_collection.assert_called_once_with(
            collection_name=mock_config_manager_settings.vector_db.collection,
            vectors_config=models.VectorParams(
                size=mock_config_manager_settings.vector_db.vector_size,
                distance=models.Distance.COSINE # Assuming COSINE is default or configured
            )
        )
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "__init__", 
            f"Collection '{mock_config_manager_settings.vector_db.collection}' not found. Attempting to create."
        )
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "__init__", 
            f"Collection '{mock_config_manager_settings.vector_db.collection}' created successfully."
        )

    def test_init_collection_creation_fails(
            self, mock_qdrant_client_class, mock_qdrant_client_instance, 
            mock_embedding_manager, mock_data_structure_checker, 
            mock_config_manager_settings, mock_log_wrapper):
        """
        Scenario 3: Default collection creation fails.
        `get_collection` raises an error.
        `create_collection` also raises an error.
        `pytest.raises` during VectorIndex instantiation.
        """
        mock_qdrant_client_instance.get_collection.side_effect = Exception("Collection not found initially")
        mock_qdrant_client_instance.create_collection.side_effect = Exception("Failed to create collection")
        
        with pytest.raises(Exception, match="Failed to create collection"):
            VectorIndex(
                embedding_manager=mock_embedding_manager,
                data_checker=mock_data_structure_checker
            )
        
        mock_log_wrapper.error.assert_any_call(
            "VectorIndex", "__init__", 
            f"Failed to create collection '{mock_config_manager_settings.vector_db.collection}'. Error: Failed to create collection"
        )

class TestVectorIndexUidToPointId:
    def test_uid_to_point_id_consistency(self):
        """
        Scenario 1: Consistency.
        _uid_to_point_id("uid1") == _uid_to_point_id("uid1").
        """
        uid = "test_uid_123"
        point_id1 = VectorIndex._uid_to_point_id(uid)
        point_id2 = VectorIndex._uid_to_point_id(uid)
        assert point_id1 == point_id2
        # Also check if it's a valid UUID string (optional, but good for sanity)
        assert isinstance(uuid.UUID(point_id1), uuid.UUID)

    def test_uid_to_point_id_uniqueness(self):
        """
        Scenario 2: Uniqueness.
        _uid_to_point_id("uid1") != _uid_to_point_id("uid2").
        """
        uid1 = "test_uid_1"
        uid2 = "test_uid_2"
        point_id1 = VectorIndex._uid_to_point_id(uid1)
        point_id2 = VectorIndex._uid_to_point_id(uid2)
        assert point_id1 != point_id2

class TestVectorIndexCreateCollectionPublic:
    # This method is very similar to the __init__ logic for collection creation.
    # We can simplify tests if the core logic is robustly tested in __init__.
    
    def test_create_collection_new_name_success(
            self, mock_qdrant_client_class, mock_qdrant_client_instance, 
            mock_embedding_manager, mock_data_structure_checker, 
            mock_config_manager_settings, mock_log_wrapper):
        """
        Scenario 1 (similar to __init__ Scen 2): New collection name, create success.
        """
        new_collection_name = "new_custom_collection"
        # Simulate get_collection failing for the new name
        mock_qdrant_client_instance.get_collection.side_effect = Exception("Collection not found")
        
        vector_index = VectorIndex( # Initialize with default collection first
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )
        # Reset mocks for the specific call to the public create_collection
        mock_qdrant_client_instance.get_collection.reset_mock(side_effect=True)
        mock_qdrant_client_instance.create_collection.reset_mock()
        mock_qdrant_client_instance.get_collection.side_effect = Exception("New collection not found")


        vector_index.create_collection(collection_name=new_collection_name, vector_size=128)
        
        mock_qdrant_client_instance.get_collection.assert_called_once_with(
            collection_name=new_collection_name
        )
        mock_qdrant_client_instance.create_collection.assert_called_once_with(
            collection_name=new_collection_name,
            vectors_config=models.VectorParams(
                size=128, # Explicitly passed vector_size
                distance=models.Distance.COSINE
            )
        )
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "create_collection", 
            f"Collection '{new_collection_name}' not found. Attempting to create."
        )
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "create_collection", 
            f"Collection '{new_collection_name}' created successfully."
        )

    def test_create_collection_already_exists(
            self, mock_qdrant_client_class, mock_qdrant_client_instance, 
            mock_embedding_manager, mock_data_structure_checker, 
            mock_config_manager_settings, mock_log_wrapper):
        """
        Scenario (similar to __init__ Scen 1): Collection with given name already exists.
        """
        existing_collection_name = "already_exists_collection"
        # get_collection will succeed by default from main fixture for this name
        mock_qdrant_client_instance.get_collection.side_effect = None 
        
        vector_index = VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )
        # Reset mocks for the specific call
        mock_qdrant_client_instance.get_collection.reset_mock()
        mock_qdrant_client_instance.create_collection.reset_mock()

        vector_index.create_collection(collection_name=existing_collection_name, vector_size=128)
        
        mock_qdrant_client_instance.get_collection.assert_called_once_with(
            collection_name=existing_collection_name
        )
        mock_qdrant_client_instance.create_collection.assert_not_called()
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "create_collection", 
            f"Collection '{existing_collection_name}' already exists. No action taken."
        )

    def test_create_collection_fails_on_creation_attempt(
            self, mock_qdrant_client_class, mock_qdrant_client_instance, 
            mock_embedding_manager, mock_data_structure_checker, 
            mock_config_manager_settings, mock_log_wrapper):
        """
        Scenario (similar to __init__ Scen 3): Creation attempt fails.
        """
        failing_collection_name = "failing_creation_collection"
        mock_qdrant_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_qdrant_client_instance.create_collection.side_effect = Exception("Creation RPC error")
        
        vector_index = VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )
        mock_qdrant_client_instance.get_collection.reset_mock(side_effect=True) # Reset for the specific call
        mock_qdrant_client_instance.create_collection.reset_mock(side_effect=True)
        mock_qdrant_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_qdrant_client_instance.create_collection.side_effect = Exception("Creation RPC error")


        with pytest.raises(Exception, match="Creation RPC error"):
            vector_index.create_collection(collection_name=failing_collection_name, vector_size=128)
            
        mock_log_wrapper.error.assert_any_call(
            "VectorIndex", "create_collection", 
            f"Failed to create collection '{failing_collection_name}'. Error: Creation RPC error"
        )

class TestVectorIndexIngestJson:
    @pytest.fixture
    def vector_index_instance(self, mock_qdrant_client_class, mock_embedding_manager, 
                              mock_data_structure_checker, mock_config_manager_settings):
        # This fixture provides a VectorIndex instance for ingest_json tests
        # It relies on the other fixtures to set up mocks correctly.
        return VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )

    def test_ingest_json_success(self, vector_index_instance, mock_embedding_manager, 
                                 mock_qdrant_client_instance, mock_data_structure_checker, 
                                 mock_log_wrapper, mock_uuid5):
        """
        Scenario 1: Success. Valid data.
        `embedding_manager.generate_embedding` called per item.
        `qdrant_client.upsert` called with correct `PointStruct`s.
        """
        valid_data = [
            {"uid": "doc1", "text": "Text for document 1", "orig_sid": "orig_s1", "side": "A"},
            {"uid": "doc2", "text": "Text for document 2", "orig_sid": "orig_s2", "side": "B", "extra_payload": "value"}
        ]
        # Mock embeddings for each document
        mock_embedding_manager.generate_embedding.side_effect = [
            [0.1] * 10, # Embedding for doc1
            [0.2] * 10  # Embedding for doc2
        ]
        
        vector_index_instance.ingest_json(valid_data)

        # Verify data_structure_checker.validate was called
        mock_data_structure_checker.validate.assert_called_once_with(valid_data, mode="input")

        # Verify generate_embedding calls
        assert mock_embedding_manager.generate_embedding.call_count == 2
        mock_embedding_manager.generate_embedding.assert_any_call("Text for document 1")
        mock_embedding_manager.generate_embedding.assert_any_call("Text for document 2")

        # Verify upsert call
        mock_qdrant_client_instance.upsert.assert_called_once()
        args, kwargs = mock_qdrant_client_instance.upsert.call_args
        
        assert kwargs['collection_name'] == vector_index_instance.collection_name
        
        points_arg = kwargs['points']
        assert len(points_arg) == 2
        
        # Check point 1
        assert points_arg[0].id == VectorIndex._uid_to_point_id("doc1") # Using the actual static method for consistency
        assert points_arg[0].vector == [0.1] * 10
        assert points_arg[0].payload == {"uid": "doc1", "text": "Text for document 1", "orig_sid": "orig_s1", "side": "A"}
        
        # Check point 2
        assert points_arg[1].id == VectorIndex._uid_to_point_id("doc2")
        assert points_arg[1].vector == [0.2] * 10
        assert points_arg[1].payload == {"uid": "doc2", "text": "Text for document 2", "orig_sid": "orig_s2", "side": "B", "extra_payload": "value"}
        
        mock_log_wrapper.info.assert_any_call(
            "VectorIndex", "ingest_json", 
            f"Successfully ingested {len(valid_data)} documents into '{vector_index_instance.collection_name}'."
        )

    def test_ingest_json_empty_data(self, vector_index_instance, mock_qdrant_client_instance, mock_log_wrapper):
        """
        Scenario 2: Empty data. `log_wrapper.warning` called, `upsert` not called.
        """
        vector_index_instance.ingest_json([])
        
        mock_log_wrapper.warning.assert_called_once_with(
            "VectorIndex", "ingest_json", "No documents provided for ingestion."
        )
        mock_qdrant_client_instance.upsert.assert_not_called()

    @pytest.mark.parametrize("missing_key_data, missing_field", [
        ([{"text": "No uid here"}], "uid"),
        ([{"uid": "doc_no_text"}], "text")
    ])
    def test_ingest_json_missing_uid_or_text(self, vector_index_instance, mock_qdrant_client_instance, 
                                             mock_log_wrapper, missing_key_data, missing_field):
        """
        Scenario 3: Missing `uid` or `text`. `log_wrapper.error` called, `upsert` not called.
        """
        vector_index_instance.ingest_json(missing_key_data)
        
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "ingest_json", 
            f"Document missing '{missing_field}' field: {missing_key_data[0]}"
        )
        mock_qdrant_client_instance.upsert.assert_not_called()

    def test_ingest_json_upsert_fails(self, vector_index_instance, mock_embedding_manager, 
                                      mock_qdrant_client_instance, mock_data_structure_checker, 
                                      mock_log_wrapper):
        """
        Scenario 4: Upsert fails. `qdrant_client.upsert` raises error. `log_wrapper.error` called.
        """
        valid_data = [{"uid": "doc1", "text": "Text for document 1"}]
        mock_embedding_manager.generate_embedding.return_value = [0.1] * 10
        
        original_error_message = "Qdrant upsert failed due to network issue"
        mock_qdrant_client_instance.upsert.side_effect = Exception(original_error_message)
        
        with pytest.raises(Exception, match=original_error_message): # Ensure the original exception is re-raised
            vector_index_instance.ingest_json(valid_data)
            
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "ingest_json", 
            f"Failed to ingest documents into '{vector_index_instance.collection_name}'. Error: {original_error_message}"
        )

    def test_ingest_json_data_structure_validation_fails(
        self, vector_index_instance, mock_data_structure_checker, 
        mock_qdrant_client_instance, mock_log_wrapper
    ):
        """
        Test that if data_structure_checker.validate fails, an error is logged and upsert is not called.
        """
        invalid_structured_data = [{"uid": 123, "text": "Bad UID type"}] # Example of invalid structure
        from src.rag_core.domain.schema_checker import DataSchemaError # Import for raising
        mock_data_structure_checker.validate.side_effect = DataSchemaError(details=["Invalid UID type"])

        with pytest.raises(DataSchemaError): # Assuming DataSchemaError should propagate
             vector_index_instance.ingest_json(invalid_structured_data)

        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "ingest_json",
            "Data structure validation failed. Details: Invalid UID type"
        )
        mock_qdrant_client_instance.upsert.assert_not_called()

class TestVectorIndexSearch:
    @pytest.fixture
    def vector_index_instance(self, mock_qdrant_client_class, mock_embedding_manager, 
                              mock_data_structure_checker, mock_config_manager_settings):
        return VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )

    def test_search_success_no_filters(self, vector_index_instance, mock_qdrant_client_instance, 
                                       mock_embedding_manager, mock_log_wrapper):
        """
        Scenario 1: Success, no filters.
        `qdrant_client.search` returns mock `ScoredPoint`s.
        Check `search` args (filter is `None`). Check formatted results.
        """
        query_text = "Search for this text"
        query_vector = [0.5] * 10 # Example vector
        k = 5
        
        mock_embedding_manager.generate_embedding.return_value = query_vector
        
        # mock_qdrant_client_instance.search is already set up in the main fixture
        # to return two ScoredPoint mocks.
        
        results = vector_index_instance.search(query_text, k=k)
        
        mock_embedding_manager.generate_embedding.assert_called_once_with(query_text)
        mock_qdrant_client_instance.search.assert_called_once_with(
            collection_name=vector_index_instance.collection_name,
            query_vector=query_vector,
            query_filter=None, # Assert filter is None
            limit=k,
            with_payload=True,
            with_vectors=True # Assuming default or standard practice
        )
        
        assert len(results) == 2 # Based on mock_qdrant_client_instance fixture
        assert results[0]['uid'] == "doc1"
        assert results[0]['text'] == "Document 1 text"
        assert results[0]['score'] == 0.9
        assert 'vector' in results[0] # Check if vector is included
        assert results[0]['orig_sid'] == "orig_doc1_sid"
        assert results[0]['side'] == "default_side"


    def test_search_success_with_filters(self, vector_index_instance, mock_qdrant_client_instance, 
                                         mock_embedding_manager, mock_log_wrapper):
        """
        Scenario 2: Success, with filters.
        Provide filters. Check `qdrant_client.search` called with `models.Filter`.
        """
        query_text = "Search with filters"
        query_vector = [0.6] * 10
        k = 3
        filters = {"orig_sid": "specific_sid", "custom_field": "custom_value"}
        
        mock_embedding_manager.generate_embedding.return_value = query_vector
        
        results = vector_index_instance.search(query_text, k=k, filter_expr=filters)
        
        mock_embedding_manager.generate_embedding.assert_called_once_with(query_text)
        
        # Construct the expected filter structure
        expected_qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="orig_sid", match=models.MatchValue(value="specific_sid")),
                models.FieldCondition(key="custom_field", match=models.MatchValue(value="custom_value"))
            ]
        )
        
        mock_qdrant_client_instance.search.assert_called_once_with(
            collection_name=vector_index_instance.collection_name,
            query_vector=query_vector,
            query_filter=expected_qdrant_filter, # Assert filter structure
            limit=k,
            with_payload=True,
            with_vectors=True
        )
        assert len(results) == 2 # Still using the default mock return for search

    def test_search_fails(self, vector_index_instance, mock_qdrant_client_instance, 
                          mock_embedding_manager, mock_log_wrapper):
        """
        Scenario 3: Search fails. `qdrant_client.search` raises error.
        `log_wrapper.error` called, returns `[]`.
        """
        query_text = "Search that will fail"
        query_vector = [0.7] * 10
        k = 5
        original_error_message = "Qdrant search connection timeout"
        
        mock_embedding_manager.generate_embedding.return_value = query_vector
        mock_qdrant_client_instance.search.side_effect = Exception(original_error_message)
        
        results = vector_index_instance.search(query_text, k=k)
        
        assert results == []
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "search", 
            f"Error searching in collection '{vector_index_instance.collection_name}'. Error: {original_error_message}"
        )

    def test_search_embedding_fails(self, vector_index_instance, mock_qdrant_client_instance,
                                   mock_embedding_manager, mock_log_wrapper):
        """ Test search when embedding generation fails. """
        query_text = "Search with failing embedding"
        k = 3
        original_embedding_error = "Embedding model not available"
        mock_embedding_manager.generate_embedding.side_effect = Exception(original_embedding_error)

        results = vector_index_instance.search(query_text, k=k)

        assert results == []
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "search",
            f"Failed to generate embedding for query: '{query_text}'. Error: {original_embedding_error}"
        )
        mock_qdrant_client_instance.search.assert_not_called()

@pytest.mark.asyncio
class TestVectorIndexSearchAsync:
    @pytest.fixture # Ensure this fixture is defined if not already, or use the one from TestVectorIndexSearch
    def vector_index_instance(self, mock_qdrant_client_class, mock_embedding_manager, 
                              mock_data_structure_checker, mock_config_manager_settings):
        return VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )

    async def test_search_async_successful_call(self, vector_index_instance, mocker):
        """
        Test successful async search call.
        Mocks the synchronous `search` method.
        """
        query_text = "Async search query"
        k = 5
        filters = {"field": "value"}
        expected_sync_result = [{"uid": "async_doc1", "text": "Async content", "score": 0.88}]
        
        # Mock the synchronous search method on the instance
        mock_sync_search = mocker.patch.object(
            vector_index_instance, 
            'search', 
            return_value=expected_sync_result
        )
        
        result = await vector_index_instance.search_async(query_text, k=k, filter_expr=filters)
        
        mock_sync_search.assert_called_once_with(query_text, k, filters)
        assert result == expected_sync_result

    async def test_search_async_exception_in_sync_call(self, vector_index_instance, mocker, mock_log_wrapper):
        """
        Test exception propagation from synchronous search in async call.
        """
        query_text = "Async search causing error"
        k = 3
        original_exception = Exception("Error in sync search during async call")
        
        mocker.patch.object(
            vector_index_instance,
            'search',
            side_effect=original_exception
        )
        
        # The async wrapper in VectorIndex is expected to catch the exception, log it, and return [].
        # This differs from some other async wrappers that might re-raise.
        # Let's assume the VectorIndex.search_async is designed to be resilient and return empty on error.
        # If it's supposed to re-raise, this test needs to change.
        # Based on the provided source for search_async, it logs and returns [].
        
        result = await vector_index_instance.search_async(query_text, k=k)
        
        assert result == [] # Expect empty list on error as per VectorIndex.search_async implementation
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "search_async",
            f"Error during async search for query '{query_text}'. Error: {original_exception}"
        )

class TestVectorIndexRemoveDocument:
    @pytest.fixture # Ensure this fixture is defined if not already
    def vector_index_instance(self, mock_qdrant_client_class, mock_embedding_manager, 
                              mock_data_structure_checker, mock_config_manager_settings):
        return VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )

    def test_remove_document_success(self, vector_index_instance, mock_qdrant_client_instance, mock_log_wrapper):
        """
        Scenario 1: Success. `qdrant_client.delete` called with filter for `orig_sid`.
        """
        orig_sid_to_remove = "doc_to_delete_orig_sid"
        
        vector_index_instance.remove_document(orig_sid_to_remove)
        
        expected_filter = models.Filter(
            must=[
                models.FieldCondition(key="orig_sid", match=models.MatchValue(value=orig_sid_to_remove))
            ]
        )
        
        mock_qdrant_client_instance.delete.assert_called_once_with(
            collection_name=vector_index_instance.collection_name,
            points_selector=expected_filter
        )
        mock_log_wrapper.info.assert_called_once_with(
            "VectorIndex", "remove_document",
            f"Successfully removed document with orig_sid '{orig_sid_to_remove}' from '{vector_index_instance.collection_name}'."
        )

    def test_remove_document_deletion_fails(self, vector_index_instance, mock_qdrant_client_instance, mock_log_wrapper):
        """
        Scenario 2: Deletion fails. `qdrant_client.delete` raises error. `log_wrapper.error` called.
        """
        orig_sid_to_remove = "doc_fail_delete_orig_sid"
        original_error_message = "Qdrant delete operation failed"
        mock_qdrant_client_instance.delete.side_effect = Exception(original_error_message)
        
        with pytest.raises(Exception, match=original_error_message): # Assuming error is re-raised
            vector_index_instance.remove_document(orig_sid_to_remove)
            
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "remove_document",
            f"Failed to remove document with orig_sid '{orig_sid_to_remove}' from '{vector_index_instance.collection_name}'. Error: {original_error_message}"
        )

class TestVectorIndexUpdateDocument:
    @pytest.fixture # Ensure this fixture is defined if not already
    def vector_index_instance(self, mock_qdrant_client_class, mock_embedding_manager, 
                              mock_data_structure_checker, mock_config_manager_settings):
        return VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )

    def test_update_document_success(self, vector_index_instance, mock_qdrant_client_instance, 
                                     mock_embedding_manager, mock_log_wrapper, mock_uuid5):
        """
        Scenario 1: Success. `embedding_manager.generate_embedding` called.
        `qdrant_client.upsert` called with correct `PointStruct`.
        """
        doc_uid = "doc_to_update_uid"
        doc_text = "Updated text for the document."
        doc_payload = {"orig_sid": "orig_sid_updated", "side": "updated_side", "custom": "info"}
        
        expected_vector = [0.9, 0.8, 0.7]
        mock_embedding_manager.generate_embedding.return_value = expected_vector
        
        vector_index_instance.update_document(doc_uid, doc_text, doc_payload)
        
        mock_embedding_manager.generate_embedding.assert_called_once_with(doc_text)
        
        expected_point_id = VectorIndex._uid_to_point_id(doc_uid)
        expected_full_payload = {"uid": doc_uid, "text": doc_text, **doc_payload}
        
        # Check upsert call
        mock_qdrant_client_instance.upsert.assert_called_once()
        args, kwargs = mock_qdrant_client_instance.upsert.call_args
        
        assert kwargs['collection_name'] == vector_index_instance.collection_name
        
        points_arg = kwargs['points']
        assert len(points_arg) == 1
        
        assert points_arg[0].id == expected_point_id
        assert points_arg[0].vector == expected_vector
        assert points_arg[0].payload == expected_full_payload
        
        mock_log_wrapper.info.assert_called_once_with(
            "VectorIndex", "update_document",
            f"Successfully updated document with uid '{doc_uid}' in '{vector_index_instance.collection_name}'."
        )

    def test_update_document_upsert_fails(self, vector_index_instance, mock_qdrant_client_instance, 
                                          mock_embedding_manager, mock_log_wrapper):
        """
        Scenario 2: Update fails. `qdrant_client.upsert` raises error. `log_wrapper.error` called.
        """
        doc_uid = "doc_fail_update_uid"
        doc_text = "Text for failing update."
        doc_payload = {"orig_sid": "orig_sid_fail"}
        
        mock_embedding_manager.generate_embedding.return_value = [0.5] * 10
        original_error_message = "Qdrant upsert failed during update"
        mock_qdrant_client_instance.upsert.side_effect = Exception(original_error_message)
        
        with pytest.raises(Exception, match=original_error_message): # Assuming error is re-raised
            vector_index_instance.update_document(doc_uid, doc_text, doc_payload)
            
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "update_document",
            f"Failed to update document with uid '{doc_uid}' in '{vector_index_instance.collection_name}'. Error: {original_error_message}"
        )

    def test_update_document_embedding_fails(self, vector_index_instance, mock_qdrant_client_instance,
                                             mock_embedding_manager, mock_log_wrapper):
        """ Test update_document when embedding generation fails. """
        doc_uid = "doc_update_embed_fail"
        doc_text = "Text for embedding failure update"
        doc_payload = {"orig_sid": "orig_sid_embed_fail"}
        original_embedding_error = "Embedding service unavailable for update"
        mock_embedding_manager.generate_embedding.side_effect = Exception(original_embedding_error)

        with pytest.raises(Exception, match=original_embedding_error):
            vector_index_instance.update_document(doc_uid, doc_text, doc_payload)

        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "update_document",
            f"Failed to generate embedding for document uid '{doc_uid}'. Error: {original_embedding_error}"
        )
        mock_qdrant_client_instance.upsert.assert_not_called()

class TestVectorIndexGetDocumentById:
    @pytest.fixture # Ensure this fixture is defined if not already
    def vector_index_instance(self, mock_qdrant_client_class, mock_embedding_manager, 
                              mock_data_structure_checker, mock_config_manager_settings):
        return VectorIndex(
            embedding_manager=mock_embedding_manager,
            data_checker=mock_data_structure_checker
        )

    def test_get_document_by_id_found(self, vector_index_instance, mock_qdrant_client_instance, mock_log_wrapper):
        """
        Scenario 1: Found. `qdrant_client.retrieve` returns mock `PointRecord`s.
        Check `retrieve` args. Check formatted results (score 1.0).
        """
        doc_uid = "doc_to_get_uid"
        expected_point_id = VectorIndex._uid_to_point_id(doc_uid)
        
        # mock_qdrant_client_instance.retrieve is already configured in the main fixture
        # to return one PointRecord. Let's adjust its payload for clarity.
        mock_retrieved_point = mock_qdrant_client_instance.retrieve.return_value[0]
        mock_retrieved_point.id = expected_point_id # Ensure ID matches
        mock_retrieved_point.payload = {
            "uid": doc_uid, 
            "text": "Retrieved document text for get_document_by_id",
            "orig_sid": "orig_sid_get",
            "side": "get_side"
        }
        mock_retrieved_point.vector = [0.1,0.2] # Example vector

        results = vector_index_instance.get_document_by_id(doc_uid)
        
        mock_qdrant_client_instance.retrieve.assert_called_once_with(
            collection_name=vector_index_instance.collection_name,
            ids=[expected_point_id],
            with_payload=True,
            with_vectors=True
        )
        
        assert len(results) == 1
        assert results[0]['uid'] == doc_uid
        assert results[0]['text'] == "Retrieved document text for get_document_by_id"
        assert results[0]['score'] == 1.0 # Score should be 1.0 for direct retrieval
        assert results[0]['vector'] == [0.1,0.2]
        assert results[0]['orig_sid'] == "orig_sid_get"
        assert results[0]['side'] == "get_side"

    def test_get_document_by_id_not_found(self, vector_index_instance, mock_qdrant_client_instance, mock_log_wrapper):
        """
        Scenario 2: Not found. `qdrant_client.retrieve` returns `[]`. Result is `[]`.
        """
        doc_uid = "doc_not_found_uid"
        expected_point_id = VectorIndex._uid_to_point_id(doc_uid)
        
        mock_qdrant_client_instance.retrieve.return_value = [] # Simulate not found
        
        results = vector_index_instance.get_document_by_id(doc_uid)
        
        mock_qdrant_client_instance.retrieve.assert_called_once_with(
            collection_name=vector_index_instance.collection_name,
            ids=[expected_point_id],
            with_payload=True,
            with_vectors=True
        )
        assert results == []
        # No error log should be called for a simple "not found" case.
        # log_wrapper.info might be called if there was specific logging for not found, but not error.
        mock_log_wrapper.error.assert_not_called()


    def test_get_document_by_id_retrieve_fails(self, vector_index_instance, mock_qdrant_client_instance, mock_log_wrapper):
        """
        Scenario 3: Retrieve fails. `qdrant_client.retrieve` raises error.
        `log_wrapper.error` called, returns `[]`.
        """
        doc_uid = "doc_retrieve_fail_uid"
        expected_point_id = VectorIndex._uid_to_point_id(doc_uid)
        original_error_message = "Qdrant retrieve operation failed"
        
        mock_qdrant_client_instance.retrieve.side_effect = Exception(original_error_message)
        
        results = vector_index_instance.get_document_by_id(doc_uid)
        
        mock_qdrant_client_instance.retrieve.assert_called_once_with(
            collection_name=vector_index_instance.collection_name,
            ids=[expected_point_id],
            with_payload=True,
            with_vectors=True
        )
        assert results == []
        mock_log_wrapper.error.assert_called_once_with(
            "VectorIndex", "get_document_by_id",
            f"Error retrieving document with uid '{doc_uid}' (point_id '{expected_point_id}') from '{vector_index_instance.collection_name}'. Error: {original_error_message}"
        )

# End of test file.
