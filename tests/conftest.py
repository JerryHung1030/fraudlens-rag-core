import os
import sys

# Ensure the src directory is on sys.path for tests
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Provide a simple stub for the tiktoken package used in token counting
import types
if 'tiktoken' not in sys.modules:
    tiktoken_stub = types.ModuleType('tiktoken')

    class _DummyEncoder:
        def encode(self, text):
            return list(text)

    def encoding_for_model(model):
        return _DummyEncoder()

    def get_encoding(name):
        return _DummyEncoder()

    tiktoken_stub.encoding_for_model = encoding_for_model
    tiktoken_stub.get_encoding = get_encoding
    sys.modules['tiktoken'] = tiktoken_stub

# Minimal stub for loguru.logger used in logging utilities
if 'loguru' not in sys.modules:
    loguru_stub = types.ModuleType('loguru')
    class _DummyLogger:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                pass
            return _noop
    loguru_stub.logger = _DummyLogger()
    sys.modules['loguru'] = loguru_stub

# Stubs for external libraries not installed in the test environment
if 'pydantic_settings' not in sys.modules:
    from pydantic import BaseSettings as _PydanticBaseSettings
    pyd_stub = types.ModuleType('pydantic_settings')

    class BaseSettings(_PydanticBaseSettings):
        class Config:
            extra = 'allow'

    pyd_stub.BaseSettings = BaseSettings
    sys.modules['pydantic_settings'] = pyd_stub

if 'langchain_openai' not in sys.modules:
    lc_stub = types.ModuleType('langchain_openai')
    class OpenAIEmbeddings:
        def __init__(self, *args, **kwargs):
            pass
        def embed_query(self, text):
            return [0.0]
    lc_stub.OpenAIEmbeddings = OpenAIEmbeddings
    sys.modules['langchain_openai'] = lc_stub

if 'qdrant_client' not in sys.modules:
    qdrant_stub = types.ModuleType('qdrant_client')
    models_stub = types.ModuleType('qdrant_client.models')

    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass

    class VectorParams:
        def __init__(self, *args, **kwargs):
            pass

    class PointStruct:
        def __init__(self, id=None, vector=None, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload

    class FieldCondition:
        def __init__(self, *args, **kwargs):
            pass

    class MatchValue:
        def __init__(self, *args, **kwargs):
            pass

    class Distance:
        COSINE = 'cosine'

    class Filter:
        def __init__(self, *args, **kwargs):
            pass

    models_stub.VectorParams = VectorParams
    models_stub.PointStruct = PointStruct
    models_stub.FieldCondition = FieldCondition
    models_stub.MatchValue = MatchValue
    models_stub.Distance = Distance
    models_stub.Filter = Filter

    qdrant_stub.QdrantClient = QdrantClient
    qdrant_stub.models = models_stub
    sys.modules['qdrant_client'] = qdrant_stub
    sys.modules['qdrant_client.models'] = models_stub

if 'yaml' not in sys.modules:
    yaml_stub = types.ModuleType('yaml')
    def safe_load(*args, **kwargs):
        return {}
    def dump(*args, **kwargs):
        return ''
    yaml_stub.safe_load = safe_load
    yaml_stub.dump = dump
    sys.modules['yaml'] = yaml_stub

# ---------------------------------------------------------------------------
# Fallback implementation of the 'mocker' fixture normally provided by
# the pytest-mock plugin. This minimal version supports the patch helpers
# commonly used throughout the tests.

import pytest
from unittest.mock import patch, MagicMock


class _SimpleMocker:
    """Minimal mocker with patch helpers."""

    def __init__(self):
        self._patches = []

        class _PatchProxy:
            def __init__(self, outer):
                self._outer = outer

            def __call__(self, target, *args, **kwargs):
                p = patch(target, *args, **kwargs)
                obj = p.start()
                self._outer._patches.append(p)
                return obj

            def object(self, target, attribute, *args, **kwargs):
                p = patch.object(target, attribute, *args, **kwargs)
                obj = p.start()
                self._outer._patches.append(p)
                return obj

            def dict(self, in_dict, values, **kwargs):
                p = patch.dict(in_dict, values, **kwargs)
                obj = p.start()
                self._outer._patches.append(p)
                return obj

        self.patch = _PatchProxy(self)
        self.Mock = MagicMock
        self.MagicMock = MagicMock

    def stopall(self):
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()


@pytest.fixture
def mocker():
    sm = _SimpleMocker()
    yield sm
    sm.stopall()
