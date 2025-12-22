"""Tests for Phase 4: Query Interface."""

import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.llm import RAG_SYSTEM_PROMPT, OllamaClient, format_rag_prompt
from core.query_engine import QueryEngine, QueryResult

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "content": "Python is a high-level programming language known for simplicity.",
            "metadata": {"filename": "python_guide.pdf", "section": "Introduction"},
            "similarity": 0.92,
        },
        {
            "content": "Python supports multiple programming paradigms including OOP.",
            "metadata": {"filename": "python_guide.pdf", "section": "Features"},
            "similarity": 0.88,
        },
        {
            "content": "Python was created by Guido van Rossum in 1991.",
            "metadata": {"filename": "python_history.pdf", "section": "Origins"},
            "similarity": 0.75,
        },
    ]


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama response."""
    return {
        "response": "Python is a versatile programming language created by Guido van Rossum.",
        "done": True,
    }


# =============================================================================
# OllamaClient Tests
# =============================================================================


class TestOllamaClient:
    """Tests for OllamaClient class."""

    def test_client_initialization(self):
        """Test client initializes with correct defaults."""
        client = OllamaClient()
        assert client.model == "mistral"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 120

    def test_client_custom_settings(self):
        """Test client with custom settings."""
        client = OllamaClient(
            model="llama2",
            base_url="http://custom:8080",
            timeout=60,
        )
        assert client.model == "llama2"
        assert client.base_url == "http://custom:8080"
        assert client.timeout == 60

    @patch("requests.get")
    def test_is_available_true(self, mock_get):
        """Test is_available when server responds."""
        mock_get.return_value.status_code = 200
        client = OllamaClient()
        assert client.is_available() is True

    @patch("core.llm.requests.get")
    def test_is_available_false(self, mock_get):
        """Test is_available when server doesn't respond."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection refused")
        client = OllamaClient()
        assert client.is_available() is False

    @patch("requests.get")
    def test_list_models(self, mock_get):
        """Test listing available models."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "mistral"},
                {"name": "llama2"},
            ]
        }
        client = OllamaClient()
        models = client.list_models()
        assert "mistral" in models
        assert "llama2" in models

    @patch("requests.post")
    def test_generate_sync(self, mock_post, mock_ollama_response):
        """Test synchronous generation."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_ollama_response

        client = OllamaClient()
        response = client.generate("What is Python?", stream=False)

        assert "Python" in response
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_generate_with_system_prompt(self, mock_post, mock_ollama_response):
        """Test generation with system prompt."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_ollama_response

        client = OllamaClient()
        client.generate(
            "What is Python?",
            system="You are a programming expert.",
            stream=False,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["system"] == "You are a programming expert."

    @patch("requests.post")
    def test_chat_sync(self, mock_post):
        """Test synchronous chat completion."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "Python is great for beginners."},
            "done": True,
        }

        client = OllamaClient()
        messages = [{"role": "user", "content": "Tell me about Python."}]
        response = client.chat(messages, stream=False)

        assert "Python" in response


# =============================================================================
# RAG Prompt Tests
# =============================================================================


class TestRAGPrompts:
    """Tests for RAG prompt formatting."""

    def test_rag_system_prompt_exists(self):
        """Test RAG system prompt is defined."""
        assert RAG_SYSTEM_PROMPT is not None
        assert len(RAG_SYSTEM_PROMPT) > 0
        assert "context" in RAG_SYSTEM_PROMPT.lower()

    def test_format_rag_prompt_basic(self, sample_chunks):
        """Test basic RAG prompt formatting."""
        prompt = format_rag_prompt(
            question="What is Python?",
            context_chunks=sample_chunks,
        )

        assert "What is Python?" in prompt
        assert "Context:" in prompt
        assert "python_guide.pdf" in prompt

    def test_format_rag_prompt_includes_sections(self, sample_chunks):
        """Test that sections are included in prompt."""
        prompt = format_rag_prompt(
            question="What is Python?",
            context_chunks=sample_chunks,
        )

        assert "Introduction" in prompt
        assert "Features" in prompt

    def test_format_rag_prompt_max_context(self, sample_chunks):
        """Test context truncation with max_context_chars."""
        prompt = format_rag_prompt(
            question="What is Python?",
            context_chunks=sample_chunks,
            max_context_chars=100,
        )

        # Should be truncated
        assert len(prompt) < 500

    def test_format_rag_prompt_empty_chunks(self):
        """Test formatting with empty chunks."""
        prompt = format_rag_prompt(
            question="What is Python?",
            context_chunks=[],
        )

        assert "What is Python?" in prompt
        # Context section should be empty
        assert "Context:\n\n" in prompt


# =============================================================================
# QueryResult Tests
# =============================================================================


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self, sample_chunks):
        """Test QueryResult creation."""
        result = QueryResult(
            question="What is Python?",
            answer="Python is a programming language.",
            sources=sample_chunks,
            metadata={"model": "mistral"},
        )

        assert result.question == "What is Python?"
        assert result.answer == "Python is a programming language."
        assert len(result.sources) == 3

    def test_query_result_to_dict(self, sample_chunks):
        """Test QueryResult.to_dict() method."""
        result = QueryResult(
            question="What is Python?",
            answer="Python is a programming language.",
            sources=sample_chunks,
            metadata={"model": "mistral"},
        )

        result_dict = result.to_dict()
        assert "question" in result_dict
        assert "answer" in result_dict
        assert "sources" in result_dict
        assert "metadata" in result_dict

    def test_query_result_defaults(self):
        """Test QueryResult with defaults."""
        result = QueryResult(
            question="Test?",
            answer="Answer.",
        )

        assert result.sources == []
        assert result.metadata == {}


# =============================================================================
# QueryEngine Tests
# =============================================================================


class TestQueryEngine:
    """Tests for QueryEngine class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = Mock()
        mock.query.return_value = {
            "ids": ["chunk1", "chunk2"],
            "documents": ["Content 1", "Content 2"],
            "metadatas": [
                {"filename": "doc1.pdf", "section": "Intro"},
                {"filename": "doc1.pdf", "section": "Body"},
            ],
            "distances": [0.1, 0.2],
        }
        mock.list_documents.return_value = [
            {"doc_id": "doc1", "filename": "doc1.pdf", "chunk_count": 2}
        ]
        mock.count = 2
        return mock

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        mock = Mock()
        mock.embed.return_value = np.random.rand(768)
        mock.model_name = "test-model"
        mock.dimension = 768
        return mock

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = Mock()
        mock.generate.return_value = "This is a generated answer about the content."
        mock.is_available.return_value = True
        mock.list_models.return_value = ["mistral"]
        mock.model = "mistral"
        mock._cache = None  # Disable LLM caching in tests
        return mock

    def test_query_engine_initialization(self, mock_vector_store, mock_embedder, mock_llm):
        """Test QueryEngine initialization."""
        engine = QueryEngine(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm=mock_llm,
        )

        assert engine.vector_store is mock_vector_store
        assert engine.embedder is mock_embedder
        assert engine.llm is mock_llm
        assert engine.n_results == 5
        assert engine.temperature == 0.7

    def test_query_returns_result(self, mock_vector_store, mock_embedder, mock_llm):
        """Test query returns QueryResult."""
        engine = QueryEngine(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        result = engine.query("What is the content about?")

        assert isinstance(result, QueryResult)
        assert result.question == "What is the content about?"
        assert len(result.answer) > 0
        assert len(result.sources) > 0

    def test_query_with_doc_filter(self, mock_vector_store, mock_embedder, mock_llm):
        """Test query with document filter."""
        engine = QueryEngine(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        engine.query("Question?", doc_id="specific-doc")

        # Check that where clause was passed
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["where"] == {"doc_id": "specific-doc"}

    def test_query_no_results(self, mock_embedder, mock_llm):
        """Test query when no chunks found."""
        mock_store = Mock()
        mock_store.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        # Mock the _collection.get for special query detection (summary/TOC)
        mock_store._collection = Mock()
        mock_store._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        engine = QueryEngine(
            vector_store=mock_store,
            embedder=mock_embedder,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        result = engine.query("Unknown question?")

        assert "couldn't find" in result.answer.lower()
        assert len(result.sources) == 0

    def test_get_similar_chunks(self, mock_vector_store, mock_embedder, mock_llm):
        """Test get_similar_chunks method."""
        # Mock the _collection.get for special query detection (summary/TOC)
        mock_vector_store._collection = Mock()
        mock_vector_store._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        engine = QueryEngine(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        chunks = engine.get_similar_chunks("Test query", n_results=3)

        assert len(chunks) == 2  # Mock returns 2
        assert "chunk_id" in chunks[0]
        assert "content" in chunks[0]
        assert "similarity" in chunks[0]

    def test_health_check(self, mock_vector_store, mock_embedder, mock_llm):
        """Test health_check method."""
        engine = QueryEngine(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        health = engine.health_check()

        assert "vector_store" in health
        assert "llm" in health
        assert "embedder" in health
        assert health["llm"]["status"] == "ok"


# =============================================================================
# API Tests
# =============================================================================


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self, mock_vector_store, mock_embedder, mock_llm):
        """Create test client with mocked dependencies."""
        from fastapi.testclient import TestClient

        import api.endpoints as endpoints
        from api.endpoints import app

        # Mock the _collection.get for special query detection (summary/TOC)
        mock_vector_store._collection = Mock()
        mock_vector_store._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        # Inject mocks
        endpoints.query_engine = QueryEngine(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        return TestClient(app)

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store for API tests."""
        mock = Mock()
        mock.query.return_value = {
            "ids": ["chunk1"],
            "documents": ["Test content"],
            "metadatas": [{"filename": "test.pdf", "doc_id": "doc1"}],
            "distances": [0.1],
        }
        mock.list_documents.return_value = [
            {
                "doc_id": "doc1",
                "filename": "test.pdf",
                "file_path": "/tmp/test.pdf",
                "chunk_count": 1,
            }
        ]
        mock.count = 1
        mock.stats.return_value = {
            "collection_name": "test",
            "total_chunks": 1,
            "documents": 1,
        }
        mock.get_by_doc_id.return_value = {
            "ids": ["chunk1"],
            "metadatas": [{"filename": "test.pdf"}],
        }
        return mock

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for API tests."""
        mock = Mock()
        mock.embed.return_value = np.random.rand(768)
        mock.model_name = "test-model"
        mock.dimension = 768
        return mock

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for API tests."""
        mock = Mock()
        mock.generate.return_value = "Generated answer"
        mock.is_available.return_value = True
        mock.list_models.return_value = ["mistral"]
        mock.model = "mistral"
        mock._cache = None  # Disable LLM caching in tests
        return mock

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "Workpedia" in data["name"]

    def test_query_endpoint(self, client):
        """Test query endpoint."""
        response = client.post(
            "/query",
            json={"question": "What is the test about?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data

    def test_search_endpoint(self, client):
        """Test search endpoint."""
        response = client.post(
            "/search",
            json={"query": "test search"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_documents_endpoint(self, client):
        """Test list documents endpoint."""
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "vector_store" in data


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase4Integration:
    """Integration tests for the complete Phase 4 pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_format_rag_prompt_full_workflow(self):
        """Test complete prompt formatting workflow."""
        # Simulate chunks from retrieval
        chunks = [
            {
                "content": "Machine learning is a subset of AI that enables computers to learn.",
                "metadata": {"filename": "ml_intro.pdf", "section": "Introduction"},
            },
            {
                "content": "Deep learning uses neural networks with multiple layers.",
                "metadata": {"filename": "ml_intro.pdf", "section": "Deep Learning"},
            },
        ]

        prompt = format_rag_prompt(
            question="What is machine learning?",
            context_chunks=chunks,
        )

        # Verify prompt structure
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "What is machine learning?" in prompt
        assert "Machine learning" in prompt
        assert "ml_intro.pdf" in prompt

    def test_query_result_serialization(self):
        """Test QueryResult can be serialized for API response."""
        result = QueryResult(
            question="Test question?",
            answer="Test answer.",
            sources=[
                {
                    "content": "Source content",
                    "metadata": {"filename": "test.pdf"},
                    "similarity": 0.9,
                }
            ],
            metadata={"model": "mistral", "chunks_retrieved": 1},
        )

        # Convert to dict (for JSON serialization)
        data = result.to_dict()

        # Should be JSON serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0

        # Verify structure
        parsed = json.loads(json_str)
        assert parsed["question"] == "Test question?"
        assert parsed["answer"] == "Test answer."

    @patch("requests.get")
    @patch("requests.post")
    def test_ollama_client_integration(self, mock_post, mock_get):
        """Test OllamaClient with mocked HTTP."""
        # Setup mocks
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "mistral"}]}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "Test response from Ollama",
            "done": True,
        }

        client = OllamaClient()

        # Test availability
        assert client.is_available() is True

        # Test model listing
        models = client.list_models()
        assert "mistral" in models

        # Test generation
        response = client.generate("Hello", stream=False)
        assert "Test response" in response
