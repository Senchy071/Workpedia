"""Tests for cross-encoder reranker functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.reranker import CrossEncoderReranker, RerankResult, rerank_chunks


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test creating a RerankResult."""
        chunk = {"content": "Test content", "metadata": {"doc_id": "doc1"}}
        result = RerankResult(
            chunk=chunk,
            original_rank=5,
            original_score=0.75,
            rerank_score=0.92,
            new_rank=1,
        )

        assert result.chunk == chunk
        assert result.original_rank == 5
        assert result.original_score == 0.75
        assert result.rerank_score == 0.92
        assert result.new_rank == 1

    def test_rerank_result_rank_improvement(self):
        """Test that rank improvement is tracked correctly."""
        result = RerankResult(
            chunk={"content": "Test"},
            original_rank=10,
            original_score=0.5,
            rerank_score=0.95,
            new_rank=0,
        )

        # Chunk moved from rank 10 to rank 0 (big improvement)
        rank_improvement = result.original_rank - result.new_rank
        assert rank_improvement == 10


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked model."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()
        return reranker

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "chunk_id": "chunk1",
                "content": "Machine learning is a subset of AI.",
                "metadata": {"doc_id": "doc1", "page": 1},
                "similarity": 0.85,
            },
            {
                "chunk_id": "chunk2",
                "content": "Deep learning uses neural networks.",
                "metadata": {"doc_id": "doc1", "page": 2},
                "similarity": 0.82,
            },
            {
                "chunk_id": "chunk3",
                "content": "Natural language processing enables text understanding.",
                "metadata": {"doc_id": "doc1", "page": 3},
                "similarity": 0.78,
            },
            {
                "chunk_id": "chunk4",
                "content": "Computer vision processes images.",
                "metadata": {"doc_id": "doc1", "page": 4},
                "similarity": 0.75,
            },
            {
                "chunk_id": "chunk5",
                "content": "Reinforcement learning trains agents.",
                "metadata": {"doc_id": "doc1", "page": 5},
                "similarity": 0.72,
            },
        ]

    def test_init_default_values(self):
        """Test initialization with default values."""
        reranker = CrossEncoderReranker()

        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.enabled is True
        assert reranker.top_n == 20
        assert reranker.top_k == 5
        assert reranker._model is None  # Lazy loading

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        reranker = CrossEncoderReranker(
            model_name="custom-model",
            enabled=False,
            top_n=30,
            top_k=10,
        )

        assert reranker.model_name == "custom-model"
        assert reranker.enabled is False
        assert reranker.top_n == 30
        assert reranker.top_k == 10

    def test_rerank_empty_chunks(self, mock_reranker):
        """Test reranking empty chunk list."""
        results = mock_reranker.rerank("test query", [])
        assert results == []

    def test_rerank_disabled_passthrough(self, sample_chunks):
        """Test that disabled reranker passes through original order."""
        reranker = CrossEncoderReranker(enabled=False)

        results = reranker.rerank("What is machine learning?", sample_chunks, top_k=3)

        assert len(results) == 3
        # Original order should be preserved
        assert results[0].chunk["chunk_id"] == "chunk1"
        assert results[1].chunk["chunk_id"] == "chunk2"
        assert results[2].chunk["chunk_id"] == "chunk3"
        # Scores should use original similarity
        assert results[0].rerank_score == 0.85
        assert results[0].original_score == 0.85

    def test_rerank_changes_order(self, mock_reranker, sample_chunks):
        """Test that reranking changes chunk order based on scores."""
        # Mock scores that would reorder chunks
        # Original order: chunk1, chunk2, chunk3, chunk4, chunk5
        # New order should be: chunk3, chunk5, chunk1, chunk4, chunk2
        mock_reranker._model.predict.return_value = np.array([0.6, 0.4, 0.95, 0.5, 0.8])

        results = mock_reranker.rerank("NLP question", sample_chunks, top_k=5)

        assert len(results) == 5
        # Check reordering based on scores
        assert results[0].chunk["chunk_id"] == "chunk3"  # Highest score: 0.95
        assert results[1].chunk["chunk_id"] == "chunk5"  # Second: 0.8
        assert results[2].chunk["chunk_id"] == "chunk1"  # Third: 0.6
        assert results[3].chunk["chunk_id"] == "chunk4"  # Fourth: 0.5
        assert results[4].chunk["chunk_id"] == "chunk2"  # Fifth: 0.4

    def test_rerank_top_k_limit(self, mock_reranker, sample_chunks):
        """Test that top_k limits the number of results."""
        mock_reranker._model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        results = mock_reranker.rerank("test", sample_chunks, top_k=3)

        assert len(results) == 3

    def test_rerank_preserves_original_info(self, mock_reranker, sample_chunks):
        """Test that original rank and score are preserved."""
        mock_reranker._model.predict.return_value = np.array([0.5, 0.9, 0.7, 0.6, 0.8])

        results = mock_reranker.rerank("test", sample_chunks, top_k=5)

        # Find the chunk that was originally first but is now lower
        chunk1_result = next(r for r in results if r.chunk["chunk_id"] == "chunk1")

        assert chunk1_result.original_rank == 0  # Was first
        assert chunk1_result.original_score == 0.85  # Original similarity
        assert chunk1_result.rerank_score == 0.5  # New cross-encoder score

    def test_rerank_updates_new_rank(self, mock_reranker, sample_chunks):
        """Test that new_rank is correctly assigned after sorting."""
        mock_reranker._model.predict.return_value = np.array([0.5, 0.9, 0.7, 0.6, 0.8])

        results = mock_reranker.rerank("test", sample_chunks, top_k=5)

        # Check new ranks are assigned sequentially
        for i, result in enumerate(results):
            assert result.new_rank == i

    def test_rerank_with_scores(self, mock_reranker, sample_chunks):
        """Test rerank_with_scores returns chunks and scores separately."""
        mock_reranker._model.predict.return_value = np.array([0.6, 0.9, 0.7, 0.5, 0.8])

        chunks, scores = mock_reranker.rerank_with_scores("test", sample_chunks, top_k=3)

        assert len(chunks) == 3
        assert len(scores) == 3
        # Highest scores first
        assert scores[0] == 0.9
        assert scores[1] == 0.8
        assert scores[2] == 0.7
        # Corresponding chunks
        assert chunks[0]["chunk_id"] == "chunk2"
        assert chunks[1]["chunk_id"] == "chunk5"
        assert chunks[2]["chunk_id"] == "chunk3"

    def test_rerank_model_lazy_loading(self):
        """Test that model is loaded lazily on first use."""
        reranker = CrossEncoderReranker(enabled=True)

        assert reranker._model is None

        # Mock the CrossEncoder to avoid actual model loading
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            # Access model property triggers loading
            model = reranker.model

            assert model == mock_model
            mock_cross_encoder.assert_called_once()

    def test_get_retrieval_count_enabled(self):
        """Test get_retrieval_count when reranking is enabled."""
        reranker = CrossEncoderReranker(enabled=True, top_n=25, top_k=5)

        count = reranker.get_retrieval_count()

        assert count == 25  # Returns top_n when enabled

    def test_get_retrieval_count_disabled(self):
        """Test get_retrieval_count when reranking is disabled."""
        reranker = CrossEncoderReranker(enabled=False, top_n=25, top_k=5)

        count = reranker.get_retrieval_count()

        assert count == 5  # Returns top_k when disabled

    def test_cleanup(self):
        """Test cleanup releases model resources."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()

        reranker.cleanup()

        assert reranker._model is None

    def test_rerank_handles_missing_similarity(self, mock_reranker):
        """Test reranking handles chunks without similarity field."""
        chunks = [
            {"chunk_id": "chunk1", "content": "Content 1", "metadata": {}},
            {"chunk_id": "chunk2", "content": "Content 2", "metadata": {}},
        ]
        mock_reranker._model.predict.return_value = np.array([0.8, 0.6])

        results = mock_reranker.rerank("test", chunks)

        assert len(results) == 2
        assert results[0].original_score == 0.0  # Default when missing
        assert results[1].original_score == 0.0

    def test_rerank_with_empty_content(self, mock_reranker):
        """Test reranking handles chunks with empty content."""
        chunks = [
            {"chunk_id": "chunk1", "content": "", "metadata": {}},
            {"chunk_id": "chunk2", "content": "Valid content", "metadata": {}},
        ]
        mock_reranker._model.predict.return_value = np.array([0.3, 0.9])

        results = mock_reranker.rerank("test", chunks, top_k=2)

        assert len(results) == 2
        # Second chunk should rank higher
        assert results[0].chunk["chunk_id"] == "chunk2"


class TestRerankChunksFunction:
    """Tests for the rerank_chunks convenience function."""

    def test_rerank_chunks_basic(self):
        """Test basic rerank_chunks function."""
        chunks = [
            {"content": "Machine learning basics", "metadata": {}},
            {"content": "Deep learning advanced", "metadata": {}},
            {"content": "Neural network fundamentals", "metadata": {}},
        ]

        with patch("core.reranker.CrossEncoderReranker") as mock_class:
            mock_reranker = MagicMock()
            mock_reranker.rerank_with_scores.return_value = (
                [chunks[1], chunks[0], chunks[2]],  # Reordered
                [0.9, 0.7, 0.5],
            )
            mock_class.return_value = mock_reranker

            result = rerank_chunks("deep learning", chunks, top_k=3)

            assert len(result) == 3
            assert result[0]["content"] == "Deep learning advanced"

    def test_rerank_chunks_custom_model(self):
        """Test rerank_chunks with custom model."""
        chunks = [{"content": "Test", "metadata": {}}]

        with patch("core.reranker.CrossEncoderReranker") as mock_class:
            mock_reranker = MagicMock()
            mock_reranker.rerank_with_scores.return_value = (chunks, [0.8])
            mock_class.return_value = mock_reranker

            rerank_chunks("query", chunks, model_name="custom-model")

            mock_class.assert_called_once_with(model_name="custom-model", top_k=5)


class TestRerankerIntegration:
    """Integration tests for reranker with QueryEngine."""

    @pytest.fixture
    def mock_query_engine_deps(self):
        """Create mocked dependencies for QueryEngine."""
        with patch("storage.vector_store.VectorStore") as mock_vs, \
             patch("core.embedder.Embedder") as mock_emb, \
             patch("core.llm.OllamaClient") as mock_llm:

            mock_vs_instance = MagicMock()
            mock_vs_instance.query.return_value = {
                "ids": ["id1", "id2", "id3"],
                "documents": ["doc1", "doc2", "doc3"],
                "metadatas": [{"page": 1}, {"page": 2}, {"page": 3}],
                "distances": [0.1, 0.2, 0.3],
            }
            mock_vs_instance.list_documents.return_value = []
            mock_vs_instance.count = 0
            mock_vs.return_value = mock_vs_instance

            mock_emb_instance = MagicMock()
            mock_emb_instance.embed.return_value = np.zeros(768)
            mock_emb_instance.model_name = "test-model"
            mock_emb_instance.dimension = 768
            mock_emb.return_value = mock_emb_instance

            mock_llm_instance = MagicMock()
            mock_llm_instance.model = "mistral"
            mock_llm_instance.generate.return_value = "Test answer"
            mock_llm_instance.is_available.return_value = True
            mock_llm_instance.list_models.return_value = ["mistral"]
            mock_llm_instance._cache = None
            mock_llm.return_value = mock_llm_instance

            yield mock_vs_instance, mock_emb_instance, mock_llm_instance

    def test_query_engine_with_reranking_disabled(self, mock_query_engine_deps):
        """Test QueryEngine with reranking disabled."""
        from core.query_engine import QueryEngine

        mock_vs, mock_emb, mock_llm = mock_query_engine_deps

        engine = QueryEngine(
            vector_store=mock_vs,
            embedder=mock_emb,
            llm=mock_llm,
            enable_reranking=False,
            enable_hybrid_search=False,
            enable_confidence=False,
            auto_save_history=False,
        )

        assert engine.enable_reranking is False
        assert engine.reranker is None

    def test_query_engine_with_reranking_enabled(self, mock_query_engine_deps):
        """Test QueryEngine with reranking enabled."""
        from core.query_engine import QueryEngine

        mock_vs, mock_emb, mock_llm = mock_query_engine_deps

        with patch("core.reranker.CrossEncoderReranker") as mock_reranker_class:
            mock_reranker = MagicMock()
            mock_reranker.get_retrieval_count.return_value = 20
            mock_reranker.rerank_with_scores.return_value = (
                [{"content": "reranked", "metadata": {}}],
                [0.95],
            )
            mock_reranker_class.return_value = mock_reranker

            engine = QueryEngine(
                vector_store=mock_vs,
                embedder=mock_emb,
                llm=mock_llm,
                enable_reranking=True,
                enable_hybrid_search=False,
                enable_confidence=False,
                auto_save_history=False,
            )

            assert engine.enable_reranking is True
            assert engine.reranker is not None


class TestRerankerEdgeCases:
    """Edge case tests for reranker."""

    def test_rerank_single_chunk(self):
        """Test reranking with a single chunk."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.9])

        chunks = [{"content": "Single chunk", "metadata": {}, "similarity": 0.8}]
        results = reranker.rerank("query", chunks, top_k=5)

        assert len(results) == 1
        assert results[0].new_rank == 0

    def test_rerank_more_chunks_than_top_k(self):
        """Test reranking when chunks > top_k."""
        reranker = CrossEncoderReranker(enabled=True, top_k=2)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.5, 0.9, 0.3, 0.7, 0.1])

        chunks = [
            {"content": f"Chunk {i}", "metadata": {}, "similarity": 0.5}
            for i in range(5)
        ]

        results = reranker.rerank("query", chunks)

        assert len(results) == 2
        # Best two chunks
        assert results[0].rerank_score == 0.9
        assert results[1].rerank_score == 0.7

    def test_rerank_with_identical_scores(self):
        """Test reranking when all scores are identical."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.5, 0.5, 0.5])

        chunks = [
            {"content": f"Chunk {i}", "metadata": {}, "similarity": 0.5}
            for i in range(3)
        ]

        results = reranker.rerank("query", chunks, top_k=3)

        assert len(results) == 3
        # All scores should be 0.5
        for result in results:
            assert result.rerank_score == 0.5

    def test_rerank_with_negative_scores(self):
        """Test reranking handles negative cross-encoder scores."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()
        # Cross-encoders can return negative scores
        reranker._model.predict.return_value = np.array([-0.5, 0.3, -0.1])

        chunks = [
            {"content": f"Chunk {i}", "metadata": {}}
            for i in range(3)
        ]

        results = reranker.rerank("query", chunks, top_k=3)

        assert len(results) == 3
        # Highest score first
        assert results[0].rerank_score == 0.3
        assert results[1].rerank_score == -0.1
        assert results[2].rerank_score == -0.5

    def test_rerank_very_long_query(self):
        """Test reranking with very long query."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.8, 0.6])

        chunks = [
            {"content": "Short chunk", "metadata": {}},
            {"content": "Another chunk", "metadata": {}},
        ]

        long_query = "This is a very long query " * 100  # ~2500 chars

        results = reranker.rerank(long_query, chunks, top_k=2)

        assert len(results) == 2
        # Model should be called with the long query
        call_args = reranker._model.predict.call_args[0][0]
        assert call_args[0][0] == long_query

    def test_rerank_unicode_content(self):
        """Test reranking with unicode content."""
        reranker = CrossEncoderReranker(enabled=True)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.9, 0.7])

        chunks = [
            {"content": "日本語テスト", "metadata": {}},
            {"content": "Ελληνικά κείμενο", "metadata": {}},
        ]

        results = reranker.rerank("multilingual query", chunks, top_k=2)

        assert len(results) == 2
        assert results[0].chunk["content"] == "日本語テスト"
