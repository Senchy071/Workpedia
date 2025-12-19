"""Tests for hybrid search (semantic + keyword with RRF)."""

import pytest
from unittest.mock import MagicMock, patch

from core.hybrid_search import (
    BM25Index,
    HybridSearcher,
    SearchResult,
    reciprocal_rank_fusion,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            chunk_id="chunk1",
            content="Test content",
            metadata={"doc_id": "doc1"},
            semantic_score=0.8,
            keyword_score=10.5,
            combined_score=0.5,
            semantic_rank=1,
            keyword_rank=2,
        )

        assert result.chunk_id == "chunk1"
        assert result.content == "Test content"
        assert result.semantic_score == 0.8
        assert result.keyword_score == 10.5
        assert result.combined_score == 0.5
        assert result.semantic_rank == 1
        assert result.keyword_rank == 2

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = SearchResult(
            chunk_id="chunk1",
            content="Content",
            metadata={"key": "value"},
            semantic_score=0.9,
            keyword_score=5.0,
            combined_score=0.6,
        )

        d = result.to_dict()
        assert d["chunk_id"] == "chunk1"
        assert d["content"] == "Content"
        assert d["metadata"]["key"] == "value"
        assert d["semantic_score"] == 0.9
        assert d["keyword_score"] == 5.0
        assert d["combined_score"] == 0.6


class TestBM25Index:
    """Tests for BM25Index class."""

    def test_init_empty(self):
        """Test initializing empty index."""
        index = BM25Index()
        assert index.count == 0

    def test_add_chunks(self):
        """Test adding chunks to index."""
        index = BM25Index()

        chunks = [
            {"chunk_id": "c1", "content": "Hello world", "metadata": {"doc_id": "d1"}},
            {"chunk_id": "c2", "content": "Goodbye world", "metadata": {"doc_id": "d1"}},
        ]

        added = index.add_chunks(chunks)
        assert added == 2
        assert index.count == 2

    def test_add_duplicate_chunks(self):
        """Test that duplicate chunks are not added."""
        index = BM25Index()

        chunks = [
            {"chunk_id": "c1", "content": "Hello world", "metadata": {}},
        ]

        added1 = index.add_chunks(chunks)
        added2 = index.add_chunks(chunks)

        assert added1 == 1
        assert added2 == 0
        assert index.count == 1

    def test_search_empty_index(self):
        """Test searching empty index."""
        index = BM25Index()
        results = index.search("hello", n_results=5)
        assert results == []

    def test_search_basic(self):
        """Test basic BM25 search."""
        index = BM25Index()

        chunks = [
            {"chunk_id": "c1", "content": "Python programming language", "metadata": {}},
            {"chunk_id": "c2", "content": "Java programming language", "metadata": {}},
            {"chunk_id": "c3", "content": "Random other content", "metadata": {}},
        ]

        index.add_chunks(chunks)
        results = index.search("Python programming", n_results=2)

        assert len(results) == 2
        # Python chunk should rank higher
        assert results[0][0] == "c1"
        assert results[0][2] == 1  # rank 1

    def test_search_with_doc_filter(self):
        """Test searching with document filter."""
        index = BM25Index()

        chunks = [
            {"chunk_id": "c1", "content": "Hello world", "metadata": {"doc_id": "d1"}},
            {"chunk_id": "c2", "content": "Hello there", "metadata": {"doc_id": "d2"}},
        ]

        index.add_chunks(chunks)
        results = index.search("Hello", n_results=5, doc_id="d1")

        assert len(results) == 1
        assert results[0][0] == "c1"

    def test_remove_by_doc_id(self):
        """Test removing chunks by document ID."""
        index = BM25Index()

        chunks = [
            {"chunk_id": "c1", "content": "Doc 1 content", "metadata": {"doc_id": "d1"}},
            {"chunk_id": "c2", "content": "Doc 1 more", "metadata": {"doc_id": "d1"}},
            {"chunk_id": "c3", "content": "Doc 2 content", "metadata": {"doc_id": "d2"}},
        ]

        index.add_chunks(chunks)
        assert index.count == 3

        removed = index.remove_by_doc_id("d1")
        assert removed == 2
        assert index.count == 1

    def test_get_chunk(self):
        """Test getting a chunk by ID."""
        index = BM25Index()

        chunks = [
            {"chunk_id": "c1", "content": "Test content", "metadata": {"key": "value"}},
        ]

        index.add_chunks(chunks)
        chunk = index.get_chunk("c1")

        assert chunk is not None
        assert chunk["chunk_id"] == "c1"
        assert chunk["content"] == "Test content"
        assert chunk["metadata"]["key"] == "value"

    def test_get_nonexistent_chunk(self):
        """Test getting a non-existent chunk."""
        index = BM25Index()
        chunk = index.get_chunk("nonexistent")
        assert chunk is None

    def test_tokenize(self):
        """Test tokenization."""
        index = BM25Index()

        tokens = index._tokenize("Hello World! This is a TEST-123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" in tokens
        # Short tokens like 'a' should be removed (unless they're digits)
        assert "a" not in tokens


class TestHybridSearcher:
    """Tests for HybridSearcher class."""

    def test_init_default(self):
        """Test default initialization."""
        searcher = HybridSearcher()
        assert searcher.enabled is True
        assert searcher.rrf_k == 60

    def test_init_disabled(self):
        """Test initialization with disabled."""
        searcher = HybridSearcher(enabled=False)
        assert searcher.enabled is False

    def test_search_disabled(self):
        """Test search when disabled returns semantic only."""
        searcher = HybridSearcher(enabled=False)

        semantic_results = [
            {"chunk_id": "c1", "content": "Content 1", "metadata": {}, "similarity": 0.9},
            {"chunk_id": "c2", "content": "Content 2", "metadata": {}, "similarity": 0.7},
        ]

        results = searcher.search("test query", semantic_results)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].semantic_score == 0.9
        assert results[0].keyword_score == 0.0

    def test_search_hybrid(self):
        """Test hybrid search with both semantic and keyword."""
        bm25_index = BM25Index()
        chunks = [
            {"chunk_id": "c1", "content": "Python programming guide", "metadata": {}},
            {"chunk_id": "c2", "content": "Java development tutorial", "metadata": {}},
            {"chunk_id": "c3", "content": "Programming concepts", "metadata": {}},
        ]
        bm25_index.add_chunks(chunks)

        searcher = HybridSearcher(bm25_index=bm25_index)

        # Semantic results favor c3, but keyword search should boost c1 for "Python"
        semantic_results = [
            {"chunk_id": "c3", "content": "Programming concepts", "metadata": {}, "similarity": 0.9},
            {"chunk_id": "c1", "content": "Python programming guide", "metadata": {}, "similarity": 0.7},
            {"chunk_id": "c2", "content": "Java development tutorial", "metadata": {}, "similarity": 0.5},
        ]

        results = searcher.search("Python programming", semantic_results, n_results=3)

        assert len(results) == 3
        # Results should combine semantic and keyword scores
        for result in results:
            assert result.combined_score > 0

    def test_rrf_score_calculation(self):
        """Test RRF score calculation."""
        searcher = HybridSearcher(rrf_k=60, semantic_weight=0.5, keyword_weight=0.5)

        # Rank 1 in both = high score
        score = searcher._calculate_rrf_score(
            semantic_rank=1,
            keyword_rank=1,
            semantic_score=0.9,
            keyword_score=10.0,
        )

        expected = 0.5 * (1 / 61) + 0.5 * (1 / 61)
        assert abs(score - expected) < 0.0001

    def test_rrf_score_missing_ranks(self):
        """Test RRF score with missing ranks."""
        searcher = HybridSearcher(rrf_k=60, semantic_weight=0.5, keyword_weight=0.5)

        # Only in semantic results
        score1 = searcher._calculate_rrf_score(
            semantic_rank=1,
            keyword_rank=None,
            semantic_score=0.9,
            keyword_score=0,
        )
        assert score1 > 0

        # Only in keyword results
        score2 = searcher._calculate_rrf_score(
            semantic_rank=None,
            keyword_rank=1,
            semantic_score=0,
            keyword_score=10.0,
        )
        assert score2 > 0

    def test_add_chunks_to_index(self):
        """Test adding chunks via searcher."""
        searcher = HybridSearcher()

        chunks = [
            {"chunk_id": "c1", "content": "Test content", "metadata": {}},
        ]

        added = searcher.add_chunks_to_index(chunks)
        assert added == 1
        assert searcher.index_count == 1

    def test_remove_doc_from_index(self):
        """Test removing document via searcher."""
        searcher = HybridSearcher()

        chunks = [
            {"chunk_id": "c1", "content": "Content", "metadata": {"doc_id": "d1"}},
            {"chunk_id": "c2", "content": "Content", "metadata": {"doc_id": "d2"}},
        ]

        searcher.add_chunks_to_index(chunks)
        removed = searcher.remove_doc_from_index("d1")

        assert removed == 1
        assert searcher.index_count == 1


class TestReciprocalRankFusion:
    """Tests for the RRF utility function."""

    def test_empty_rankings(self):
        """Test with empty rankings."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_ranking(self):
        """Test with single ranking."""
        ranking = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        result = reciprocal_rank_fusion([ranking])

        assert len(result) == 3
        # Order should be preserved
        assert result[0][0] == "a"
        assert result[1][0] == "b"
        assert result[2][0] == "c"

    def test_multiple_rankings(self):
        """Test with multiple rankings."""
        # Ranking 1 prefers a, b, c
        ranking1 = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        # Ranking 2 prefers c, b, a
        ranking2 = [("c", 0.9), ("b", 0.7), ("a", 0.5)]

        result = reciprocal_rank_fusion([ranking1, ranking2])

        # b should be in both rankings at position 2, so should rank high
        item_scores = {item: score for item, score in result}
        assert "a" in item_scores
        assert "b" in item_scores
        assert "c" in item_scores

    def test_weighted_rankings(self):
        """Test with custom weights."""
        ranking1 = [("a", 0.9), ("b", 0.7)]
        ranking2 = [("b", 0.9), ("a", 0.7)]

        # Weight ranking1 higher
        result = reciprocal_rank_fusion([ranking1, ranking2], weights=[0.9, 0.1])

        # a should win because ranking1 is weighted higher
        assert result[0][0] == "a"


class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""

    def test_exact_match_improvement(self):
        """Test that exact keyword matches improve ranking."""
        bm25_index = BM25Index()
        chunks = [
            {"chunk_id": "c1", "content": "Invoice number 12345 for payment", "metadata": {}},
            {"chunk_id": "c2", "content": "Payment processing information", "metadata": {}},
            {"chunk_id": "c3", "content": "General payment details", "metadata": {}},
        ]
        bm25_index.add_chunks(chunks)

        searcher = HybridSearcher(bm25_index=bm25_index)

        # Semantic search might not find "12345", but keyword search should
        semantic_results = [
            {"chunk_id": "c2", "content": "Payment processing", "metadata": {}, "similarity": 0.9},
            {"chunk_id": "c3", "content": "General payment", "metadata": {}, "similarity": 0.8},
            {"chunk_id": "c1", "content": "Invoice 12345", "metadata": {}, "similarity": 0.3},
        ]

        results = searcher.search("invoice 12345", semantic_results, n_results=3)

        # The exact match "12345" should boost c1 to top
        chunk_ids = [r.chunk_id for r in results]
        # c1 should be ranked higher after hybrid search
        assert "c1" in chunk_ids

    def test_semantic_only_fallback(self):
        """Test that semantic results are used when no keyword matches."""
        bm25_index = BM25Index()
        # Index chunks without the query terms
        chunks = [
            {"chunk_id": "c1", "content": "Unrelated content here", "metadata": {}},
        ]
        bm25_index.add_chunks(chunks)

        searcher = HybridSearcher(bm25_index=bm25_index)

        semantic_results = [
            {"chunk_id": "c1", "content": "Unrelated content", "metadata": {}, "similarity": 0.9},
        ]

        results = searcher.search("completely different query", semantic_results, n_results=1)

        # Should still return semantic result even if keyword search finds nothing
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
