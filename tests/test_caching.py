"""Tests for performance caching (embeddings and LLM responses)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.caching import (
    EmbeddingCache,
    LLMCache,
    clear_all_caches,
    get_cache_stats,
)


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, cache_dir):
        """Create EmbeddingCache instance."""
        return EmbeddingCache(cache_dir=cache_dir, ttl=3600)

    def test_init_enabled(self, cache_dir):
        """Test cache initialization when enabled."""
        cache = EmbeddingCache(cache_dir=cache_dir, enabled=True)
        assert cache.enabled is True
        assert cache._cache is not None
        assert cache.ttl == 3600

    def test_init_disabled(self, cache_dir):
        """Test cache initialization when disabled."""
        cache = EmbeddingCache(cache_dir=cache_dir, enabled=False)
        assert cache.enabled is False
        assert cache._cache is None

    def test_normalize_query(self, cache):
        """Test query normalization."""
        assert cache._normalize_query("  Hello World  ") == "hello world"
        assert cache._normalize_query("UPPERCASE") == "uppercase"
        assert cache._normalize_query("Mixed Case") == "mixed case"

    def test_generate_key(self, cache):
        """Test cache key generation."""
        key1 = cache._generate_key("test query")
        key2 = cache._generate_key("test query")
        key3 = cache._generate_key("different query")

        # Same query should produce same key
        assert key1 == key2
        # Different query should produce different key
        assert key1 != key3
        # Keys should be SHA-256 hashes (64 hex chars)
        assert len(key1) == 64

    def test_cache_hit(self, cache):
        """Test cache hit."""
        embedding = np.array([0.1, 0.2, 0.3])

        # First call - cache miss
        result = cache.get("test query")
        assert result is None

        # Set value
        cache.set("test query", embedding)

        # Second call - cache hit
        result = cache.get("test query")
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, embedding)

    def test_cache_miss(self, cache):
        """Test cache miss."""
        result = cache.get("nonexistent query")
        assert result is None

    def test_cache_normalization(self, cache):
        """Test that normalized queries hit the same cache entry."""
        embedding = np.array([0.5, 0.6, 0.7])

        cache.set("Test Query", embedding)

        # Different casing/spacing should hit same cache entry
        result1 = cache.get("test query")
        result2 = cache.get("  TEST QUERY  ")
        result3 = cache.get("Test Query")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        np.testing.assert_array_equal(result1, embedding)
        np.testing.assert_array_equal(result2, embedding)
        np.testing.assert_array_equal(result3, embedding)

    def test_clear_cache(self, cache):
        """Test cache clearing."""
        cache.set("query1", np.array([1.0, 2.0]))
        cache.set("query2", np.array([3.0, 4.0]))

        # Verify cached
        assert cache.get("query1") is not None
        assert cache.get("query2") is not None

        # Clear cache
        cache.clear()

        # Verify cleared
        assert cache.get("query1") is None
        assert cache.get("query2") is None

    def test_stats(self, cache):
        """Test cache statistics."""
        stats = cache.stats()
        assert stats["enabled"] is True
        assert stats["size"] == 0

        # Add items
        cache.set("query1", np.array([1.0]))
        cache.set("query2", np.array([2.0]))

        stats = cache.stats()
        assert stats["size"] == 2
        assert "volume" in stats
        assert "ttl" in stats

    def test_disabled_cache_operations(self, cache_dir):
        """Test operations on disabled cache."""
        cache = EmbeddingCache(cache_dir=cache_dir, enabled=False)

        # get() should return None
        assert cache.get("query") is None

        # set() should not raise error
        cache.set("query", np.array([1.0]))

        # stats() should show disabled
        stats = cache.stats()
        assert stats["enabled"] is False
        assert stats["size"] == 0


class TestLLMCache:
    """Tests for LLMCache class."""

    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, cache_dir):
        """Create LLMCache instance."""
        return LLMCache(cache_dir=cache_dir, ttl=3600)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample context chunks."""
        return [
            {"content": "Chunk 1 content", "doc_id": "doc1"},
            {"content": "Chunk 2 content", "doc_id": "doc1"},
        ]

    def test_init_enabled(self, cache_dir):
        """Test cache initialization when enabled."""
        cache = LLMCache(cache_dir=cache_dir, enabled=True)
        assert cache.enabled is True
        assert cache._cache is not None
        assert cache.ttl == 3600

    def test_init_disabled(self, cache_dir):
        """Test cache initialization when disabled."""
        cache = LLMCache(cache_dir=cache_dir, enabled=False)
        assert cache.enabled is False
        assert cache._cache is None

    def test_generate_key(self, cache, sample_chunks):
        """Test cache key generation."""
        key1 = cache._generate_key("test query", sample_chunks, 0.7, None)
        key2 = cache._generate_key("test query", sample_chunks, 0.7, None)
        key3 = cache._generate_key("different query", sample_chunks, 0.7, None)

        # Same inputs should produce same key
        assert key1 == key2
        # Different query should produce different key
        assert key1 != key3
        # Keys should be SHA-256 hashes (64 hex chars)
        assert len(key1) == 64

    def test_key_includes_temperature(self, cache, sample_chunks):
        """Test that temperature affects cache key."""
        key1 = cache._generate_key("query", sample_chunks, 0.5, None)
        key2 = cache._generate_key("query", sample_chunks, 0.7, None)

        assert key1 != key2

    def test_key_includes_max_tokens(self, cache, sample_chunks):
        """Test that max_tokens affects cache key."""
        key1 = cache._generate_key("query", sample_chunks, 0.7, 100)
        key2 = cache._generate_key("query", sample_chunks, 0.7, 200)

        assert key1 != key2

    def test_key_includes_context(self, cache):
        """Test that context chunks affect cache key."""
        chunks1 = [{"content": "Content A", "doc_id": "doc1"}]
        chunks2 = [{"content": "Content B", "doc_id": "doc1"}]

        key1 = cache._generate_key("query", chunks1, 0.7, None)
        key2 = cache._generate_key("query", chunks2, 0.7, None)

        assert key1 != key2

    def test_cache_hit(self, cache, sample_chunks):
        """Test cache hit."""
        response = "This is the answer."

        # First call - cache miss
        result = cache.get("query", sample_chunks, 0.7, None)
        assert result is None

        # Set value
        cache.set("query", sample_chunks, response, 0.7, None)

        # Second call - cache hit
        result = cache.get("query", sample_chunks, 0.7, None)
        assert result == response

    def test_cache_miss(self, cache, sample_chunks):
        """Test cache miss."""
        result = cache.get("nonexistent query", sample_chunks, 0.7, None)
        assert result is None

    def test_query_normalization(self, cache, sample_chunks):
        """Test that normalized queries hit the same cache entry."""
        response = "Answer"

        cache.set("Test Query", sample_chunks, response, 0.7, None)

        # Different casing/spacing should hit same cache entry
        result1 = cache.get("test query", sample_chunks, 0.7, None)
        result2 = cache.get("  TEST QUERY  ", sample_chunks, 0.7, None)

        assert result1 == response
        assert result2 == response

    def test_clear_cache(self, cache, sample_chunks):
        """Test cache clearing."""
        cache.set("query1", sample_chunks, "answer1", 0.7, None)
        cache.set("query2", sample_chunks, "answer2", 0.7, None)

        # Verify cached
        assert cache.get("query1", sample_chunks, 0.7, None) is not None
        assert cache.get("query2", sample_chunks, 0.7, None) is not None

        # Clear cache
        cache.clear()

        # Verify cleared
        assert cache.get("query1", sample_chunks, 0.7, None) is None
        assert cache.get("query2", sample_chunks, 0.7, None) is None

    def test_stats(self, cache, sample_chunks):
        """Test cache statistics."""
        stats = cache.stats()
        assert stats["enabled"] is True
        assert stats["size"] == 0

        # Add items
        cache.set("query1", sample_chunks, "answer1", 0.7, None)
        cache.set("query2", sample_chunks, "answer2", 0.7, None)

        stats = cache.stats()
        assert stats["size"] == 2
        assert "volume" in stats
        assert "ttl" in stats

    def test_disabled_cache_operations(self, cache_dir, sample_chunks):
        """Test operations on disabled cache."""
        cache = LLMCache(cache_dir=cache_dir, enabled=False)

        # get() should return None
        assert cache.get("query", sample_chunks, 0.7, None) is None

        # set() should not raise error
        cache.set("query", sample_chunks, "answer", 0.7, None)

        # stats() should show disabled
        stats = cache.stats()
        assert stats["enabled"] is False
        assert stats["size"] == 0


class TestCacheUtilities:
    """Tests for cache utility functions."""

    @pytest.fixture
    def cache_base_dir(self):
        """Create temporary cache base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_clear_all_caches(self, cache_base_dir):
        """Test clearing all caches."""
        # Create and populate both caches
        emb_cache = EmbeddingCache(cache_dir=cache_base_dir / "embeddings")
        llm_cache = LLMCache(cache_dir=cache_base_dir / "llm")

        emb_cache.set("query1", np.array([1.0, 2.0]))
        llm_cache.set("query2", [{"content": "chunk"}], "answer", 0.7, None)

        # Verify caches have data
        assert emb_cache.get("query1") is not None
        assert llm_cache.get("query2", [{"content": "chunk"}], 0.7, None) is not None

        # Clear all caches
        clear_all_caches(cache_base_dir)

        # Verify caches are cleared
        assert emb_cache.get("query1") is None
        assert llm_cache.get("query2", [{"content": "chunk"}], 0.7, None) is None

    def test_get_cache_stats(self, cache_base_dir):
        """Test getting cache statistics."""
        # Create and populate both caches
        emb_cache = EmbeddingCache(cache_dir=cache_base_dir / "embeddings")
        llm_cache = LLMCache(cache_dir=cache_base_dir / "llm")

        emb_cache.set("query1", np.array([1.0, 2.0]))
        emb_cache.set("query2", np.array([3.0, 4.0]))
        llm_cache.set("query3", [{"content": "chunk"}], "answer", 0.7, None)

        # Get stats
        stats = get_cache_stats(cache_base_dir)

        assert "embeddings" in stats
        assert "llm" in stats
        assert "total" in stats

        assert stats["embeddings"]["size"] == 2
        assert stats["llm"]["size"] == 1
        assert stats["total"]["size"] == 3

    def test_get_cache_stats_empty(self, cache_base_dir):
        """Test stats for empty caches."""
        stats = get_cache_stats(cache_base_dir)

        assert stats["embeddings"]["size"] == 0
        assert stats["llm"]["size"] == 0
        assert stats["total"]["size"] == 0


class TestEmbedderCacheIntegration:
    """Tests for Embedder cache integration."""

    def test_embedder_with_cache(self):
        """Test Embedder with caching enabled."""
        from core.embedder import Embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = Embedder(enable_cache=True, cache_ttl=60)

            # Mock the actual embedding to avoid loading the model
            mock_embedding = np.array([0.1, 0.2, 0.3])
            embedder._model = MagicMock()
            embedder._model.encode = MagicMock(return_value=mock_embedding)

            # First call - cache miss
            result1 = embedder.embed("test query")
            assert embedder._model.encode.call_count == 1

            # Second call - cache hit (model should not be called again)
            result2 = embedder.embed("test query")
            assert embedder._model.encode.call_count == 1  # Still 1

            # Results should match
            np.testing.assert_array_equal(result1, result2)

    def test_embedder_without_cache(self):
        """Test Embedder with caching disabled."""
        from core.embedder import Embedder

        embedder = Embedder(enable_cache=False)
        assert embedder._cache is None

    def test_embedder_cache_stats(self):
        """Test getting cache stats from Embedder."""
        from core.embedder import Embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = Embedder(enable_cache=True)

            stats = embedder.get_cache_stats()
            assert stats["enabled"] is True
            assert "size" in stats

    def test_embedder_clear_cache(self):
        """Test clearing Embedder cache."""
        from core.embedder import Embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = Embedder(enable_cache=True)

            # Mock embedding
            mock_embedding = np.array([0.1, 0.2, 0.3])
            embedder._model = MagicMock()
            embedder._model.encode = MagicMock(return_value=mock_embedding)

            # Add to cache
            embedder.embed("query")
            assert embedder.get_cache_stats()["size"] > 0

            # Clear cache
            embedder.clear_cache()
            assert embedder.get_cache_stats()["size"] == 0


class TestOllamaClientCacheIntegration:
    """Tests for OllamaClient cache integration."""

    def test_ollama_client_with_cache(self):
        """Test OllamaClient with caching enabled."""
        from core.llm import OllamaClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OllamaClient(enable_cache=True, cache_ttl=60)
            assert client._cache is not None

    def test_ollama_client_without_cache(self):
        """Test OllamaClient with caching disabled."""
        from core.llm import OllamaClient

        client = OllamaClient(enable_cache=False)
        assert client._cache is None

    def test_ollama_client_cache_stats(self):
        """Test getting cache stats from OllamaClient."""
        from core.llm import OllamaClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OllamaClient(enable_cache=True)

            stats = client.get_cache_stats()
            assert stats["enabled"] is True
            assert "size" in stats

    def test_ollama_client_clear_cache(self):
        """Test clearing OllamaClient cache."""
        from core.llm import OllamaClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OllamaClient(enable_cache=True)

            # Add to cache manually
            client._cache.set(
                "query", [{"content": "chunk"}], "answer", 0.7, None
            )
            assert client.get_cache_stats()["size"] > 0

            # Clear cache
            client.clear_cache()
            assert client.get_cache_stats()["size"] == 0
