"""Performance caching for embeddings and LLM responses.

This module provides caching functionality to improve performance for repeated queries:
- EmbeddingCache: Caches query embeddings to avoid recomputation
- LLMCache: Caches LLM responses for identical context+question pairs

Both caches use diskcache for persistent, local storage with TTL-based eviction.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from diskcache import Cache

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for query embeddings.

    Stores embeddings for query strings to avoid recomputation. Cache keys are
    generated from normalized query text using SHA-256 hashing.

    Args:
        cache_dir: Directory for cache storage
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        enabled: Whether caching is enabled (default: True)

    Example:
        cache = EmbeddingCache(cache_dir="cache/embeddings")

        # Check cache first
        embedding = cache.get("What is machine learning?")
        if embedding is None:
            embedding = embedder.embed("What is machine learning?")
            cache.set("What is machine learning?", embedding)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        ttl: int = 3600,
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(self.cache_dir))
            logger.info(
                f"EmbeddingCache initialized: dir={cache_dir}, ttl={ttl}s, "
                f"size={len(self._cache)}"
            )
        else:
            self._cache = None
            logger.info("EmbeddingCache disabled")

    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key generation.

        Args:
            query: Query string

        Returns:
            Normalized query (lowercase, stripped)
        """
        return query.lower().strip()

    def _generate_key(self, query: str) -> str:
        """Generate cache key from query.

        Args:
            query: Query string

        Returns:
            SHA-256 hash of normalized query
        """
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding for query.

        Args:
            query: Query string

        Returns:
            Cached embedding as numpy array, or None if not found
        """
        if not self.enabled or self._cache is None:
            return None

        key = self._generate_key(query)
        cached = self._cache.get(key)

        if cached is not None:
            logger.debug(f"EmbeddingCache HIT: query='{query[:50]}...'")
            # Convert list back to numpy array
            return np.array(cached)

        logger.debug(f"EmbeddingCache MISS: query='{query[:50]}...'")
        return None

    def set(self, query: str, embedding: np.ndarray) -> None:
        """Cache embedding for query.

        Args:
            query: Query string
            embedding: Embedding vector as numpy array
        """
        if not self.enabled or self._cache is None:
            return

        key = self._generate_key(query)
        # Convert numpy array to list for JSON serialization
        self._cache.set(key, embedding.tolist(), expire=self.ttl)
        logger.debug(f"EmbeddingCache SET: query='{query[:50]}...', ttl={self.ttl}s")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        if self.enabled and self._cache is not None:
            self._cache.clear()
            logger.info("EmbeddingCache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hits, misses)
        """
        if not self.enabled or self._cache is None:
            return {"enabled": False, "size": 0}

        return {
            "enabled": True,
            "size": len(self._cache),
            "volume": self._cache.volume(),
            "ttl": self.ttl,
        }


class LLMCache:
    """Cache for LLM responses.

    Stores LLM-generated answers for identical context+question pairs. Cache keys
    are generated from the query, context chunks, and generation parameters.

    Args:
        cache_dir: Directory for cache storage
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        enabled: Whether caching is enabled (default: True)

    Example:
        cache = LLMCache(cache_dir="cache/llm")

        # Check cache first
        answer = cache.get(query="What is RAG?", context_chunks=chunks, temperature=0.7)
        if answer is None:
            answer = llm.generate(query="What is RAG?", context=context)
            cache.set(query="What is RAG?", context_chunks=chunks,
                     temperature=0.7, response=answer)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        ttl: int = 3600,
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(self.cache_dir))
            logger.info(
                f"LLMCache initialized: dir={cache_dir}, ttl={ttl}s, "
                f"size={len(self._cache)}"
            )
        else:
            self._cache = None
            logger.info("LLMCache disabled")

    def _generate_key(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate cache key from query, context, and parameters.

        Args:
            query: Query string
            context_chunks: List of context chunks used for generation
            temperature: LLM temperature parameter
            max_tokens: Max tokens parameter

        Returns:
            SHA-256 hash of serialized inputs
        """
        # Create deterministic representation
        key_data = {
            "query": query.lower().strip(),
            "context": [
                {
                    "content": chunk.get("content", ""),
                    "doc_id": chunk.get("doc_id", ""),
                }
                for chunk in context_chunks
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Serialize and hash
        serialized = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Get cached LLM response.

        Args:
            query: Query string
            context_chunks: List of context chunks
            temperature: LLM temperature parameter
            max_tokens: Max tokens parameter

        Returns:
            Cached response string, or None if not found
        """
        if not self.enabled or self._cache is None:
            return None

        key = self._generate_key(query, context_chunks, temperature, max_tokens)
        cached = self._cache.get(key)

        if cached is not None:
            logger.debug(f"LLMCache HIT: query='{query[:50]}...'")
            return cached

        logger.debug(f"LLMCache MISS: query='{query[:50]}...'")
        return None

    def set(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        response: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Cache LLM response.

        Args:
            query: Query string
            context_chunks: List of context chunks
            response: LLM response to cache
            temperature: LLM temperature parameter
            max_tokens: Max tokens parameter
        """
        if not self.enabled or self._cache is None:
            return

        key = self._generate_key(query, context_chunks, temperature, max_tokens)
        self._cache.set(key, response, expire=self.ttl)
        logger.debug(f"LLMCache SET: query='{query[:50]}...', ttl={self.ttl}s")

    def clear(self) -> None:
        """Clear all cached responses."""
        if self.enabled and self._cache is not None:
            self._cache.clear()
            logger.info("LLMCache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hits, misses)
        """
        if not self.enabled or self._cache is None:
            return {"enabled": False, "size": 0}

        return {
            "enabled": True,
            "size": len(self._cache),
            "volume": self._cache.volume(),
            "ttl": self.ttl,
        }


def clear_all_caches(cache_base_dir: Union[str, Path]) -> None:
    """Clear all caches in the base directory.

    Args:
        cache_base_dir: Base directory containing cache subdirectories
    """
    cache_base = Path(cache_base_dir)

    # Clear embedding cache
    embedding_cache_dir = cache_base / "embeddings"
    if embedding_cache_dir.exists():
        cache = Cache(str(embedding_cache_dir))
        cache.clear()
        logger.info(f"Cleared embedding cache: {embedding_cache_dir}")

    # Clear LLM cache
    llm_cache_dir = cache_base / "llm"
    if llm_cache_dir.exists():
        cache = Cache(str(llm_cache_dir))
        cache.clear()
        logger.info(f"Cleared LLM cache: {llm_cache_dir}")


def get_cache_stats(cache_base_dir: Union[str, Path]) -> Dict[str, Any]:
    """Get statistics for all caches.

    Args:
        cache_base_dir: Base directory containing cache subdirectories

    Returns:
        Dictionary with stats for each cache type
    """
    cache_base = Path(cache_base_dir)
    stats = {}

    # Embedding cache stats
    embedding_cache_dir = cache_base / "embeddings"
    if embedding_cache_dir.exists():
        cache = Cache(str(embedding_cache_dir))
        stats["embeddings"] = {
            "size": len(cache),
            "volume": cache.volume(),
        }
    else:
        stats["embeddings"] = {"size": 0, "volume": 0}

    # LLM cache stats
    llm_cache_dir = cache_base / "llm"
    if llm_cache_dir.exists():
        cache = Cache(str(llm_cache_dir))
        stats["llm"] = {
            "size": len(cache),
            "volume": cache.volume(),
        }
    else:
        stats["llm"] = {"size": 0, "volume": 0}

    # Total stats
    stats["total"] = {
        "size": stats["embeddings"]["size"] + stats["llm"]["size"],
        "volume": stats["embeddings"]["volume"] + stats["llm"]["volume"],
    }

    return stats
