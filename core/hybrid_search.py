"""Hybrid search combining semantic and keyword search with Reciprocal Rank Fusion.

This module provides:
- BM25 keyword search for exact matches (names, codes, IDs)
- Integration with ChromaDB semantic search
- Reciprocal Rank Fusion (RRF) to merge results
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Okapi

from config.config import (
    HYBRID_SEARCH_ENABLED,
    HYBRID_SEARCH_K,
    HYBRID_SEARCH_KEYWORD_WEIGHT,
    HYBRID_SEARCH_SEMANTIC_WEIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with combined score."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
        }


class BM25Index:
    """
    BM25 keyword index for a document collection.

    Features:
    - Token-based indexing with preprocessing
    - Persistent storage to disk
    - Incremental updates (add/remove documents)
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize BM25 index.

        Args:
            persist_path: Path to persist index (optional)
        """
        self.persist_path = persist_path
        self.corpus: List[List[str]] = []  # Tokenized documents
        self.chunk_ids: List[str] = []  # Chunk IDs in same order
        self.chunk_contents: Dict[str, str] = {}  # chunk_id -> content
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata
        self.bm25: Optional[BM25Okapi] = None

        # Load existing index if available
        if persist_path and Path(persist_path).exists():
            self._load()
            logger.info(f"Loaded BM25 index from {persist_path}: {len(self.chunk_ids)} chunks")
        else:
            logger.info("Initialized empty BM25 index")

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        rebuild: bool = True,
    ) -> int:
        """
        Add chunks to the BM25 index.

        Args:
            chunks: List of chunk dicts with chunk_id, content, metadata
            rebuild: Rebuild BM25 index after adding (set False for batch adds)

        Returns:
            Number of chunks added
        """
        added = 0
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            if not chunk_id or not content:
                continue

            # Skip if already indexed
            if chunk_id in self.chunk_contents:
                continue

            # Tokenize and add
            tokens = self._tokenize(content)
            self.corpus.append(tokens)
            self.chunk_ids.append(chunk_id)
            self.chunk_contents[chunk_id] = content
            self.chunk_metadata[chunk_id] = metadata
            added += 1

        if rebuild and added > 0:
            self._rebuild_index()

        return added

    def remove_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks for a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of chunks removed
        """
        # Find indices to remove
        indices_to_remove = []
        for i, chunk_id in enumerate(self.chunk_ids):
            metadata = self.chunk_metadata.get(chunk_id, {})
            if metadata.get("doc_id") == doc_id:
                indices_to_remove.append(i)

        if not indices_to_remove:
            return 0

        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            chunk_id = self.chunk_ids[i]
            del self.corpus[i]
            del self.chunk_ids[i]
            del self.chunk_contents[chunk_id]
            del self.chunk_metadata[chunk_id]

        # Rebuild index
        self._rebuild_index()

        logger.info(f"Removed {len(indices_to_remove)} chunks for doc_id={doc_id}")
        return len(indices_to_remove)

    def search(
        self,
        query: str,
        n_results: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[Tuple[str, float, int]]:
        """
        Search the BM25 index.

        Args:
            query: Search query
            n_results: Number of results to return
            doc_id: Optional filter by document ID

        Returns:
            List of (chunk_id, score, rank) tuples sorted by score descending
        """
        if self.bm25 is None or len(self.corpus) == 0:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Create (chunk_id, score, index) tuples
        results = []
        for i, (chunk_id, score) in enumerate(zip(self.chunk_ids, scores)):
            # Filter by doc_id if specified
            if doc_id:
                metadata = self.chunk_metadata.get(chunk_id, {})
                if metadata.get("doc_id") != doc_id:
                    continue
            results.append((chunk_id, float(score), i))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Add ranks and limit
        ranked_results = []
        for rank, (chunk_id, score, _) in enumerate(results[:n_results], start=1):
            ranked_results.append((chunk_id, score, rank))

        return ranked_results

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        if chunk_id not in self.chunk_contents:
            return None
        return {
            "chunk_id": chunk_id,
            "content": self.chunk_contents[chunk_id],
            "metadata": self.chunk_metadata.get(chunk_id, {}),
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Preprocessing:
        - Lowercase
        - Split on whitespace and punctuation
        - Remove very short tokens
        - Keep alphanumeric tokens
        """
        if not text:
            return []

        # Lowercase
        text = text.lower()

        # Split on non-alphanumeric (but keep numbers for IDs like "12345")
        tokens = re.findall(r"[a-z0-9]+", text)

        # Filter very short tokens (except numbers)
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]

        return tokens

    def _rebuild_index(self):
        """Rebuild the BM25 index from corpus."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            logger.debug(f"Rebuilt BM25 index with {len(self.corpus)} documents")
        else:
            self.bm25 = None

    def _save(self):
        """Save index to disk."""
        if not self.persist_path:
            return

        data = {
            "corpus": self.corpus,
            "chunk_ids": self.chunk_ids,
            "chunk_contents": self.chunk_contents,
            "chunk_metadata": self.chunk_metadata,
        }

        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

        logger.debug(f"Saved BM25 index to {self.persist_path}")

    def _load(self):
        """Load index from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return

        with open(self.persist_path, "r") as f:
            data = json.load(f)

        self.corpus = data.get("corpus", [])
        self.chunk_ids = data.get("chunk_ids", [])
        self.chunk_contents = data.get("chunk_contents", {})
        self.chunk_metadata = data.get("chunk_metadata", {})

        self._rebuild_index()

    def save(self):
        """Public method to persist index."""
        self._save()

    @property
    def count(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunk_ids)


class HybridSearcher:
    """
    Combines semantic and keyword search using Reciprocal Rank Fusion.

    Features:
    - Semantic search via ChromaDB embeddings
    - Keyword search via BM25
    - Configurable weights for each method
    - Reciprocal Rank Fusion (RRF) for score combination
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        enabled: bool = HYBRID_SEARCH_ENABLED,
        rrf_k: int = HYBRID_SEARCH_K,
        semantic_weight: float = HYBRID_SEARCH_SEMANTIC_WEIGHT,
        keyword_weight: float = HYBRID_SEARCH_KEYWORD_WEIGHT,
    ):
        """
        Initialize hybrid searcher.

        Args:
            bm25_index: BM25Index instance (creates new if None)
            enabled: Enable hybrid search (False = semantic only)
            rrf_k: RRF constant k (typically 60)
            semantic_weight: Weight for semantic search scores
            keyword_weight: Weight for keyword search scores
        """
        self.bm25_index = bm25_index or BM25Index()
        self.enabled = enabled
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            self.semantic_weight = semantic_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight

        logger.info(
            f"HybridSearcher initialized: enabled={enabled}, k={rrf_k}, "
            f"semantic_weight={self.semantic_weight:.2f}, "
            f"keyword_weight={self.keyword_weight:.2f}"
        )

    def search(
        self,
        query: str,
        semantic_results: List[Dict[str, Any]],
        n_results: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword results.

        Args:
            query: Search query
            semantic_results: Results from semantic search (ChromaDB)
                Each result should have: chunk_id, content, metadata, similarity
            n_results: Number of results to return
            doc_id: Optional filter by document ID

        Returns:
            List of SearchResult objects sorted by combined score
        """
        if not self.enabled:
            # Just return semantic results as SearchResult objects
            return [
                SearchResult(
                    chunk_id=r.get("chunk_id", ""),
                    content=r.get("content", ""),
                    metadata=r.get("metadata", {}),
                    semantic_score=r.get("similarity", 0.0),
                    keyword_score=0.0,
                    combined_score=r.get("similarity", 0.0),
                    semantic_rank=i + 1,
                    keyword_rank=None,
                )
                for i, r in enumerate(semantic_results[:n_results])
            ]

        # Get keyword search results
        keyword_results = self.bm25_index.search(
            query,
            n_results=n_results * 2,  # Get more for better fusion
            doc_id=doc_id,
        )

        # Build lookup for semantic results
        semantic_lookup: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                semantic_lookup[chunk_id] = (
                    result.get("similarity", 0.0),
                    rank,
                    result,
                )

        # Build lookup for keyword results
        keyword_lookup: Dict[str, Tuple[float, int]] = {}
        for chunk_id, score, rank in keyword_results:
            keyword_lookup[chunk_id] = (score, rank)

        # Collect all unique chunk IDs
        all_chunk_ids: Set[str] = set(semantic_lookup.keys()) | set(keyword_lookup.keys())

        # Calculate RRF scores
        results: List[SearchResult] = []
        for chunk_id in all_chunk_ids:
            semantic_score = 0.0
            semantic_rank = None
            keyword_score = 0.0
            keyword_rank = None
            content = ""
            metadata = {}

            if chunk_id in semantic_lookup:
                semantic_score, semantic_rank, result = semantic_lookup[chunk_id]
                content = result.get("content", "")
                metadata = result.get("metadata", {})

            if chunk_id in keyword_lookup:
                keyword_score, keyword_rank = keyword_lookup[chunk_id]
                # Get content from BM25 index if not from semantic
                if not content:
                    chunk_data = self.bm25_index.get_chunk(chunk_id)
                    if chunk_data:
                        content = chunk_data.get("content", "")
                        metadata = chunk_data.get("metadata", {})

            # Calculate RRF combined score
            combined_score = self._calculate_rrf_score(
                semantic_rank, keyword_rank, semantic_score, keyword_score
            )

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                    semantic_rank=semantic_rank,
                    keyword_rank=keyword_rank,
                )
            )

        # Sort by combined score descending
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Limit results
        return results[:n_results]

    def _calculate_rrf_score(
        self,
        semantic_rank: Optional[int],
        keyword_rank: Optional[int],
        semantic_score: float,
        keyword_score: float,
    ) -> float:
        """
        Calculate Reciprocal Rank Fusion score.

        RRF formula: 1 / (k + rank)
        Combined with weights for each search method.

        Args:
            semantic_rank: Rank in semantic search (1-indexed, None if not found)
            keyword_rank: Rank in keyword search (1-indexed, None if not found)
            semantic_score: Raw semantic similarity score
            keyword_score: Raw BM25 score

        Returns:
            Combined RRF score
        """
        score = 0.0

        # RRF contribution from semantic search
        if semantic_rank is not None:
            rrf_semantic = 1.0 / (self.rrf_k + semantic_rank)
            score += self.semantic_weight * rrf_semantic

        # RRF contribution from keyword search
        if keyword_rank is not None:
            rrf_keyword = 1.0 / (self.rrf_k + keyword_rank)
            score += self.keyword_weight * rrf_keyword

        return score

    def add_chunks_to_index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add chunks to the BM25 index.

        Args:
            chunks: List of chunk dicts

        Returns:
            Number of chunks added
        """
        return self.bm25_index.add_chunks(chunks)

    def remove_doc_from_index(self, doc_id: str) -> int:
        """
        Remove document from BM25 index.

        Args:
            doc_id: Document ID

        Returns:
            Number of chunks removed
        """
        return self.bm25_index.remove_by_doc_id(doc_id)

    def save_index(self):
        """Persist the BM25 index to disk."""
        self.bm25_index.save()

    @property
    def index_count(self) -> int:
        """Number of chunks in BM25 index."""
        return self.bm25_index.count


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    Args:
        rankings: List of rankings, each a list of (item_id, score) tuples
        k: RRF constant (default 60)
        weights: Optional weights for each ranking

    Returns:
        Combined ranking as list of (item_id, rrf_score) tuples
    """
    if not rankings:
        return []

    # Default equal weights
    if weights is None:
        weights = [1.0 / len(rankings)] * len(rankings)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Calculate RRF scores
    scores: Dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (item_id, _) in enumerate(ranking, start=1):
            rrf_score = weight * (1.0 / (k + rank))
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score

    # Sort by score
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result
