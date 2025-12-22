"""Cross-encoder reranker for improving retrieval quality.

This module provides a cross-encoder based reranking system that significantly
improves answer quality by re-scoring retrieved chunks using a more powerful
model that considers query-document pairs jointly.

The cross-encoder approach:
1. Initial retrieval returns top-N candidates (e.g., 20 chunks)
2. Cross-encoder scores each (query, chunk) pair for relevance
3. Return top-K chunks (e.g., 5) with highest scores to LLM

This two-stage approach provides better quality than single-stage retrieval
while maintaining reasonable latency.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.config import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    RERANKER_TOP_N,
)

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation.

    Attributes:
        chunk: The original chunk data
        original_rank: Position in original retrieval results (0-indexed)
        original_score: Original similarity score from vector search
        rerank_score: New score from cross-encoder (higher = more relevant)
        new_rank: Position after reranking (0-indexed)
    """

    chunk: Dict[str, Any]
    original_rank: int
    original_score: float
    rerank_score: float
    new_rank: int


class CrossEncoderReranker:
    """Cross-encoder based reranker for improving retrieval quality.

    Uses a cross-encoder model to re-score query-document pairs, providing
    more accurate relevance scores than bi-encoder similarity search alone.

    Features:
    - Lazy model loading (loads on first use)
    - GPU acceleration when available
    - Configurable top-N retrieval and top-K return
    - Detailed scoring with original vs reranked comparisons

    Example:
        reranker = CrossEncoderReranker()

        # Get initial retrieval results (more than you need)
        chunks = vector_store.query(query_embedding, n_results=20)

        # Rerank to get best matches
        reranked = reranker.rerank(query, chunks, top_k=5)

        # Use top-k reranked chunks for generation
        for result in reranked:
            print(f"Score: {result.rerank_score:.3f} - {result.chunk['content'][:100]}")
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: Optional[str] = None,
        enabled: bool = RERANKER_ENABLED,
        top_n: int = RERANKER_TOP_N,
        top_k: int = RERANKER_TOP_K,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            enabled: Whether reranking is enabled (disabled = passthrough)
            top_n: Number of candidates to retrieve for reranking
            top_k: Number of results to return after reranking
        """
        self.model_name = model_name
        self._device = device
        self.enabled = enabled
        self.top_n = top_n
        self.top_k = top_k
        self._model = None

        logger.info(
            f"CrossEncoderReranker initialized: model={model_name}, "
            f"enabled={enabled}, top_n={top_n}, top_k={top_k}"
        )

    @property
    def model(self):
        """Lazy load cross-encoder model on first use."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self._device)
            logger.info(f"Cross-encoder model loaded: {self.model_name}")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank chunks based on relevance to query.

        Args:
            query: The search query
            chunks: List of chunk dictionaries with 'content' and optional 'similarity' keys
            top_k: Number of results to return (overrides instance default)

        Returns:
            List of RerankResult objects sorted by rerank_score (highest first)
        """
        if not chunks:
            return []

        top_k = top_k or self.top_k

        # If disabled, return original chunks as-is (passthrough mode)
        if not self.enabled:
            logger.debug("Reranking disabled, returning original order")
            return self._passthrough(chunks, top_k)

        logger.info(f"Reranking {len(chunks)} chunks for query: '{query[:50]}...'")

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, chunk.get("content", "")) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Create results with both original and reranked info
        results = []
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            original_score = chunk.get("similarity", 0.0)
            results.append(
                RerankResult(
                    chunk=chunk,
                    original_rank=i,
                    original_score=original_score,
                    rerank_score=float(score),
                    new_rank=-1,  # Will be set after sorting
                )
            )

        # Sort by rerank score (highest first)
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update new ranks and take top-k
        for i, result in enumerate(results):
            result.new_rank = i

        top_results = results[:top_k]

        # Log reranking effect
        self._log_rerank_effect(top_results)

        return top_results

    def rerank_with_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Rerank chunks and return in simpler format for QueryEngine integration.

        Args:
            query: The search query
            chunks: List of chunk dictionaries
            top_k: Number of results to return

        Returns:
            Tuple of (reranked_chunks, rerank_scores)
        """
        results = self.rerank(query, chunks, top_k)

        reranked_chunks = [r.chunk for r in results]
        rerank_scores = [r.rerank_score for r in results]

        return reranked_chunks, rerank_scores

    def _passthrough(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[RerankResult]:
        """Return original chunks without reranking (disabled mode)."""
        results = []
        for i, chunk in enumerate(chunks[:top_k]):
            original_score = chunk.get("similarity", 0.0)
            results.append(
                RerankResult(
                    chunk=chunk,
                    original_rank=i,
                    original_score=original_score,
                    rerank_score=original_score,  # Use original score
                    new_rank=i,
                )
            )
        return results

    def _log_rerank_effect(self, results: List[RerankResult]) -> None:
        """Log the effect of reranking for debugging."""
        if not results:
            return

        # Count how many chunks moved significantly
        significant_moves = sum(
            1 for r in results if abs(r.new_rank - r.original_rank) >= 3
        )

        # Find biggest rank change (promotion)
        biggest_promotion = min(
            results, key=lambda r: r.new_rank - r.original_rank
        )

        logger.info(
            f"Reranking complete: {len(results)} results, "
            f"{significant_moves} significant rank changes"
        )

        if biggest_promotion.original_rank > biggest_promotion.new_rank:
            logger.debug(
                f"Biggest promotion: rank {biggest_promotion.original_rank} → "
                f"{biggest_promotion.new_rank} (score: {biggest_promotion.rerank_score:.3f})"
            )

    def get_retrieval_count(self) -> int:
        """Get the number of candidates to retrieve for reranking.

        Use this to know how many results to request from vector search.

        Returns:
            Number of candidates (top_n if enabled, top_k if disabled)
        """
        if self.enabled:
            return self.top_n
        return self.top_k

    def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            logger.info("Cleaning up cross-encoder resources")
            del self._model
            self._model = None


# Convenience function for one-off reranking
def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = RERANKER_TOP_K,
    model_name: str = RERANKER_MODEL,
) -> List[Dict[str, Any]]:
    """Quick function to rerank chunks without managing Reranker instance.

    Args:
        query: The search query
        chunks: List of chunk dictionaries
        top_k: Number of results to return
        model_name: Cross-encoder model to use

    Returns:
        List of reranked chunk dictionaries
    """
    reranker = CrossEncoderReranker(model_name=model_name, top_k=top_k)
    reranked, _ = reranker.rerank_with_scores(query, chunks, top_k)
    return reranked
