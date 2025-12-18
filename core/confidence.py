"""Answer confidence scoring for RAG queries.

This module calculates confidence scores based on:
1. Source similarity scores (average of top chunks)
2. Source agreement (multiple sources saying the same thing)
3. Score distribution (how consistent are the top results)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from config.config import (
    CONFIDENCE_AGREEMENT_WEIGHT,
    CONFIDENCE_COVERAGE_WEIGHT,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_MIN_SOURCES,
    CONFIDENCE_SIMILARITY_WEIGHT,
)

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfidenceScore:
    """
    Confidence score result.

    Attributes:
        overall_score: Combined confidence score (0.0 - 1.0)
        level: Categorical confidence level (high/medium/low)
        similarity_score: Average similarity of retrieved chunks
        agreement_score: How much sources agree (0.0 - 1.0)
        coverage_score: How well sources cover the query (0.0 - 1.0)
        factors: Detailed breakdown of scoring factors
    """

    overall_score: float
    level: ConfidenceLevel
    similarity_score: float
    agreement_score: float
    coverage_score: float
    factors: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": round(self.overall_score, 3),
            "level": self.level.value,
            "similarity_score": round(self.similarity_score, 3),
            "agreement_score": round(self.agreement_score, 3),
            "coverage_score": round(self.coverage_score, 3),
            "factors": self.factors,
        }


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG query results.

    The confidence score combines multiple factors:
    1. Similarity Score: How similar are the retrieved chunks to the query?
    2. Agreement Score: Do multiple chunks point to consistent information?
    3. Coverage Score: How many relevant sources were found?

    Usage:
        scorer = ConfidenceScorer()
        confidence = scorer.calculate(chunks)
        print(f"Confidence: {confidence.level} ({confidence.overall_score:.2f})")
    """

    def __init__(
        self,
        high_threshold: float = CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold: float = CONFIDENCE_MEDIUM_THRESHOLD,
        similarity_weight: float = CONFIDENCE_SIMILARITY_WEIGHT,
        agreement_weight: float = CONFIDENCE_AGREEMENT_WEIGHT,
        coverage_weight: float = CONFIDENCE_COVERAGE_WEIGHT,
        min_sources: int = CONFIDENCE_MIN_SOURCES,
    ):
        """
        Initialize confidence scorer.

        Args:
            high_threshold: Score threshold for HIGH confidence (default: 0.75)
            medium_threshold: Score threshold for MEDIUM confidence (default: 0.50)
            similarity_weight: Weight for similarity score component (default: 0.5)
            agreement_weight: Weight for agreement score component (default: 0.3)
            coverage_weight: Weight for coverage score component (default: 0.2)
            min_sources: Minimum sources needed for full coverage score (default: 3)
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.similarity_weight = similarity_weight
        self.agreement_weight = agreement_weight
        self.coverage_weight = coverage_weight
        self.min_sources = min_sources

        # Normalize weights to sum to 1.0
        total_weight = similarity_weight + agreement_weight + coverage_weight
        self.similarity_weight /= total_weight
        self.agreement_weight /= total_weight
        self.coverage_weight /= total_weight

        logger.debug(
            f"ConfidenceScorer initialized: thresholds=({high_threshold}, {medium_threshold}), "
            f"weights=(sim={self.similarity_weight:.2f}, agree={self.agreement_weight:.2f}, "
            f"cov={self.coverage_weight:.2f})"
        )

    def calculate(
        self,
        chunks: List[Dict[str, Any]],
        n_requested: int = 5,
    ) -> ConfidenceScore:
        """
        Calculate confidence score for retrieved chunks.

        Args:
            chunks: List of retrieved chunks with similarity scores
            n_requested: Number of chunks originally requested

        Returns:
            ConfidenceScore with overall score and breakdown
        """
        if not chunks:
            return ConfidenceScore(
                overall_score=0.0,
                level=ConfidenceLevel.LOW,
                similarity_score=0.0,
                agreement_score=0.0,
                coverage_score=0.0,
                factors={
                    "reason": "No relevant chunks found",
                    "chunk_count": 0,
                },
            )

        # Extract similarity scores
        similarities = [c.get("similarity", 0.0) for c in chunks]

        # Calculate individual scores
        similarity_score = self._calculate_similarity_score(similarities)
        agreement_score = self._calculate_agreement_score(similarities, chunks)
        coverage_score = self._calculate_coverage_score(len(chunks), n_requested)

        # Calculate weighted overall score
        overall_score = (
            self.similarity_weight * similarity_score
            + self.agreement_weight * agreement_score
            + self.coverage_weight * coverage_score
        )

        # Determine confidence level
        if overall_score >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif overall_score >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        # Build factors breakdown
        factors = self._build_factors(
            chunks=chunks,
            similarities=similarities,
            similarity_score=similarity_score,
            agreement_score=agreement_score,
            coverage_score=coverage_score,
            n_requested=n_requested,
        )

        confidence = ConfidenceScore(
            overall_score=overall_score,
            level=level,
            similarity_score=similarity_score,
            agreement_score=agreement_score,
            coverage_score=coverage_score,
            factors=factors,
        )

        logger.debug(
            f"Confidence calculated: {level.value} ({overall_score:.3f}) - "
            f"sim={similarity_score:.3f}, agree={agreement_score:.3f}, cov={coverage_score:.3f}"
        )

        return confidence

    def _calculate_similarity_score(self, similarities: List[float]) -> float:
        """
        Calculate similarity score from chunk similarities.

        Uses weighted average giving more importance to top results.
        """
        if not similarities:
            return 0.0

        # Weight top results more heavily (exponential decay)
        weights = [1.0 / (i + 1) for i in range(len(similarities))]
        total_weight = sum(weights)

        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        return weighted_sum / total_weight

    def _calculate_agreement_score(
        self,
        similarities: List[float],
        chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate agreement score based on:
        1. Consistency of similarity scores (low variance = high agreement)
        2. Multiple sources from different documents
        """
        if len(similarities) < 2:
            # With only one source, we can't measure agreement
            return 0.5  # Neutral score

        # Factor 1: Similarity consistency (low variance = high agreement)
        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        # Convert variance to score (0 variance = 1.0, high variance = lower score)
        # Max expected variance for [0,1] range is ~0.25
        consistency_score = max(0, 1 - (variance / 0.25))

        # Factor 2: Source diversity (different documents)
        doc_ids = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id", "unknown")
            doc_ids.add(doc_id)

        # More unique documents = more agreement across sources
        # Cap at min_sources for full score
        diversity_score = min(len(doc_ids), self.min_sources) / self.min_sources

        # Factor 3: Top results should have high similarity (agreement indicator)
        # If top 2 chunks have high similarity, sources likely agree
        top_agreement = 0.0
        if len(similarities) >= 2:
            top_agreement = min(similarities[0], similarities[1])

        # Combine factors
        agreement = (consistency_score * 0.3 + diversity_score * 0.3 + top_agreement * 0.4)

        return agreement

    def _calculate_coverage_score(
        self,
        n_found: int,
        n_requested: int,
    ) -> float:
        """
        Calculate coverage score based on how many relevant sources were found.
        """
        if n_requested <= 0:
            return 1.0 if n_found > 0 else 0.0

        # Score based on finding requested number of chunks
        request_coverage = min(n_found, n_requested) / n_requested

        # Bonus for finding at least min_sources
        source_coverage = min(n_found, self.min_sources) / self.min_sources

        # Combine with emphasis on meeting the request
        return request_coverage * 0.7 + source_coverage * 0.3

    def _build_factors(
        self,
        chunks: List[Dict[str, Any]],
        similarities: List[float],
        similarity_score: float,
        agreement_score: float,
        coverage_score: float,
        n_requested: int,
    ) -> Dict[str, Any]:
        """Build detailed factors breakdown for debugging/display."""
        # Count unique documents
        doc_ids = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id", "unknown")
            doc_ids.add(doc_id)

        # Get top similarity scores
        top_similarities = similarities[:3] if similarities else []

        factors = {
            "chunk_count": len(chunks),
            "chunks_requested": n_requested,
            "unique_documents": len(doc_ids),
            "top_similarities": [round(s, 3) for s in top_similarities],
            "avg_similarity": (
                round(sum(similarities) / len(similarities), 3) if similarities else 0
            ),
            "min_similarity": round(min(similarities), 3) if similarities else 0,
            "max_similarity": round(max(similarities), 3) if similarities else 0,
            "weights": {
                "similarity": round(self.similarity_weight, 2),
                "agreement": round(self.agreement_weight, 2),
                "coverage": round(self.coverage_weight, 2),
            },
            "thresholds": {
                "high": self.high_threshold,
                "medium": self.medium_threshold,
            },
        }

        # Add reason based on scores
        if similarity_score < 0.5:
            factors["primary_concern"] = "Low similarity between query and retrieved chunks"
        elif agreement_score < 0.5:
            factors["primary_concern"] = "Sources do not strongly agree"
        elif coverage_score < 0.5:
            factors["primary_concern"] = "Limited relevant sources found"
        else:
            factors["primary_concern"] = None

        return factors

    def get_level_emoji(self, level: ConfidenceLevel) -> str:
        """Get emoji representation for confidence level."""
        return {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴",
        }.get(level, "⚪")

    def format_confidence(self, confidence: ConfidenceScore) -> str:
        """Format confidence for display."""
        emoji = self.get_level_emoji(confidence.level)
        return (
            f"{emoji} {confidence.level.value.capitalize()} confidence "
            f"({confidence.overall_score:.0%})"
        )
