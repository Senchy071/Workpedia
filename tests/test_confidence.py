"""Tests for confidence scoring module."""

from core.confidence import (
    ConfidenceLevel,
    ConfidenceScore,
    ConfidenceScorer,
)


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_level_values(self):
        """Test confidence level string values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"

    def test_level_is_string_enum(self):
        """Test that ConfidenceLevel is a string enum."""
        assert isinstance(ConfidenceLevel.HIGH, str)
        assert ConfidenceLevel.HIGH == "high"


class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = ConfidenceScore(
            overall_score=0.85,
            level=ConfidenceLevel.HIGH,
            similarity_score=0.9,
            agreement_score=0.8,
            coverage_score=0.75,
            factors={"chunk_count": 5},
        )

        result = score.to_dict()

        assert result["overall_score"] == 0.85
        assert result["level"] == "high"
        assert result["similarity_score"] == 0.9
        assert result["agreement_score"] == 0.8
        assert result["coverage_score"] == 0.75
        assert result["factors"]["chunk_count"] == 5

    def test_score_rounding(self):
        """Test that scores are rounded to 3 decimal places."""
        score = ConfidenceScore(
            overall_score=0.123456789,
            level=ConfidenceLevel.MEDIUM,
            similarity_score=0.987654321,
            agreement_score=0.555555555,
            coverage_score=0.111111111,
            factors={},
        )

        result = score.to_dict()

        assert result["overall_score"] == 0.123
        assert result["similarity_score"] == 0.988
        assert result["agreement_score"] == 0.556
        assert result["coverage_score"] == 0.111


class TestConfidenceScorer:
    """Tests for ConfidenceScorer class."""

    def test_default_initialization(self):
        """Test scorer with default parameters."""
        scorer = ConfidenceScorer()

        assert scorer.high_threshold == 0.75
        assert scorer.medium_threshold == 0.50
        assert scorer.min_sources == 3

    def test_custom_thresholds(self):
        """Test scorer with custom thresholds."""
        scorer = ConfidenceScorer(
            high_threshold=0.8,
            medium_threshold=0.6,
        )

        assert scorer.high_threshold == 0.8
        assert scorer.medium_threshold == 0.6

    def test_empty_chunks_returns_low(self):
        """Test that empty chunks returns low confidence."""
        scorer = ConfidenceScorer()

        result = scorer.calculate([], n_requested=5)

        assert result.level == ConfidenceLevel.LOW
        assert result.overall_score == 0.0
        assert result.factors["reason"] == "No relevant chunks found"

    def test_high_confidence_chunks(self):
        """Test high confidence with high similarity chunks."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test content 1", "similarity": 0.95, "metadata": {"doc_id": "doc1"}},
            {"content": "Test content 2", "similarity": 0.92, "metadata": {"doc_id": "doc2"}},
            {"content": "Test content 3", "similarity": 0.90, "metadata": {"doc_id": "doc3"}},
            {"content": "Test content 4", "similarity": 0.88, "metadata": {"doc_id": "doc4"}},
            {"content": "Test content 5", "similarity": 0.85, "metadata": {"doc_id": "doc5"}},
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert result.level == ConfidenceLevel.HIGH
        assert result.overall_score >= 0.75

    def test_medium_confidence_chunks(self):
        """Test medium confidence with moderate similarity."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test content 1", "similarity": 0.70, "metadata": {"doc_id": "doc1"}},
            {"content": "Test content 2", "similarity": 0.65, "metadata": {"doc_id": "doc2"}},
            {"content": "Test content 3", "similarity": 0.60, "metadata": {"doc_id": "doc3"}},
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert result.level == ConfidenceLevel.MEDIUM
        assert 0.50 <= result.overall_score < 0.75

    def test_low_confidence_chunks(self):
        """Test low confidence with low similarity."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test content 1", "similarity": 0.30, "metadata": {"doc_id": "doc1"}},
            {"content": "Test content 2", "similarity": 0.25, "metadata": {"doc_id": "doc2"}},
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert result.level == ConfidenceLevel.LOW
        assert result.overall_score < 0.50

    def test_single_chunk_agreement(self):
        """Test agreement score with single chunk (neutral)."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Single chunk", "similarity": 0.95, "metadata": {"doc_id": "doc1"}},
        ]

        result = scorer.calculate(chunks, n_requested=1)

        # With single chunk, agreement is neutral (0.5)
        assert result.agreement_score == 0.5

    def test_source_diversity(self):
        """Test that multiple documents improve agreement score."""
        scorer = ConfidenceScorer()

        # Same document chunks
        same_doc_chunks = [
            {"content": "Test 1", "similarity": 0.8, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 2", "similarity": 0.8, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 3", "similarity": 0.8, "metadata": {"doc_id": "doc1"}},
        ]

        # Different document chunks
        diff_doc_chunks = [
            {"content": "Test 1", "similarity": 0.8, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 2", "similarity": 0.8, "metadata": {"doc_id": "doc2"}},
            {"content": "Test 3", "similarity": 0.8, "metadata": {"doc_id": "doc3"}},
        ]

        same_result = scorer.calculate(same_doc_chunks, n_requested=3)
        diff_result = scorer.calculate(diff_doc_chunks, n_requested=3)

        # More diverse sources should have higher agreement
        assert diff_result.agreement_score >= same_result.agreement_score

    def test_coverage_score_full(self):
        """Test full coverage when all requested chunks found."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": f"Test {i}", "similarity": 0.8, "metadata": {"doc_id": f"doc{i}"}}
            for i in range(5)
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert result.coverage_score >= 0.9

    def test_coverage_score_partial(self):
        """Test partial coverage when fewer chunks found."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test 1", "similarity": 0.8, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 2", "similarity": 0.75, "metadata": {"doc_id": "doc2"}},
        ]

        result = scorer.calculate(chunks, n_requested=5)

        # Only 2 of 5 requested chunks found
        assert result.coverage_score < 0.7

    def test_factors_breakdown(self):
        """Test that factors include detailed breakdown."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test 1", "similarity": 0.9, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 2", "similarity": 0.85, "metadata": {"doc_id": "doc2"}},
            {"content": "Test 3", "similarity": 0.8, "metadata": {"doc_id": "doc3"}},
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert "chunk_count" in result.factors
        assert "unique_documents" in result.factors
        assert "top_similarities" in result.factors
        assert "avg_similarity" in result.factors
        assert "min_similarity" in result.factors
        assert "max_similarity" in result.factors
        assert "weights" in result.factors
        assert "thresholds" in result.factors

        assert result.factors["chunk_count"] == 3
        assert result.factors["unique_documents"] == 3

    def test_emoji_representation(self):
        """Test emoji helper method."""
        scorer = ConfidenceScorer()

        assert scorer.get_level_emoji(ConfidenceLevel.HIGH) == "🟢"
        assert scorer.get_level_emoji(ConfidenceLevel.MEDIUM) == "🟡"
        assert scorer.get_level_emoji(ConfidenceLevel.LOW) == "🔴"

    def test_format_confidence(self):
        """Test formatted confidence display."""
        scorer = ConfidenceScorer()
        score = ConfidenceScore(
            overall_score=0.85,
            level=ConfidenceLevel.HIGH,
            similarity_score=0.9,
            agreement_score=0.8,
            coverage_score=0.75,
            factors={},
        )

        formatted = scorer.format_confidence(score)

        assert "🟢" in formatted
        assert "High" in formatted
        assert "85%" in formatted

    def test_weighted_similarity(self):
        """Test that top results have more weight in similarity score."""
        scorer = ConfidenceScorer()

        # High top similarity, low bottom
        high_top = [
            {"content": "Test 1", "similarity": 0.95, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 2", "similarity": 0.40, "metadata": {"doc_id": "doc2"}},
        ]

        # Low top similarity, high bottom
        low_top = [
            {"content": "Test 1", "similarity": 0.40, "metadata": {"doc_id": "doc1"}},
            {"content": "Test 2", "similarity": 0.95, "metadata": {"doc_id": "doc2"}},
        ]

        high_result = scorer.calculate(high_top, n_requested=2)
        low_result = scorer.calculate(low_top, n_requested=2)

        # High top similarity should score better
        assert high_result.similarity_score > low_result.similarity_score

    def test_custom_weights(self):
        """Test scorer with custom weight configuration."""
        # All weight on similarity
        similarity_scorer = ConfidenceScorer(
            similarity_weight=1.0,
            agreement_weight=0.0,
            coverage_weight=0.0,
        )

        chunks = [
            {"content": "Test", "similarity": 0.95, "metadata": {"doc_id": "doc1"}},
        ]

        result = similarity_scorer.calculate(chunks, n_requested=5)

        # Should be primarily determined by similarity
        assert abs(result.overall_score - result.similarity_score) < 0.01

    def test_missing_metadata_handled(self):
        """Test handling of chunks with missing metadata."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test 1", "similarity": 0.8, "metadata": {}},
            {"content": "Test 2", "similarity": 0.75},  # No metadata at all
        ]

        # Should not raise an error
        result = scorer.calculate(chunks, n_requested=2)

        assert result.level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]

    def test_zero_n_requested(self):
        """Test handling of zero n_requested."""
        scorer = ConfidenceScorer()
        chunks = [
            {"content": "Test", "similarity": 0.8, "metadata": {"doc_id": "doc1"}},
        ]

        result = scorer.calculate(chunks, n_requested=0)

        # Should handle gracefully
        assert result.coverage_score == 1.0


class TestConfidenceScorerIntegration:
    """Integration tests for confidence scoring in context."""

    def test_realistic_high_confidence_scenario(self):
        """Test realistic high confidence RAG result."""
        scorer = ConfidenceScorer()

        # Simulate finding 5 highly relevant chunks from multiple documents
        chunks = [
            {
                "content": "Python is a programming language known for its simplicity.",
                "similarity": 0.92,
                "metadata": {"doc_id": "python_guide", "filename": "python_guide.pdf"},
            },
            {
                "content": "Python emphasizes code readability and simplicity.",
                "similarity": 0.89,
                "metadata": {"doc_id": "python_intro", "filename": "python_intro.pdf"},
            },
            {
                "content": "Python was created by Guido van Rossum in 1991.",
                "similarity": 0.87,
                "metadata": {"doc_id": "python_history", "filename": "history.pdf"},
            },
            {
                "content": "Python's design philosophy emphasizes code readability.",
                "similarity": 0.85,
                "metadata": {"doc_id": "python_philosophy", "filename": "philosophy.pdf"},
            },
            {
                "content": "Python supports multiple programming paradigms.",
                "similarity": 0.82,
                "metadata": {"doc_id": "python_features", "filename": "features.pdf"},
            },
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert result.level == ConfidenceLevel.HIGH
        assert result.overall_score > 0.8
        assert result.factors["unique_documents"] == 5

    def test_realistic_low_confidence_scenario(self):
        """Test realistic low confidence RAG result."""
        scorer = ConfidenceScorer()

        # Simulate finding only marginally relevant chunks
        chunks = [
            {
                "content": "The weather forecast shows rain tomorrow.",
                "similarity": 0.35,
                "metadata": {"doc_id": "news", "filename": "news.pdf"},
            },
            {
                "content": "Traffic conditions may affect commute times.",
                "similarity": 0.28,
                "metadata": {"doc_id": "news", "filename": "news.pdf"},
            },
        ]

        result = scorer.calculate(chunks, n_requested=5)

        assert result.level == ConfidenceLevel.LOW
        assert result.overall_score < 0.5
        assert result.factors["primary_concern"] is not None
