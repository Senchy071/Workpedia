"""Tests for query suggestion generation."""

import pytest

from core.suggestions import (
    QuerySuggestion,
    QuerySuggestionGenerator,
    generate_suggestions,
)


class TestQuerySuggestion:
    """Tests for QuerySuggestion dataclass."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = QuerySuggestion(
            suggestion_id="doc1_heading_0",
            doc_id="doc1",
            text="What is covered in the Introduction?",
            source_type="heading",
            source_text="Introduction",
            priority=8,
        )

        assert suggestion.suggestion_id == "doc1_heading_0"
        assert suggestion.doc_id == "doc1"
        assert suggestion.text == "What is covered in the Introduction?"
        assert suggestion.source_type == "heading"
        assert suggestion.source_text == "Introduction"
        assert suggestion.priority == 8

    def test_to_dict(self):
        """Test converting suggestion to dictionary."""
        suggestion = QuerySuggestion(
            suggestion_id="doc1_heading_0",
            doc_id="doc1",
            text="What is X?",
            source_type="heading",
            source_text="X",
            priority=5,
            metadata={"heading_level": 2},
        )

        d = suggestion.to_dict()
        assert d["suggestion_id"] == "doc1_heading_0"
        assert d["doc_id"] == "doc1"
        assert d["text"] == "What is X?"
        assert d["source_type"] == "heading"
        assert d["source_text"] == "X"
        assert d["priority"] == 5
        assert d["metadata"]["heading_level"] == 2

    def test_default_metadata(self):
        """Test default empty metadata."""
        suggestion = QuerySuggestion(
            suggestion_id="id",
            doc_id="doc",
            text="text",
            source_type="heading",
            source_text="source",
        )
        assert suggestion.metadata == {}


class TestQuerySuggestionGenerator:
    """Tests for QuerySuggestionGenerator class."""

    def test_init_default(self):
        """Test default initialization."""
        generator = QuerySuggestionGenerator()
        assert generator.enabled is True
        assert generator.max_suggestions == 15
        assert generator.min_heading_length == 5

    def test_init_disabled(self):
        """Test initialization with disabled."""
        generator = QuerySuggestionGenerator(enabled=False)
        assert generator.enabled is False

    def test_generate_disabled(self):
        """Test generation when disabled."""
        generator = QuerySuggestionGenerator(enabled=False)
        suggestions = generator.generate_suggestions({"doc_id": "test"})
        assert suggestions == []

    def test_extract_from_headings(self):
        """Test extracting suggestions from markdown headings."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# Chapter One

Some content here.

## Getting Started

More content.

### Configuration Options

Details about configuration.
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(
            parsed_doc,
            include_headings=True,
            include_toc=False,
            include_concepts=False,
        )

        assert len(suggestions) > 0
        # Check that headings were converted to questions
        texts = [s.text for s in suggestions]
        assert any("Chapter One" in t for t in texts)
        assert any("Getting Started" in t for t in texts)

    def test_skip_generic_headings(self):
        """Test that generic headings are skipped."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# Introduction

## References

### Appendix

## Real Content Section
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(
            parsed_doc,
            include_headings=True,
            include_toc=False,
            include_concepts=False,
        )

        texts = [s.text for s in suggestions]
        # Generic headings should be skipped
        assert not any("Introduction" in t and "what is" in t.lower() for t in texts)
        assert not any("References" in t for t in texts)
        assert not any("Appendix" in t for t in texts)
        # Real content should be included
        assert any("Real Content Section" in t for t in texts)

    def test_skip_short_headings(self):
        """Test that short headings are skipped."""
        generator = QuerySuggestionGenerator(min_heading_length=5)
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# ABC

## Real Heading Here
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(
            parsed_doc,
            include_headings=True,
            include_toc=False,
            include_concepts=False,
        )

        texts = [s.text for s in suggestions]
        # Short heading "ABC" should be skipped
        assert not any("ABC" in t and len(t) < 20 for t in texts)
        # Long heading should be included
        assert any("Real Heading Here" in t for t in texts)

    def test_extract_concepts(self):
        """Test extracting key concepts from content."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
The Machine Learning algorithm uses Natural Language Processing.
Machine Learning is important. Natural Language Processing is used.
Machine Learning appears frequently in this document.
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(
            parsed_doc,
            include_headings=False,
            include_toc=False,
            include_concepts=True,
        )

        texts = [s.text for s in suggestions]
        # Frequent concepts should be extracted
        assert any("Machine Learning" in t for t in texts)

    def test_priority_ordering(self):
        """Test that suggestions are ordered by priority."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# Top Level Chapter

## Second Level Section

### Third Level Subsection
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(
            parsed_doc,
            include_headings=True,
            include_toc=False,
            include_concepts=False,
        )

        # Higher level headings should have higher priority
        if len(suggestions) >= 2:
            assert suggestions[0].priority >= suggestions[-1].priority

    def test_max_suggestions_limit(self):
        """Test that max_suggestions is respected."""
        generator = QuerySuggestionGenerator(max_suggestions=3)
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# Chapter One
## Section A
## Section B
## Section C
## Section D
## Section E
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(parsed_doc)
        assert len(suggestions) <= 3

    def test_default_suggestions(self):
        """Test generating default suggestions."""
        generator = QuerySuggestionGenerator()
        suggestions = generator.get_default_suggestions("doc123", "my_document.pdf")

        assert len(suggestions) > 0
        assert all(s.doc_id == "doc123" for s in suggestions)
        assert all(s.source_type == "default" for s in suggestions)
        # Check expected default questions
        texts = [s.text for s in suggestions]
        assert any("about" in t.lower() for t in texts)
        assert any("main topics" in t.lower() or "key points" in t.lower() for t in texts)

    def test_heading_to_question_patterns(self):
        """Test various heading to question conversions."""
        generator = QuerySuggestionGenerator()

        # Test numbered chapter
        parsed_doc = {
            "doc_id": "test",
            "raw_text": "# Chapter 1 Getting Started",
            "metadata": {},
        }
        suggestions = generator.generate_suggestions(
            parsed_doc, include_headings=True, include_toc=False, include_concepts=False
        )
        if suggestions:
            assert "?" in suggestions[0].text

    def test_clean_heading(self):
        """Test heading cleaning."""
        generator = QuerySuggestionGenerator()

        # Test with markdown formatting
        heading = "**Bold Heading**"
        cleaned = generator._clean_heading(heading)
        assert cleaned == "Bold Heading"

        # Test with link
        heading = "[Link Text](http://example.com)"
        cleaned = generator._clean_heading(heading)
        assert cleaned == "Link Text"

        # Test with trailing punctuation
        heading = "Heading:"
        cleaned = generator._clean_heading(heading)
        assert cleaned == "Heading"

    def test_is_generic_heading(self):
        """Test generic heading detection."""
        generator = QuerySuggestionGenerator()

        assert generator._is_generic_heading("introduction") is True
        assert generator._is_generic_heading("References") is True
        assert generator._is_generic_heading("appendix") is True
        assert generator._is_generic_heading("Machine Learning") is False
        assert generator._is_generic_heading("12345") is True  # Mostly numbers

    def test_avoid_duplicates(self):
        """Test that duplicate suggestions are avoided."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# Same Heading

## Same Heading

### Same Heading
""",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(
            parsed_doc,
            include_headings=True,
            include_toc=False,
            include_concepts=False,
        )

        # Should only have one suggestion for "Same Heading"
        texts = [s.text for s in suggestions]
        same_heading_count = sum(1 for t in texts if "Same Heading" in t)
        assert same_heading_count == 1

    def test_empty_document(self):
        """Test handling of empty document."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": "",
            "metadata": {"filename": "empty.pdf"},
        }

        suggestions = generator.generate_suggestions(parsed_doc)
        assert suggestions == []

    def test_no_raw_text(self):
        """Test handling of document without raw_text."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test_doc",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generator.generate_suggestions(parsed_doc)
        assert suggestions == []


class TestGenerateSuggestionsFunction:
    """Tests for the generate_suggestions convenience function."""

    def test_generate_suggestions_function(self):
        """Test the convenience function."""
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": "# Test Heading\n\nSome content.",
            "metadata": {"filename": "test.pdf"},
        }

        suggestions = generate_suggestions(parsed_doc)
        assert isinstance(suggestions, list)
        if suggestions:
            assert isinstance(suggestions[0], dict)
            assert "text" in suggestions[0]
            assert "source_type" in suggestions[0]

    def test_generate_suggestions_max_limit(self):
        """Test max_suggestions parameter."""
        parsed_doc = {
            "doc_id": "test_doc",
            "raw_text": """
# H1
## H2
## H3
## H4
## H5
""",
            "metadata": {},
        }

        suggestions = generate_suggestions(parsed_doc, max_suggestions=2)
        assert len(suggestions) <= 2


class TestSuggestionSourceTypes:
    """Tests for different suggestion source types."""

    def test_heading_source_type(self):
        """Test heading source type."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test",
            "raw_text": "# Real Heading",
            "metadata": {},
        }
        suggestions = generator.generate_suggestions(
            parsed_doc, include_headings=True, include_toc=False, include_concepts=False
        )
        if suggestions:
            assert suggestions[0].source_type == "heading"

    def test_concept_source_type(self):
        """Test concept source type."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test",
            "raw_text": "The Machine Learning algorithm. Machine Learning is great.",
            "metadata": {},
        }
        suggestions = generator.generate_suggestions(
            parsed_doc, include_headings=False, include_toc=False, include_concepts=True
        )
        if suggestions:
            assert any(s.source_type == "concept" for s in suggestions)

    def test_default_source_type(self):
        """Test default source type."""
        generator = QuerySuggestionGenerator()
        suggestions = generator.get_default_suggestions("doc1", "file.pdf")
        assert all(s.source_type == "default" for s in suggestions)


class TestSuggestionMetadata:
    """Tests for suggestion metadata."""

    def test_heading_level_metadata(self):
        """Test heading level in metadata."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test",
            "raw_text": "## Level Two Heading",
            "metadata": {},
        }
        suggestions = generator.generate_suggestions(
            parsed_doc, include_headings=True, include_toc=False, include_concepts=False
        )
        if suggestions:
            assert suggestions[0].metadata.get("heading_level") == 2

    def test_concept_frequency_metadata(self):
        """Test frequency in concept metadata."""
        generator = QuerySuggestionGenerator()
        parsed_doc = {
            "doc_id": "test",
            "raw_text": "Machine Learning is used. Machine Learning is great. Machine Learning.",
            "metadata": {},
        }
        suggestions = generator.generate_suggestions(
            parsed_doc, include_headings=False, include_toc=False, include_concepts=True
        )
        concept_suggestions = [s for s in suggestions if s.source_type == "concept"]
        if concept_suggestions:
            assert "frequency" in concept_suggestions[0].metadata
