"""Tests for document summarizer module."""

from unittest.mock import MagicMock, patch

from core.summarizer import (
    SUMMARY_SYSTEM_PROMPT,
    DocumentSummarizer,
    DocumentSummary,
)


class TestDocumentSummary:
    """Tests for DocumentSummary dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = DocumentSummary(
            doc_id="test123",
            summary="- Point 1\n- Point 2\n- Point 3",
            bullets=["Point 1", "Point 2", "Point 3"],
            metadata={"filename": "test.pdf", "num_bullets": 3},
        )

        result = summary.to_dict()

        assert result["doc_id"] == "test123"
        assert result["summary"] == "- Point 1\n- Point 2\n- Point 3"
        assert result["bullets"] == ["Point 1", "Point 2", "Point 3"]
        assert result["metadata"]["filename"] == "test.pdf"
        assert result["metadata"]["num_bullets"] == 3

    def test_empty_bullets(self):
        """Test summary with no bullets."""
        summary = DocumentSummary(
            doc_id="test123",
            summary="",
            bullets=[],
            metadata={},
        )

        result = summary.to_dict()

        assert result["bullets"] == []


class TestDocumentSummarizer:
    """Tests for DocumentSummarizer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch("core.summarizer.OllamaClient"):
            summarizer = DocumentSummarizer()

            assert summarizer.max_bullets == 5
            assert summarizer.enabled is True

    def test_initialization_custom(self):
        """Test custom initialization."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(
            llm=mock_llm,
            max_bullets=3,
            max_input_chars=5000,
            temperature=0.5,
            enabled=False,
        )

        assert summarizer.max_bullets == 3
        assert summarizer.max_input_chars == 5000
        assert summarizer.temperature == 0.5
        assert summarizer.enabled is False

    def test_max_bullets_clamped(self):
        """Test that max_bullets is clamped between 3-7."""
        mock_llm = MagicMock()

        # Too low
        summarizer = DocumentSummarizer(llm=mock_llm, max_bullets=1)
        assert summarizer.max_bullets == 3

        # Too high
        summarizer = DocumentSummarizer(llm=mock_llm, max_bullets=10)
        assert summarizer.max_bullets == 7

        # Valid
        summarizer = DocumentSummarizer(llm=mock_llm, max_bullets=5)
        assert summarizer.max_bullets == 5

    def test_summarize_disabled(self):
        """Test that summarize returns None when disabled."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm, enabled=False)

        result = summarizer.summarize({"doc_id": "test", "raw_text": "content"})

        assert result is None
        mock_llm.generate.assert_not_called()

    def test_summarize_force_when_disabled(self):
        """Test force summarization even when disabled."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "- Point 1\n- Point 2\n- Point 3"
        mock_llm.model = "mistral"

        summarizer = DocumentSummarizer(llm=mock_llm, enabled=False)

        result = summarizer.summarize(
            {
                "doc_id": "test123",
                "raw_text": "This is a test document with sufficient content." * 10,
                "metadata": {"filename": "test.pdf"},
            },
            force=True,
        )

        assert result is not None
        assert result.doc_id == "test123"
        mock_llm.generate.assert_called_once()

    def test_summarize_insufficient_content(self):
        """Test that short content returns None."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm, enabled=True)

        result = summarizer.summarize({"doc_id": "test", "raw_text": "Short"})

        assert result is None
        mock_llm.generate.assert_not_called()

    def test_summarize_success(self):
        """Test successful summarization."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """- First key point about the document
- Second important finding
- Third main topic covered
- Fourth relevant detail
- Fifth concluding point"""
        mock_llm.model = "mistral"

        summarizer = DocumentSummarizer(llm=mock_llm, enabled=True)

        result = summarizer.summarize(
            {
                "doc_id": "test123",
                "raw_text": "This is a comprehensive test document." * 20,
                "metadata": {"filename": "report.pdf", "pages": 50},
            }
        )

        assert result is not None
        assert result.doc_id == "test123"
        assert len(result.bullets) == 5
        assert result.bullets[0] == "First key point about the document"
        assert result.metadata["filename"] == "report.pdf"
        assert result.metadata["model"] == "mistral"

    def test_parse_bullets_dash_format(self):
        """Test parsing bullets with dash format."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        bullets = summarizer._parse_bullets(
            "- First point\n- Second point\n- Third point"
        )

        assert len(bullets) == 3
        assert bullets[0] == "First point"
        assert bullets[1] == "Second point"
        assert bullets[2] == "Third point"

    def test_parse_bullets_asterisk_format(self):
        """Test parsing bullets with asterisk format."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        bullets = summarizer._parse_bullets(
            "* First point\n* Second point\n* Third point"
        )

        assert len(bullets) == 3
        assert bullets[0] == "First point"

    def test_parse_bullets_numbered_format(self):
        """Test parsing bullets with numbered format."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        bullets = summarizer._parse_bullets(
            "1. First point\n2. Second point\n3. Third point"
        )

        assert len(bullets) == 3
        assert bullets[0] == "First point"

    def test_parse_bullets_mixed_content(self):
        """Test parsing bullets with extra content."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        bullets = summarizer._parse_bullets(
            "Here are the key points:\n\n- First point\n- Second point\n\nAdditional notes..."
        )

        assert len(bullets) == 2
        assert bullets[0] == "First point"

    def test_extract_content_basic(self):
        """Test basic content extraction."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm, max_input_chars=1000)

        content = summarizer._extract_content(
            {
                "doc_id": "test",
                "raw_text": "This is the document content.",
                "metadata": {"filename": "test.pdf", "pages": 10},
            }
        )

        assert "Document: test.pdf" in content
        assert "Pages: 10" in content
        assert "This is the document content." in content

    def test_extract_content_truncation(self):
        """Test content truncation for long documents."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm, max_input_chars=500)

        long_text = "This is a sentence. " * 100  # ~2000 chars

        content = summarizer._extract_content(
            {
                "doc_id": "test",
                "raw_text": long_text,
                "metadata": {},
            }
        )

        assert len(content) <= 600  # Some buffer for truncation message
        assert "[Content truncated" in content

    def test_format_summary_for_chunk(self):
        """Test formatting summary as chunk content."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        summary = DocumentSummary(
            doc_id="test123",
            summary="- Point 1\n- Point 2",
            bullets=["Point 1", "Point 2"],
            metadata={"filename": "test.pdf"},
        )

        chunk_content = summarizer.format_summary_for_chunk(summary)

        assert "# DOCUMENT SUMMARY" in chunk_content
        assert "Document: test.pdf" in chunk_content
        assert "1. Point 1" in chunk_content
        assert "2. Point 2" in chunk_content
        assert "executive summary" in chunk_content.lower()

    def test_is_summary_query_positive(self):
        """Test detection of summary queries."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        assert summarizer.is_summary_query("What is in this document?")
        assert summarizer.is_summary_query("Give me a document summary")
        assert summarizer.is_summary_query("What's this document about?")
        assert summarizer.is_summary_query("Summarize this document")
        assert summarizer.is_summary_query("What are the main topics?")
        assert summarizer.is_summary_query("Tell me about this document")

    def test_is_summary_query_negative(self):
        """Test that non-summary queries are not detected."""
        mock_llm = MagicMock()
        summarizer = DocumentSummarizer(llm=mock_llm)

        assert not summarizer.is_summary_query("What is Python?")
        assert not summarizer.is_summary_query("How do I install the software?")
        assert not summarizer.is_summary_query("List the chapters")
        assert not summarizer.is_summary_query("Find errors in the code")

    def test_system_prompt_contains_guidelines(self):
        """Test that system prompt contains proper guidelines."""
        assert "bullet" in SUMMARY_SYSTEM_PROMPT.lower()
        assert "summar" in SUMMARY_SYSTEM_PROMPT.lower()  # Matches summarization, summaries, etc.


class TestDocumentSummarizerIntegration:
    """Integration tests for document summarizer."""

    def test_end_to_end_summarization(self):
        """Test complete summarization flow."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """- The document covers Python programming fundamentals
- Key sections include data types, functions, and classes
- Practical examples demonstrate each concept
- Best practices for code organization are discussed
- The guide targets intermediate developers"""
        mock_llm.model = "mistral"

        summarizer = DocumentSummarizer(llm=mock_llm, enabled=True)

        parsed_doc = {
            "doc_id": "python_guide_001",
            "raw_text": """
            Python Programming Guide

            Chapter 1: Introduction to Python
            Python is a versatile programming language...

            Chapter 2: Data Types
            Python supports various data types including...

            Chapter 3: Functions
            Functions allow you to organize code...
            """ * 10,  # Make it long enough
            "metadata": {
                "filename": "python_guide.pdf",
                "pages": 150,
            },
        }

        result = summarizer.summarize(parsed_doc)

        assert result is not None
        assert result.doc_id == "python_guide_001"
        assert len(result.bullets) == 5
        assert "Python programming" in result.bullets[0]
        assert result.metadata["filename"] == "python_guide.pdf"

        # Verify LLM was called with appropriate content
        call_args = mock_llm.generate.call_args
        assert "python_guide.pdf" in call_args.kwargs.get("prompt", "").lower() or \
               "python_guide.pdf" in str(call_args)
