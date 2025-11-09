"""Tests for document parser and related components."""

import pytest
from pathlib import Path
import tempfile
import os

from core.parser import DocumentParser
from core.large_doc_handler import LargeDocumentHandler
from core.analyzer import StructureAnalyzer
from core.validator import DocumentValidator
from core.progress_tracker import ProgressTracker
from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DOCXProcessor
from processors.html_processor import HTMLProcessor
from processors.image_processor import ImageProcessor


class TestDocumentParser:
    """Test DocumentParser class."""

    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = DocumentParser()
        assert parser is not None
        assert parser._converter is None  # Lazy initialization
        assert parser._docs_processed == 0

    def test_parser_cleanup(self):
        """Test parser cleanup."""
        parser = DocumentParser()
        parser.cleanup()
        assert parser._converter is None
        assert parser._docs_processed == 0


class TestLargeDocumentHandler:
    """Test LargeDocumentHandler class."""

    def test_handler_initialization(self):
        """Test handler can be initialized."""
        handler = LargeDocumentHandler()
        assert handler is not None
        assert handler.max_pages == 100
        assert handler.max_size_mb == 50
        assert handler.chunk_size_pages == 75

    def test_calculate_chunks(self):
        """Test chunk calculation."""
        handler = LargeDocumentHandler(chunk_size_pages=50)

        # Test 150 pages -> 3 chunks
        chunks = handler.calculate_chunks(150)
        assert len(chunks) == 3
        assert chunks[0] == (1, 50)
        assert chunks[1] == (51, 100)
        assert chunks[2] == (101, 150)

        # Test 175 pages -> 4 chunks (last chunk smaller)
        chunks = handler.calculate_chunks(175)
        assert len(chunks) == 4
        assert chunks[3] == (151, 175)

    def test_is_large_document_small_file(self):
        """Test detection of small documents."""
        handler = LargeDocumentHandler()

        # Create a small temporary file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Small test file")
            temp_path = Path(f.name)

        try:
            is_large, info = handler.is_large_document(temp_path)
            assert not is_large
            assert info["file_size_mb"] < 1
        finally:
            temp_path.unlink()


class TestStructureAnalyzer:
    """Test StructureAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer can be initialized."""
        analyzer = StructureAnalyzer()
        assert analyzer is not None

    def test_element_classification(self):
        """Test element type classification."""
        analyzer = StructureAnalyzer()

        assert analyzer._classify_element("heading-1") == "heading"
        assert analyzer._classify_element("table") == "table"
        assert analyzer._classify_element("figure") == "figure"
        assert analyzer._classify_element("list-item") == "list"
        assert analyzer._classify_element("code-block") == "code"
        assert analyzer._classify_element("paragraph") == "text"

    def test_infer_heading_level(self):
        """Test heading level inference."""
        analyzer = StructureAnalyzer()

        # Create mock items with labels
        class MockItem:
            def __init__(self, label):
                self.label = label

        assert analyzer._infer_heading_level(MockItem("h1")) == 1
        assert analyzer._infer_heading_level(MockItem("h2")) == 2
        assert analyzer._infer_heading_level(MockItem("h3")) == 3
        assert analyzer._infer_heading_level(MockItem("heading-level2")) == 2


class TestDocumentValidator:
    """Test DocumentValidator class."""

    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = DocumentValidator()
        assert validator is not None
        assert validator.require_text is True
        assert validator.min_text_length == 10

    def test_validate_minimal_valid_result(self):
        """Test validation of minimal valid result."""
        validator = DocumentValidator(require_text=False)

        parse_result = {
            "doc_id": "test-123",
            "metadata": {
                "filename": "test.pdf",
                "file_size_mb": 1.5,
                "processing_time_seconds": 2.3,
                "parsed_at": "2025-01-01T00:00:00",
            },
            "structure": {"pages": 10},
        }

        report = validator.validate(parse_result)
        assert report.is_valid
        assert report.summary["errors"] == 0

    def test_validate_missing_metadata(self):
        """Test validation catches missing metadata."""
        validator = DocumentValidator(require_text=False)

        parse_result = {
            "doc_id": "test-123",
            "structure": {"pages": 10},
            # Missing metadata
        }

        report = validator.validate(parse_result)
        assert not report.is_valid
        assert report.summary["errors"] > 0

    def test_validate_missing_text_content(self):
        """Test validation catches missing text."""
        validator = DocumentValidator(require_text=True)

        parse_result = {
            "doc_id": "test-123",
            "metadata": {
                "filename": "test.pdf",
                "file_size_mb": 1.5,
                "processing_time_seconds": 2.3,
                "parsed_at": "2025-01-01T00:00:00",
            },
            "structure": {"pages": 10},
            "raw_text": "",  # Empty text
        }

        report = validator.validate(parse_result)
        assert not report.is_valid
        assert report.summary["errors"] > 0


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_tracker_initialization(self):
        """Test tracker can be initialized."""
        tracker = ProgressTracker("test.pdf", total_pages=100)
        assert tracker.document_name == "test.pdf"
        assert tracker.total_pages == 100
        assert tracker.current_page == 0

    def test_stage_lifecycle(self):
        """Test stage start/complete lifecycle."""
        tracker = ProgressTracker("test.pdf")

        # Start stage
        tracker.start_stage("parsing", {"detail": "test"})
        assert "parsing" in tracker.stages
        assert tracker.current_stage == "parsing"
        assert tracker.stages["parsing"].status == "in_progress"

        # Complete stage
        tracker.complete_stage("parsing")
        assert tracker.stages["parsing"].status == "completed"
        assert tracker.stages["parsing"].duration_seconds is not None

    def test_stage_failure(self):
        """Test stage failure handling."""
        tracker = ProgressTracker("test.pdf")

        tracker.start_stage("parsing")
        tracker.fail_stage("parsing", "Test error")

        assert tracker.stages["parsing"].status == "failed"
        assert len(tracker.errors) == 1

    def test_page_progress(self):
        """Test page progress tracking."""
        tracker = ProgressTracker("test.pdf", total_pages=100)

        tracker.update_page_progress(25)
        assert tracker.current_page == 25

        status = tracker.get_status()
        assert status["progress_pct"] == 25.0

    def test_time_estimation(self):
        """Test time estimation."""
        import time

        tracker = ProgressTracker("test.pdf", total_pages=100)

        # Simulate processing
        tracker.update_page_progress(50)
        time.sleep(0.1)  # Small delay

        eta = tracker.estimated_time_remaining
        assert eta is not None
        assert eta >= 0


class TestProcessors:
    """Test format-specific processors."""

    def test_pdf_processor_initialization(self):
        """Test PDF processor initialization."""
        processor = PDFProcessor()
        assert processor is not None
        assert processor.enable_table_structure is True

    def test_docx_processor_initialization(self):
        """Test DOCX processor initialization."""
        processor = DOCXProcessor()
        assert processor is not None

    def test_html_processor_initialization(self):
        """Test HTML processor initialization."""
        processor = HTMLProcessor()
        assert processor is not None

    def test_image_processor_initialization(self):
        """Test image processor initialization."""
        processor = ImageProcessor()
        assert processor is not None
        assert processor.enable_ocr is True

    def test_image_processor_no_ocr(self):
        """Test image processor without OCR."""
        processor = ImageProcessor(enable_ocr=False)
        assert processor.enable_ocr is False


def test_config_imports():
    """Test that configuration can be imported."""
    from config.config import (
        MAX_PAGES_SINGLE_PASS,
        MAX_FILE_SIZE_MB,
        CHUNK_SIZE_PAGES,
    )

    assert MAX_PAGES_SINGLE_PASS == 100
    assert MAX_FILE_SIZE_MB == 50
    assert CHUNK_SIZE_PAGES == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
