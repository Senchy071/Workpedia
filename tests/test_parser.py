"""Tests for document parser and related components."""

import tempfile
from pathlib import Path

import pytest

from core.analyzer import StructureAnalyzer
from core.large_doc_handler import LargeDocumentHandler
from core.parser import DocumentParser
from core.progress_tracker import ProgressTracker
from core.validator import DocumentValidator
from processors.docx_processor import DOCXProcessor
from processors.html_processor import HTMLProcessor
from processors.image_processor import ImageProcessor
from processors.pdf_processor import PDFProcessor


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
        CHUNK_SIZE_PAGES,
        MAX_FILE_SIZE_MB,
        MAX_PAGES_SINGLE_PASS,
    )

    assert MAX_PAGES_SINGLE_PASS == 100
    assert MAX_FILE_SIZE_MB == 50
    assert CHUNK_SIZE_PAGES == 75


class TestPDFSplitter:
    """Test PDFSplitter class."""

    def test_splitter_initialization(self):
        """Test splitter can be initialized."""
        from core.pdf_splitter import PDFSplitter

        splitter = PDFSplitter()
        assert splitter is not None
        assert splitter.chunk_size == 75

        splitter_custom = PDFSplitter(chunk_size=50)
        assert splitter_custom.chunk_size == 50

    def test_calculate_splits(self):
        """Test split calculation."""
        from core.pdf_splitter import PDFSplitter

        splitter = PDFSplitter(chunk_size=50)

        # Test 150 pages -> 3 splits
        splits = splitter.calculate_splits(150)
        assert len(splits) == 3
        assert splits[0] == (1, 50)
        assert splits[1] == (51, 100)
        assert splits[2] == (101, 150)

        # Test 75 pages -> 2 splits
        splits = splitter.calculate_splits(75)
        assert len(splits) == 2
        assert splits[0] == (1, 50)
        assert splits[1] == (51, 75)

        # Test exact chunk size
        splits = splitter.calculate_splits(50)
        assert len(splits) == 1
        assert splits[0] == (1, 50)


class TestDocumentMerger:
    """Test DocumentMerger class."""

    def test_merger_initialization(self):
        """Test merger can be initialized."""
        from core.doc_merger import DocumentMerger

        merger = DocumentMerger()
        assert merger is not None

    def test_chunk_info_creation(self):
        """Test ChunkInfo dataclass."""
        from core.doc_merger import create_chunk_info

        chunk = create_chunk_info(
            chunk_index=1,
            start_page=1,
            end_page=50,
            parse_result={"raw_text": "test"},
            processing_time=5.0,
            success=True,
        )

        assert chunk.chunk_index == 1
        assert chunk.start_page == 1
        assert chunk.end_page == 50
        assert chunk.success is True
        assert chunk.processing_time == 5.0

    def test_merge_text(self):
        """Test text merging functionality."""
        from core.doc_merger import DocumentMerger, create_chunk_info

        merger = DocumentMerger()

        chunks = [
            create_chunk_info(
                chunk_index=1,
                start_page=1,
                end_page=50,
                parse_result={"raw_text": "Content from chunk 1"},
            ),
            create_chunk_info(
                chunk_index=2,
                start_page=51,
                end_page=100,
                parse_result={"raw_text": "Content from chunk 2"},
            ),
        ]

        merged = merger._merge_text(chunks)

        assert "Content from chunk 1" in merged
        assert "Content from chunk 2" in merged
        assert "Pages 1-50" in merged
        assert "Pages 51-100" in merged


class TestCrossReferenceExtraction:
    """Test cross-reference extraction in StructureAnalyzer."""

    def test_cross_ref_patterns(self):
        """Test cross-reference regex patterns."""

        from core.analyzer import StructureAnalyzer

        analyzer = StructureAnalyzer()

        # Test table references
        table_pattern = analyzer.CROSS_REF_PATTERNS["table"]
        assert table_pattern.search("See Table 1 for details")
        assert table_pattern.search("As shown in Tab. 2.3")
        assert table_pattern.search("Refer to Table 1a")

        # Test figure references
        figure_pattern = analyzer.CROSS_REF_PATTERNS["figure"]
        assert figure_pattern.search("See Figure 1")
        assert figure_pattern.search("As shown in Fig. 2.3")
        assert figure_pattern.search("Figures 1 and 2")

        # Test section references
        section_pattern = analyzer.CROSS_REF_PATTERNS["section"]
        assert section_pattern.search("See Section 1")
        assert section_pattern.search("As described in Section 3.2.1")
        assert section_pattern.search("§ 4.5")

        # Test equation references
        equation_pattern = analyzer.CROSS_REF_PATTERNS["equation"]
        assert equation_pattern.search("See Equation 1")
        assert equation_pattern.search("From Eq. (2)")
        assert equation_pattern.search("Equations 3.1")

    def test_cross_reference_dataclass(self):
        """Test CrossReference dataclass."""
        from core.analyzer import CrossReference

        ref = CrossReference(
            ref_type="table",
            ref_id="1",
            source_page=5,
            source_text="See Table 1 for details",
            target_id="table_1",
        )

        assert ref.ref_type == "table"
        assert ref.ref_id == "1"
        assert ref.source_page == 5


class TestTableStructureExtraction:
    """Test table structure extraction features."""

    def test_parse_table_from_markdown(self):
        """Test parsing table from markdown format."""
        from core.analyzer import StructureAnalyzer

        analyzer = StructureAnalyzer()

        markdown_table = """| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |"""

        rows, cols, headers = analyzer._parse_table_from_text(markdown_table)

        assert rows == 3  # Header + 2 data rows (separator excluded)
        assert cols == 3
        assert headers == ["Name", "Age", "City"]

    def test_parse_table_from_tsv(self):
        """Test parsing table from tab-separated format."""
        from core.analyzer import StructureAnalyzer

        analyzer = StructureAnalyzer()

        tsv_table = "Name\tAge\tCity\nJohn\t30\tNYC\nJane\t25\tLA"

        rows, cols, headers = analyzer._parse_table_from_text(tsv_table)

        assert rows == 3
        assert cols == 3
        assert headers == ["Name", "Age", "City"]

    def test_table_info_dataclass(self):
        """Test TableInfo dataclass with headers."""
        from core.analyzer import TableInfo

        table = TableInfo(
            table_id="table_1",
            page_numbers=[1, 2],
            num_rows=10,
            num_cols=5,
            headers=["Col1", "Col2", "Col3", "Col4", "Col5"],
            bounding_boxes=[],
            is_multi_page=True,
            content="table content",
            header_row_index=0,
        )

        assert table.table_id == "table_1"
        assert len(table.headers) == 5
        assert table.is_multi_page is True
        assert table.num_rows == 10
        assert table.num_cols == 5


class TestMultiPageTableDetection:
    """Test multi-page table detection."""

    def test_detect_multi_page_single_table(self):
        """Test that single tables are not marked as multi-page."""
        from core.analyzer import StructureAnalyzer, TableInfo

        analyzer = StructureAnalyzer()

        tables = [
            TableInfo(
                table_id="table_1",
                page_numbers=[1],
                num_rows=5,
                num_cols=3,
                headers=["A", "B", "C"],
                bounding_boxes=[],
                is_multi_page=False,
                content="",
            )
        ]

        result = analyzer._detect_multi_page_tables(tables)
        assert len(result) == 1
        assert result[0].is_multi_page is False

    def test_detect_multi_page_consecutive_tables(self):
        """Test detection of tables on consecutive pages."""
        from core.analyzer import StructureAnalyzer, TableInfo

        analyzer = StructureAnalyzer()

        tables = [
            TableInfo(
                table_id="table_1",
                page_numbers=[1],
                num_rows=5,
                num_cols=3,
                headers=["A", "B", "C"],
                bounding_boxes=[{"x": 0, "y": 0.8, "width": 1, "height": 0.2}],
                is_multi_page=False,
                content="Row 1",
            ),
            TableInfo(
                table_id="table_2",
                page_numbers=[2],
                num_rows=5,
                num_cols=3,
                headers=["A", "B", "C"],  # Same headers
                bounding_boxes=[{"x": 0, "y": 0.1, "width": 1, "height": 0.2}],
                is_multi_page=False,
                content="Row 6",
            ),
        ]

        result = analyzer._detect_multi_page_tables(tables)

        # Both should be marked as multi-page
        assert result[0].is_multi_page is True
        assert result[1].is_multi_page is True
        # First table should have pages from both
        assert 1 in result[0].page_numbers
        assert 2 in result[0].page_numbers


class TestElementClassification:
    """Test enhanced element classification."""

    def test_classify_equation(self):
        """Test equation element classification."""
        from core.analyzer import StructureAnalyzer

        analyzer = StructureAnalyzer()

        assert analyzer._classify_element("formula") == "equation"
        assert analyzer._classify_element("equation-block") == "equation"

    def test_classify_caption(self):
        """Test caption element classification."""
        from core.analyzer import StructureAnalyzer

        analyzer = StructureAnalyzer()

        assert analyzer._classify_element("caption") == "caption"
        assert analyzer._classify_element("figure-caption") == "caption"

    def test_classify_picture(self):
        """Test picture element classification."""
        from core.analyzer import StructureAnalyzer

        analyzer = StructureAnalyzer()

        assert analyzer._classify_element("picture") == "figure"
        assert analyzer._classify_element("image") == "figure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
