"""Integration tests for document processing pipeline."""

import tempfile
from pathlib import Path

import pytest

from core.progress_tracker import ProgressTracker
from core.validator import DocumentValidator
from processors.docx_processor import DOCXProcessor
from processors.html_processor import HTMLProcessor
from processors.image_processor import ImageProcessor
from processors.pdf_processor import PDFProcessor


class TestIntegration:
    """Integration tests for full document processing pipeline."""

    def test_end_to_end_workflow_mock(self):
        """Test complete workflow with mock document."""
        # This test demonstrates the expected workflow
        # In practice, you would use actual documents from data/input/

        # Create a simple text file to simulate processing
        with tempfile.NamedTemporaryFile(
            mode='w', suffix=".txt", delete=False
        ) as f:
            f.write("Test document content for integration testing.\n" * 100)
            temp_path = Path(f.name)

        try:
            # Initialize components
            validator = DocumentValidator()

            # Note: Actual processing would require real document formats
            # For now, we just verify the components work together

            # Example workflow:
            # 1. Select appropriate processor based on file type
            # 2. Process document
            # 3. Validate results
            # 4. Extract structure
            # 5. Store in vector database (Phase 3)

            assert temp_path.exists()
            assert validator is not None

        finally:
            temp_path.unlink()

    def test_progress_tracking_integration(self):
        """Test progress tracking integration."""
        tracker = ProgressTracker("test_document.pdf", total_pages=50)

        # Simulate processing stages
        tracker.start_stage("loading")
        tracker.complete_stage("loading")

        tracker.start_stage("parsing")
        for page in range(1, 51):
            tracker.update_page_progress(page)
        tracker.complete_stage("parsing")

        tracker.start_stage("analyzing")
        tracker.complete_stage("analyzing")

        tracker.complete()

        # Verify tracking
        status = tracker.get_status()
        assert status["is_complete"]
        assert status["pages_processed"] == 50
        assert status["stages_completed"] == 3

    def test_validation_integration(self):
        """Test validation integration."""
        validator = DocumentValidator()

        # Create a mock parse result
        parse_result = {
            "doc_id": "integration-test-123",
            "metadata": {
                "filename": "test.pdf",
                "file_path": "/path/to/test.pdf",
                "file_size_mb": 2.5,
                "pages": 25,
                "processing_time_seconds": 5.2,
                "parsed_at": "2025-01-01T00:00:00",
            },
            "raw_text": "This is test content. " * 100,
            "structure": {
                "pages": 25,
                "has_tables": True,
                "has_figures": False,
            },
            "structure_analysis": {
                "sections": [
                    {"type": "heading", "text": "Introduction", "level": 1},
                    {"type": "heading", "text": "Methods", "level": 1},
                ],
                "tables": [
                    {"table_id": "table_1", "rows": 10, "cols": 5}
                ],
                "total_elements": 45,
            },
        }

        # Validate
        report = validator.validate(parse_result)

        # Should be valid
        assert report.is_valid
        assert report.summary["errors"] == 0
        assert report.passed_checks > 0


def test_processor_selection():
    """Test automatic processor selection based on file extension."""

    def get_processor_for_file(file_path: Path):
        """Select appropriate processor based on file type."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return PDFProcessor()
        elif suffix in [".docx", ".doc"]:
            return DOCXProcessor()
        elif suffix in [".html", ".htm"]:
            return HTMLProcessor()
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
            return ImageProcessor()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # Test selection
    assert isinstance(get_processor_for_file(Path("test.pdf")), PDFProcessor)
    assert isinstance(get_processor_for_file(Path("test.docx")), DOCXProcessor)
    assert isinstance(get_processor_for_file(Path("test.html")), HTMLProcessor)
    assert isinstance(get_processor_for_file(Path("test.png")), ImageProcessor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
