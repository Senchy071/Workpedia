"""Tests for XLSX/XLS document processor."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from processors.xlsx_processor import XLSXProcessor


class TestXLSXProcessor:
    """Tests for XLSXProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create XLSXProcessor instance."""
        return XLSXProcessor()

    @pytest.fixture
    def sample_xlsx(self):
        """Create a sample XLSX file for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.xlsx"

            # Create sample data
            df1 = pd.DataFrame({
                "Name": ["Alice", "Bob", "Charlie"],
                "Age": [25, 30, 35],
                "City": ["New York", "Los Angeles", "Chicago"],
            })

            df2 = pd.DataFrame({
                "Product": ["Widget", "Gadget", "Gizmo"],
                "Price": [9.99, 19.99, 29.99],
                "Quantity": [100, 50, 75],
            })

            # Write to Excel with multiple sheets
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df1.to_excel(writer, sheet_name="Employees", index=False)
                df2.to_excel(writer, sheet_name="Products", index=False)

            yield file_path

    @pytest.fixture
    def single_sheet_xlsx(self):
        """Create a single-sheet XLSX file for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "single_sheet.xlsx"

            df = pd.DataFrame({
                "ID": [1, 2, 3, 4, 5],
                "Value": ["A", "B", "C", "D", "E"],
            })

            df.to_excel(file_path, sheet_name="Data", index=False, engine="openpyxl")
            yield file_path

    @pytest.fixture
    def empty_sheet_xlsx(self):
        """Create an XLSX file with an empty sheet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "empty_sheet.xlsx"

            df1 = pd.DataFrame({"Col1": [1, 2, 3]})
            df2 = pd.DataFrame()  # Empty dataframe

            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df1.to_excel(writer, sheet_name="HasData", index=False)
                df2.to_excel(writer, sheet_name="Empty", index=False)

            yield file_path

    @pytest.fixture
    def large_sheet_xlsx(self):
        """Create an XLSX file with many rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "large_sheet.xlsx"

            # Create 250 rows
            df = pd.DataFrame({
                "ID": range(1, 251),
                "Name": [f"Item_{i}" for i in range(1, 251)],
                "Value": [i * 1.5 for i in range(1, 251)],
            })

            df.to_excel(file_path, sheet_name="LargeData", index=False, engine="openpyxl")
            yield file_path

    # =========================================================================
    # Basic Processing Tests
    # =========================================================================

    def test_process_multi_sheet_xlsx(self, processor, sample_xlsx):
        """Test processing XLSX with multiple sheets."""
        result = processor.process(sample_xlsx)

        assert result is not None
        assert "doc_id" in result
        assert "metadata" in result
        assert "raw_text" in result
        assert "structure" in result

        # Check metadata
        assert result["metadata"]["sheet_count"] == 2
        assert result["metadata"]["filename"] == "test_data.xlsx"
        assert result["metadata"]["format"] == ".xlsx"

        # Check structure
        assert len(result["structure"]["sheets"]) == 2
        assert "Employees" in result["structure"]["sheet_names"]
        assert "Products" in result["structure"]["sheet_names"]

    def test_process_single_sheet_xlsx(self, processor, single_sheet_xlsx):
        """Test processing XLSX with single sheet."""
        result = processor.process(single_sheet_xlsx)

        assert result["metadata"]["sheet_count"] == 1
        assert result["structure"]["sheet_names"] == ["Data"]

        # Check sheet info
        sheet = result["structure"]["sheets"][0]
        assert sheet["name"] == "Data"
        assert sheet["rows"] == 5
        assert sheet["cols"] == 2
        assert "ID" in sheet["columns"]
        assert "Value" in sheet["columns"]

    def test_process_creates_unique_doc_id(self, processor, sample_xlsx):
        """Test that processing creates a unique doc_id."""
        result1 = processor.process(sample_xlsx)
        result2 = processor.process(sample_xlsx)

        assert result1["doc_id"] != result2["doc_id"]

    def test_docling_document_is_none(self, processor, sample_xlsx):
        """Test that docling_document is None for XLSX."""
        result = processor.process(sample_xlsx)
        assert result["docling_document"] is None

    # =========================================================================
    # Content Extraction Tests
    # =========================================================================

    def test_raw_text_contains_data(self, processor, sample_xlsx):
        """Test that raw_text contains the spreadsheet data."""
        result = processor.process(sample_xlsx)

        # Check employee data is present
        assert "Alice" in result["raw_text"]
        assert "Bob" in result["raw_text"]
        assert "New York" in result["raw_text"]

        # Check product data is present
        assert "Widget" in result["raw_text"]
        assert "Gadget" in result["raw_text"]

    def test_markdown_table_format(self, processor, sample_xlsx):
        """Test that content is formatted as markdown tables."""
        result = processor.process(sample_xlsx)

        # Markdown tables use | as column separator
        assert "|" in result["raw_text"]

        # Should have sheet headers
        assert "## Sheet:" in result["raw_text"]

    def test_sheet_content_preserved(self, processor, sample_xlsx):
        """Test that each sheet's content is in the result."""
        result = processor.process(sample_xlsx)

        for sheet in result["structure"]["sheets"]:
            assert sheet["content"] in result["raw_text"]

    # =========================================================================
    # Empty Sheet Handling Tests
    # =========================================================================

    def test_empty_sheet_skipped(self, processor, empty_sheet_xlsx):
        """Test that empty sheets are skipped."""
        result = processor.process(empty_sheet_xlsx)

        # Only the non-empty sheet should be in results
        assert result["metadata"]["sheet_count"] == 1
        assert result["structure"]["sheet_names"] == ["HasData"]

    # =========================================================================
    # Large Sheet Chunking Tests
    # =========================================================================

    def test_large_sheet_chunking(self, large_sheet_xlsx):
        """Test that large sheets are chunked."""
        processor = XLSXProcessor(max_rows_per_chunk=100)
        result = processor.process(large_sheet_xlsx)

        # With 250 rows and 100 rows per chunk, should have 3 chunks
        raw_text = result["raw_text"]
        assert "Chunk 1/3" in raw_text
        assert "Chunk 2/3" in raw_text
        assert "Chunk 3/3" in raw_text

    def test_small_sheet_no_chunking(self, processor, single_sheet_xlsx):
        """Test that small sheets are not chunked."""
        result = processor.process(single_sheet_xlsx)

        # Should not have chunk markers
        assert "Chunk" not in result["raw_text"]

    # =========================================================================
    # Metadata Tests
    # =========================================================================

    def test_metadata_fields(self, processor, sample_xlsx):
        """Test that all required metadata fields are present."""
        result = processor.process(sample_xlsx)
        metadata = result["metadata"]

        assert "filename" in metadata
        assert "file_path" in metadata
        assert "file_size_mb" in metadata
        assert "sheet_count" in metadata
        assert "total_rows" in metadata
        assert "total_cols" in metadata
        assert "processing_time_seconds" in metadata
        assert "parsed_at" in metadata
        assert "parser_version" in metadata
        assert "format" in metadata

    def test_total_rows_cols_calculated(self, processor, sample_xlsx):
        """Test that total rows and cols are calculated correctly."""
        result = processor.process(sample_xlsx)

        # 3 rows in Employees + 3 rows in Products = 6 total rows
        assert result["metadata"]["total_rows"] == 6

        # Max cols is 3 (both sheets have 3 columns)
        assert result["metadata"]["total_cols"] == 3

    def test_custom_metadata_merged(self, processor, sample_xlsx):
        """Test that custom metadata is merged."""
        custom_meta = {"source": "test", "version": "1.0"}
        result = processor.process(sample_xlsx, metadata=custom_meta)

        assert result["metadata"]["source"] == "test"
        assert result["metadata"]["version"] == "1.0"
        # Original metadata should still be present
        assert result["metadata"]["sheet_count"] == 2

    # =========================================================================
    # Structure Tests
    # =========================================================================

    def test_structure_has_tables(self, processor, sample_xlsx):
        """Test that structure marks has_tables as True."""
        result = processor.process(sample_xlsx)

        assert result["structure"]["has_tables"] is True
        assert result["structure"]["has_figures"] is False

    def test_tables_in_structure(self, processor, sample_xlsx):
        """Test that tables array has correct info."""
        result = processor.process(sample_xlsx)
        tables = result["structure"]["tables"]

        assert len(tables) == 2

        # Find Employees table
        emp_table = next(t for t in tables if t["name"] == "Employees")
        assert emp_table["rows"] == 3
        assert emp_table["cols"] == 3
        assert "Name" in emp_table["columns"]

    def test_sheet_preview(self, processor, sample_xlsx):
        """Test that sheets have preview data."""
        result = processor.process(sample_xlsx)

        for sheet in result["structure"]["sheets"]:
            assert "preview" in sheet
            assert len(sheet["preview"]) > 0

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_invalid_extension_raises_error(self, processor):
        """Test that non-Excel files raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            with pytest.raises(ValueError, match="Expected Excel file"):
                processor.process(f.name)

    def test_file_not_found_raises_error(self, processor):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            processor.process("/nonexistent/file.xlsx")

    # =========================================================================
    # get_sheet_data Tests
    # =========================================================================

    def test_get_sheet_data_default(self, processor, sample_xlsx):
        """Test getting raw DataFrame for default (first) sheet."""
        df = processor.get_sheet_data(sample_xlsx)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Name" in df.columns

    def test_get_sheet_data_by_name(self, processor, sample_xlsx):
        """Test getting raw DataFrame for specific sheet."""
        df = processor.get_sheet_data(sample_xlsx, sheet_name="Products")

        assert isinstance(df, pd.DataFrame)
        assert "Product" in df.columns
        assert "Widget" in df["Product"].values

    # =========================================================================
    # Cleanup Tests
    # =========================================================================

    def test_cleanup_method_exists(self, processor):
        """Test that cleanup method exists and doesn't raise."""
        processor.cleanup()  # Should not raise


class TestXLSXProcessorIntegration:
    """Integration tests for XLSXProcessor with DocumentIndexer."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_xlsx(self, temp_dirs):
        """Create a sample XLSX file for integration testing."""
        file_path = temp_dirs / "integration_test.xlsx"

        df = pd.DataFrame({
            "Question": ["What is Python?", "What is RAG?", "What is Workpedia?"],
            "Answer": [
                "Python is a programming language",
                "RAG is Retrieval-Augmented Generation",
                "Workpedia is a document Q&A system",
            ],
        })

        df.to_excel(file_path, sheet_name="FAQ", index=False, engine="openpyxl")
        return file_path

    def test_xlsx_output_compatible_with_indexer(self, sample_xlsx):
        """Test that XLSX processor output is compatible with DocumentIndexer."""
        processor = XLSXProcessor()
        result = processor.process(sample_xlsx)

        # Check required fields for DocumentIndexer
        assert "doc_id" in result
        assert "metadata" in result
        assert "raw_text" in result
        assert "structure" in result

        # Check metadata has required fields
        assert "filename" in result["metadata"]

        # raw_text should be non-empty
        assert len(result["raw_text"]) > 0

    def test_xlsx_chunking_compatibility(self, sample_xlsx, temp_dirs):
        """Test that XLSX output can be chunked by SemanticChunker."""
        from core.chunker import SemanticChunker

        processor = XLSXProcessor()
        result = processor.process(sample_xlsx)

        chunker = SemanticChunker()
        chunks = chunker.chunk_document(result)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == result["doc_id"]
            assert len(chunk.content) > 0

    def test_xlsx_full_indexing_pipeline(self, sample_xlsx, temp_dirs):
        """Test full indexing pipeline with XLSX file."""
        from core.chunker import SemanticChunker
        from core.embedder import Embedder
        from storage.vector_store import VectorStore

        # Process XLSX
        processor = XLSXProcessor()
        parsed_doc = processor.process(sample_xlsx)

        # Chunk
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(parsed_doc)

        # Embed
        embedder = Embedder()
        embeddings = embedder.embed_chunks(chunks)

        # Store
        vector_store = VectorStore(
            persist_directory=str(temp_dirs / "chroma"),
            collection_name="test_xlsx",
        )
        added = vector_store.add_chunks(chunks, embeddings)

        assert added == len(chunks)

        # Query
        query_embedding = embedder.embed("What is Python?")
        results = vector_store.query(query_embedding, n_results=3)

        assert len(results["documents"]) > 0
        # Should find Python-related content
        assert any("Python" in doc for doc in results["documents"])
