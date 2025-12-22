"""Tests for CSV/TSV document processor."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from processors.csv_processor import CSVProcessor


class TestCSVProcessor:
    """Tests for CSVProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create CSVProcessor instance."""
        return CSVProcessor()

    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.csv"

            df = pd.DataFrame({
                "Name": ["Alice", "Bob", "Charlie"],
                "Age": [25, 30, 35],
                "City": ["New York", "Los Angeles", "Chicago"],
                "Salary": [50000.0, 60000.0, 70000.0],
            })

            df.to_csv(file_path, index=False)
            yield file_path

    @pytest.fixture
    def sample_tsv(self):
        """Create a sample TSV file for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.tsv"

            df = pd.DataFrame({
                "Product": ["Widget", "Gadget", "Gizmo"],
                "Price": [9.99, 19.99, 29.99],
                "Quantity": [100, 50, 75],
            })

            df.to_csv(file_path, sep="\t", index=False)
            yield file_path

    @pytest.fixture
    def large_csv(self):
        """Create a large CSV file for chunking tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "large_data.csv"

            df = pd.DataFrame({
                "ID": range(1, 251),
                "Name": [f"Item_{i}" for i in range(1, 251)],
                "Value": [i * 1.5 for i in range(1, 251)],
            })

            df.to_csv(file_path, index=False)
            yield file_path

    @pytest.fixture
    def csv_with_special_chars(self):
        """Create a CSV with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "special_chars.csv"

            df = pd.DataFrame({
                "Name": ["Müller", "O'Brien", "García"],
                "Description": ["Über good", "It's great", "¡Fantástico!"],
            })

            df.to_csv(file_path, index=False, encoding="utf-8")
            yield file_path

    # =========================================================================
    # Basic Processing Tests
    # =========================================================================

    def test_process_csv(self, processor, sample_csv):
        """Test processing a basic CSV file."""
        result = processor.process(sample_csv)

        assert result is not None
        assert "doc_id" in result
        assert "metadata" in result
        assert "raw_text" in result
        assert "structure" in result

        # Check metadata
        assert result["metadata"]["rows"] == 3
        assert result["metadata"]["cols"] == 4
        assert result["metadata"]["filename"] == "test_data.csv"
        assert result["metadata"]["format"] == ".csv"
        assert result["metadata"]["delimiter"] == ","

    def test_process_tsv(self, processor, sample_tsv):
        """Test processing a TSV file."""
        result = processor.process(sample_tsv)

        assert result["metadata"]["rows"] == 3
        assert result["metadata"]["cols"] == 3
        assert result["metadata"]["format"] == ".tsv"
        assert result["metadata"]["delimiter"] == "\t"

    def test_process_creates_unique_doc_id(self, processor, sample_csv):
        """Test that processing creates a unique doc_id."""
        result1 = processor.process(sample_csv)
        result2 = processor.process(sample_csv)

        assert result1["doc_id"] != result2["doc_id"]

    def test_docling_document_is_none(self, processor, sample_csv):
        """Test that docling_document is None for CSV."""
        result = processor.process(sample_csv)
        assert result["docling_document"] is None

    # =========================================================================
    # Content Extraction Tests
    # =========================================================================

    def test_raw_text_contains_data(self, processor, sample_csv):
        """Test that raw_text contains the CSV data."""
        result = processor.process(sample_csv)

        assert "Alice" in result["raw_text"]
        assert "Bob" in result["raw_text"]
        assert "New York" in result["raw_text"]
        assert "50000" in result["raw_text"]

    def test_markdown_table_format(self, processor, sample_csv):
        """Test that content is formatted as markdown table."""
        result = processor.process(sample_csv)

        # Markdown tables use | as column separator
        assert "|" in result["raw_text"]

        # Should have table header
        assert "## Table:" in result["raw_text"]

    def test_special_characters_handled(self, processor, csv_with_special_chars):
        """Test that special characters are handled correctly."""
        result = processor.process(csv_with_special_chars)

        assert "Müller" in result["raw_text"]
        assert "O'Brien" in result["raw_text"]
        assert "García" in result["raw_text"]

    # =========================================================================
    # Large File Chunking Tests
    # =========================================================================

    def test_large_file_chunking(self, large_csv):
        """Test that large files are chunked."""
        processor = CSVProcessor(max_rows_per_chunk=100)
        result = processor.process(large_csv)

        raw_text = result["raw_text"]
        assert "Chunk 1/3" in raw_text
        assert "Chunk 2/3" in raw_text
        assert "Chunk 3/3" in raw_text

    def test_small_file_no_chunking(self, processor, sample_csv):
        """Test that small files are not chunked."""
        result = processor.process(sample_csv)

        assert "Chunk" not in result["raw_text"]

    # =========================================================================
    # Metadata Tests
    # =========================================================================

    def test_metadata_fields(self, processor, sample_csv):
        """Test that all required metadata fields are present."""
        result = processor.process(sample_csv)
        metadata = result["metadata"]

        assert "filename" in metadata
        assert "file_path" in metadata
        assert "file_size_mb" in metadata
        assert "rows" in metadata
        assert "cols" in metadata
        assert "delimiter" in metadata
        assert "encoding" in metadata
        assert "processing_time_seconds" in metadata
        assert "parsed_at" in metadata
        assert "parser_version" in metadata
        assert "format" in metadata

    def test_custom_metadata_merged(self, processor, sample_csv):
        """Test that custom metadata is merged."""
        custom_meta = {"source": "test", "version": "1.0"}
        result = processor.process(sample_csv, metadata=custom_meta)

        assert result["metadata"]["source"] == "test"
        assert result["metadata"]["version"] == "1.0"
        assert result["metadata"]["rows"] == 3  # Original metadata preserved

    # =========================================================================
    # Structure Tests
    # =========================================================================

    def test_structure_has_tables(self, processor, sample_csv):
        """Test that structure marks has_tables as True."""
        result = processor.process(sample_csv)

        assert result["structure"]["has_tables"] is True
        assert result["structure"]["has_figures"] is False

    def test_structure_columns_info(self, processor, sample_csv):
        """Test that structure contains column information."""
        result = processor.process(sample_csv)

        columns = result["structure"]["columns"]
        assert len(columns) == 4

        # Find Name column
        name_col = next(c for c in columns if c["name"] == "Name")
        assert name_col["dtype"] == "object"
        assert name_col["non_null"] == 3
        assert name_col["sample"] == "Alice"

    def test_structure_column_names(self, processor, sample_csv):
        """Test that column names are in structure."""
        result = processor.process(sample_csv)

        assert result["structure"]["column_names"] == ["Name", "Age", "City", "Salary"]

    def test_tables_in_structure(self, processor, sample_csv):
        """Test that tables array has correct info."""
        result = processor.process(sample_csv)
        tables = result["structure"]["tables"]

        assert len(tables) == 1
        assert tables[0]["table_id"] == "main"
        assert tables[0]["rows"] == 3
        assert tables[0]["cols"] == 4

    # =========================================================================
    # Delimiter Tests
    # =========================================================================

    def test_custom_delimiter(self):
        """Test using custom delimiter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "semicolon.csv"

            # Create file with semicolon delimiter
            with open(file_path, "w") as f:
                f.write("Name;Age;City\n")
                f.write("Alice;25;Boston\n")

            processor = CSVProcessor()
            result = processor.process(file_path, delimiter=";")

            assert "Alice" in result["raw_text"]
            assert "Boston" in result["raw_text"]
            assert result["metadata"]["delimiter"] == ";"

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_invalid_extension_raises_error(self, processor):
        """Test that non-CSV files raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            with pytest.raises(ValueError, match="Expected CSV/TSV file"):
                processor.process(f.name)

    def test_file_not_found_raises_error(self, processor):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            processor.process("/nonexistent/file.csv")

    # =========================================================================
    # get_dataframe Tests
    # =========================================================================

    def test_get_dataframe(self, processor, sample_csv):
        """Test getting raw DataFrame from CSV."""
        df = processor.get_dataframe(sample_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Name" in df.columns
        assert df["Name"].iloc[0] == "Alice"

    def test_get_dataframe_tsv(self, processor, sample_tsv):
        """Test getting raw DataFrame from TSV."""
        df = processor.get_dataframe(sample_tsv)

        assert isinstance(df, pd.DataFrame)
        assert "Product" in df.columns
        assert "Widget" in df["Product"].values

    # =========================================================================
    # Cleanup Tests
    # =========================================================================

    def test_cleanup_method_exists(self, processor):
        """Test that cleanup method exists and doesn't raise."""
        processor.cleanup()  # Should not raise


class TestCSVProcessorIntegration:
    """Integration tests for CSVProcessor with DocumentIndexer."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_csv(self, temp_dirs):
        """Create a sample CSV file for integration testing."""
        file_path = temp_dirs / "integration_test.csv"

        df = pd.DataFrame({
            "Question": ["What is Python?", "What is RAG?", "What is Workpedia?"],
            "Answer": [
                "Python is a programming language",
                "RAG is Retrieval-Augmented Generation",
                "Workpedia is a document Q&A system",
            ],
        })

        df.to_csv(file_path, index=False)
        return file_path

    def test_csv_output_compatible_with_indexer(self, sample_csv):
        """Test that CSV processor output is compatible with DocumentIndexer."""
        processor = CSVProcessor()
        result = processor.process(sample_csv)

        # Check required fields for DocumentIndexer
        assert "doc_id" in result
        assert "metadata" in result
        assert "raw_text" in result
        assert "structure" in result

        # Check metadata has required fields
        assert "filename" in result["metadata"]

        # raw_text should be non-empty
        assert len(result["raw_text"]) > 0

    def test_csv_chunking_compatibility(self, sample_csv):
        """Test that CSV output can be chunked by SemanticChunker."""
        from core.chunker import SemanticChunker

        processor = CSVProcessor()
        result = processor.process(sample_csv)

        chunker = SemanticChunker()
        chunks = chunker.chunk_document(result)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == result["doc_id"]
            assert len(chunk.content) > 0

    def test_csv_full_indexing_pipeline(self, sample_csv, temp_dirs):
        """Test full indexing pipeline with CSV file."""
        from core.chunker import SemanticChunker
        from core.embedder import Embedder
        from storage.vector_store import VectorStore

        # Process CSV
        processor = CSVProcessor()
        parsed_doc = processor.process(sample_csv)

        # Chunk
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(parsed_doc)

        # Embed
        embedder = Embedder()
        embeddings = embedder.embed_chunks(chunks)

        # Store
        vector_store = VectorStore(
            persist_directory=str(temp_dirs / "chroma"),
            collection_name="test_csv",
        )
        added = vector_store.add_chunks(chunks, embeddings)

        assert added == len(chunks)

        # Query
        query_embedding = embedder.embed("What is Python?")
        results = vector_store.query(query_embedding, n_results=3)

        assert len(results["documents"]) > 0
        # Should find Python-related content
        assert any("Python" in doc for doc in results["documents"])
