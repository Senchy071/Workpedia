"""CSV/TSV-specific document processor for delimited text files."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CSVProcessor:
    """
    CSV/TSV-specific processor for delimited text files.

    Features:
    - CSV and TSV support with automatic delimiter detection
    - Markdown table conversion for chunking compatibility
    - Large file chunking by row count
    - Column metadata extraction
    - Encoding detection and handling
    """

    def __init__(
        self,
        max_rows_per_chunk: int = 100,
        encoding: Optional[str] = None,
    ):
        """
        Initialize CSV processor.

        Args:
            max_rows_per_chunk: Maximum rows per chunk for large files
            encoding: File encoding (None for auto-detect)
        """
        self.max_rows_per_chunk = max_rows_per_chunk
        self.encoding = encoding
        logger.info(
            f"CSVProcessor initialized: max_rows_per_chunk={max_rows_per_chunk}, "
            f"encoding={encoding or 'auto'}"
        )

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        delimiter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process CSV/TSV file.

        Args:
            file_path: Path to CSV/TSV file
            metadata: Optional additional metadata
            delimiter: Column delimiter (None for auto-detect based on extension)

        Returns:
            Processing result with parsed document and structure
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in [".csv", ".tsv"]:
            raise ValueError(f"Expected CSV/TSV file, got: {file_path.suffix}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing CSV: {file_path.name}")
        start_time = datetime.now()

        # Get file stats
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Determine delimiter
        if delimiter is None:
            delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","

        # Read CSV file
        try:
            # Try common encodings
            encodings_to_try = [self.encoding] if self.encoding else ["utf-8", "latin-1", "cp1252"]

            df = None
            used_encoding = None

            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(
                        file_path,
                        delimiter=delimiter,
                        encoding=enc,
                        on_bad_lines="warn",
                    )
                    used_encoding = enc
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError(f"Could not decode file with encodings: {encodings_to_try}")

            # Get dimensions
            num_rows = len(df)
            num_cols = len(df.columns)

            # Convert to markdown
            markdown_content = self._dataframe_to_markdown(df, file_path.stem)

            # Build column info
            columns_info = [
                {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "sample": str(df[col].iloc[0]) if len(df) > 0 else None,
                }
                for col in df.columns
            ]

        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            raise

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Build metadata
        doc_metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size_mb": round(file_size_mb, 2),
            "rows": num_rows,
            "cols": num_cols,
            "delimiter": delimiter,
            "encoding": used_encoding,
            "processing_time_seconds": round(processing_time, 2),
            "parsed_at": datetime.now().isoformat(),
            "parser_version": "csv_processor_v1",
            "format": file_path.suffix.lower(),
        }

        # Add custom metadata if provided
        if metadata:
            doc_metadata.update(metadata)

        # Build structure info
        structure = {
            "columns": columns_info,
            "column_names": [str(col) for col in df.columns],
            "has_tables": True,  # CSV is inherently tabular
            "has_figures": False,
            "tables": [
                {
                    "table_id": "main",
                    "name": file_path.stem,
                    "rows": num_rows,
                    "cols": num_cols,
                    "columns": [str(col) for col in df.columns],
                }
            ],
        }

        logger.info(
            f"CSV processing complete: {num_rows} rows, {num_cols} cols in {processing_time:.2f}s"
        )

        return {
            "doc_id": doc_id,
            "metadata": doc_metadata,
            "raw_text": markdown_content,
            "structure": structure,
            "docling_document": None,  # Not using Docling for CSV
        }

    def _dataframe_to_markdown(self, df: pd.DataFrame, name: str) -> str:
        """
        Convert DataFrame to markdown format.

        Args:
            df: Pandas DataFrame
            name: Name for the table header

        Returns:
            Markdown formatted string
        """
        parts = [f"## Table: {name}\n"]

        # Handle large files by chunking
        if len(df) > self.max_rows_per_chunk:
            logger.info(
                f"File has {len(df)} rows, splitting into chunks of {self.max_rows_per_chunk}"
            )

            for i in range(0, len(df), self.max_rows_per_chunk):
                chunk = df.iloc[i : i + self.max_rows_per_chunk]
                chunk_num = i // self.max_rows_per_chunk + 1
                total_chunks = (len(df) + self.max_rows_per_chunk - 1) // self.max_rows_per_chunk

                parts.append(
                    f"\n### Rows {i + 1}-{min(i + self.max_rows_per_chunk, len(df))} "
                    f"(Chunk {chunk_num}/{total_chunks})\n"
                )
                parts.append(chunk.to_markdown(index=False))
        else:
            parts.append(df.to_markdown(index=False))

        return "\n".join(parts)

    def get_dataframe(
        self,
        file_path: Path | str,
        delimiter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get raw DataFrame from CSV file.

        Args:
            file_path: Path to CSV file
            delimiter: Column delimiter (None for auto-detect)

        Returns:
            Pandas DataFrame
        """
        file_path = Path(file_path)

        if delimiter is None:
            delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","

        encodings_to_try = [self.encoding] if self.encoding else ["utf-8", "latin-1", "cp1252"]

        for enc in encodings_to_try:
            try:
                return pd.read_csv(file_path, delimiter=delimiter, encoding=enc)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode file with any encoding: {encodings_to_try}")

    def cleanup(self):
        """Clean up resources."""
        # Nothing to clean up for this processor
        pass
