"""XLSX/XLS-specific document processor for Excel spreadsheets."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class XLSXProcessor:
    """
    XLSX/XLS-specific processor for Excel spreadsheets.

    Features:
    - Multi-sheet support
    - Table structure extraction
    - Markdown table conversion for chunking compatibility
    - Named range detection
    - Sheet metadata extraction
    """

    def __init__(
        self,
        max_rows_per_chunk: int = 100,
        include_formulas: bool = False,
    ):
        """
        Initialize XLSX processor.

        Args:
            max_rows_per_chunk: Maximum rows per chunk for large sheets
            include_formulas: Whether to include formula text (not values)
        """
        self.max_rows_per_chunk = max_rows_per_chunk
        self.include_formulas = include_formulas
        logger.info(
            f"XLSXProcessor initialized: max_rows_per_chunk={max_rows_per_chunk}, "
            f"include_formulas={include_formulas}"
        )

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process Excel spreadsheet.

        Args:
            file_path: Path to XLSX/XLS file
            metadata: Optional additional metadata

        Returns:
            Processing result with parsed document and structure
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in [".xlsx", ".xls"]:
            raise ValueError(f"Expected Excel file, got: {file_path.suffix}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Excel: {file_path.name}")
        start_time = datetime.now()

        # Get file stats
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Read Excel file
        try:
            # Use openpyxl for .xlsx, xlrd for .xls
            engine = "openpyxl" if file_path.suffix.lower() == ".xlsx" else "xlrd"

            # Read all sheet names first
            excel_file = pd.ExcelFile(file_path, engine=engine)
            sheet_names = excel_file.sheet_names

            sheets: List[Dict[str, Any]] = []
            total_rows = 0
            total_cols = 0
            all_content_parts: List[str] = []

            for sheet_name in sheet_names:
                logger.debug(f"Processing sheet: {sheet_name}")

                # Read sheet data
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    header=0,  # First row as header
                )

                # Skip empty sheets
                if df.empty:
                    logger.debug(f"Skipping empty sheet: {sheet_name}")
                    continue

                # Get sheet dimensions
                num_rows = len(df)
                num_cols = len(df.columns)
                total_rows += num_rows
                total_cols = max(total_cols, num_cols)

                # Convert to markdown table
                markdown_content = self._dataframe_to_markdown(df, sheet_name)

                # Build sheet info
                sheet_info = {
                    "name": sheet_name,
                    "rows": num_rows,
                    "cols": num_cols,
                    "columns": list(df.columns.astype(str)),
                    "content": markdown_content,
                    "preview": self._get_preview(df),
                }

                sheets.append(sheet_info)
                all_content_parts.append(markdown_content)

            excel_file.close()

        except ImportError as e:
            if "openpyxl" in str(e):
                raise ImportError(
                    "openpyxl is required for .xlsx files. "
                    "Install with: pip install openpyxl"
                ) from e
            elif "xlrd" in str(e):
                raise ImportError(
                    "xlrd is required for .xls files. "
                    "Install with: pip install xlrd"
                ) from e
            raise
        except Exception as e:
            logger.error(f"Failed to process Excel file: {e}")
            raise

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Combine all content
        raw_text = "\n\n".join(all_content_parts)

        # Build metadata
        doc_metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size_mb": round(file_size_mb, 2),
            "sheet_count": len(sheets),
            "total_rows": total_rows,
            "total_cols": total_cols,
            "processing_time_seconds": round(processing_time, 2),
            "parsed_at": datetime.now().isoformat(),
            "parser_version": "xlsx_processor_v1",
            "format": file_path.suffix.lower(),
        }

        # Add custom metadata if provided
        if metadata:
            doc_metadata.update(metadata)

        # Build structure info
        structure = {
            "sheets": sheets,
            "sheet_names": [s["name"] for s in sheets],
            "has_tables": True,  # Excel is inherently tabular
            "has_figures": False,
            "tables": [
                {
                    "table_id": f"sheet_{i}",
                    "name": s["name"],
                    "rows": s["rows"],
                    "cols": s["cols"],
                    "columns": s["columns"],
                }
                for i, s in enumerate(sheets)
            ],
        }

        logger.info(
            f"Excel processing complete: {len(sheets)} sheets, "
            f"{total_rows} total rows in {processing_time:.2f}s"
        )

        return {
            "doc_id": doc_id,
            "metadata": doc_metadata,
            "raw_text": raw_text,
            "structure": structure,
            "docling_document": None,  # Not using Docling for Excel
        }

    def _dataframe_to_markdown(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        Convert DataFrame to markdown format.

        Args:
            df: Pandas DataFrame
            sheet_name: Name of the sheet

        Returns:
            Markdown formatted string
        """
        # Add sheet header
        parts = [f"## Sheet: {sheet_name}\n"]

        # Handle large sheets by chunking
        if len(df) > self.max_rows_per_chunk:
            logger.info(
                f"Sheet {sheet_name} has {len(df)} rows, "
                f"splitting into chunks of {self.max_rows_per_chunk}"
            )

            for i in range(0, len(df), self.max_rows_per_chunk):
                chunk = df.iloc[i : i + self.max_rows_per_chunk]
                chunk_num = i // self.max_rows_per_chunk + 1
                total_chunks = (len(df) + self.max_rows_per_chunk - 1) // self.max_rows_per_chunk

                parts.append(f"\n### Rows {i + 1}-{min(i + self.max_rows_per_chunk, len(df))} "
                           f"(Chunk {chunk_num}/{total_chunks})\n")
                parts.append(chunk.to_markdown(index=False))
        else:
            # Convert entire sheet to markdown
            parts.append(df.to_markdown(index=False))

        return "\n".join(parts)

    def _get_preview(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """
        Get a preview of the dataframe.

        Args:
            df: Pandas DataFrame
            max_rows: Maximum rows to include in preview

        Returns:
            Preview string
        """
        preview_df = df.head(max_rows)
        return preview_df.to_string(index=False, max_colwidth=50)

    def get_sheet_data(
        self,
        file_path: Path | str,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get raw DataFrame for a specific sheet.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (defaults to first sheet)

        Returns:
            Pandas DataFrame
        """
        file_path = Path(file_path)
        engine = "openpyxl" if file_path.suffix.lower() == ".xlsx" else "xlrd"

        if sheet_name is None:
            # Default to first sheet (index 0)
            return pd.read_excel(file_path, sheet_name=0, engine=engine)

        return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)

    def cleanup(self):
        """Clean up resources."""
        # Nothing to clean up for this processor
        pass
