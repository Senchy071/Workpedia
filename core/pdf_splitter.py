"""PDF splitter for extracting page ranges into temporary files."""

import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Result of a PDF split operation."""

    original_path: Path
    split_files: List[Path]
    page_ranges: List[Tuple[int, int]]  # (start, end) 1-indexed
    total_pages: int
    temp_dir: Optional[Path] = None


class PDFSplitter:
    """
    Splits large PDFs into smaller chunks for processing.

    Features:
    - Split by page count
    - Preserve PDF structure within chunks
    - Automatic temp file management
    - Page range tracking for later merging
    """

    def __init__(self, chunk_size: int = 75):
        """
        Initialize PDF splitter.

        Args:
            chunk_size: Maximum pages per split file
        """
        self.chunk_size = chunk_size
        logger.info(f"PDFSplitter initialized with chunk_size={chunk_size}")

    def get_page_count(self, file_path: Path) -> int:
        """Get total page count of a PDF."""
        reader = PdfReader(str(file_path))
        return len(reader.pages)

    def calculate_splits(self, total_pages: int) -> List[Tuple[int, int]]:
        """
        Calculate page ranges for splitting.

        Args:
            total_pages: Total number of pages

        Returns:
            List of (start_page, end_page) tuples (1-indexed, inclusive)
        """
        splits = []
        current = 1

        while current <= total_pages:
            end = min(current + self.chunk_size - 1, total_pages)
            splits.append((current, end))
            current = end + 1

        logger.debug(f"Calculated {len(splits)} splits for {total_pages} pages")
        return splits

    def split(
        self,
        file_path: Path,
        output_dir: Optional[Path] = None,
    ) -> SplitResult:
        """
        Split PDF into smaller files.

        Args:
            file_path: Path to source PDF
            output_dir: Directory for split files (uses temp dir if None)

        Returns:
            SplitResult with paths to split files and page ranges
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Splitting PDF: {file_path.name}")

        # Read source PDF
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)

        # Calculate splits
        page_ranges = self.calculate_splits(total_pages)

        # If only one chunk needed, return original
        if len(page_ranges) == 1:
            logger.info("PDF fits in single chunk, no split needed")
            return SplitResult(
                original_path=file_path,
                split_files=[file_path],
                page_ranges=page_ranges,
                total_pages=total_pages,
                temp_dir=None,
            )

        # Create output directory
        temp_dir = None
        if output_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="workpedia_split_"))
            output_dir = temp_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        split_files = []

        for idx, (start, end) in enumerate(page_ranges, 1):
            # Create writer for this chunk
            writer = PdfWriter()

            # Add pages (pypdf uses 0-indexed)
            for page_num in range(start - 1, end):
                writer.add_page(reader.pages[page_num])

            # Write chunk file
            chunk_name = f"{file_path.stem}_chunk{idx:03d}_p{start}-{end}.pdf"
            chunk_path = output_dir / chunk_name

            with open(chunk_path, "wb") as f:
                writer.write(f)

            split_files.append(chunk_path)
            logger.debug(f"Created chunk {idx}: pages {start}-{end} -> {chunk_name}")

        logger.info(
            f"Split complete: {total_pages} pages -> {len(split_files)} chunks "
            f"({self.chunk_size} pages each)"
        )

        return SplitResult(
            original_path=file_path,
            split_files=split_files,
            page_ranges=page_ranges,
            total_pages=total_pages,
            temp_dir=temp_dir,
        )

    def split_range(
        self,
        file_path: Path,
        start_page: int,
        end_page: int,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Extract a specific page range from a PDF.

        Args:
            file_path: Path to source PDF
            start_page: First page (1-indexed)
            end_page: Last page (1-indexed, inclusive)
            output_path: Output file path (uses temp file if None)

        Returns:
            Path to extracted PDF
        """
        file_path = Path(file_path)

        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)

        # Validate range
        if start_page < 1 or end_page > total_pages or start_page > end_page:
            raise ValueError(
                f"Invalid page range: {start_page}-{end_page} "
                f"(document has {total_pages} pages)"
            )

        writer = PdfWriter()

        # Add pages (pypdf uses 0-indexed)
        for page_num in range(start_page - 1, end_page):
            writer.add_page(reader.pages[page_num])

        # Determine output path
        if output_path is None:
            fd, temp_path = tempfile.mkstemp(
                prefix=f"{file_path.stem}_p{start_page}-{end_page}_",
                suffix=".pdf",
            )
            output_path = Path(temp_path)
            # Close the file descriptor since we'll write using pypdf
            import os
            os.close(fd)

        with open(output_path, "wb") as f:
            writer.write(f)

        logger.debug(f"Extracted pages {start_page}-{end_page} to {output_path.name}")
        return output_path

    @staticmethod
    def cleanup(split_result: SplitResult) -> None:
        """
        Clean up temporary split files.

        Args:
            split_result: Result from split() operation
        """
        if split_result.temp_dir and split_result.temp_dir.exists():
            import shutil
            shutil.rmtree(split_result.temp_dir)
            logger.debug(f"Cleaned up temp directory: {split_result.temp_dir}")
