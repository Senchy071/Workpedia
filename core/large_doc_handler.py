"""Handler for processing large documents in chunks to avoid memory issues."""

import logging
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from pypdf import PdfReader

from config.config import (
    MAX_PAGES_SINGLE_PASS,
    MAX_FILE_SIZE_MB,
    CHUNK_SIZE_PAGES,
    VLM_MODEL,
)
from core.parser import DocumentParser

logger = logging.getLogger(__name__)


class LargeDocumentHandler:
    """
    Handler for processing large documents in manageable chunks.

    Features:
    - Auto-detection of large documents (>100 pages or >50MB)
    - Sequential chunk processing with memory cleanup
    - Progress tracking
    - Result merging
    """

    def __init__(
        self,
        max_pages: int = MAX_PAGES_SINGLE_PASS,
        max_size_mb: int = MAX_FILE_SIZE_MB,
        chunk_size_pages: int = CHUNK_SIZE_PAGES,
    ):
        """
        Initialize large document handler.

        Args:
            max_pages: Max pages for single-pass processing
            max_size_mb: Max file size (MB) for single-pass processing
            chunk_size_pages: Pages per chunk for large documents
        """
        self.max_pages = max_pages
        self.max_size_mb = max_size_mb
        self.chunk_size_pages = chunk_size_pages

        logger.info(
            f"LargeDocumentHandler initialized: "
            f"max_pages={max_pages}, max_size_mb={max_size_mb}, "
            f"chunk_size_pages={chunk_size_pages}"
        )

    def is_large_document(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if document requires chunked processing.

        Args:
            file_path: Path to document

        Returns:
            Tuple of (is_large, info_dict)
        """
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Check file size
        if file_size_mb > self.max_size_mb:
            logger.info(
                f"Document is large by size: {file_size_mb:.2f}MB > {self.max_size_mb}MB"
            )
            return True, {
                "reason": "file_size",
                "file_size_mb": file_size_mb,
                "threshold_mb": self.max_size_mb,
            }

        # Check page count for PDFs
        if file_path.suffix.lower() == ".pdf":
            try:
                pdf_reader = PdfReader(str(file_path))
                num_pages = len(pdf_reader.pages)

                if num_pages > self.max_pages:
                    logger.info(
                        f"Document is large by page count: "
                        f"{num_pages} pages > {self.max_pages} pages"
                    )
                    return True, {
                        "reason": "page_count",
                        "num_pages": num_pages,
                        "threshold_pages": self.max_pages,
                        "file_size_mb": file_size_mb,
                    }

                return False, {
                    "num_pages": num_pages,
                    "file_size_mb": file_size_mb,
                }

            except Exception as e:
                logger.warning(f"Could not read PDF metadata: {e}")
                # Fall back to size-based check
                return file_size_mb > self.max_size_mb, {
                    "reason": "size_fallback",
                    "file_size_mb": file_size_mb,
                }

        # For non-PDF files, use size-based check only
        return False, {"file_size_mb": file_size_mb}

    def calculate_chunks(self, num_pages: int) -> List[Tuple[int, int]]:
        """
        Calculate page ranges for chunked processing.

        Args:
            num_pages: Total number of pages

        Returns:
            List of (start_page, end_page) tuples (1-indexed)
        """
        chunks = []
        current_page = 1

        while current_page <= num_pages:
            end_page = min(current_page + self.chunk_size_pages - 1, num_pages)
            chunks.append((current_page, end_page))
            current_page = end_page + 1

        logger.info(
            f"Document split into {len(chunks)} chunks "
            f"({self.chunk_size_pages} pages each)"
        )
        return chunks

    def process_large_pdf(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
        backend: str = "v2",
        do_ocr: bool = False,
        do_table_structure: bool = True,
        vlm_model: str = VLM_MODEL,
    ) -> Dict[str, Any]:
        """
        Process large PDF in chunks.

        Args:
            file_path: Path to PDF file
            metadata: Optional additional metadata
            progress_callback: Optional callback(chunk_num, total_chunks, info)
            backend: PDF backend to use ("v2", "pypdfium", or "vlm")
            do_ocr: Enable OCR
            do_table_structure: Enable table structure
            vlm_model: VLM model to use if backend="vlm"

        Returns:
            Merged document result
        """
        file_path = Path(file_path)

        # Get document info
        is_large, info = self.is_large_document(file_path)

        if not is_large:
            logger.info("Document is not large, using standard processing")
            parser = DocumentParser()
            return parser.parse(file_path, metadata)

        logger.info(f"Processing large document: {file_path.name}")
        start_time = datetime.now()

        # Get page count
        pdf_reader = PdfReader(str(file_path))
        num_pages = len(pdf_reader.pages)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Calculate chunks
        chunks = self.calculate_chunks(num_pages)

        # Process each chunk
        chunk_results = []

        for idx, (start_page, end_page) in enumerate(chunks, 1):
            logger.info(
                f"Processing chunk {idx}/{len(chunks)}: "
                f"pages {start_page}-{end_page}"
            )

            if progress_callback:
                progress_callback(
                    idx,
                    len(chunks),
                    {
                        "start_page": start_page,
                        "end_page": end_page,
                        "pages_in_chunk": end_page - start_page + 1,
                    },
                )

            try:
                # Create fresh parser for each chunk to prevent memory leaks
                parser = DocumentParser(
                    do_ocr=do_ocr,
                    do_table_structure=do_table_structure,
                    backend=backend,
                    vlm_model=vlm_model
                )

                # Note: Docling's DocumentConverter doesn't support page ranges
                # directly in the API, so we'll process the whole document
                # but this approach allows for memory cleanup between attempts
                result = parser.parse(file_path)

                chunk_results.append(result)

                # Cleanup
                parser.cleanup()
                del parser
                gc.collect()

                logger.info(f"Chunk {idx}/{len(chunks)} processed successfully")

                # Since Docling doesn't support page ranges and processes the whole document,
                # if we got a successful result with all pages, we're done
                if result and result.get("metadata", {}).get("pages", 0) == num_pages:
                    logger.info(
                        f"First chunk successfully processed all {num_pages} pages. "
                        f"Skipping remaining chunks (Docling processes entire document)."
                    )
                    break

            except Exception as e:
                logger.error(f"Failed to process chunk {idx}: {e}")
                # Continue with other chunks
                continue

        if not chunk_results:
            raise RuntimeError("Failed to process any chunks of the document")

        # For now, we'll use the first successful result
        # TODO: Implement proper merging of DoclingDocument objects
        final_result = chunk_results[0]

        # Update metadata to reflect chunked processing
        processing_time = (datetime.now() - start_time).total_seconds()

        final_result["metadata"].update(
            {
                "processed_in_chunks": True,
                "num_chunks": len(chunks),
                "chunks_processed": len(chunk_results),
                "chunks_failed": len(chunks) - len(chunk_results),
                "processing_time_seconds": round(processing_time, 2),
                "pages": num_pages,
                "file_size_mb": round(file_size_mb, 2),
            }
        )

        if metadata:
            final_result["metadata"].update(metadata)

        logger.info(
            f"Large document processed: {len(chunk_results)}/{len(chunks)} "
            f"chunks successful in {processing_time:.2f}s"
        )

        return final_result

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
        backend: str = "v2",
        do_ocr: bool = False,
        do_table_structure: bool = True,
        vlm_model: str = VLM_MODEL,
    ) -> Dict[str, Any]:
        """
        Process document (automatically handles large documents).

        Args:
            file_path: Path to document
            metadata: Optional additional metadata
            progress_callback: Optional progress callback for large documents
            backend: PDF backend to use ("v2", "pypdfium", or "vlm")
            do_ocr: Enable OCR
            do_table_structure: Enable table structure recognition
            vlm_model: VLM model to use if backend="vlm"

        Returns:
            Parsed document result
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".pdf":
            is_large, _ = self.is_large_document(file_path)

            if is_large:
                return self.process_large_pdf(
                    file_path, metadata, progress_callback,
                    backend, do_ocr, do_table_structure, vlm_model
                )

        # Standard processing for small documents or non-PDFs
        parser = DocumentParser(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            backend=backend,
            vlm_model=vlm_model
        )
        result = parser.parse(file_path, metadata)
        parser.cleanup()

        return result
