"""PDF-specific document processor."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pypdf import PdfReader

from config.config import (
    LARGE_DOC_PAGE_THRESHOLD,
    LARGE_DOC_SIZE_MB_THRESHOLD,
    USE_VLM_FOR_LARGE_DOCS,
    VLM_MODEL,
)
from core.analyzer import StructureAnalyzer
from core.large_doc_handler import LargeDocumentHandler

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF-specific processor with large document handling.

    Features:
    - Automatic large document detection
    - Chunked processing for 200+ page PDFs
    - Table structure preservation
    - GPU acceleration for table recognition
    """

    def __init__(
        self,
        enable_ocr: bool = False,
        enable_table_structure: bool = True,
        backend: str = "auto",
        auto_fallback: bool = True,
    ):
        """
        Initialize PDF processor.

        Args:
            enable_ocr: Enable OCR for scanned PDFs
            enable_table_structure: Enable table structure recognition
            backend: PDF backend to use ("auto", "v2", or "pypdfium")
                    "auto" = v2 for small docs, pypdfium for large docs
            auto_fallback: Automatically fallback to pypdfium on v2 crashes
        """
        self.enable_ocr = enable_ocr
        self.enable_table_structure = enable_table_structure
        self.backend = backend
        self.auto_fallback = auto_fallback

        self.doc_handler = LargeDocumentHandler()
        self.analyzer = StructureAnalyzer()

        logger.info(
            f"PDFProcessor initialized: OCR={enable_ocr}, "
            f"TableStructure={enable_table_structure}, Backend={backend}, "
            f"AutoFallback={auto_fallback}"
        )

    def _select_backend_for_document(self, file_path: Path) -> tuple[str, str]:
        """
        Select appropriate backend based on document size.

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (backend_name, reason)
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Check file size threshold
        if file_size_mb > LARGE_DOC_SIZE_MB_THRESHOLD:
            reason = f"File size {file_size_mb:.1f}MB > {LARGE_DOC_SIZE_MB_THRESHOLD}MB"
            if USE_VLM_FOR_LARGE_DOCS:
                logger.info(f"Large document detected: {reason} → using VLM ({VLM_MODEL})")
                return "vlm", reason
            else:
                logger.info(f"Large document detected: {reason} → using pypdfium backend")
                return "pypdfium", reason

        # Check page count threshold
        try:
            pdf_reader = PdfReader(str(file_path))
            num_pages = len(pdf_reader.pages)

            if num_pages > LARGE_DOC_PAGE_THRESHOLD:
                reason = f"{num_pages} pages > {LARGE_DOC_PAGE_THRESHOLD} pages"
                if USE_VLM_FOR_LARGE_DOCS:
                    logger.info(f"Large document detected: {reason} → using VLM ({VLM_MODEL})")
                    return "vlm", reason
                else:
                    logger.info(f"Large document detected: {reason} → using pypdfium backend")
                    return "pypdfium", reason

            logger.info(
                f"Small document: {num_pages} pages, {file_size_mb:.1f}MB → using v2 backend"
            )
            return "v2", f"{num_pages} pages, {file_size_mb:.1f}MB"

        except Exception as e:
            logger.warning(f"Could not read PDF metadata: {e}, using size-based decision")
            if file_size_mb > LARGE_DOC_SIZE_MB_THRESHOLD:
                backend = "vlm" if USE_VLM_FOR_LARGE_DOCS else "pypdfium"
                return backend, f"size: {file_size_mb:.1f}MB"
            else:
                return "v2", f"size: {file_size_mb:.1f}MB"

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_structure: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Process PDF document with automatic fallback on crashes.

        Args:
            file_path: Path to PDF file
            metadata: Optional additional metadata
            analyze_structure: Whether to perform structure analysis
            progress_callback: Optional progress callback

        Returns:
            Processing result with parsed document and structure
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")

        logger.info(f"Processing PDF: {file_path.name}")

        # Auto-select backend based on document size
        selected_backend = self.backend
        if self.backend == "auto":
            selected_backend, reason = self._select_backend_for_document(file_path)
            logger.info(f"Auto-selected backend: {selected_backend} ({reason})")

        result = None
        last_error = None

        # Try with selected backend first
        try:
            result = self.doc_handler.process(
                file_path=file_path,
                metadata=metadata,
                progress_callback=progress_callback,
                backend=selected_backend,
                do_ocr=self.enable_ocr,
                do_table_structure=self.enable_table_structure,
                vlm_model=VLM_MODEL,
            )
        except Exception as e:
            last_error = e
            logger.error(f"Processing failed with {selected_backend} backend: {e}")

            # Try fallback strategies if enabled
            if self.auto_fallback and selected_backend == "v2":
                logger.info("Attempting fallback strategies...")

                # Strategy 1: Try pypdfium backend
                try:
                    logger.info("Fallback 1: Trying PyPdfium backend")
                    result = self.doc_handler.process(
                        file_path=file_path,
                        metadata=metadata,
                        progress_callback=progress_callback,
                        backend="pypdfium",
                        do_ocr=self.enable_ocr,
                        do_table_structure=self.enable_table_structure,
                        vlm_model=VLM_MODEL,
                    )
                    logger.info("✓ PyPdfium backend succeeded")
                    if result:
                        result["metadata"]["fallback_used"] = "pypdfium"
                except Exception as e2:
                    logger.error(f"PyPdfium backend failed: {e2}")

                    # Strategy 2: Disable table structure and retry with v2
                    if not result and self.enable_table_structure:
                        try:
                            logger.info("Fallback 2: Disabling table structure, retrying v2")
                            result = self.doc_handler.process(
                                file_path=file_path,
                                metadata=metadata,
                                progress_callback=progress_callback,
                                backend="v2",
                                do_ocr=self.enable_ocr,
                                do_table_structure=False,
                                vlm_model=VLM_MODEL,
                            )
                            logger.info("✓ V2 without table structure succeeded")
                            if result:
                                result["metadata"]["fallback_used"] = "v2_no_tables"
                        except Exception as e3:
                            logger.error(f"V2 without table structure failed: {e3}")

                    # Strategy 3: Disable table structure with pypdfium
                    if not result and self.enable_table_structure:
                        try:
                            logger.info("Fallback 3: PyPdfium without table structure")
                            result = self.doc_handler.process(
                                file_path=file_path,
                                metadata=metadata,
                                progress_callback=progress_callback,
                                backend="pypdfium",
                                do_ocr=self.enable_ocr,
                                do_table_structure=False,
                                vlm_model=VLM_MODEL,
                            )
                            logger.info("✓ PyPdfium without table structure succeeded")
                            if result:
                                result["metadata"]["fallback_used"] = "pypdfium_no_tables"
                        except Exception as e4:
                            logger.error(f"PyPdfium without table structure failed: {e4}")

        # If all strategies failed, raise the last error
        if result is None:
            logger.error("All processing strategies failed")
            raise last_error if last_error else RuntimeError("Processing failed")

        # Add backend selection info to metadata
        if "backend_selected" not in result["metadata"]:
            result["metadata"]["backend_selected"] = selected_backend
            if self.backend == "auto":
                result["metadata"]["backend_auto_selected"] = True

        # Perform structure analysis if requested
        if analyze_structure:
            logger.info("Analyzing PDF structure")
            structure = self.analyzer.analyze(result["docling_document"])

            # Update result with detailed structure
            result["structure_analysis"] = {
                "sections": [
                    {
                        "type": s.get("type"),
                        "text": s.get("text"),
                        "level": s.get("level"),
                        "page": s.get("page"),
                    }
                    for s in structure.sections
                ],
                "tables": [
                    {
                        "table_id": t.table_id,
                        "pages": t.page_numbers,
                        "rows": t.num_rows,
                        "cols": t.num_cols,
                        "is_multi_page": t.is_multi_page,
                    }
                    for t in structure.tables
                ],
                "figures": [
                    {
                        "figure_id": f.get("figure_id"),
                        "page": f.get("page"),
                        "caption": f.get("caption"),
                    }
                    for f in structure.figures
                ],
                "total_elements": len(structure.elements),
                "hierarchy_depth": self._calculate_hierarchy_depth(
                    structure.hierarchy
                ),
            }

        logger.info(f"PDF processing complete: {file_path.name}")
        return result

    def _calculate_hierarchy_depth(self, hierarchy: Dict[str, Any]) -> int:
        """Calculate depth of document hierarchy."""

        def get_depth(node: Dict[str, Any], current_depth: int = 0) -> int:
            if not node.get("children"):
                return current_depth
            return max(
                get_depth(child, current_depth + 1) for child in node["children"]
            )

        return get_depth(hierarchy)
