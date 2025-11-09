"""PDF-specific document processor."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.large_doc_handler import LargeDocumentHandler
from core.analyzer import StructureAnalyzer

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
        backend: str = "v2",
        auto_fallback: bool = True,
    ):
        """
        Initialize PDF processor.

        Args:
            enable_ocr: Enable OCR for scanned PDFs
            enable_table_structure: Enable table structure recognition
            backend: PDF backend to use ("v2" or "pypdfium")
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

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_structure: bool = True,
        progress_callback: Optional[callable] = None,
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

        result = None
        last_error = None

        # Try with configured backend first
        try:
            result = self.doc_handler.process(
                file_path=file_path,
                metadata=metadata,
                progress_callback=progress_callback,
                backend=self.backend,
                do_ocr=self.enable_ocr,
                do_table_structure=self.enable_table_structure,
            )
        except Exception as e:
            last_error = e
            logger.error(f"Processing failed with {self.backend} backend: {e}")

            # Try fallback strategies if enabled
            if self.auto_fallback and self.backend == "v2":
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
