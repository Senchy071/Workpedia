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
    ):
        """
        Initialize PDF processor.

        Args:
            enable_ocr: Enable OCR for scanned PDFs
            enable_table_structure: Enable table structure recognition
        """
        self.enable_ocr = enable_ocr
        self.enable_table_structure = enable_table_structure

        self.doc_handler = LargeDocumentHandler()
        self.analyzer = StructureAnalyzer()

        logger.info(
            f"PDFProcessor initialized: OCR={enable_ocr}, "
            f"TableStructure={enable_table_structure}"
        )

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_structure: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process PDF document.

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

        # Process document (handles large docs automatically)
        result = self.doc_handler.process(
            file_path=file_path,
            metadata=metadata,
            progress_callback=progress_callback,
        )

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
