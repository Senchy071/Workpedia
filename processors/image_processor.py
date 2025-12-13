"""Image-specific document processor with OCR support."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.analyzer import StructureAnalyzer
from core.parser import DocumentParser

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image-specific processor with OCR.

    Features:
    - OCR for scanned documents
    - Support for PNG, JPG, TIFF formats
    - Text extraction from images
    """

    def __init__(self, enable_ocr: bool = True):
        """
        Initialize image processor.

        Args:
            enable_ocr: Enable OCR for text extraction
        """
        self.enable_ocr = enable_ocr
        # Create parser with OCR enabled
        self.parser = DocumentParser(do_ocr=enable_ocr)
        self.analyzer = StructureAnalyzer()

        logger.info(f"ImageProcessor initialized: OCR={enable_ocr}")

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_structure: bool = True,
    ) -> Dict[str, Any]:
        """
        Process image document.

        Args:
            file_path: Path to image file
            metadata: Optional additional metadata
            analyze_structure: Whether to perform structure analysis

        Returns:
            Processing result with parsed document and structure
        """
        file_path = Path(file_path)

        valid_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]
        if file_path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Expected image file ({', '.join(valid_extensions)}), "
                f"got: {file_path.suffix}"
            )

        logger.info(f"Processing image: {file_path.name}")

        # Parse document (OCR will be applied if enabled)
        result = self.parser.parse(file_path, metadata)

        # Add image-specific metadata
        result["metadata"]["ocr_enabled"] = self.enable_ocr

        # Perform structure analysis if requested
        if analyze_structure and result.get("raw_text"):
            logger.info("Analyzing extracted text structure")
            structure = self.analyzer.analyze(result["docling_document"])

            result["structure_analysis"] = {
                "total_elements": len(structure.elements),
                "text_extracted": len(result.get("raw_text", "")) > 0,
            }

        logger.info(f"Image processing complete: {file_path.name}")
        return result

    def cleanup(self):
        """Clean up resources."""
        self.parser.cleanup()
