"""DOCX-specific document processor."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.parser import DocumentParser
from core.analyzer import StructureAnalyzer

logger = logging.getLogger(__name__)


class DOCXProcessor:
    """
    DOCX-specific processor for Word documents.

    Features:
    - Style preservation
    - Table extraction
    - Structure analysis
    """

    def __init__(self):
        """Initialize DOCX processor."""
        self.parser = DocumentParser()
        self.analyzer = StructureAnalyzer()
        logger.info("DOCXProcessor initialized")

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_structure: bool = True,
    ) -> Dict[str, Any]:
        """
        Process DOCX document.

        Args:
            file_path: Path to DOCX file
            metadata: Optional additional metadata
            analyze_structure: Whether to perform structure analysis

        Returns:
            Processing result with parsed document and structure
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in [".docx", ".doc"]:
            raise ValueError(f"Expected DOCX file, got: {file_path.suffix}")

        logger.info(f"Processing DOCX: {file_path.name}")

        # Parse document
        result = self.parser.parse(file_path, metadata)

        # Perform structure analysis if requested
        if analyze_structure:
            logger.info("Analyzing DOCX structure")
            structure = self.analyzer.analyze(result["docling_document"])

            result["structure_analysis"] = {
                "sections": [
                    {"type": s.get("type"), "text": s.get("text"), "level": s.get("level")}
                    for s in structure.sections
                ],
                "tables": [
                    {"table_id": t.table_id, "rows": t.num_rows, "cols": t.num_cols}
                    for t in structure.tables
                ],
                "total_elements": len(structure.elements),
            }

        logger.info(f"DOCX processing complete: {file_path.name}")
        return result

    def cleanup(self):
        """Clean up resources."""
        self.parser.cleanup()
