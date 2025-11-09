"""HTML-specific document processor."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.parser import DocumentParser
from core.analyzer import StructureAnalyzer

logger = logging.getLogger(__name__)


class HTMLProcessor:
    """
    HTML-specific processor for web content.

    Features:
    - Semantic structure preservation
    - Link extraction
    - Nested element handling
    """

    def __init__(self):
        """Initialize HTML processor."""
        self.parser = DocumentParser()
        self.analyzer = StructureAnalyzer()
        logger.info("HTMLProcessor initialized")

    def process(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_structure: bool = True,
    ) -> Dict[str, Any]:
        """
        Process HTML document.

        Args:
            file_path: Path to HTML file
            metadata: Optional additional metadata
            analyze_structure: Whether to perform structure analysis

        Returns:
            Processing result with parsed document and structure
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in [".html", ".htm"]:
            raise ValueError(f"Expected HTML file, got: {file_path.suffix}")

        logger.info(f"Processing HTML: {file_path.name}")

        # Parse document
        result = self.parser.parse(file_path, metadata)

        # Perform structure analysis if requested
        if analyze_structure:
            logger.info("Analyzing HTML structure")
            structure = self.analyzer.analyze(result["docling_document"])

            result["structure_analysis"] = {
                "sections": [
                    {"type": s.get("type"), "text": s.get("text"), "level": s.get("level")}
                    for s in structure.sections
                ],
                "total_elements": len(structure.elements),
                "hierarchy_depth": self._calculate_hierarchy_depth(
                    structure.hierarchy
                ),
            }

        logger.info(f"HTML processing complete: {file_path.name}")
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

    def cleanup(self):
        """Clean up resources."""
        self.parser.cleanup()
