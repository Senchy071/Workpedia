"""Document structure analyzer for extracting hierarchy and metadata."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocumentElement:
    """Represents a single document element."""

    element_type: str  # text, table, figure, list, code, heading
    content: str
    page_number: Optional[int] = None
    bounding_box: Optional[Dict[str, float]] = None
    level: Optional[int] = None  # For headings
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableInfo:
    """Represents table structure information."""

    table_id: str
    page_numbers: List[int]
    num_rows: int
    num_cols: int
    headers: List[str]
    bounding_boxes: List[Dict[str, float]]
    is_multi_page: bool
    content: str


@dataclass
class DocumentStructure:
    """Complete document structure representation."""

    sections: List[Dict[str, Any]]
    tables: List[TableInfo]
    figures: List[Dict[str, Any]]
    cross_references: List[Dict[str, Any]]
    elements: List[DocumentElement]
    hierarchy: Dict[str, Any]


class StructureAnalyzer:
    """
    Analyzes DoclingDocument to extract structure and metadata.

    Features:
    - Document hierarchy extraction (sections, headings)
    - Table boundary detection (including multi-page)
    - Cross-reference mapping
    - Layout metadata extraction
    - Element classification
    """

    def __init__(self):
        """Initialize structure analyzer."""
        logger.info("StructureAnalyzer initialized")

    def analyze(self, docling_doc) -> DocumentStructure:
        """
        Analyze document structure.

        Args:
            docling_doc: Parsed DoclingDocument object

        Returns:
            DocumentStructure with complete analysis
        """
        logger.info("Analyzing document structure")

        # Extract different components
        sections = self._extract_sections(docling_doc)
        tables = self._extract_tables(docling_doc)
        figures = self._extract_figures(docling_doc)
        cross_refs = self._extract_cross_references(docling_doc)
        elements = self._extract_elements(docling_doc)
        hierarchy = self._build_hierarchy(docling_doc, sections)

        structure = DocumentStructure(
            sections=sections,
            tables=tables,
            figures=figures,
            cross_references=cross_refs,
            elements=elements,
            hierarchy=hierarchy,
        )

        logger.info(
            f"Structure analysis complete: "
            f"{len(sections)} sections, {len(tables)} tables, "
            f"{len(figures)} figures, {len(elements)} elements"
        )

        return structure

    def _extract_sections(self, docling_doc) -> List[Dict[str, Any]]:
        """Extract document sections and headings."""
        sections = []

        try:
            for item in docling_doc.iterate_items():
                # Check if item is a heading/section
                if hasattr(item, 'label'):
                    label = item.label if isinstance(item.label, str) else str(item.label)

                    if any(
                        keyword in label.lower()
                        for keyword in ['heading', 'title', 'section']
                    ):
                        section_info = {
                            "type": label,
                            "text": str(item.text) if hasattr(item, 'text') else "",
                            "level": self._infer_heading_level(item),
                            "page": self._get_page_number(item),
                        }
                        sections.append(section_info)

        except Exception as e:
            logger.warning(f"Error extracting sections: {e}")

        return sections

    def _extract_tables(self, docling_doc) -> List[TableInfo]:
        """Extract table information including multi-page tables."""
        tables = []
        table_counter = 0

        try:
            for item in docling_doc.iterate_items():
                if hasattr(item, 'label'):
                    label = item.label if isinstance(item.label, str) else str(item.label)

                    if 'table' in label.lower():
                        table_counter += 1

                        # Extract table data
                        table_text = str(item.text) if hasattr(item, 'text') else ""
                        page_num = self._get_page_number(item)
                        bbox = self._get_bounding_box(item)

                        # Try to infer table dimensions
                        num_rows, num_cols = self._infer_table_dimensions(item)

                        table_info = TableInfo(
                            table_id=f"table_{table_counter}",
                            page_numbers=[page_num] if page_num else [],
                            num_rows=num_rows,
                            num_cols=num_cols,
                            headers=[],  # TODO: Extract actual headers
                            bounding_boxes=[bbox] if bbox else [],
                            is_multi_page=False,  # TODO: Detect multi-page tables
                            content=table_text,
                        )
                        tables.append(table_info)

        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        return tables

    def _extract_figures(self, docling_doc) -> List[Dict[str, Any]]:
        """Extract figure information."""
        figures = []
        figure_counter = 0

        try:
            for item in docling_doc.iterate_items():
                if hasattr(item, 'label'):
                    label = item.label if isinstance(item.label, str) else str(item.label)

                    if 'figure' in label.lower() or 'image' in label.lower():
                        figure_counter += 1

                        figure_info = {
                            "figure_id": f"figure_{figure_counter}",
                            "type": label,
                            "caption": str(item.text) if hasattr(item, 'text') else "",
                            "page": self._get_page_number(item),
                            "bounding_box": self._get_bounding_box(item),
                        }
                        figures.append(figure_info)

        except Exception as e:
            logger.warning(f"Error extracting figures: {e}")

        return figures

    def _extract_cross_references(self, docling_doc) -> List[Dict[str, Any]]:
        """Extract cross-references and internal links."""
        cross_refs = []

        try:
            # TODO: Implement cross-reference extraction
            # This depends on Docling's representation of links/references
            pass

        except Exception as e:
            logger.warning(f"Error extracting cross-references: {e}")

        return cross_refs

    def _extract_elements(self, docling_doc) -> List[DocumentElement]:
        """Extract and classify all document elements."""
        elements = []

        try:
            for item in docling_doc.iterate_items():
                if hasattr(item, 'label'):
                    label = item.label if isinstance(item.label, str) else str(item.label)
                    text = str(item.text) if hasattr(item, 'text') else ""

                    # Classify element
                    element_type = self._classify_element(label)

                    element = DocumentElement(
                        element_type=element_type,
                        content=text,
                        page_number=self._get_page_number(item),
                        bounding_box=self._get_bounding_box(item),
                        level=self._infer_heading_level(item)
                        if element_type == "heading"
                        else None,
                        metadata={"label": label},
                    )
                    elements.append(element)

        except Exception as e:
            logger.warning(f"Error extracting elements: {e}")

        return elements

    def _build_hierarchy(
        self, docling_doc, sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build document hierarchy tree."""
        hierarchy = {
            "type": "document",
            "children": [],
        }

        # Build tree from sections
        stack = [hierarchy]

        for section in sections:
            level = section.get("level", 1)

            # Pop stack until we find the right parent
            while len(stack) > level:
                stack.pop()

            # Create section node
            section_node = {
                "type": "section",
                "level": level,
                "text": section.get("text", ""),
                "page": section.get("page"),
                "children": [],
            }

            # Add to parent
            if stack:
                stack[-1]["children"].append(section_node)

            stack.append(section_node)

        return hierarchy

    def _classify_element(self, label: str) -> str:
        """Classify element type based on label."""
        label_lower = label.lower()

        if any(kw in label_lower for kw in ['heading', 'title', 'section']):
            return "heading"
        elif 'table' in label_lower:
            return "table"
        elif any(kw in label_lower for kw in ['figure', 'image']):
            return "figure"
        elif 'list' in label_lower:
            return "list"
        elif 'code' in label_lower:
            return "code"
        else:
            return "text"

    def _infer_heading_level(self, item) -> int:
        """Infer heading level from item."""
        # Try to extract level from label
        if hasattr(item, 'label'):
            label = str(item.label).lower()

            # Look for level indicators
            for i in range(1, 7):
                if f'h{i}' in label or f'level{i}' in label or f'{i}' in label:
                    return i

        return 1  # Default to level 1

    def _get_page_number(self, item) -> Optional[int]:
        """Extract page number from item."""
        try:
            if hasattr(item, 'page'):
                return item.page
            if hasattr(item, 'prov') and hasattr(item.prov, 'page'):
                return item.prov.page
        except Exception:
            pass
        return None

    def _get_bounding_box(self, item) -> Optional[Dict[str, float]]:
        """Extract bounding box from item."""
        try:
            if hasattr(item, 'bbox'):
                bbox = item.bbox
                return {
                    "x": bbox.l if hasattr(bbox, 'l') else 0,
                    "y": bbox.t if hasattr(bbox, 't') else 0,
                    "width": bbox.r - bbox.l if hasattr(bbox, 'r') and hasattr(bbox, 'l') else 0,
                    "height": bbox.b - bbox.t if hasattr(bbox, 'b') and hasattr(bbox, 't') else 0,
                }
            if hasattr(item, 'prov') and hasattr(item.prov, 'bbox'):
                bbox = item.prov.bbox
                return {
                    "x": bbox.l if hasattr(bbox, 'l') else 0,
                    "y": bbox.t if hasattr(bbox, 't') else 0,
                    "width": bbox.r - bbox.l if hasattr(bbox, 'r') and hasattr(bbox, 'l') else 0,
                    "height": bbox.b - bbox.t if hasattr(bbox, 'b') and hasattr(bbox, 't') else 0,
                }
        except Exception:
            pass
        return None

    def _infer_table_dimensions(self, item) -> tuple[int, int]:
        """Infer table dimensions (rows, cols) from item."""
        # TODO: Extract actual table structure from Docling
        # For now, return placeholder values
        return (0, 0)
