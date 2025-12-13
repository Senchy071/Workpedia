"""Document structure analyzer for extracting hierarchy and metadata."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
    header_row_index: int = 0  # Which row contains headers (usually 0)


@dataclass
class CrossReference:
    """Represents a cross-reference within the document."""

    ref_type: str  # table, figure, section, equation, footnote, citation
    ref_id: str  # The reference identifier (e.g., "Table 1", "Fig. 2")
    source_page: Optional[int] = None
    source_text: str = ""  # Context text containing the reference
    target_id: Optional[str] = None  # ID of referenced element if resolved


@dataclass
class DocumentStructure:
    """Complete document structure representation."""

    sections: List[Dict[str, Any]]
    tables: List[TableInfo]
    figures: List[Dict[str, Any]]
    cross_references: List[CrossReference]
    elements: List[DocumentElement]
    hierarchy: Dict[str, Any]


class StructureAnalyzer:
    """
    Analyzes DoclingDocument to extract structure and metadata.

    Features:
    - Document hierarchy extraction (sections, headings)
    - Table boundary detection (including multi-page)
    - Table header extraction
    - Cross-reference mapping
    - Layout metadata extraction
    - Element classification
    """

    # Patterns for cross-reference detection
    CROSS_REF_PATTERNS = {
        "table": re.compile(
            r'\b(?:Table|Tab\.?|Tbl\.?)\s*(\d+(?:\.\d+)?(?:[a-zA-Z])?)',
            re.IGNORECASE
        ),
        "figure": re.compile(
            r'\b(?:Figures?|Fig\.?|Figs?\.?)\s*(\d+(?:\.\d+)?(?:[a-zA-Z])?)',
            re.IGNORECASE
        ),
        "section": re.compile(
            r'(?:\bSection|\bSect?\.?|§)\s*(\d+(?:\.\d+)*)',
            re.IGNORECASE
        ),
        "equation": re.compile(
            r'\b(?:Equations?|Eq\.?|Eqs?\.?)\s*[(\[]?(\d+(?:\.\d+)?(?:[a-zA-Z])?)[)\]]?',
            re.IGNORECASE
        ),
        "citation": re.compile(
            r'\[(\d+(?:[-–,]\s*\d+)*)\]|\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?)\)',
            re.IGNORECASE
        ),
        "footnote": re.compile(
            r'\b(?:footnote|note)\s*(\d+)|\[(\d+)\]$',
            re.IGNORECASE
        ),
    }

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
        elements = self._extract_elements(docling_doc)

        # Extract cross-references from all text content
        cross_refs = self._extract_cross_references(docling_doc, elements)

        # Build hierarchy
        hierarchy = self._build_hierarchy(docling_doc, sections)

        # Detect multi-page tables
        tables = self._detect_multi_page_tables(tables)

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
            f"{len(figures)} figures, {len(cross_refs)} cross-refs, "
            f"{len(elements)} elements"
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
        """Extract table information including structure and headers."""
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

                        # Extract table dimensions and headers
                        num_rows, num_cols, headers = self._extract_table_structure(
                            item, table_text
                        )

                        table_info = TableInfo(
                            table_id=f"table_{table_counter}",
                            page_numbers=[page_num] if page_num else [],
                            num_rows=num_rows,
                            num_cols=num_cols,
                            headers=headers,
                            bounding_boxes=[bbox] if bbox else [],
                            is_multi_page=False,  # Will be updated by _detect_multi_page_tables
                            content=table_text,
                        )
                        tables.append(table_info)

        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        return tables

    def _extract_table_structure(
        self, item, table_text: str
    ) -> Tuple[int, int, List[str]]:
        """
        Extract table dimensions and headers from table item.

        Args:
            item: Docling table item
            table_text: Text representation of table

        Returns:
            Tuple of (num_rows, num_cols, headers)
        """
        num_rows = 0
        num_cols = 0
        headers: List[str] = []

        try:
            # Try to get structure from Docling's table data
            if hasattr(item, 'data') and item.data is not None:
                table_data = item.data

                # Check for grid/cells structure
                if hasattr(table_data, 'grid'):
                    grid = table_data.grid
                    num_rows = len(grid)
                    num_cols = max(len(row) for row in grid) if grid else 0

                    # Extract headers from first row
                    if grid and len(grid) > 0:
                        headers = [
                            str(cell.text) if hasattr(cell, 'text') else str(cell)
                            for cell in grid[0]
                        ]

                # Alternative: check for rows/cols attributes
                elif hasattr(table_data, 'num_rows'):
                    num_rows = table_data.num_rows
                    num_cols = getattr(table_data, 'num_cols', 0)

            # Fallback: parse from markdown/text representation
            if num_rows == 0 and table_text:
                num_rows, num_cols, headers = self._parse_table_from_text(table_text)

        except Exception as e:
            logger.debug(f"Could not extract table structure: {e}")

        return num_rows, num_cols, headers

    def _parse_table_from_text(self, table_text: str) -> Tuple[int, int, List[str]]:
        """
        Parse table dimensions and headers from text representation.

        Handles markdown table format:
        | Header1 | Header2 |
        |---------|---------|
        | Cell1   | Cell2   |
        """
        lines = table_text.strip().split('\n')
        if not lines:
            return 0, 0, []

        # Filter out separator lines (like |---|---|)
        data_lines = []
        separator_pattern = re.compile(r'^[\|\s\-:]+$')

        for line in lines:
            line = line.strip()
            if line and not separator_pattern.match(line):
                data_lines.append(line)

        if not data_lines:
            return 0, 0, []

        num_rows = len(data_lines)

        # Parse first line for headers and column count
        headers: List[str] = []
        first_line = data_lines[0]

        if '|' in first_line:
            # Markdown table format
            cells = [c.strip() for c in first_line.split('|') if c.strip()]
            headers = cells
            num_cols = len(cells)
        elif '\t' in first_line:
            # Tab-separated
            cells = first_line.split('\t')
            headers = [c.strip() for c in cells]
            num_cols = len(cells)
        else:
            # Try to infer columns from spacing
            # This is a rough heuristic
            num_cols = len(first_line.split())

        return num_rows, num_cols, headers

    def _detect_multi_page_tables(self, tables: List[TableInfo]) -> List[TableInfo]:
        """
        Detect tables that span multiple pages.

        Strategy:
        - Tables on consecutive pages with similar column counts
        - Tables at page bottom followed by tables at page top
        - Headers that repeat on subsequent pages
        """
        if len(tables) < 2:
            return tables

        # Sort by page number
        sorted_tables = sorted(
            [(i, t) for i, t in enumerate(tables) if t.page_numbers],
            key=lambda x: x[1].page_numbers[0] if x[1].page_numbers else 0
        )

        # Track which tables are continuations
        continuation_groups: Dict[int, List[int]] = {}  # primary_idx -> [continuation_indices]

        for i in range(len(sorted_tables) - 1):
            idx1, table1 = sorted_tables[i]
            idx2, table2 = sorted_tables[i + 1]

            if not table1.page_numbers or not table2.page_numbers:
                continue

            page1 = table1.page_numbers[-1]
            page2 = table2.page_numbers[0]

            # Check if tables are on consecutive pages
            if page2 == page1 + 1:
                # Check if column counts match
                if table1.num_cols == table2.num_cols and table1.num_cols > 0:
                    # Check for header repetition (common in multi-page tables)
                    headers_match = (
                        table1.headers and table2.headers and
                        table1.headers == table2.headers
                    )

                    # Check bounding box positions (table2 at top of page)
                    table2_at_top = False
                    if table2.bounding_boxes:
                        bbox = table2.bounding_boxes[0]
                        # Y coordinate near top of page (assuming normalized coords)
                        if bbox and bbox.get('y', 1) < 0.3:
                            table2_at_top = True

                    if headers_match or table2_at_top:
                        # Mark as continuation
                        if idx1 not in continuation_groups:
                            continuation_groups[idx1] = []
                        continuation_groups[idx1].append(idx2)

        # Merge continuation groups
        for primary_idx, continuation_indices in continuation_groups.items():
            primary_table = tables[primary_idx]
            primary_table.is_multi_page = True

            for cont_idx in continuation_indices:
                cont_table = tables[cont_idx]
                cont_table.is_multi_page = True

                # Merge page numbers
                for page in cont_table.page_numbers:
                    if page not in primary_table.page_numbers:
                        primary_table.page_numbers.append(page)

                # Merge bounding boxes
                primary_table.bounding_boxes.extend(cont_table.bounding_boxes)

                # Merge content (excluding repeated headers)
                if cont_table.content:
                    # Skip first row if it's a header repetition
                    content_to_add = cont_table.content
                    if cont_table.headers == primary_table.headers:
                        lines = content_to_add.split('\n')
                        if len(lines) > 2:  # Header + separator + data
                            content_to_add = '\n'.join(lines[2:])
                    primary_table.content += '\n' + content_to_add

                # Update row count
                primary_table.num_rows += cont_table.num_rows

        # Sort page numbers
        for table in tables:
            table.page_numbers.sort()

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

    def _extract_cross_references(
        self, docling_doc, elements: List[DocumentElement]
    ) -> List[CrossReference]:
        """
        Extract cross-references from document text.

        Looks for patterns like:
        - "Table 1", "Fig. 2", "Section 3.1"
        - "[1]", "[Smith et al., 2023]"
        - "Equation (4)"
        """
        cross_refs: List[CrossReference] = []
        seen_refs: Set[Tuple[str, str]] = set()  # (type, id) to avoid duplicates

        try:
            # Collect all text content with page info
            text_sources: List[Tuple[str, Optional[int]]] = []

            # From elements
            for element in elements:
                if element.content:
                    text_sources.append((element.content, element.page_number))

            # Also try to get from docling_doc directly
            try:
                for item in docling_doc.iterate_items():
                    if hasattr(item, 'text') and item.text:
                        page = self._get_page_number(item)
                        text_sources.append((str(item.text), page))
            except Exception:
                pass

            # Search for cross-references in all text
            for text, page_num in text_sources:
                for ref_type, pattern in self.CROSS_REF_PATTERNS.items():
                    for match in pattern.finditer(text):
                        # Get the reference ID from the match
                        ref_id = None
                        for group in match.groups():
                            if group:
                                ref_id = group
                                break

                        if not ref_id:
                            continue

                        # Create unique key
                        ref_key = (ref_type, ref_id)
                        if ref_key in seen_refs:
                            continue
                        seen_refs.add(ref_key)

                        # Extract context (surrounding text)
                        start = max(0, match.start() - 30)
                        end = min(len(text), match.end() + 30)
                        context = text[start:end].strip()

                        cross_ref = CrossReference(
                            ref_type=ref_type,
                            ref_id=ref_id,
                            source_page=page_num,
                            source_text=context,
                            target_id=None,  # Would need element matching to resolve
                        )
                        cross_refs.append(cross_ref)

        except Exception as e:
            logger.warning(f"Error extracting cross-references: {e}")

        logger.debug(f"Found {len(cross_refs)} cross-references")
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
        # Check caption before figure to handle "figure-caption" correctly
        elif 'caption' in label_lower:
            return "caption"
        elif any(kw in label_lower for kw in ['figure', 'image', 'picture']):
            return "figure"
        elif 'list' in label_lower:
            return "list"
        elif 'code' in label_lower:
            return "code"
        elif 'formula' in label_lower or 'equation' in label_lower:
            return "equation"
        else:
            return "text"

    def _infer_heading_level(self, item) -> int:
        """Infer heading level from item."""
        # Try to extract level from label
        if hasattr(item, 'label'):
            label = str(item.label).lower()

            # Look for level indicators
            for i in range(1, 7):
                if f'h{i}' in label or f'level{i}' in label or f'level-{i}' in label:
                    return i

            # Check for numbered patterns like "heading_1" or "section1"
            match = re.search(r'(?:heading|section|h)[-_]?(\d)', label)
            if match:
                return int(match.group(1))

        # Try to infer from text formatting/length
        if hasattr(item, 'text'):
            text = str(item.text)
            # Short, all-caps text is likely a major heading
            if len(text) < 50 and text.isupper():
                return 1
            # Numbered sections like "1. Introduction" or "1.2.3 Details"
            match = re.match(r'^(\d+(?:\.\d+)*)\s*[.\-:)]?\s*\w', text)
            if match:
                level = len(match.group(1).split('.'))
                return min(level, 6)

        return 1  # Default to level 1

    def _get_page_number(self, item) -> Optional[int]:
        """Extract page number from item."""
        try:
            # Direct page attribute
            if hasattr(item, 'page'):
                return item.page

            # Through prov (provenance) object
            if hasattr(item, 'prov'):
                prov = item.prov
                if isinstance(prov, list) and len(prov) > 0:
                    prov = prov[0]
                if hasattr(prov, 'page_no'):
                    return prov.page_no
                if hasattr(prov, 'page'):
                    return prov.page

            # Through self_ref or location
            if hasattr(item, 'self_ref'):
                # Parse page from reference path
                ref = str(item.self_ref)
                match = re.search(r'page[_-]?(\d+)', ref, re.IGNORECASE)
                if match:
                    return int(match.group(1))

        except Exception:
            pass
        return None

    def _get_bounding_box(self, item) -> Optional[Dict[str, float]]:
        """Extract bounding box from item."""
        try:
            bbox = None

            # Direct bbox attribute
            if hasattr(item, 'bbox'):
                bbox = item.bbox
            # Through prov (provenance)
            elif hasattr(item, 'prov'):
                prov = item.prov
                if isinstance(prov, list) and len(prov) > 0:
                    prov = prov[0]
                if hasattr(prov, 'bbox'):
                    bbox = prov.bbox

            if bbox is not None:
                # Handle different bbox formats
                if hasattr(bbox, 'l'):
                    # Named attributes (left, top, right, bottom)
                    return {
                        "x": float(bbox.l) if hasattr(bbox, 'l') else 0,
                        "y": float(bbox.t) if hasattr(bbox, 't') else 0,
                        "width": float(bbox.r - bbox.l) if hasattr(bbox, 'r') else 0,
                        "height": float(bbox.b - bbox.t) if hasattr(bbox, 'b') else 0,
                    }
                elif hasattr(bbox, 'x0'):
                    # Alternative naming (x0, y0, x1, y1)
                    return {
                        "x": float(bbox.x0),
                        "y": float(bbox.y0),
                        "width": float(bbox.x1 - bbox.x0),
                        "height": float(bbox.y1 - bbox.y0),
                    }
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    # Array format [x0, y0, x1, y1]
                    return {
                        "x": float(bbox[0]),
                        "y": float(bbox[1]),
                        "width": float(bbox[2] - bbox[0]),
                        "height": float(bbox[3] - bbox[1]),
                    }

        except Exception:
            pass
        return None

    def _infer_table_dimensions(self, item) -> Tuple[int, int]:
        """
        Infer table dimensions (rows, cols) from item.

        Deprecated: Use _extract_table_structure instead.
        """
        table_text = str(item.text) if hasattr(item, 'text') else ""
        rows, cols, _ = self._parse_table_from_text(table_text)
        return rows, cols
