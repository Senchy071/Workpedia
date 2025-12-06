"""Document merger for combining parsed chunk results."""

import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a processed chunk."""

    chunk_index: int
    start_page: int  # 1-indexed
    end_page: int  # 1-indexed, inclusive
    parse_result: Dict[str, Any]
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class MergeResult:
    """Result of merging document chunks."""

    doc_id: str
    metadata: Dict[str, Any]
    raw_text: str
    structure: Dict[str, Any]
    chunks: List[ChunkInfo]
    total_pages: int
    merge_time: float = 0.0


class DocumentMerger:
    """
    Merges parsed document chunks back into a unified result.

    Features:
    - Text concatenation with page boundary markers
    - Metadata aggregation with proper page offsets
    - Structure merging for tables, sections, figures
    - Cross-reference resolution across chunks
    """

    def __init__(self):
        """Initialize document merger."""
        logger.info("DocumentMerger initialized")

    def merge(
        self,
        chunks: List[ChunkInfo],
        original_path: Path,
        total_pages: int,
    ) -> MergeResult:
        """
        Merge parsed chunk results into unified document.

        Args:
            chunks: List of ChunkInfo with parse results
            original_path: Path to original document
            total_pages: Total pages in original document

        Returns:
            MergeResult with unified document data
        """
        start_time = datetime.now()

        if not chunks:
            raise ValueError("No chunks to merge")

        # Filter successful chunks
        successful_chunks = [c for c in chunks if c.success and c.parse_result]

        if not successful_chunks:
            raise ValueError("No successful chunks to merge")

        logger.info(
            f"Merging {len(successful_chunks)}/{len(chunks)} chunks "
            f"for {original_path.name}"
        )

        # Generate new doc_id for merged document
        doc_id = str(uuid.uuid4())

        # Merge text with page markers
        merged_text = self._merge_text(successful_chunks)

        # Merge structure information
        merged_structure = self._merge_structure(successful_chunks, total_pages)

        # Build merged metadata
        merged_metadata = self._merge_metadata(
            successful_chunks,
            original_path,
            total_pages,
            len(chunks),
            len(successful_chunks),
        )

        merge_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Merge complete: {total_pages} pages from {len(successful_chunks)} chunks "
            f"in {merge_time:.2f}s"
        )

        return MergeResult(
            doc_id=doc_id,
            metadata=merged_metadata,
            raw_text=merged_text,
            structure=merged_structure,
            chunks=chunks,
            total_pages=total_pages,
            merge_time=merge_time,
        )

    def _merge_text(self, chunks: List[ChunkInfo]) -> str:
        """Merge text content from all chunks."""
        text_parts = []

        for chunk in sorted(chunks, key=lambda c: c.start_page):
            chunk_text = chunk.parse_result.get("raw_text", "")

            if chunk_text:
                # Add page range marker
                marker = f"\n\n<!-- Pages {chunk.start_page}-{chunk.end_page} -->\n\n"
                text_parts.append(marker + chunk_text)

        return "".join(text_parts).strip()

    def _merge_structure(
        self, chunks: List[ChunkInfo], total_pages: int
    ) -> Dict[str, Any]:
        """Merge structure information from all chunks."""
        merged = {
            "pages": total_pages,
            "has_tables": False,
            "has_figures": False,
            "sections": [],
            "tables": [],
            "figures": [],
            "chunk_boundaries": [],
        }

        for chunk in sorted(chunks, key=lambda c: c.start_page):
            chunk_structure = chunk.parse_result.get("structure", {})
            page_offset = chunk.start_page - 1

            # Track table/figure presence
            if chunk_structure.get("has_tables"):
                merged["has_tables"] = True
            if chunk_structure.get("has_figures"):
                merged["has_figures"] = True

            # Record chunk boundary for reference
            merged["chunk_boundaries"].append({
                "chunk_index": chunk.chunk_index,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
            })

            # Merge sections with page offset
            for section in chunk_structure.get("sections", []):
                adjusted_section = section.copy()
                if adjusted_section.get("page"):
                    adjusted_section["page"] += page_offset
                adjusted_section["source_chunk"] = chunk.chunk_index
                merged["sections"].append(adjusted_section)

            # Merge tables with page offset
            for table in chunk_structure.get("tables", []):
                adjusted_table = self._adjust_table_pages(table, page_offset)
                adjusted_table["source_chunk"] = chunk.chunk_index
                merged["tables"].append(adjusted_table)

            # Merge figures with page offset
            for figure in chunk_structure.get("figures", []):
                adjusted_figure = figure.copy()
                if adjusted_figure.get("page"):
                    adjusted_figure["page"] += page_offset
                adjusted_figure["source_chunk"] = chunk.chunk_index
                merged["figures"].append(adjusted_figure)

        # Detect multi-page tables at chunk boundaries
        merged["tables"] = self._detect_boundary_tables(
            merged["tables"], merged["chunk_boundaries"]
        )

        return merged

    def _adjust_table_pages(
        self, table: Dict[str, Any], page_offset: int
    ) -> Dict[str, Any]:
        """Adjust table page numbers with offset."""
        adjusted = table.copy()

        # Handle single page
        if adjusted.get("page"):
            adjusted["page"] += page_offset

        # Handle page list (for multi-page tables)
        if adjusted.get("page_numbers"):
            adjusted["page_numbers"] = [
                p + page_offset for p in adjusted["page_numbers"]
            ]

        return adjusted

    def _detect_boundary_tables(
        self,
        tables: List[Dict[str, Any]],
        chunk_boundaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect tables that may span chunk boundaries.

        Marks tables near boundaries as potentially incomplete.
        """
        if len(chunk_boundaries) <= 1:
            return tables

        # Get boundary pages
        boundary_pages = set()
        for boundary in chunk_boundaries:
            boundary_pages.add(boundary["end_page"])
            boundary_pages.add(boundary["start_page"])

        # Mark tables near boundaries
        for table in tables:
            table_pages = table.get("page_numbers", [])
            if not table_pages and table.get("page"):
                table_pages = [table["page"]]

            # Check if table is at a chunk boundary
            for page in table_pages:
                if page in boundary_pages:
                    table["at_chunk_boundary"] = True
                    table["may_be_incomplete"] = True
                    break

        return tables

    def _merge_metadata(
        self,
        chunks: List[ChunkInfo],
        original_path: Path,
        total_pages: int,
        total_chunks: int,
        successful_chunks: int,
    ) -> Dict[str, Any]:
        """Build merged metadata from chunk results."""
        # Calculate total processing time
        total_processing_time = sum(c.processing_time for c in chunks)

        # Get file size from first chunk (should be same for all)
        first_result = chunks[0].parse_result
        file_size_mb = first_result.get("metadata", {}).get("file_size_mb", 0)

        return {
            "filename": original_path.name,
            "file_path": str(original_path),
            "file_size_mb": file_size_mb,
            "pages": total_pages,
            "processing_time_seconds": round(total_processing_time, 2),
            "processed_in_chunks": True,
            "num_chunks": total_chunks,
            "chunks_processed": successful_chunks,
            "chunks_failed": total_chunks - successful_chunks,
            "parsed_at": datetime.now().isoformat(),
            "parser_version": "docling_v2_chunked",
            "chunk_details": [
                {
                    "index": c.chunk_index,
                    "pages": f"{c.start_page}-{c.end_page}",
                    "success": c.success,
                    "time": round(c.processing_time, 2),
                }
                for c in chunks
            ],
        }

    def to_parser_result(self, merge_result: MergeResult) -> Dict[str, Any]:
        """
        Convert MergeResult to standard parser result format.

        This makes merged results compatible with the rest of the pipeline.
        """
        return {
            "doc_id": merge_result.doc_id,
            "metadata": merge_result.metadata,
            "docling_document": None,  # Not available for merged docs
            "raw_text": merge_result.raw_text,
            "structure": merge_result.structure,
        }


def create_chunk_info(
    chunk_index: int,
    start_page: int,
    end_page: int,
    parse_result: Optional[Dict[str, Any]] = None,
    processing_time: float = 0.0,
    success: bool = True,
    error: Optional[str] = None,
) -> ChunkInfo:
    """Helper function to create ChunkInfo objects."""
    return ChunkInfo(
        chunk_index=chunk_index,
        start_page=start_page,
        end_page=end_page,
        parse_result=parse_result or {},
        processing_time=processing_time,
        success=success,
        error=error,
    )
