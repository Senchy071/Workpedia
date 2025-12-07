"""Semantic chunker that preserves document structure for RAG."""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a document chunk with metadata.

    Attributes:
        chunk_id: Unique identifier for this chunk
        doc_id: Parent document ID
        content: Text content of the chunk
        token_count: Approximate token count
        metadata: Chunk metadata including position, type, etc.
    """
    chunk_id: str
    doc_id: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class SemanticChunker:
    """
    Semantic chunker that preserves document structure.

    Features:
    - Structure-aware chunking (respects section boundaries)
    - Configurable chunk size with overlap
    - Table and figure preservation
    - Metadata tracking (page numbers, section hierarchy)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: float = CHUNK_OVERLAP,
        min_chunk_size: int = 50,
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            overlap: Overlap ratio between chunks (default: 0.15)
            min_chunk_size: Minimum chunk size to avoid tiny chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.overlap_tokens = int(chunk_size * overlap)
        self.min_chunk_size = min_chunk_size

        logger.info(
            f"SemanticChunker initialized: size={chunk_size}, "
            f"overlap={overlap:.0%} ({self.overlap_tokens} tokens)"
        )

    def chunk_document(
        self,
        parsed_doc: Dict[str, Any],
        preserve_tables: bool = True,
        preserve_figures: bool = True,
    ) -> List[Chunk]:
        """
        Chunk a parsed document while preserving structure.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            preserve_tables: Keep tables as single chunks
            preserve_figures: Keep figures with captions as single chunks

        Returns:
            List of Chunk objects
        """
        doc_id = parsed_doc.get("doc_id", "unknown")
        raw_text = parsed_doc.get("raw_text", "")
        metadata = parsed_doc.get("metadata", {})

        if not raw_text:
            logger.warning(f"Document {doc_id} has no text content")
            return []

        # Extract structural elements
        elements = self._extract_elements(raw_text, preserve_tables, preserve_figures)

        # Create chunks from elements
        chunks = self._create_chunks(elements, doc_id, metadata)

        logger.info(
            f"Document {doc_id} chunked: {len(chunks)} chunks from "
            f"{self._estimate_tokens(raw_text)} tokens"
        )

        return chunks

    def _extract_elements(
        self,
        text: str,
        preserve_tables: bool,
        preserve_figures: bool,
    ) -> List[Dict[str, Any]]:
        """
        Extract structural elements from markdown text.

        Identifies:
        - Sections (headers)
        - Tables
        - Figures (images with captions)
        - Text paragraphs
        """
        elements = []
        current_section = None
        current_level = 0

        # Split by lines for processing
        lines = text.split('\n')
        current_text = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for section headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save accumulated text
                if current_text:
                    elements.append({
                        "type": "text",
                        "content": '\n'.join(current_text).strip(),
                        "section": current_section,
                        "level": current_level,
                    })
                    current_text = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = title
                current_level = level

                elements.append({
                    "type": "section",
                    "content": line,
                    "title": title,
                    "level": level,
                })
                i += 1
                continue

            # Check for tables (markdown table format)
            if preserve_tables and line.strip().startswith('|'):
                # Save accumulated text
                if current_text:
                    elements.append({
                        "type": "text",
                        "content": '\n'.join(current_text).strip(),
                        "section": current_section,
                        "level": current_level,
                    })
                    current_text = []

                # Collect entire table
                table_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1

                elements.append({
                    "type": "table",
                    "content": '\n'.join(table_lines),
                    "section": current_section,
                    "level": current_level,
                })
                continue

            # Check for figures (markdown images)
            if preserve_figures and re.match(r'!\[.*\]\(.*\)', line):
                # Save accumulated text
                if current_text:
                    elements.append({
                        "type": "text",
                        "content": '\n'.join(current_text).strip(),
                        "section": current_section,
                        "level": current_level,
                    })
                    current_text = []

                # Include figure with potential caption
                figure_content = [line]
                i += 1
                # Check for caption on next line (often italic text)
                if i < len(lines) and lines[i].strip().startswith('*'):
                    figure_content.append(lines[i])
                    i += 1

                elements.append({
                    "type": "figure",
                    "content": '\n'.join(figure_content),
                    "section": current_section,
                    "level": current_level,
                })
                continue

            # Regular text line
            current_text.append(line)
            i += 1

        # Don't forget remaining text
        if current_text:
            content = '\n'.join(current_text).strip()
            if content:
                elements.append({
                    "type": "text",
                    "content": content,
                    "section": current_section,
                    "level": current_level,
                })

        return elements

    def _create_chunks(
        self,
        elements: List[Dict[str, Any]],
        doc_id: str,
        doc_metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create chunks from structural elements.

        - Tables and figures are kept as single chunks (if under 2x chunk_size)
        - Text is split with overlap
        - Section context is preserved in metadata
        """
        chunks = []
        chunk_index = 0
        current_content = []
        current_tokens = 0
        current_section = None

        for element in elements:
            elem_type = element["type"]
            content = element["content"]
            elem_tokens = self._estimate_tokens(content)

            # Update section context
            if elem_type == "section":
                current_section = element.get("title")

            # Tables and figures: keep as single chunks
            if elem_type in ("table", "figure"):
                # Flush current text chunk first
                if current_content:
                    chunk = self._create_chunk(
                        '\n\n'.join(current_content),
                        doc_id,
                        chunk_index,
                        doc_metadata,
                        current_section,
                        "text",
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_content = []
                    current_tokens = 0

                # Create chunk for table/figure
                chunk = self._create_chunk(
                    content,
                    doc_id,
                    chunk_index,
                    doc_metadata,
                    current_section,
                    elem_type,
                )
                chunks.append(chunk)
                chunk_index += 1
                continue

            # Text and section headers: accumulate with overlap handling
            # If adding this element would exceed chunk size, flush first
            if current_tokens + elem_tokens > self.chunk_size and current_content:
                chunk = self._create_chunk(
                    '\n\n'.join(current_content),
                    doc_id,
                    chunk_index,
                    doc_metadata,
                    current_section,
                    "text",
                )
                chunks.append(chunk)
                chunk_index += 1

                # Keep overlap from end of current content
                overlap_content = self._get_overlap_content(current_content)
                current_content = overlap_content
                current_tokens = self._estimate_tokens('\n\n'.join(current_content))

            current_content.append(content)
            current_tokens += elem_tokens

        # Flush remaining content
        if current_content:
            content = '\n\n'.join(current_content)
            if self._estimate_tokens(content) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    content,
                    doc_id,
                    chunk_index,
                    doc_metadata,
                    current_section,
                    "text",
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        index: int,
        doc_metadata: Dict[str, Any],
        section: Optional[str],
        chunk_type: str,
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        chunk_id = f"{doc_id}_chunk_{index}"
        token_count = self._estimate_tokens(content)

        metadata = {
            "chunk_index": index,
            "chunk_type": chunk_type,
            "section": section,
            "filename": doc_metadata.get("filename", ""),
            "file_path": doc_metadata.get("file_path", ""),
            "doc_pages": doc_metadata.get("pages", 0),
        }

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            token_count=token_count,
            metadata=metadata,
        )

    def _get_overlap_content(self, content_list: List[str]) -> List[str]:
        """Get content for overlap from the end of content list."""
        if not content_list:
            return []

        # Take content from end that fits in overlap
        overlap_parts = []
        total_tokens = 0

        for content in reversed(content_list):
            tokens = self._estimate_tokens(content)
            if total_tokens + tokens <= self.overlap_tokens:
                overlap_parts.insert(0, content)
                total_tokens += tokens
            else:
                break

        return overlap_parts

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.

        Uses simple heuristic: ~4 characters per token for English text.
        This is an approximation; actual tokenization depends on the model.
        """
        if not text:
            return 0
        # Rough estimate: 4 chars per token is common for English
        return len(text) // 4


def chunk_document(
    parsed_doc: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    overlap: float = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Convenience function to chunk a document.

    Args:
        parsed_doc: Output from DocumentParser.parse()
        chunk_size: Target chunk size in tokens
        overlap: Overlap ratio between chunks

    Returns:
        List of Chunk objects
    """
    chunker = SemanticChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_document(parsed_doc)
