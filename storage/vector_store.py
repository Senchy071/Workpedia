"""ChromaDB vector store for document chunk storage and retrieval."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from config.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from core.chunker import Chunk
from core.exceptions import (
    IndexingError,
    VectorStoreConnectionError,
    VectorStoreQueryError,
)

if TYPE_CHECKING:
    from core.chunker import SemanticChunker
    from core.embedder import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for RAG.

    Features:
    - Persistent storage to disk
    - Similarity search with metadata filtering
    - Batch insertion for efficiency
    - Document management (add, delete, update)
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Ensure persist directory exists
        try:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise VectorStoreConnectionError(
                store_path=persist_directory, reason=f"Failed to create persist directory: {e}"
            ) from e

        # Initialize ChromaDB client with persistence
        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

            logger.info(
                f"VectorStore initialized: {persist_directory}/{collection_name} "
                f"({self._collection.count()} documents)"
            )
        except Exception as e:
            raise VectorStoreConnectionError(
                store_path=persist_directory, reason=f"Failed to initialize ChromaDB: {e}"
            ) from e

    @property
    def count(self) -> int:
        """Get total number of documents in the collection."""
        return self._collection.count()

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[np.ndarray],
        batch_size: int = 100,
    ) -> int:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)
            batch_size: Batch size for insertion

        Returns:
            Number of chunks added

        Raises:
            IndexingError: If adding chunks fails
        """
        if len(chunks) != len(embeddings):
            reason = (
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have same length"
            )
            raise IndexingError(
                doc_id=chunks[0].doc_id if chunks else "unknown",
                reason=reason,
            )

        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "chunk_type": chunk.metadata.get("chunk_type", "text"),
                "section": chunk.metadata.get("section", ""),
                "filename": chunk.metadata.get("filename", ""),
                "file_path": chunk.metadata.get("file_path", ""),
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Add in batches
        total_added = 0
        try:
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))

                self._collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings_list[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                )
                total_added += end_idx - i

                if len(chunks) > batch_size:
                    logger.debug(f"Added batch {i//batch_size + 1}: {total_added}/{len(chunks)}")

            logger.info(f"Added {total_added} chunks to vector store")
            return total_added
        except Exception as e:
            raise IndexingError(
                doc_id=chunks[0].doc_id if chunks else "unknown",
                reason=f"Failed to add chunks to ChromaDB: {e}",
            ) from e

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"doc_id": "abc123"})
            include: What to include in results (default: documents, metadatas, distances)

        Returns:
            Dictionary with keys: ids, documents, metadatas, distances

        Raises:
            VectorStoreQueryError: If query fails
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=include,
            )

            # Flatten results (query returns nested lists)
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results.get("documents") else [],
                "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
                "distances": results["distances"][0] if results.get("distances") else [],
            }
        except Exception as e:
            raise VectorStoreQueryError(
                reason=f"ChromaDB query failed: {e}", query_length=len(query_embedding)
            ) from e

    def query_text(
        self,
        query_text: str,
        embedder: "Embedder",
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query using text (embedder generates the embedding).

        Args:
            query_text: Text query
            embedder: Embedder instance to generate query embedding
            n_results: Number of results to return
            where: Metadata filter

        Returns:
            Query results with ids, documents, metadatas, distances
        """
        query_embedding = embedder.embed(query_text)
        return self.query(query_embedding, n_results=n_results, where=where)

    def get_by_doc_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Get all chunks for a specific document.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Dictionary with ids, documents, metadatas for the document

        Raises:
            VectorStoreQueryError: If get operation fails
        """
        try:
            results = self._collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
            )
            return results
        except Exception as e:
            raise VectorStoreQueryError(reason=f"Failed to get document by ID: {e}") from e

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks for a specific document.

        Args:
            doc_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        # Get chunks for this document first
        existing = self.get_by_doc_id(doc_id)
        chunk_ids = existing.get("ids", [])

        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for document {doc_id}")

        return len(chunk_ids)

    def delete_by_ids(self, chunk_ids: List[str]) -> int:
        """
        Delete specific chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0

        self._collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks")
        return len(chunk_ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the store.

        Returns:
            List of document info dicts with doc_id, filename, chunk_count
        """
        # Get all metadata
        all_data = self._collection.get(include=["metadatas"])

        # Group by doc_id
        docs = {}
        for metadata in all_data.get("metadatas", []):
            doc_id = metadata.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename", ""),
                    "file_path": metadata.get("file_path", ""),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def clear(self) -> int:
        """
        Clear all documents from the collection.

        Returns:
            Number of documents deleted
        """
        count = self._collection.count()

        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"Cleared {count} documents from vector store")
        return count

    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection stats
        """
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks": self._collection.count(),
            "documents": len(self.list_documents()),
        }


class DocumentIndexer:
    """
    High-level interface for indexing documents into the vector store.

    Combines chunking, embedding, and storage into a single workflow.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional["Embedder"] = None,
        chunker: Optional["SemanticChunker"] = None,
    ):
        """
        Initialize document indexer.

        Args:
            vector_store: VectorStore instance (creates default if None)
            embedder: Embedder instance (creates default if None)
            chunker: SemanticChunker instance (creates default if None)
        """
        # Lazy imports to avoid circular dependencies
        if vector_store is None:
            vector_store = VectorStore()
        if embedder is None:
            from core.embedder import Embedder

            embedder = Embedder()
        if chunker is None:
            from core.chunker import SemanticChunker

            chunker = SemanticChunker()

        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker

        logger.info("DocumentIndexer initialized")

    def index_document(
        self,
        parsed_doc: Dict[str, Any],
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Index a parsed document into the vector store.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            replace_existing: Delete existing chunks for this doc first

        Returns:
            Indexing result with doc_id, chunks_added, etc.
        """
        doc_id = parsed_doc.get("doc_id", "unknown")
        filename = parsed_doc.get("metadata", {}).get("filename", "unknown")

        logger.info(f"Indexing document: {filename} ({doc_id})")

        # Delete existing if requested
        if replace_existing:
            deleted = self.vector_store.delete_by_doc_id(doc_id)
            if deleted > 0:
                logger.info(f"Deleted {deleted} existing chunks for {doc_id}")

        # Chunk the document
        chunks = self.chunker.chunk_document(parsed_doc)

        if not chunks:
            logger.warning(f"No chunks generated for document {doc_id}")
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks_added": 0,
                "status": "empty",
            }

        # Create synthetic Table of Contents chunk
        toc_chunk = self._create_toc_chunk(parsed_doc, doc_id)
        if toc_chunk:
            chunks.insert(0, toc_chunk)  # Add at beginning
            logger.info(f"Created synthetic TOC chunk with {len(toc_chunk.content)} characters")

        # Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks)

        # Store in vector store
        added = self.vector_store.add_chunks(chunks, embeddings)

        logger.info(
            f"Indexed document {filename}: {added} chunks, "
            f"total store size: {self.vector_store.count}"
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_added": added,
            "total_tokens": sum(c.token_count for c in chunks),
            "status": "success",
        }

    def _create_toc_chunk(
        self,
        parsed_doc: Dict[str, Any],
        doc_id: str,
    ) -> Optional[Chunk]:
        """
        Create a synthetic Table of Contents chunk from document structure.

        This chunk lists all major sections/chapters and is helpful for
        queries like "List main chapters" or "What are the sections?".

        Strategy:
        1. First try StructureAnalyzer to extract sections from parsed structure
        2. If no sections found, parse the actual TOC from raw text

        Args:
            parsed_doc: Output from DocumentParser.parse()
            doc_id: Document ID

        Returns:
            Chunk object with TOC, or None if no structure found
        """
        toc_content = None
        sections_count = 0
        extraction_method = None

        # Strategy 1: Try StructureAnalyzer
        try:
            from core.analyzer import StructureAnalyzer

            docling_doc = parsed_doc.get("docling_document")
            if docling_doc:
                analyzer = StructureAnalyzer()
                structure = analyzer.analyze(docling_doc)
                sections = structure.sections

                if sections:
                    toc_lines = [
                        "# Table of Contents",
                        "",
                        "This document contains the following sections and chapters:",
                        "",
                    ]

                    for section in sections:
                        level = section.get("level", 1)
                        text = section.get("text", "").strip()
                        page = section.get("page")

                        if not text:
                            continue

                        indent = "  " * (level - 1)
                        if page:
                            toc_lines.append(f"{indent}- {text} (page {page})")
                        else:
                            toc_lines.append(f"{indent}- {text}")

                    if len(toc_lines) > 4:
                        toc_content = "\n".join(toc_lines)
                        sections_count = len(sections)
                        extraction_method = "structure_analyzer"
                        logger.info(
                            f"TOC extracted via StructureAnalyzer: {sections_count} sections"
                        )

        except Exception as e:
            logger.debug(f"StructureAnalyzer failed: {e}")

        # Strategy 2: Parse TOC from raw text (fallback)
        if not toc_content:
            raw_text = parsed_doc.get("raw_text", "")
            if raw_text:
                toc_content, sections_count = self._extract_toc_from_text(raw_text)
                if toc_content:
                    extraction_method = "text_parser"
                    logger.info(f"TOC extracted via text parsing: {sections_count} entries")

        if not toc_content:
            logger.debug("No TOC could be extracted from document")
            return None

        # Create chunk with special metadata
        chunk = Chunk(
            chunk_id=f"{doc_id}_toc",
            doc_id=doc_id,
            content=toc_content,
            token_count=len(toc_content) // 4,
            metadata={
                "chunk_index": -1,
                "chunk_type": "table_of_contents",
                "section": "Table of Contents",
                "filename": parsed_doc.get("metadata", {}).get("filename", ""),
                "file_path": parsed_doc.get("metadata", {}).get("file_path", ""),
                "doc_pages": parsed_doc.get("metadata", {}).get("pages", 0),
                "sections_count": sections_count,
                "extraction_method": extraction_method,
            },
        )

        return chunk

    def _extract_toc_from_text(self, raw_text: str) -> tuple[Optional[str], int]:
        """
        Extract Table of Contents from raw document text.

        Looks for common TOC markers and parses the structured content that follows.
        Handles both plain text and markdown table formats.

        Args:
            raw_text: Full document text

        Returns:
            Tuple of (toc_content, entry_count) or (None, 0) if not found
        """
        import re

        # Common TOC header patterns (including markdown headings)
        toc_markers = [
            r"(?:^|\n)\s*#{1,3}\s*(?:TABLE\s+OF\s+CONTENTS|CONTENTS|Table\s+of\s+Contents)\s*\n",
            r"(?:^|\n)\s*(?:TABLE\s+OF\s+CONTENTS|CONTENTS|Table\s+of\s+Contents|Contents)\s*\n",
        ]

        toc_start = None
        for pattern in toc_markers:
            match = re.search(pattern, raw_text[:100000], re.IGNORECASE)  # Search first 100k chars
            if match:
                toc_start = match.end()
                break

        if toc_start is None:
            return None, 0

        # Extract TOC section (typically ends at a major section or after many lines)
        toc_section = raw_text[toc_start : toc_start + 30000]  # Max 30k chars for TOC

        # Parse TOC entries - look for lines with page numbers or chapter patterns
        toc_entries = []
        lines = toc_section.split("\n")

        # Patterns for TOC entries (handle both plain text and markdown table formats)
        entry_patterns = [
            # Markdown table: "| TITLE  .  .  .  . 15 |" (dots with spaces between)
            re.compile(
                r"^\|\s*([A-Za-z][A-Za-z\s\-/,\(\)\']+?)(?:\s+\.)+\s*(\d+)\s*\|", re.IGNORECASE
            ),
            # Markdown table: "| Title . . . . . . .15 |"
            re.compile(r"^\|\s*([A-Za-z][^|]+?)\s*[\.]+\s*(\d+)\s*\|"),
            # "Chapter 1 Title ... 15" or "1. Introduction ... 5"
            re.compile(
                r"^[\|\s]*(?:Chapter\s+)?(\d+[\.\d]*\.?\s+.+?)\s*[\.…\s]+(\d+)\s*\|?\s*$",
                re.IGNORECASE,
            ),
            # Markdown table format: "| TITLE . . . . 15 |" or "| TITLE ... 15 |"
            re.compile(r"^\|?\s*(?:CHAPTER\s+\d+\s+)?([A-Z][A-Z\s\-/,]+?)[\s\.]+(\d+)\s*\|?\s*$"),
            # Title with dots and page number (handles markdown table cell)
            re.compile(r"^\|?\s*([A-Za-z][A-Za-z\s\-/,\(\)]+?)\s*[\.\s]{2,}(\d+)\s*\|?\s*$"),
            # "INTRODUCTION ... 1" (all caps heading)
            re.compile(r"^\|?\s*([A-Z][A-Z\s\-/]+(?:[A-Z]|\d))\s*[\.…\s]+(\d+)\s*\|?\s*$"),
            # "Appendix A: Title ... 150"
            re.compile(
                r"^\|?\s*((?:Appendix|Annex|Part|Chapter)\s+[A-Z\d]+[:\.\s].+?)\s*[\.…\s]+(\d+)\s*\|?\s*$",
                re.IGNORECASE,
            ),
            # Generic: "Title text ... 15" with dots/spaces before page number
            re.compile(r"^\|?\s*([A-Z][^|]{5,80}?)\s*[\.…\s]{3,}(\d+)\s*\|?\s*$"),
        ]

        # Also detect chapter headers like "CHAPTER 1 THE ALLIANCE'S..."
        chapter_header_pattern = re.compile(
            r"^\|?\s*(CHAPTER\s+\d+\s+[A-Z][A-Z\s\-/,]+)", re.IGNORECASE
        )

        consecutive_non_toc = 0
        seen_entries = set()  # Avoid duplicates

        for line in lines:
            line = line.strip()

            # Skip empty lines and table separators
            if not line or line.startswith("|--") or line.startswith("|-"):
                continue

            # Skip lines that are just table formatting
            if re.match(r"^[\|\s\-:]+$", line):
                continue

            # Check for chapter header first
            chapter_match = chapter_header_pattern.match(line)
            if chapter_match:
                title = chapter_match.group(1).strip()
                title = re.sub(r"[\.\s|]+$", "", title)
                if title and title not in seen_entries and len(title) > 5:
                    seen_entries.add(title)
                    toc_entries.append(f"- {title}")
                    consecutive_non_toc = 0
                    continue

            # Check if line matches a TOC entry pattern
            matched = False
            for pattern in entry_patterns:
                match = pattern.match(line)
                if match:
                    title = match.group(1).strip()
                    page = match.group(2).strip()

                    # Clean up title (remove excessive dots/spaces/pipes)
                    title = re.sub(r"[\.\s\|]+$", "", title)
                    title = re.sub(r"\s+", " ", title)

                    # Filter out garbage entries
                    if len(title) > 3 and len(title) < 100 and title not in seen_entries:
                        # Skip entries that look like just numbers or symbols
                        if re.search(r"[a-zA-Z]{2,}", title):
                            seen_entries.add(title)
                            toc_entries.append(f"- {title} (page {page})")
                            matched = True
                            consecutive_non_toc = 0
                            break

            if not matched:
                consecutive_non_toc += 1
                # Stop if we hit many non-TOC lines (likely end of TOC)
                if consecutive_non_toc > 15:
                    break

        if len(toc_entries) < 3:
            return None, 0

        # Build TOC content with semantic keywords for better retrieval
        toc_lines = [
            "# TABLE OF CONTENTS - Document Structure and Main Chapters",
            "",
            "TABLE OF CONTENTS: This is the complete list of main chapters and sections.",
            "The book/document contains the following chapters, sections, and topics:",
            "Main chapters, headings, and outline of this indexed book/document:",
            "",
        ] + toc_entries

        return "\n".join(toc_lines), len(toc_entries)

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            n_results: Number of results to return
            doc_id: Optional filter to specific document

        Returns:
            List of result dicts with content, metadata, and similarity score
        """
        where = {"doc_id": doc_id} if doc_id else None

        results = self.vector_store.query_text(
            query_text=query,
            embedder=self.embedder,
            n_results=n_results,
            where=where,
        )

        # Format results
        formatted = []
        for i, (doc_id, content, metadata, distance) in enumerate(
            zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                results["distances"],
            )
        ):
            formatted.append(
                {
                    "rank": i + 1,
                    "chunk_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity": 1 - distance,  # Convert distance to similarity
                }
            )

        return formatted
