"""ChromaDB vector store for document chunk storage and retrieval."""

import logging
import re
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
    from core.hybrid_search import HybridSearcher
    from core.suggestions import QuerySuggestionGenerator
    from core.summarizer import DocumentSummarizer

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
        collection_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)
            batch_size: Batch size for insertion
            collection_name: Optional collection name to assign to all chunks
            tags: Optional tags to assign to all chunks (stored as comma-separated)

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

        # Prepare tags string for ChromaDB (comma-separated for filtering)
        tags_str = ",".join(sorted(set(t.lower().strip() for t in tags if t.strip()))) if tags else ""

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
                "collection": collection_name or "",
                "tags": tags_str,
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

    def update_document_collection(
        self,
        doc_id: str,
        collection_name: Optional[str],
    ) -> int:
        """
        Update the collection for all chunks of a document.

        Args:
            doc_id: Document ID
            collection_name: New collection name (or None/empty to remove)

        Returns:
            Number of chunks updated
        """
        existing = self.get_by_doc_id(doc_id)
        chunk_ids = existing.get("ids", [])

        if not chunk_ids:
            return 0

        # Update each chunk's metadata
        for chunk_id, metadata in zip(chunk_ids, existing.get("metadatas", [])):
            metadata["collection"] = collection_name or ""
            self._collection.update(
                ids=[chunk_id],
                metadatas=[metadata],
            )

        logger.info(f"Updated collection for {len(chunk_ids)} chunks of doc {doc_id}")
        return len(chunk_ids)

    def update_document_tags(
        self,
        doc_id: str,
        tags: List[str],
    ) -> int:
        """
        Update the tags for all chunks of a document.

        Args:
            doc_id: Document ID
            tags: New tags list

        Returns:
            Number of chunks updated
        """
        existing = self.get_by_doc_id(doc_id)
        chunk_ids = existing.get("ids", [])

        if not chunk_ids:
            return 0

        # Prepare tags string
        tags_str = ",".join(sorted(set(t.lower().strip() for t in tags if t.strip())))

        # Update each chunk's metadata
        for chunk_id, metadata in zip(chunk_ids, existing.get("metadatas", [])):
            metadata["tags"] = tags_str
            self._collection.update(
                ids=[chunk_id],
                metadatas=[metadata],
            )

        logger.info(f"Updated tags for {len(chunk_ids)} chunks of doc {doc_id}")
        return len(chunk_ids)

    def list_documents_by_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        List all documents in a specific collection.

        Args:
            collection_name: Collection name to filter by

        Returns:
            List of document info dicts
        """
        all_data = self._collection.get(
            where={"collection": collection_name},
            include=["metadatas"],
        )

        # Group by doc_id
        docs = {}
        for metadata in all_data.get("metadatas", []):
            doc_id = metadata.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename", ""),
                    "file_path": metadata.get("file_path", ""),
                    "collection": metadata.get("collection", ""),
                    "tags": metadata.get("tags", ""),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def list_documents_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        List all documents that have a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of document info dicts
        """
        tag = tag.lower().strip()

        # Get all chunks and filter by tag (ChromaDB doesn't support contains for strings)
        all_data = self._collection.get(include=["metadatas"])

        # Group by doc_id, filtering by tag
        docs = {}
        for metadata in all_data.get("metadatas", []):
            tags_str = metadata.get("tags", "")
            tags_list = [t.strip() for t in tags_str.split(",") if t.strip()]

            if tag in tags_list:
                doc_id = metadata.get("doc_id", "unknown")
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": metadata.get("filename", ""),
                        "file_path": metadata.get("file_path", ""),
                        "collection": metadata.get("collection", ""),
                        "tags": tags_str,
                        "chunk_count": 0,
                    }
                docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def get_all_collections(self) -> List[Dict[str, Any]]:
        """
        Get all unique collection names with document counts.

        Returns:
            List of dicts with collection name and document count
        """
        all_data = self._collection.get(include=["metadatas"])

        # Group by collection
        collections = {}
        for metadata in all_data.get("metadatas", []):
            collection = metadata.get("collection", "")
            if collection:
                doc_id = metadata.get("doc_id", "unknown")
                if collection not in collections:
                    collections[collection] = {"name": collection, "doc_ids": set()}
                collections[collection]["doc_ids"].add(doc_id)

        return [
            {"name": name, "document_count": len(data["doc_ids"])}
            for name, data in sorted(collections.items())
        ]

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """
        Get all unique tags with document counts.

        Returns:
            List of dicts with tag and document count
        """
        all_data = self._collection.get(include=["metadatas"])

        # Collect tags with doc_ids
        tags = {}
        for metadata in all_data.get("metadatas", []):
            tags_str = metadata.get("tags", "")
            doc_id = metadata.get("doc_id", "unknown")

            for tag in tags_str.split(","):
                tag = tag.strip()
                if tag:
                    if tag not in tags:
                        tags[tag] = {"tag": tag, "doc_ids": set()}
                    tags[tag]["doc_ids"].add(doc_id)

        return [
            {"tag": name, "document_count": len(data["doc_ids"])}
            for name, data in sorted(tags.items())
        ]

    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the summary chunk for a specific document.

        Args:
            doc_id: Document ID

        Returns:
            Summary chunk data with content and metadata, or None if not found
        """
        try:
            results = self._collection.get(
                where={"$and": [{"doc_id": doc_id}, {"chunk_type": "document_summary"}]},
                include=["documents", "metadatas"],
            )

            if results["ids"]:
                return {
                    "chunk_id": results["ids"][0],
                    "content": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {},
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get document summary: {e}")
            return None

    def get_document_suggestions(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the suggestions chunk for a specific document.

        Args:
            doc_id: Document ID

        Returns:
            Suggestions chunk data with content (JSON list) and metadata, or None if not found
        """
        try:
            results = self._collection.get(
                where={"$and": [{"doc_id": doc_id}, {"chunk_type": "query_suggestions"}]},
                include=["documents", "metadatas"],
            )

            if results["ids"]:
                import json

                content = results["documents"][0] if results.get("documents") else "[]"
                try:
                    suggestions = json.loads(content)
                except json.JSONDecodeError:
                    suggestions = []

                return {
                    "chunk_id": results["ids"][0],
                    "suggestions": suggestions,
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {},
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get document suggestions: {e}")
            return None


class DocumentIndexer:
    """
    High-level interface for indexing documents into the vector store.

    Combines chunking, embedding, and storage into a single workflow.
    Optionally generates document summaries, query suggestions, and hybrid search index.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional["Embedder"] = None,
        chunker: Optional["SemanticChunker"] = None,
        summarizer: Optional["DocumentSummarizer"] = None,
        suggestion_generator: Optional["QuerySuggestionGenerator"] = None,
        hybrid_searcher: Optional["HybridSearcher"] = None,
        enable_summaries: bool = True,
        enable_suggestions: bool = True,
        enable_hybrid_search: bool = True,
    ):
        """
        Initialize document indexer.

        Args:
            vector_store: VectorStore instance (creates default if None)
            embedder: Embedder instance (creates default if None)
            chunker: SemanticChunker instance (creates default if None)
            summarizer: DocumentSummarizer instance (creates default if None)
            suggestion_generator: QuerySuggestionGenerator (creates default if None)
            hybrid_searcher: HybridSearcher for BM25 indexing (creates default if None)
            enable_summaries: Generate summaries during indexing
            enable_suggestions: Generate query suggestions during indexing
            enable_hybrid_search: Index chunks in BM25 for hybrid search
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
        if summarizer is None and enable_summaries:
            from core.summarizer import DocumentSummarizer

            summarizer = DocumentSummarizer()
        if suggestion_generator is None and enable_suggestions:
            from core.suggestions import QuerySuggestionGenerator

            suggestion_generator = QuerySuggestionGenerator()
        if hybrid_searcher is None and enable_hybrid_search:
            from config.config import HYBRID_SEARCH_INDEX_PATH
            from core.hybrid_search import BM25Index, HybridSearcher

            bm25_index = BM25Index(persist_path=HYBRID_SEARCH_INDEX_PATH)
            hybrid_searcher = HybridSearcher(bm25_index=bm25_index)

        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker
        self.summarizer = summarizer
        self.suggestion_generator = suggestion_generator
        self.hybrid_searcher = hybrid_searcher
        self.enable_summaries = enable_summaries
        self.enable_suggestions = enable_suggestions
        self.enable_hybrid_search = enable_hybrid_search

        logger.info(
            f"DocumentIndexer initialized (summaries={enable_summaries}, "
            f"suggestions={enable_suggestions}, hybrid_search={enable_hybrid_search})"
        )

    def index_document(
        self,
        parsed_doc: Dict[str, Any],
        replace_existing: bool = True,
        generate_summary: Optional[bool] = None,
        generate_suggestions: Optional[bool] = None,
        collection_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Index a parsed document into the vector store.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            replace_existing: Delete existing chunks for this doc first
            generate_summary: Generate document summary (None = use default setting)
            generate_suggestions: Generate query suggestions (None = use default setting)
            collection_name: Optional collection name to assign to the document
            tags: Optional list of tags to assign to the document

        Returns:
            Indexing result with doc_id, chunks_added, summary, suggestions, etc.
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
                "summary": None,
                "suggestions": None,
            }

        # Create synthetic Table of Contents chunk
        toc_chunk = self._create_toc_chunk(parsed_doc, doc_id)
        if toc_chunk:
            chunks.insert(0, toc_chunk)  # Add at beginning
            logger.info(f"Created synthetic TOC chunk with {len(toc_chunk.content)} characters")

        # Create document summary chunk if enabled
        summary_result = None
        should_summarize = (
            generate_summary if generate_summary is not None else self.enable_summaries
        )
        if should_summarize and self.summarizer:
            summary_chunk, summary_result = self._create_summary_chunk(parsed_doc, doc_id)
            if summary_chunk:
                chunks.insert(0, summary_chunk)  # Add at beginning (before TOC)
                logger.info(
                    f"Created summary chunk with {len(summary_chunk.content)} characters"
                )

        # Create query suggestions chunk if enabled
        suggestions_result = None
        should_suggest = (
            generate_suggestions if generate_suggestions is not None else self.enable_suggestions
        )
        if should_suggest and self.suggestion_generator:
            suggestions_chunk, suggestions_result = self._create_suggestions_chunk(
                parsed_doc, doc_id, filename
            )
            if suggestions_chunk:
                chunks.insert(0, suggestions_chunk)  # Add at beginning
                logger.info(
                    f"Created suggestions chunk with {len(suggestions_result)} suggestions"
                )

        # Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks)

        # Store in vector store with collection and tags
        added = self.vector_store.add_chunks(
            chunks,
            embeddings,
            collection_name=collection_name,
            tags=tags,
        )

        # Index in BM25 for hybrid search
        bm25_indexed = 0
        if self.enable_hybrid_search and self.hybrid_searcher:
            chunk_dicts = [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "metadata": {**c.metadata, "doc_id": c.doc_id},
                }
                for c in chunks
            ]
            bm25_indexed = self.hybrid_searcher.add_chunks_to_index(chunk_dicts)
            self.hybrid_searcher.save_index()
            logger.info(f"Indexed {bm25_indexed} chunks in BM25 for hybrid search")

        logger.info(
            f"Indexed document {filename}: {added} chunks, "
            f"total store size: {self.vector_store.count}"
        )

        result = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_added": added,
            "total_tokens": sum(c.token_count for c in chunks),
            "status": "success",
            "summary": summary_result.to_dict() if summary_result else None,
            "suggestions": (
                [s.to_dict() for s in suggestions_result] if suggestions_result else None
            ),
            "bm25_indexed": bm25_indexed,
            "collection": collection_name,
            "tags": tags or [],
        }

        return result

    def _create_summary_chunk(
        self,
        parsed_doc: Dict[str, Any],
        doc_id: str,
    ) -> tuple[Optional[Chunk], Optional[Any]]:
        """
        Create a document summary chunk using the LLM.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            doc_id: Document ID

        Returns:
            Tuple of (Chunk object, DocumentSummary) or (None, None) if failed
        """
        if not self.summarizer:
            return None, None

        try:
            summary = self.summarizer.summarize(parsed_doc)

            if not summary:
                return None, None

            # Format summary for storage as chunk
            summary_content = self.summarizer.format_summary_for_chunk(summary)

            chunk = Chunk(
                chunk_id=f"{doc_id}_summary",
                doc_id=doc_id,
                content=summary_content,
                token_count=len(summary_content) // 4,
                metadata={
                    "chunk_index": -2,  # Before TOC (-1)
                    "chunk_type": "document_summary",
                    "section": "Document Summary",
                    "filename": parsed_doc.get("metadata", {}).get("filename", ""),
                    "file_path": parsed_doc.get("metadata", {}).get("file_path", ""),
                    "num_bullets": len(summary.bullets),
                    "summary_model": summary.metadata.get("model", ""),
                },
            )

            return chunk, summary

        except Exception as e:
            logger.error(f"Failed to create summary chunk: {e}")
            return None, None

    def _create_suggestions_chunk(
        self,
        parsed_doc: Dict[str, Any],
        doc_id: str,
        filename: str,
    ) -> tuple[Optional[Chunk], Optional[List[Any]]]:
        """
        Create a query suggestions chunk from document structure.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            doc_id: Document ID
            filename: Document filename

        Returns:
            Tuple of (Chunk object, list of QuerySuggestion) or (None, None) if failed
        """
        if not self.suggestion_generator:
            return None, None

        try:
            import json

            suggestions = self.suggestion_generator.generate_suggestions(parsed_doc)

            # If no suggestions from content, use defaults
            if not suggestions:
                suggestions = self.suggestion_generator.get_default_suggestions(doc_id, filename)

            if not suggestions:
                return None, None

            # Store suggestions as JSON in the chunk content
            suggestions_json = json.dumps([s.to_dict() for s in suggestions], indent=2)

            # Create a human-readable content as well for embedding
            readable_content = "# SUGGESTED QUESTIONS\n\n"
            readable_content += "Based on this document, you might ask:\n\n"
            for s in suggestions:
                readable_content += f"- {s.text}\n"

            chunk = Chunk(
                chunk_id=f"{doc_id}_suggestions",
                doc_id=doc_id,
                content=suggestions_json,  # Store JSON for retrieval
                token_count=len(suggestions_json) // 4,
                metadata={
                    "chunk_index": -3,  # Before summary (-2) and TOC (-1)
                    "chunk_type": "query_suggestions",
                    "section": "Query Suggestions",
                    "filename": parsed_doc.get("metadata", {}).get("filename", ""),
                    "file_path": parsed_doc.get("metadata", {}).get("file_path", ""),
                    "suggestion_count": len(suggestions),
                    "readable_content": readable_content,
                },
            )

            return chunk, suggestions

        except Exception as e:
            logger.error(f"Failed to create suggestions chunk: {e}")
            return None, None

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
                        "# TABLE OF CONTENTS",
                        "",
                        "This is the table of contents for this document.",
                        "Lists all chapters, sections, and topics.",
                        "Document outline and structure.",
                        "",
                        "Complete chapter list and document outline:",
                        "",
                    ]

                    # Sections to exclude from TOC
                    exclude_patterns = [
                        r"^(?:LIST\s+OF\s+)?ABBREVIATIONS?$",
                        r"^LIST\s+OF\s+(?:FIGURES?|TABLES?|ACRONYMS?)$",
                        r"^(?:REFERENCE|REFERENCES|BIBLIOGRAPHY)$",
                        r"^ACRONYMS?$",
                        r"^GLOSSARY$",
                        r"^INDEX$",
                    ]

                    included_count = 0
                    for section in sections:
                        level = section.get("level", 1)
                        text = section.get("text", "").strip()
                        page = section.get("page")

                        if not text:
                            continue

                        # Skip excluded sections
                        skip = False
                        for pattern in exclude_patterns:
                            if re.match(pattern, text, re.IGNORECASE):
                                logger.debug(f"Excluding section from TOC: {text}")
                                skip = True
                                break

                        if skip:
                            continue

                        # Skip very short section names (likely abbreviations)
                        if len(text) < 3:
                            continue

                        # Skip ALL CAPS short entries (likely abbreviations)
                        if text.isupper() and len(text) < 15:
                            continue

                        indent = "  " * (level - 1)
                        if page:
                            toc_lines.append(f"{indent}- {text} (page {page})")
                        else:
                            toc_lines.append(f"{indent}- {text}")
                        included_count += 1

                    if len(toc_lines) > 8:  # 8 header lines + at least 1 entry
                        toc_content = "\n".join(toc_lines)
                        sections_count = included_count
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
        # Note: Use specific "TABLE OF CONTENTS" to avoid matching "List of Contents", etc.
        toc_markers = [
            r"(?:^|\n)\s*#{1,3}\s*(?:TABLE\s+OF\s+CONTENTS|Table\s+of\s+Contents)\s*\n",
            r"(?:^|\n)\s*(?:TABLE\s+OF\s+CONTENTS|Table\s+of\s+Contents)\s*\n",
        ]

        toc_start = None
        for pattern in toc_markers:
            match = re.search(pattern, raw_text[:100000], re.IGNORECASE)  # Search first 100k chars
            if match:
                toc_start = match.end()
                logger.info(f"Found TOC marker at position {toc_start}")
                break

        if toc_start is None:
            logger.info("No TOC marker found in document")
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
        # Updated to stop at pipe (for markdown tables with duplicate columns)
        chapter_header_pattern = re.compile(
            r"^\|?\s*(CHAPTER\s+\d+\s+[A-Z][A-Z\s\-/,]+?)(?:\s*\||$)", re.IGNORECASE
        )

        consecutive_non_toc = 0
        seen_entries = set()  # Avoid duplicates

        # Patterns that indicate we've left the TOC section
        # IMPORTANT: Only match on standalone lines (NOT inside table cells)
        # Table cells start with |, so we need to make sure line doesn't start with |
        stop_patterns = [
            # Markdown headings
            r"^#{1,3}\s*(?:LIST\s+OF\s+(?:ABBREVIATIONS|FIGURES|TABLES)|ABBREVIATIONS|PREFACE|FOREWORD)",
            # Plain text (but NOT if it starts with | indicating table cell)
            r"^(?![\|\s])(?:LIST\s+OF\s+(?:ABBREVIATIONS|FIGURES|TABLES)|ABBREVIATIONS)\s*$",
        ]

        for line in lines:
            original_line = line
            line = line.strip()

            # Skip empty lines and table separators
            if not line or line.startswith("|--") or line.startswith("|-"):
                continue

            # Skip lines that are just table formatting
            if re.match(r"^[\|\s\-:]+$", line):
                continue

            # Check if we've hit a section that's not the TOC
            # Only check lines that are NOT in table cells (don't start with |)
            if not original_line.lstrip().startswith("|"):
                for stop_pattern in stop_patterns:
                    if re.match(stop_pattern, line, re.IGNORECASE):
                        logger.info(f"Stopped TOC extraction at: {line[:50]}")
                        # Only stop if we already have some entries
                        if len(toc_entries) >= 3:
                            return "\n".join(
                                [
                                    "# TABLE OF CONTENTS",
                                    "",
                                    "This is the table of contents for this document.",
                                    "Lists all chapters, sections, and topics.",
                                    "Document outline and structure.",
                                    "",
                                    "Complete chapter list and document outline:",
                                    "",
                                ]
                                + toc_entries
                            ), len(toc_entries)

            # For markdown table format, extract first cell only (avoid duplicates)
            # Table format: | Cell 1 | Cell 2 |
            # We only want Cell 1
            processing_line = line
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.split("|")]
                # cells[0] is empty (before first |)
                # cells[1] is first cell content
                # cells[2+] might be duplicates or empty
                if len(cells) > 1 and cells[1]:
                    processing_line = cells[1]
                else:
                    continue  # Skip if first cell is empty

            # Check for chapter header first
            chapter_match = chapter_header_pattern.match(processing_line)
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
                match = pattern.match(processing_line)
                if match:
                    title = match.group(1).strip()
                    page = match.group(2).strip()

                    # Clean up title (remove excessive dots/spaces/pipes)
                    title = re.sub(r"[\.\s\|]+$", "", title)
                    title = re.sub(r"\s+", " ", title)

                    # Filter out garbage entries and abbreviations
                    if len(title) > 3 and len(title) < 100 and title not in seen_entries:
                        # Skip entries that look like just numbers or symbols
                        if re.search(r"[a-zA-Z]{2,}", title):
                            # Skip abbreviations (usually ALL CAPS and short)
                            # TOC entries are typically sentence case or title case
                            if not (title.isupper() and len(title) < 15):
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
            logger.info(f"Insufficient TOC entries found: {len(toc_entries)}")
            return None, 0

        logger.info(f"Successfully extracted {len(toc_entries)} TOC entries")

        # Build TOC content with semantic keywords for better retrieval
        # Include many variations of how users might ask for TOC
        toc_lines = [
            "# TABLE OF CONTENTS",
            "",
            "This is the table of contents for this document.",
            "Lists all chapters, sections, and topics.",
            "Document outline and structure.",
            "",
            "Complete chapter list and document outline:",
            "",
        ] + toc_entries

        return "\n".join(toc_lines), len(toc_entries)

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_id: Optional[str] = None,
        use_hybrid: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid search (semantic + keyword).

        Args:
            query: Search query text
            n_results: Number of results to return
            doc_id: Optional filter to specific document
            use_hybrid: Use hybrid search (None = use default setting)

        Returns:
            List of result dicts with content, metadata, and similarity score
        """
        formatted = []

        # Check if this is a TOC-related query
        # Semantic search fails for TOC because the detailed TOC chunk embedding
        # is diluted across 88+ chapter names, while short chunks with just
        # "TABLE OF CONTENTS" rank higher
        toc_keywords = [
            "table of contents",
            "toc",
            "list of chapters",
            "all chapters",
            "main chapters",
            "chapter list",
            "what chapters",
            "show chapters",
            "document outline",
            "document structure",
            "all sections",
            "main sections",
        ]
        query_lower = query.lower()
        is_toc_query = any(kw in query_lower for kw in toc_keywords)

        if is_toc_query:
            # Fetch TOC chunk directly by metadata
            toc_filter = {"chunk_type": "table_of_contents"}
            if doc_id:
                toc_filter["doc_id"] = doc_id

            toc_results = self.vector_store._collection.get(
                where=toc_filter,
                include=["documents", "metadatas"],
            )

            if toc_results["ids"]:
                # Add TOC chunk as first result
                for i, (chunk_id, content, metadata) in enumerate(
                    zip(
                        toc_results["ids"],
                        toc_results["documents"],
                        toc_results["metadatas"],
                    )
                ):
                    formatted.append(
                        {
                            "rank": len(formatted) + 1,
                            "chunk_id": chunk_id,
                            "content": content,
                            "metadata": metadata,
                            "similarity": 1.0,  # Perfect match for explicit TOC request
                        }
                    )
                logger.info(f"TOC query detected, retrieved {len(formatted)} TOC chunk(s)")

        # Determine if hybrid search should be used
        should_use_hybrid = (
            use_hybrid if use_hybrid is not None else self.enable_hybrid_search
        )

        # Fill remaining slots with search
        remaining_slots = n_results - len(formatted)
        if remaining_slots > 0:
            where = {"doc_id": doc_id} if doc_id else None

            # Get semantic search results
            semantic_results = self.vector_store.query_text(
                query_text=query,
                embedder=self.embedder,
                n_results=remaining_slots * 2,  # Get extra for fusion
                where=where,
            )

            # Format semantic results
            semantic_formatted = []
            for chunk_id, content, metadata, distance in zip(
                semantic_results["ids"],
                semantic_results["documents"],
                semantic_results["metadatas"],
                semantic_results["distances"],
            ):
                semantic_formatted.append(
                    {
                        "chunk_id": chunk_id,
                        "content": content,
                        "metadata": metadata,
                        "similarity": 1 - distance,
                    }
                )

            # Apply hybrid search if enabled
            if should_use_hybrid and self.hybrid_searcher:
                hybrid_results = self.hybrid_searcher.search(
                    query=query,
                    semantic_results=semantic_formatted,
                    n_results=remaining_slots + len(formatted),
                    doc_id=doc_id,
                )

                # Add hybrid results, skipping any already added (TOC chunks)
                existing_ids = {r["chunk_id"] for r in formatted}
                for result in hybrid_results:
                    if result.chunk_id not in existing_ids and len(formatted) < n_results:
                        formatted.append(
                            {
                                "rank": len(formatted) + 1,
                                "chunk_id": result.chunk_id,
                                "content": result.content,
                                "metadata": result.metadata,
                                "similarity": result.combined_score,
                                "semantic_score": result.semantic_score,
                                "keyword_score": result.keyword_score,
                            }
                        )
                logger.debug(
                    f"Hybrid search: {len(hybrid_results)} results, "
                    f"returned {len(formatted)} after filtering"
                )
            else:
                # Semantic-only search
                existing_ids = {r["chunk_id"] for r in formatted}
                for result in semantic_formatted:
                    chunk_id = result["chunk_id"]
                    if chunk_id not in existing_ids and len(formatted) < n_results:
                        formatted.append(
                            {
                                "rank": len(formatted) + 1,
                                "chunk_id": chunk_id,
                                "content": result["content"],
                                "metadata": result["metadata"],
                                "similarity": result["similarity"],
                            }
                        )

        return formatted
