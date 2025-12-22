"""Document collections and tags management for Workpedia.

This module provides organization capabilities for documents through:
- Collections: Group related documents (e.g., "Legal Docs", "Project Alpha")
- Tags: Flexible labels for documents (e.g., "project:alpha", "type:contract")

Features:
- SQLite persistence for collections and tags
- Document-to-collection associations
- Multi-tag support per document
- Filtering queries by collection or tags
- Collection statistics and management

Usage:
    manager = CollectionManager()

    # Create collection
    collection_id = manager.create_collection(
        name="Legal Documents",
        description="All legal contracts and agreements"
    )

    # Add document to collection
    manager.add_document_to_collection(doc_id="doc-123", collection_id=collection_id)

    # Tag documents
    manager.add_tags(doc_id="doc-123", tags=["project:alpha", "type:contract", "year:2024"])

    # Query by collection or tags
    docs = manager.get_documents_in_collection(collection_id)
    docs = manager.get_documents_by_tags(tags=["project:alpha"])
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import COLLECTIONS_DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class Collection:
    """Represents a document collection."""

    collection_id: str
    name: str
    description: str
    created_at: float
    updated_at: float
    document_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_count": self.document_count,
        }


@dataclass
class DocumentMetadata:
    """Document metadata including collection and tags."""

    doc_id: str
    collection_id: Optional[str] = None
    collection_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    added_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "tags": self.tags,
            "added_at": self.added_at,
        }


class CollectionManager:
    """Manages document collections and tags.

    Provides CRUD operations for collections, tag management,
    and document organization capabilities.
    """

    def __init__(self, db_path: str = COLLECTIONS_DB_PATH):
        """Initialize collection manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"CollectionManager initialized: {db_path}")

    def _init_db(self) -> None:
        """Create database schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")

        conn.executescript(
            """
            -- Collections table
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id TEXT UNIQUE NOT NULL,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Document-Collection associations
            CREATE TABLE IF NOT EXISTS document_collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                added_at REAL NOT NULL,
                UNIQUE(doc_id, collection_id),
                FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE CASCADE
            );

            -- Document tags
            CREATE TABLE IF NOT EXISTS document_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                added_at REAL NOT NULL,
                UNIQUE(doc_id, tag)
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_doc_collections_doc_id ON document_collections(doc_id);
            CREATE INDEX IF NOT EXISTS idx_doc_collections_collection_id ON document_collections(collection_id);
            CREATE INDEX IF NOT EXISTS idx_doc_tags_doc_id ON document_tags(doc_id);
            CREATE INDEX IF NOT EXISTS idx_doc_tags_tag ON document_tags(tag);
            CREATE INDEX IF NOT EXISTS idx_collections_name ON collections(name);
            """
        )

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # Collection Management
    # =========================================================================

    def create_collection(
        self,
        name: str,
        description: str = "",
    ) -> str:
        """Create a new collection.

        Args:
            name: Unique collection name
            description: Optional description

        Returns:
            Collection ID

        Raises:
            ValueError: If collection name already exists
        """
        collection_id = str(uuid.uuid4())
        now = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO collections (collection_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (collection_id, name, description, now, now),
            )
            conn.commit()
            logger.info(f"Created collection: {name} ({collection_id})")
            return collection_id
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Collection '{name}' already exists") from e
        finally:
            conn.close()

    def get_collection(self, collection_id: str) -> Optional[Collection]:
        """Get collection by ID.

        Args:
            collection_id: Collection ID

        Returns:
            Collection object or None if not found
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                """
                SELECT c.*, COUNT(dc.doc_id) as document_count
                FROM collections c
                LEFT JOIN document_collections dc ON c.collection_id = dc.collection_id
                WHERE c.collection_id = ?
                GROUP BY c.collection_id
                """,
                (collection_id,),
            ).fetchone()

            if row:
                return Collection(
                    collection_id=row["collection_id"],
                    name=row["name"],
                    description=row["description"] or "",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    document_count=row["document_count"],
                )
            return None
        finally:
            conn.close()

    def get_collection_by_name(self, name: str) -> Optional[Collection]:
        """Get collection by name.

        Args:
            name: Collection name

        Returns:
            Collection object or None if not found
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                """
                SELECT c.*, COUNT(dc.doc_id) as document_count
                FROM collections c
                LEFT JOIN document_collections dc ON c.collection_id = dc.collection_id
                WHERE c.name = ?
                GROUP BY c.collection_id
                """,
                (name,),
            ).fetchone()

            if row:
                return Collection(
                    collection_id=row["collection_id"],
                    name=row["name"],
                    description=row["description"] or "",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    document_count=row["document_count"],
                )
            return None
        finally:
            conn.close()

    def list_collections(self) -> List[Collection]:
        """List all collections.

        Returns:
            List of Collection objects
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT c.*, COUNT(dc.doc_id) as document_count
                FROM collections c
                LEFT JOIN document_collections dc ON c.collection_id = dc.collection_id
                GROUP BY c.collection_id
                ORDER BY c.name
                """
            ).fetchall()

            return [
                Collection(
                    collection_id=row["collection_id"],
                    name=row["name"],
                    description=row["description"] or "",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    document_count=row["document_count"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def update_collection(
        self,
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update collection details.

        Args:
            collection_id: Collection ID
            name: New name (optional)
            description: New description (optional)

        Returns:
            True if updated, False if not found
        """
        if name is None and description is None:
            return False

        conn = self._get_connection()
        try:
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)

            updates.append("updated_at = ?")
            params.append(time.time())
            params.append(collection_id)

            cursor = conn.execute(
                f"UPDATE collections SET {', '.join(updates)} WHERE collection_id = ?",
                params,
            )
            conn.commit()

            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection.

        Note: Documents in the collection will be disassociated but not deleted.

        Args:
            collection_id: Collection ID

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM collections WHERE collection_id = ?",
                (collection_id,),
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Deleted collection: {collection_id}")
                return True
            return False
        finally:
            conn.close()

    # =========================================================================
    # Document-Collection Management
    # =========================================================================

    def add_document_to_collection(
        self,
        doc_id: str,
        collection_id: str,
    ) -> bool:
        """Add a document to a collection.

        Args:
            doc_id: Document ID
            collection_id: Collection ID

        Returns:
            True if added, False if already in collection
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO document_collections (doc_id, collection_id, added_at)
                VALUES (?, ?, ?)
                """,
                (doc_id, collection_id, time.time()),
            )
            conn.commit()
            logger.debug(f"Added document {doc_id} to collection {collection_id}")
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def remove_document_from_collection(
        self,
        doc_id: str,
        collection_id: str,
    ) -> bool:
        """Remove a document from a collection.

        Args:
            doc_id: Document ID
            collection_id: Collection ID

        Returns:
            True if removed, False if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM document_collections WHERE doc_id = ? AND collection_id = ?",
                (doc_id, collection_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_document_collection(self, doc_id: str) -> Optional[Collection]:
        """Get the collection a document belongs to.

        Args:
            doc_id: Document ID

        Returns:
            Collection object or None if not in any collection
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                """
                SELECT c.*, COUNT(dc2.doc_id) as document_count
                FROM document_collections dc
                JOIN collections c ON dc.collection_id = c.collection_id
                LEFT JOIN document_collections dc2 ON c.collection_id = dc2.collection_id
                WHERE dc.doc_id = ?
                GROUP BY c.collection_id
                """,
                (doc_id,),
            ).fetchone()

            if row:
                return Collection(
                    collection_id=row["collection_id"],
                    name=row["name"],
                    description=row["description"] or "",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    document_count=row["document_count"],
                )
            return None
        finally:
            conn.close()

    def get_documents_in_collection(self, collection_id: str) -> List[str]:
        """Get all document IDs in a collection.

        Args:
            collection_id: Collection ID

        Returns:
            List of document IDs
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT doc_id FROM document_collections WHERE collection_id = ?",
                (collection_id,),
            ).fetchall()
            return [row["doc_id"] for row in rows]
        finally:
            conn.close()

    # =========================================================================
    # Tag Management
    # =========================================================================

    def add_tags(self, doc_id: str, tags: List[str]) -> int:
        """Add tags to a document.

        Args:
            doc_id: Document ID
            tags: List of tags to add

        Returns:
            Number of tags added (excludes duplicates)
        """
        if not tags:
            return 0

        conn = self._get_connection()
        try:
            now = time.time()
            added = 0
            for tag in tags:
                tag = tag.strip().lower()  # Normalize tags
                if not tag:
                    continue
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO document_tags (doc_id, tag, added_at) VALUES (?, ?, ?)",
                        (doc_id, tag, now),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    pass  # Tag already exists

            conn.commit()
            logger.debug(f"Added {added} tags to document {doc_id}")
            return added
        finally:
            conn.close()

    def remove_tags(self, doc_id: str, tags: List[str]) -> int:
        """Remove tags from a document.

        Args:
            doc_id: Document ID
            tags: List of tags to remove

        Returns:
            Number of tags removed
        """
        if not tags:
            return 0

        conn = self._get_connection()
        try:
            removed = 0
            for tag in tags:
                tag = tag.strip().lower()
                cursor = conn.execute(
                    "DELETE FROM document_tags WHERE doc_id = ? AND tag = ?",
                    (doc_id, tag),
                )
                removed += cursor.rowcount

            conn.commit()
            return removed
        finally:
            conn.close()

    def get_document_tags(self, doc_id: str) -> List[str]:
        """Get all tags for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of tags
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT tag FROM document_tags WHERE doc_id = ? ORDER BY tag",
                (doc_id,),
            ).fetchall()
            return [row["tag"] for row in rows]
        finally:
            conn.close()

    def get_documents_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
    ) -> List[str]:
        """Get documents that have specific tags.

        Args:
            tags: List of tags to filter by
            match_all: If True, documents must have ALL tags. If False, any tag matches.

        Returns:
            List of document IDs
        """
        if not tags:
            return []

        # Normalize tags
        tags = [t.strip().lower() for t in tags if t.strip()]

        conn = self._get_connection()
        try:
            if match_all:
                # Documents must have ALL specified tags
                placeholders = ",".join("?" * len(tags))
                rows = conn.execute(
                    f"""
                    SELECT doc_id
                    FROM document_tags
                    WHERE tag IN ({placeholders})
                    GROUP BY doc_id
                    HAVING COUNT(DISTINCT tag) = ?
                    """,
                    (*tags, len(tags)),
                ).fetchall()
            else:
                # Documents with ANY of the specified tags
                placeholders = ",".join("?" * len(tags))
                rows = conn.execute(
                    f"SELECT DISTINCT doc_id FROM document_tags WHERE tag IN ({placeholders})",
                    tags,
                ).fetchall()

            return [row["doc_id"] for row in rows]
        finally:
            conn.close()

    def list_all_tags(self) -> List[Dict[str, Any]]:
        """List all unique tags with document counts.

        Returns:
            List of dicts with tag and document_count
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT tag, COUNT(DISTINCT doc_id) as document_count
                FROM document_tags
                GROUP BY tag
                ORDER BY document_count DESC, tag
                """
            ).fetchall()

            return [
                {"tag": row["tag"], "document_count": row["document_count"]}
                for row in rows
            ]
        finally:
            conn.close()

    # =========================================================================
    # Document Metadata
    # =========================================================================

    def get_document_metadata(self, doc_id: str) -> DocumentMetadata:
        """Get full metadata for a document (collection and tags).

        Args:
            doc_id: Document ID

        Returns:
            DocumentMetadata object
        """
        collection = self.get_document_collection(doc_id)
        tags = self.get_document_tags(doc_id)

        return DocumentMetadata(
            doc_id=doc_id,
            collection_id=collection.collection_id if collection else None,
            collection_name=collection.name if collection else None,
            tags=tags,
        )

    def set_document_collection(
        self,
        doc_id: str,
        collection_id: Optional[str],
    ) -> bool:
        """Set the collection for a document (replaces any existing).

        Args:
            doc_id: Document ID
            collection_id: Collection ID (None to remove from collection)

        Returns:
            True if updated
        """
        conn = self._get_connection()
        try:
            # Remove from any existing collection
            conn.execute(
                "DELETE FROM document_collections WHERE doc_id = ?",
                (doc_id,),
            )

            # Add to new collection if specified
            if collection_id:
                conn.execute(
                    """
                    INSERT INTO document_collections (doc_id, collection_id, added_at)
                    VALUES (?, ?, ?)
                    """,
                    (doc_id, collection_id, time.time()),
                )

            conn.commit()
            return True
        finally:
            conn.close()

    def set_document_tags(self, doc_id: str, tags: List[str]) -> bool:
        """Set tags for a document (replaces all existing tags).

        Args:
            doc_id: Document ID
            tags: List of tags

        Returns:
            True if updated
        """
        conn = self._get_connection()
        try:
            # Remove all existing tags
            conn.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc_id,))

            # Add new tags
            now = time.time()
            for tag in tags:
                tag = tag.strip().lower()
                if tag:
                    conn.execute(
                        "INSERT INTO document_tags (doc_id, tag, added_at) VALUES (?, ?, ?)",
                        (doc_id, tag, now),
                    )

            conn.commit()
            return True
        finally:
            conn.close()

    def delete_document_metadata(self, doc_id: str) -> bool:
        """Delete all collection and tag associations for a document.

        Call this when a document is deleted from the vector store.

        Args:
            doc_id: Document ID

        Returns:
            True if any metadata was deleted
        """
        conn = self._get_connection()
        try:
            c1 = conn.execute(
                "DELETE FROM document_collections WHERE doc_id = ?",
                (doc_id,),
            )
            c2 = conn.execute(
                "DELETE FROM document_tags WHERE doc_id = ?",
                (doc_id,),
            )
            conn.commit()

            deleted = c1.rowcount + c2.rowcount
            if deleted > 0:
                logger.debug(f"Deleted metadata for document {doc_id}")
            return deleted > 0
        finally:
            conn.close()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get collection and tag statistics.

        Returns:
            Dictionary with statistics
        """
        conn = self._get_connection()
        try:
            collections_count = conn.execute(
                "SELECT COUNT(*) FROM collections"
            ).fetchone()[0]

            documents_in_collections = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM document_collections"
            ).fetchone()[0]

            total_tags = conn.execute(
                "SELECT COUNT(*) FROM document_tags"
            ).fetchone()[0]

            unique_tags = conn.execute(
                "SELECT COUNT(DISTINCT tag) FROM document_tags"
            ).fetchone()[0]

            documents_with_tags = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM document_tags"
            ).fetchone()[0]

            return {
                "collections_count": collections_count,
                "documents_in_collections": documents_in_collections,
                "total_tags": total_tags,
                "unique_tags": unique_tags,
                "documents_with_tags": documents_with_tags,
            }
        finally:
            conn.close()
