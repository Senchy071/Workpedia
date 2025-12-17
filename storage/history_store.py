"""Query history and bookmark storage for Workpedia RAG system.

This module provides persistent storage for query history and bookmarks using SQLite.
All queries can be automatically saved with full context including sources and metadata.
Users can bookmark favorite Q&A pairs with notes and tags for organization.

Features:
- Automatic query history with sources and metadata
- Bookmark management with notes and tags
- Search and filter capabilities
- Export to Markdown, JSON, and PDF formats
- Session tracking for conversation grouping
- Efficient indexing for fast retrieval

Usage:
    store = HistoryStore()

    # Save query
    query_id = store.add_query(
        question="What is X?",
        answer="X is...",
        sources=[...],
        session_id="session123"
    )

    # Search history
    queries = store.search_queries("X", limit=10)

    # Bookmark query
    bookmark_id = store.add_bookmark(
        query_id=query_id,
        notes="Important finding",
        tags=["research", "key-concept"]
    )

    # Export
    markdown = store.export_queries_markdown([query_id])
    pdf_bytes = store.export_queries_pdf([query_id])
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import HISTORY_DB_PATH
from core.exceptions import (
    HistoryDatabaseError,
    QueryNotFoundError,
    BookmarkNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass
class HistoryQuery:
    """Represents a saved query with metadata."""

    query_id: str
    session_id: Optional[str]
    timestamp: float
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata,
        }


@dataclass
class Bookmark:
    """Represents a bookmarked query."""

    bookmark_id: str
    query_id: str
    timestamp: float
    notes: Optional[str]
    tags: List[str]
    query: Optional[HistoryQuery] = None  # Populated with query data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "bookmark_id": self.bookmark_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "tags": self.tags,
        }
        if self.query:
            result["query"] = self.query.to_dict()
        return result


class HistoryStore:
    """
    SQLite-based storage for query history and bookmarks.

    Features:
    - Persistent query history across sessions
    - Bookmark management with notes and tags
    - Search and filter capabilities
    - Export to JSON, Markdown, and PDF

    The database uses Write-Ahead Logging (WAL) mode for better concurrent
    read performance. All timestamps are stored as Unix time (float).
    """

    def __init__(self, db_path: str = HISTORY_DB_PATH):
        """
        Initialize history store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"HistoryStore initialized: {db_path}")

    def _init_db(self):
        """Create database schema if not exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes

            # Create tables
            conn.executescript(
                """
                -- Main queries table
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT UNIQUE NOT NULL,
                    session_id TEXT,
                    timestamp REAL NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Sources for each query (1-to-many)
                CREATE TABLE IF NOT EXISTS query_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    chunk_id TEXT,
                    content TEXT,
                    metadata_json TEXT,
                    similarity REAL,
                    source_index INTEGER,
                    FOREIGN KEY (query_id) REFERENCES query_history(query_id) ON DELETE CASCADE
                );

                -- Bookmarks table
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bookmark_id TEXT UNIQUE NOT NULL,
                    query_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    notes TEXT,
                    tags_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES query_history(query_id) ON DELETE CASCADE
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_history(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_query_session ON query_history(session_id);
                CREATE INDEX IF NOT EXISTS idx_sources_query ON query_sources(query_id);
                CREATE INDEX IF NOT EXISTS idx_bookmark_query ON bookmarks(query_id);
                CREATE INDEX IF NOT EXISTS idx_bookmark_timestamp ON bookmarks(timestamp DESC);
                """
            )

            conn.commit()
            conn.close()

            logger.debug("Database schema initialized")

        except sqlite3.Error as e:
            raise HistoryDatabaseError("initialization", str(e))

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # =============================================================================
    # Query Management Methods
    # =============================================================================

    def add_query(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Add query to history.

        Args:
            question: User's question
            answer: Generated answer
            sources: List of source chunks with metadata
            metadata: Query metadata (temperature, model, etc.)
            session_id: Session identifier for grouping

        Returns:
            query_id: UUID for this query
        """
        query_id = str(uuid.uuid4())
        timestamp = time.time()
        metadata = metadata or {}

        try:
            conn = self._get_connection()

            # Insert query
            conn.execute(
                """
                INSERT INTO query_history (query_id, session_id, timestamp, question, answer, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (query_id, session_id, timestamp, question, answer, json.dumps(metadata)),
            )

            # Insert sources
            for idx, source in enumerate(sources):
                conn.execute(
                    """
                    INSERT INTO query_sources (query_id, chunk_id, content, metadata_json, similarity, source_index)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        source.get("chunk_id"),
                        source.get("content", ""),
                        json.dumps(source.get("metadata", {})),
                        source.get("similarity"),
                        idx,
                    ),
                )

            conn.commit()
            conn.close()

            logger.debug(f"Added query to history: {query_id}")
            return query_id

        except sqlite3.Error as e:
            raise HistoryDatabaseError("add_query", str(e))

    def get_query(self, query_id: str) -> Optional[HistoryQuery]:
        """
        Get query by ID with sources.

        Args:
            query_id: Query identifier

        Returns:
            HistoryQuery object or None if not found
        """
        try:
            conn = self._get_connection()

            # Get query
            row = conn.execute(
                "SELECT * FROM query_history WHERE query_id = ?", (query_id,)
            ).fetchone()

            if not row:
                conn.close()
                return None

            # Get sources
            source_rows = conn.execute(
                "SELECT * FROM query_sources WHERE query_id = ? ORDER BY source_index",
                (query_id,),
            ).fetchall()

            conn.close()

            # Build sources list
            sources = []
            for source_row in source_rows:
                sources.append(
                    {
                        "chunk_id": source_row["chunk_id"],
                        "content": source_row["content"],
                        "metadata": json.loads(source_row["metadata_json"] or "{}"),
                        "similarity": source_row["similarity"],
                    }
                )

            return HistoryQuery(
                query_id=row["query_id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                question=row["question"],
                answer=row["answer"],
                sources=sources,
                metadata=json.loads(row["metadata_json"] or "{}"),
            )

        except sqlite3.Error as e:
            raise HistoryDatabaseError("get_query", str(e))

    def list_queries(
        self,
        limit: int = 50,
        offset: int = 0,
        session_id: Optional[str] = None,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
    ) -> List[HistoryQuery]:
        """
        List queries with pagination and filtering.

        Args:
            limit: Maximum queries to return
            offset: Skip this many queries
            session_id: Filter by session
            start_date: Unix timestamp for date range start
            end_date: Unix timestamp for date range end

        Returns:
            List of HistoryQuery objects
        """
        try:
            conn = self._get_connection()

            # Build query
            query = "SELECT * FROM query_history WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()

            # Get queries with sources
            queries = []
            for row in rows:
                query_obj = self.get_query(row["query_id"])
                if query_obj:
                    queries.append(query_obj)

            conn.close()
            return queries

        except sqlite3.Error as e:
            raise HistoryDatabaseError("list_queries", str(e))

    def search_queries(
        self,
        search_text: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[HistoryQuery]:
        """
        Search queries by text in question or answer.

        Uses SQLite LIKE operator for text search.

        Args:
            search_text: Text to search for
            limit: Maximum queries to return
            offset: Skip this many queries

        Returns:
            List of matching HistoryQuery objects
        """
        try:
            conn = self._get_connection()

            # Search in question and answer
            search_pattern = f"%{search_text}%"
            rows = conn.execute(
                """
                SELECT * FROM query_history
                WHERE question LIKE ? OR answer LIKE ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (search_pattern, search_pattern, limit, offset),
            ).fetchall()

            # Get queries with sources
            queries = []
            for row in rows:
                query_obj = self.get_query(row["query_id"])
                if query_obj:
                    queries.append(query_obj)

            conn.close()
            return queries

        except sqlite3.Error as e:
            raise HistoryDatabaseError("search_queries", str(e))

    def delete_query(self, query_id: str) -> bool:
        """
        Delete query and associated sources (CASCADE).

        Args:
            query_id: Query identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            conn = self._get_connection()

            cursor = conn.execute("DELETE FROM query_history WHERE query_id = ?", (query_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if deleted:
                logger.debug(f"Deleted query: {query_id}")

            return deleted

        except sqlite3.Error as e:
            raise HistoryDatabaseError("delete_query", str(e))

    def clear_history(
        self,
        before_date: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Clear history with optional filters.

        Args:
            before_date: Delete queries before this timestamp
            session_id: Delete only queries from this session

        Returns:
            Number of queries deleted
        """
        try:
            conn = self._get_connection()

            query = "DELETE FROM query_history WHERE 1=1"
            params = []

            if before_date:
                query += " AND timestamp < ?"
                params.append(before_date)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            cursor = conn.execute(query, params)
            count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Cleared {count} queries from history")
            return count

        except sqlite3.Error as e:
            raise HistoryDatabaseError("clear_history", str(e))

    # =============================================================================
    # Bookmark Management Methods
    # =============================================================================

    def add_bookmark(
        self,
        query_id: str,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Bookmark a query.

        Args:
            query_id: Query identifier to bookmark
            notes: Optional notes about this bookmark
            tags: Optional tags for organization

        Returns:
            bookmark_id: UUID for this bookmark

        Raises:
            QueryNotFoundError: If query_id doesn't exist
        """
        # Verify query exists
        if not self.get_query(query_id):
            raise QueryNotFoundError(query_id)

        bookmark_id = str(uuid.uuid4())
        timestamp = time.time()
        tags = tags or []

        try:
            conn = self._get_connection()

            conn.execute(
                """
                INSERT INTO bookmarks (bookmark_id, query_id, timestamp, notes, tags_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (bookmark_id, query_id, timestamp, notes, json.dumps(tags)),
            )

            conn.commit()
            conn.close()

            logger.debug(f"Created bookmark: {bookmark_id}")
            return bookmark_id

        except sqlite3.Error as e:
            raise HistoryDatabaseError("add_bookmark", str(e))

    def get_bookmark(self, bookmark_id: str) -> Optional[Bookmark]:
        """
        Get bookmark by ID with query data.

        Args:
            bookmark_id: Bookmark identifier

        Returns:
            Bookmark object or None if not found
        """
        try:
            conn = self._get_connection()

            row = conn.execute(
                "SELECT * FROM bookmarks WHERE bookmark_id = ?", (bookmark_id,)
            ).fetchone()

            if not row:
                conn.close()
                return None

            conn.close()

            # Get associated query
            query = self.get_query(row["query_id"])

            return Bookmark(
                bookmark_id=row["bookmark_id"],
                query_id=row["query_id"],
                timestamp=row["timestamp"],
                notes=row["notes"],
                tags=json.loads(row["tags_json"] or "[]"),
                query=query,
            )

        except sqlite3.Error as e:
            raise HistoryDatabaseError("get_bookmark", str(e))

    def list_bookmarks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Bookmark]:
        """
        List bookmarks with optional tag filtering.

        Args:
            tags: Filter by these tags (OR condition)
            limit: Maximum bookmarks to return
            offset: Skip this many bookmarks

        Returns:
            List of Bookmark objects
        """
        try:
            conn = self._get_connection()

            rows = conn.execute(
                """
                SELECT * FROM bookmarks
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            conn.close()

            # Filter by tags if specified
            bookmarks = []
            for row in rows:
                bookmark = self.get_bookmark(row["bookmark_id"])
                if bookmark:
                    # Filter by tags
                    if tags:
                        if any(tag in bookmark.tags for tag in tags):
                            bookmarks.append(bookmark)
                    else:
                        bookmarks.append(bookmark)

            return bookmarks

        except sqlite3.Error as e:
            raise HistoryDatabaseError("list_bookmarks", str(e))

    def update_bookmark(
        self,
        bookmark_id: str,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Update bookmark notes or tags.

        Args:
            bookmark_id: Bookmark identifier
            notes: New notes (or None to keep unchanged)
            tags: New tags (or None to keep unchanged)

        Returns:
            True if updated, False if not found
        """
        try:
            conn = self._get_connection()

            # Build update query
            updates = []
            params = []

            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)

            if tags is not None:
                updates.append("tags_json = ?")
                params.append(json.dumps(tags))

            if not updates:
                conn.close()
                return True  # Nothing to update

            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(bookmark_id)

            query = f"UPDATE bookmarks SET {', '.join(updates)} WHERE bookmark_id = ?"
            cursor = conn.execute(query, params)

            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if updated:
                logger.debug(f"Updated bookmark: {bookmark_id}")

            return updated

        except sqlite3.Error as e:
            raise HistoryDatabaseError("update_bookmark", str(e))

    def delete_bookmark(self, bookmark_id: str) -> bool:
        """
        Delete a bookmark (query remains in history).

        Args:
            bookmark_id: Bookmark identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            conn = self._get_connection()

            cursor = conn.execute("DELETE FROM bookmarks WHERE bookmark_id = ?", (bookmark_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if deleted:
                logger.debug(f"Deleted bookmark: {bookmark_id}")

            return deleted

        except sqlite3.Error as e:
            raise HistoryDatabaseError("delete_bookmark", str(e))

    # =============================================================================
    # Export Methods
    # =============================================================================

    def export_query_markdown(self, query_id: str) -> str:
        """
        Export single query as Markdown.

        Args:
            query_id: Query identifier

        Returns:
            Markdown string

        Raises:
            QueryNotFoundError: If query not found
        """
        query = self.get_query(query_id)
        if not query:
            raise QueryNotFoundError(query_id)

        timestamp_str = datetime.fromtimestamp(query.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# Query: {query.question}

**Date:** {timestamp_str}
**Session:** {query.session_id or 'N/A'}

## Question
{query.question}

## Answer
{query.answer}

## Sources
"""

        for i, source in enumerate(query.sources, 1):
            metadata = source.get("metadata", {})
            md += f"""
### Source {i}
- **File:** {metadata.get('filename', 'Unknown')}
- **Section:** {metadata.get('section', 'N/A')}
- **Similarity:** {source.get('similarity', 0):.2%}

{source.get('content', '')[:500]}...

---
"""

        if query.metadata:
            md += "\n## Metadata\n"
            md += f"- **Chunks Retrieved:** {query.metadata.get('chunks_retrieved', 'N/A')}\n"
            md += f"- **Model:** {query.metadata.get('model', 'N/A')}\n"
            md += f"- **Temperature:** {query.metadata.get('temperature', 'N/A')}\n"

        return md

    def export_queries_markdown(
        self,
        query_ids: List[str],
        title: str = "Workpedia Query History",
    ) -> str:
        """
        Export multiple queries as Markdown document.

        Args:
            query_ids: List of query identifiers
            title: Document title

        Returns:
            Markdown string
        """
        md = f"# {title}\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"**Total Queries:** {len(query_ids)}\n\n"
        md += "---\n\n"

        for i, query_id in enumerate(query_ids, 1):
            try:
                query_md = self.export_query_markdown(query_id)
                md += f"## Query {i}\n\n{query_md}\n\n---\n\n"
            except QueryNotFoundError:
                logger.warning(f"Query not found during export: {query_id}")
                continue

        return md

    def export_queries_json(
        self,
        query_ids: Optional[List[str]] = None,
        include_sources: bool = True,
    ) -> str:
        """
        Export queries as JSON.

        Args:
            query_ids: List of query identifiers (or None for all recent)
            include_sources: Include source data

        Returns:
            JSON string
        """
        if query_ids is None:
            queries = self.list_queries(limit=1000)
            query_ids = [q.query_id for q in queries]

        export_data = {
            "export_metadata": {
                "title": "Workpedia Query History",
                "exported_at": datetime.now().isoformat(),
                "total_queries": len(query_ids),
                "include_sources": include_sources,
            },
            "queries": [],
        }

        for query_id in query_ids:
            try:
                query = self.get_query(query_id)
                if not query:
                    continue

                query_dict = query.to_dict()
                if not include_sources:
                    query_dict["sources"] = []

                export_data["queries"].append(query_dict)

            except QueryNotFoundError:
                logger.warning(f"Query not found during export: {query_id}")
                continue

        return json.dumps(export_data, indent=2)

    def export_queries_pdf(
        self,
        query_ids: List[str],
        title: str = "Workpedia Query History",
    ) -> bytes:
        """
        Export queries as PDF document.

        Args:
            query_ids: List of query identifiers
            title: Document title

        Returns:
            PDF bytes

        Raises:
            ImportError: If reportlab not installed
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                PageBreak,
                Table,
                TableStyle,
            )
            from io import BytesIO
        except ImportError:
            raise ImportError(
                "reportlab is required for PDF export. Install with: pip install reportlab"
            )

        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Heading1"], fontSize=24, textColor=colors.HexColor("#2c3e50")
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3 * inch))

        # Metadata
        meta_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Total Queries: {len(query_ids)}"
        story.append(Paragraph(meta_text, styles["Normal"]))
        story.append(Spacer(1, 0.5 * inch))

        # Queries
        for i, query_id in enumerate(query_ids, 1):
            try:
                query = self.get_query(query_id)
                if not query:
                    continue

                # Query header
                timestamp_str = datetime.fromtimestamp(query.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                header_text = f"<b>Query {i}</b> - {timestamp_str}"
                story.append(Paragraph(header_text, styles["Heading2"]))
                story.append(Spacer(1, 0.2 * inch))

                # Question
                story.append(Paragraph("<b>Question:</b>", styles["Heading3"]))
                story.append(Paragraph(query.question, styles["Normal"]))
                story.append(Spacer(1, 0.2 * inch))

                # Answer
                story.append(Paragraph("<b>Answer:</b>", styles["Heading3"]))
                story.append(Paragraph(query.answer, styles["Normal"]))
                story.append(Spacer(1, 0.2 * inch))

                # Sources
                if query.sources:
                    story.append(Paragraph(f"<b>Sources ({len(query.sources)}):</b>", styles["Heading3"]))
                    for j, source in enumerate(query.sources, 1):
                        metadata = source.get("metadata", {})
                        source_text = f"""
                        <b>Source {j}:</b><br/>
                        File: {metadata.get('filename', 'Unknown')}<br/>
                        Section: {metadata.get('section', 'N/A')}<br/>
                        Similarity: {source.get('similarity', 0):.2%}<br/>
                        <i>{source.get('content', '')[:300]}...</i>
                        """
                        story.append(Paragraph(source_text, styles["Normal"]))
                        story.append(Spacer(1, 0.1 * inch))

                if i < len(query_ids):
                    story.append(PageBreak())

            except QueryNotFoundError:
                logger.warning(f"Query not found during PDF export: {query_id}")
                continue

        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    # =============================================================================
    # Statistics and Utilities
    # =============================================================================

    def stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            conn = self._get_connection()

            total_queries = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
            total_bookmarks = conn.execute("SELECT COUNT(*) FROM bookmarks").fetchone()[0]
            total_sources = conn.execute("SELECT COUNT(*) FROM query_sources").fetchone()[0]

            # Get oldest and newest query timestamps
            oldest_row = conn.execute("SELECT MIN(timestamp) FROM query_history").fetchone()
            newest_row = conn.execute("SELECT MAX(timestamp) FROM query_history").fetchone()

            oldest_timestamp = oldest_row[0] if oldest_row[0] else None
            newest_timestamp = newest_row[0] if newest_row[0] else None

            # Get session count
            session_count = conn.execute("SELECT COUNT(DISTINCT session_id) FROM query_history WHERE session_id IS NOT NULL").fetchone()[0]

            conn.close()

            return {
                "total_queries": total_queries,
                "total_bookmarks": total_bookmarks,
                "total_sources": total_sources,
                "session_count": session_count,
                "oldest_query": datetime.fromtimestamp(oldest_timestamp).isoformat() if oldest_timestamp else None,
                "newest_query": datetime.fromtimestamp(newest_timestamp).isoformat() if newest_timestamp else None,
                "db_path": self.db_path,
            }

        except sqlite3.Error as e:
            raise HistoryDatabaseError("stats", str(e))
