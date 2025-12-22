"""Tests for query history and bookmarks functionality."""

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Import directly to avoid circular import issues
sys.path.insert(0, str(Path(__file__).parent.parent))
from storage.history_store import HistoryStore, HistoryQuery, Bookmark
from core.exceptions import QueryNotFoundError, BookmarkNotFoundError


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def history_store(temp_db):
    """Create HistoryStore instance."""
    return HistoryStore(db_path=temp_db)


# =============================================================================
# Database Initialization Tests
# =============================================================================

def test_init_db(history_store):
    """Test database initialization."""
    stats = history_store.stats()
    assert stats["total_queries"] == 0
    assert stats["total_bookmarks"] == 0
    assert stats["total_sources"] == 0


def test_stats(history_store):
    """Test statistics retrieval."""
    stats = history_store.stats()
    assert "total_queries" in stats
    assert "total_bookmarks" in stats
    assert "db_path" in stats


# =============================================================================
# Query Management Tests
# =============================================================================

def test_add_query(history_store):
    """Test adding query to history."""
    query_id = history_store.add_query(
        question="What is Python?",
        answer="Python is a programming language.",
        sources=[{"content": "test content", "metadata": {"file": "test.pdf"}, "similarity": 0.95}],
        metadata={"model": "mistral", "temperature": 0.7}
    )

    assert query_id is not None
    assert len(query_id) == 36  # UUID length

    # Verify it was added
    query = history_store.get_query(query_id)
    assert query is not None
    assert query.question == "What is Python?"
    assert query.answer == "Python is a programming language."
    assert len(query.sources) == 1
    assert query.metadata["model"] == "mistral"


def test_add_query_with_session(history_store):
    """Test adding query with session ID."""
    session_id = "test-session-123"
    query_id = history_store.add_query(
        question="Test question",
        answer="Test answer",
        sources=[],
        session_id=session_id
    )

    query = history_store.get_query(query_id)
    assert query.session_id == session_id


def test_get_query_not_found(history_store):
    """Test getting non-existent query."""
    result = history_store.get_query("non-existent-id")
    assert result is None


def test_list_queries(history_store):
    """Test listing queries."""
    # Add multiple queries
    for i in range(5):
        history_store.add_query(
            question=f"Question {i}",
            answer=f"Answer {i}",
            sources=[]
        )

    # List all
    queries = history_store.list_queries(limit=10)
    assert len(queries) == 5

    # Verify sorted by timestamp (newest first)
    assert queries[0].question == "Question 4"
    assert queries[-1].question == "Question 0"


def test_list_queries_pagination(history_store):
    """Test pagination in list queries."""
    # Add 10 queries
    for i in range(10):
        history_store.add_query(f"Q{i}", f"A{i}", [])
        time.sleep(0.01)  # Ensure different timestamps

    # Get first page
    page1 = history_store.list_queries(limit=5, offset=0)
    assert len(page1) == 5

    # Get second page
    page2 = history_store.list_queries(limit=5, offset=5)
    assert len(page2) == 5

    # Ensure different queries
    assert page1[0].query_id != page2[0].query_id


def test_list_queries_by_session(history_store):
    """Test filtering queries by session."""
    session1 = "session-1"
    session2 = "session-2"

    # Add queries to different sessions
    history_store.add_query("Q1", "A1", [], session_id=session1)
    history_store.add_query("Q2", "A2", [], session_id=session1)
    history_store.add_query("Q3", "A3", [], session_id=session2)

    # Filter by session1
    results = history_store.list_queries(session_id=session1)
    assert len(results) == 2
    assert all(q.session_id == session1 for q in results)


def test_list_queries_by_date_range(history_store):
    """Test filtering queries by date range."""
    # Add query
    query_id1 = history_store.add_query("Old", "Answer", [])

    # Wait and capture timestamp
    time.sleep(0.1)
    start_time = time.time()
    time.sleep(0.1)

    # Add newer query
    query_id2 = history_store.add_query("New", "Answer", [])

    # Filter to get only new query
    results = history_store.list_queries(start_date=start_time)
    assert len(results) == 1
    assert results[0].query_id == query_id2


def test_search_queries(history_store):
    """Test searching queries by text."""
    # Add queries with different content
    history_store.add_query("What is Python?", "Python is a language", [])
    history_store.add_query("What is Java?", "Java is a language", [])
    history_store.add_query("Explain databases", "Databases store data", [])

    # Search for Python
    results = history_store.search_queries("Python")
    assert len(results) == 1
    assert "Python" in results[0].question

    # Search for language (in answer)
    results = history_store.search_queries("language")
    assert len(results) == 2


def test_delete_query(history_store):
    """Test deleting a query."""
    query_id = history_store.add_query("Test", "Answer", [])

    # Verify it exists
    assert history_store.get_query(query_id) is not None

    # Delete it
    deleted = history_store.delete_query(query_id)
    assert deleted is True

    # Verify it's gone
    assert history_store.get_query(query_id) is None


def test_delete_nonexistent_query(history_store):
    """Test deleting non-existent query."""
    deleted = history_store.delete_query("fake-id")
    assert deleted is False


def test_clear_history(history_store):
    """Test clearing history."""
    # Add queries
    for i in range(5):
        history_store.add_query(f"Q{i}", f"A{i}", [])

    # Clear all
    count = history_store.clear_history()
    assert count == 5

    # Verify empty
    queries = history_store.list_queries()
    assert len(queries) == 0


def test_clear_history_by_date(history_store):
    """Test clearing history before a date."""
    # Add old query
    history_store.add_query("Old", "Answer", [])

    time.sleep(0.1)
    cutoff = time.time()
    time.sleep(0.1)

    # Add new query
    history_store.add_query("New", "Answer", [])

    # Clear old queries
    count = history_store.clear_history(before_date=cutoff)
    assert count == 1

    # Verify only new query remains
    queries = history_store.list_queries()
    assert len(queries) == 1
    assert queries[0].question == "New"


# =============================================================================
# Bookmark Management Tests
# =============================================================================

def test_add_bookmark(history_store):
    """Test creating a bookmark."""
    # Add query first
    query_id = history_store.add_query("Test Q", "Test A", [])

    # Bookmark it
    bookmark_id = history_store.add_bookmark(
        query_id=query_id,
        notes="Important finding",
        tags=["test", "important"]
    )

    assert bookmark_id is not None

    # Verify bookmark
    bookmark = history_store.get_bookmark(bookmark_id)
    assert bookmark is not None
    assert bookmark.query_id == query_id
    assert bookmark.notes == "Important finding"
    assert "test" in bookmark.tags
    assert bookmark.query is not None  # Should have query data


def test_add_bookmark_nonexistent_query(history_store):
    """Test bookmarking non-existent query raises error."""
    with pytest.raises(QueryNotFoundError):
        history_store.add_bookmark("fake-query-id", notes="Test")


def test_get_bookmark_not_found(history_store):
    """Test getting non-existent bookmark."""
    result = history_store.get_bookmark("fake-id")
    assert result is None


def test_list_bookmarks(history_store):
    """Test listing bookmarks."""
    # Add queries and bookmark them
    query_id1 = history_store.add_query("Q1", "A1", [])
    query_id2 = history_store.add_query("Q2", "A2", [])

    history_store.add_bookmark(query_id1, tags=["tag1"])
    history_store.add_bookmark(query_id2, tags=["tag2"])

    # List all
    bookmarks = history_store.list_bookmarks()
    assert len(bookmarks) == 2


def test_list_bookmarks_by_tags(history_store):
    """Test filtering bookmarks by tags."""
    query_id1 = history_store.add_query("Q1", "A1", [])
    query_id2 = history_store.add_query("Q2", "A2", [])
    query_id3 = history_store.add_query("Q3", "A3", [])

    history_store.add_bookmark(query_id1, tags=["python", "code"])
    history_store.add_bookmark(query_id2, tags=["java", "code"])
    history_store.add_bookmark(query_id3, tags=["data"])

    # Filter by "code" tag
    results = history_store.list_bookmarks(tags=["code"])
    assert len(results) == 2

    # Filter by "python" tag
    results = history_store.list_bookmarks(tags=["python"])
    assert len(results) == 1


def test_update_bookmark(history_store):
    """Test updating bookmark notes and tags."""
    query_id = history_store.add_query("Q", "A", [])
    bookmark_id = history_store.add_bookmark(query_id, notes="Old notes", tags=["old"])

    # Update
    success = history_store.update_bookmark(
        bookmark_id,
        notes="New notes",
        tags=["new", "updated"]
    )
    assert success is True

    # Verify changes
    bookmark = history_store.get_bookmark(bookmark_id)
    assert bookmark.notes == "New notes"
    assert "new" in bookmark.tags
    assert "updated" in bookmark.tags
    assert "old" not in bookmark.tags


def test_update_bookmark_partial(history_store):
    """Test updating only notes or only tags."""
    query_id = history_store.add_query("Q", "A", [])
    bookmark_id = history_store.add_bookmark(query_id, notes="Original", tags=["tag1"])

    # Update only notes
    history_store.update_bookmark(bookmark_id, notes="Updated notes")
    bookmark = history_store.get_bookmark(bookmark_id)
    assert bookmark.notes == "Updated notes"
    assert "tag1" in bookmark.tags  # Tags unchanged

    # Update only tags
    history_store.update_bookmark(bookmark_id, tags=["tag2"])
    bookmark = history_store.get_bookmark(bookmark_id)
    assert "tag2" in bookmark.tags
    assert bookmark.notes == "Updated notes"  # Notes unchanged


def test_update_nonexistent_bookmark(history_store):
    """Test updating non-existent bookmark."""
    success = history_store.update_bookmark("fake-id", notes="Test")
    assert success is False


def test_delete_bookmark(history_store):
    """Test deleting a bookmark."""
    query_id = history_store.add_query("Q", "A", [])
    bookmark_id = history_store.add_bookmark(query_id)

    # Delete bookmark
    deleted = history_store.delete_bookmark(bookmark_id)
    assert deleted is True

    # Verify it's gone
    assert history_store.get_bookmark(bookmark_id) is None

    # Query should still exist
    assert history_store.get_query(query_id) is not None


def test_delete_query_cascade_bookmarks(history_store):
    """Test that deleting query also deletes its bookmarks."""
    query_id = history_store.add_query("Q", "A", [])
    bookmark_id = history_store.add_bookmark(query_id)

    # Delete query
    history_store.delete_query(query_id)

    # Bookmark should be gone too (CASCADE)
    assert history_store.get_bookmark(bookmark_id) is None


# =============================================================================
# Export Tests
# =============================================================================

def test_export_query_markdown(history_store):
    """Test exporting query as Markdown."""
    query_id = history_store.add_query(
        "What is X?",
        "X is a variable.",
        [{"content": "Test source", "metadata": {"filename": "test.pdf", "section": "Intro"}, "similarity": 0.9}]
    )

    markdown = history_store.export_query_markdown(query_id)

    assert "What is X?" in markdown
    assert "X is a variable" in markdown
    assert "test.pdf" in markdown
    assert "Intro" in markdown


def test_export_query_markdown_not_found(history_store):
    """Test exporting non-existent query raises error."""
    with pytest.raises(QueryNotFoundError):
        history_store.export_query_markdown("fake-id")


def test_export_queries_markdown(history_store):
    """Test exporting multiple queries as Markdown."""
    query_id1 = history_store.add_query("Q1", "A1", [])
    query_id2 = history_store.add_query("Q2", "A2", [])

    markdown = history_store.export_queries_markdown(
        [query_id1, query_id2],
        title="Test Export"
    )

    assert "Test Export" in markdown
    assert "Q1" in markdown
    assert "Q2" in markdown
    assert "Total Queries:** 2" in markdown  # Markdown format: **Total Queries:** 2


def test_export_queries_json(history_store):
    """Test exporting queries as JSON."""
    query_id = history_store.add_query(
        "Test Q",
        "Test A",
        [{"content": "source"}],
        metadata={"model": "mistral"}
    )

    json_str = history_store.export_queries_json([query_id])
    data = json.loads(json_str)

    assert "queries" in data
    assert len(data["queries"]) == 1
    assert data["queries"][0]["question"] == "Test Q"
    assert data["queries"][0]["metadata"]["model"] == "mistral"


def test_export_queries_json_without_sources(history_store):
    """Test exporting JSON without sources."""
    query_id = history_store.add_query("Q", "A", [{"content": "source"}])

    json_str = history_store.export_queries_json([query_id], include_sources=False)
    data = json.loads(json_str)

    assert data["queries"][0]["sources"] == []


def test_export_queries_pdf(history_store):
    """Test exporting queries as PDF."""
    pytest.importorskip("reportlab")  # Skip if reportlab not installed

    query_id = history_store.add_query(
        "Test question",
        "Test answer",
        [{"content": "Test source", "metadata": {"filename": "test.pdf"}, "similarity": 0.9}]
    )

    pdf_bytes = history_store.export_queries_pdf([query_id], title="Test PDF")

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")  # PDF header


# =============================================================================
# Data Integrity Tests
# =============================================================================

def test_query_with_sources(history_store):
    """Test that sources are properly stored and retrieved."""
    sources = [
        {
            "content": "First source",
            "metadata": {"filename": "doc1.pdf", "page": 1},
            "similarity": 0.95
        },
        {
            "content": "Second source",
            "metadata": {"filename": "doc2.pdf", "page": 5},
            "similarity": 0.85
        }
    ]

    query_id = history_store.add_query("Q", "A", sources)
    query = history_store.get_query(query_id)

    assert len(query.sources) == 2
    assert query.sources[0]["content"] == "First source"
    assert query.sources[0]["metadata"]["filename"] == "doc1.pdf"
    assert query.sources[0]["similarity"] == 0.95


def test_to_dict_methods(history_store):
    """Test to_dict() methods on dataclasses."""
    query_id = history_store.add_query("Q", "A", [], metadata={"test": "value"})
    query = history_store.get_query(query_id)

    query_dict = query.to_dict()
    assert isinstance(query_dict, dict)
    assert query_dict["question"] == "Q"
    assert query_dict["metadata"]["test"] == "value"

    # Test bookmark to_dict
    bookmark_id = history_store.add_bookmark(query_id, notes="Test")
    bookmark = history_store.get_bookmark(bookmark_id)

    bookmark_dict = bookmark.to_dict()
    assert isinstance(bookmark_dict, dict)
    assert bookmark_dict["notes"] == "Test"
    assert "query" in bookmark_dict


def test_concurrent_operations(history_store):
    """Test that multiple operations work correctly."""
    # Add multiple queries
    query_ids = []
    for i in range(3):
        qid = history_store.add_query(f"Q{i}", f"A{i}", [])
        query_ids.append(qid)

    # Bookmark some
    bookmark_id1 = history_store.add_bookmark(query_ids[0], tags=["tag1"])
    bookmark_id2 = history_store.add_bookmark(query_ids[1], tags=["tag2"])

    # Verify stats
    stats = history_store.stats()
    assert stats["total_queries"] == 3
    assert stats["total_bookmarks"] == 2

    # Delete one query
    history_store.delete_query(query_ids[2])

    # Verify
    assert len(history_store.list_queries()) == 2
    assert len(history_store.list_bookmarks()) == 2
