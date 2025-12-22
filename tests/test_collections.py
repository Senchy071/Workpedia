"""Tests for document collections and tags functionality."""

import tempfile
from pathlib import Path

import pytest

from storage.collections import Collection, CollectionManager, DocumentMetadata


class TestCollection:
    """Tests for Collection dataclass."""

    def test_collection_creation(self):
        """Test creating a Collection."""
        collection = Collection(
            collection_id="test-id",
            name="Test Collection",
            description="A test collection",
            created_at=1000.0,
            updated_at=1000.0,
            document_count=5,
        )

        assert collection.collection_id == "test-id"
        assert collection.name == "Test Collection"
        assert collection.description == "A test collection"
        assert collection.document_count == 5

    def test_collection_to_dict(self):
        """Test Collection.to_dict method."""
        collection = Collection(
            collection_id="test-id",
            name="Test",
            description="Desc",
            created_at=1000.0,
            updated_at=1000.0,
            document_count=0,
        )

        d = collection.to_dict()
        assert d["collection_id"] == "test-id"
        assert d["name"] == "Test"
        assert "document_count" in d


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_document_metadata_creation(self):
        """Test creating DocumentMetadata."""
        meta = DocumentMetadata(
            doc_id="doc-123",
            collection_id="coll-456",
            collection_name="Test Collection",
            tags=["tag1", "tag2"],
        )

        assert meta.doc_id == "doc-123"
        assert meta.collection_id == "coll-456"
        assert meta.tags == ["tag1", "tag2"]

    def test_document_metadata_to_dict(self):
        """Test DocumentMetadata.to_dict method."""
        meta = DocumentMetadata(doc_id="doc-123", tags=["a", "b"])

        d = meta.to_dict()
        assert d["doc_id"] == "doc-123"
        assert d["tags"] == ["a", "b"]


class TestCollectionManager:
    """Tests for CollectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create CollectionManager with temp database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test_collections.db")
            yield CollectionManager(db_path=db_path)

    # =========================================================================
    # Collection CRUD Tests
    # =========================================================================

    def test_create_collection(self, manager):
        """Test creating a collection."""
        collection_id = manager.create_collection(
            name="Legal Documents",
            description="All legal contracts",
        )

        assert collection_id is not None
        assert len(collection_id) > 0

    def test_create_collection_duplicate_name(self, manager):
        """Test creating collection with duplicate name fails."""
        manager.create_collection(name="Test")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_collection(name="Test")

    def test_get_collection(self, manager):
        """Test getting a collection by ID."""
        collection_id = manager.create_collection(name="Test", description="Desc")

        collection = manager.get_collection(collection_id)

        assert collection is not None
        assert collection.name == "Test"
        assert collection.description == "Desc"
        assert collection.document_count == 0

    def test_get_collection_not_found(self, manager):
        """Test getting non-existent collection returns None."""
        collection = manager.get_collection("nonexistent-id")
        assert collection is None

    def test_get_collection_by_name(self, manager):
        """Test getting collection by name."""
        manager.create_collection(name="My Collection")

        collection = manager.get_collection_by_name("My Collection")

        assert collection is not None
        assert collection.name == "My Collection"

    def test_list_collections(self, manager):
        """Test listing all collections."""
        manager.create_collection(name="Alpha")
        manager.create_collection(name="Beta")
        manager.create_collection(name="Gamma")

        collections = manager.list_collections()

        assert len(collections) == 3
        names = [c.name for c in collections]
        assert "Alpha" in names
        assert "Beta" in names
        assert "Gamma" in names

    def test_update_collection(self, manager):
        """Test updating a collection."""
        collection_id = manager.create_collection(name="Original", description="Old")

        manager.update_collection(collection_id, name="Updated", description="New")

        collection = manager.get_collection(collection_id)
        assert collection.name == "Updated"
        assert collection.description == "New"

    def test_update_collection_partial(self, manager):
        """Test updating only some fields."""
        collection_id = manager.create_collection(name="Test", description="Original")

        manager.update_collection(collection_id, description="Changed")

        collection = manager.get_collection(collection_id)
        assert collection.name == "Test"  # Unchanged
        assert collection.description == "Changed"

    def test_update_nonexistent_collection(self, manager):
        """Test updating non-existent collection returns False."""
        result = manager.update_collection("fake-id", name="New")
        assert result is False

    def test_delete_collection(self, manager):
        """Test deleting a collection."""
        collection_id = manager.create_collection(name="ToDelete")

        result = manager.delete_collection(collection_id)

        assert result is True
        assert manager.get_collection(collection_id) is None

    def test_delete_nonexistent_collection(self, manager):
        """Test deleting non-existent collection returns False."""
        result = manager.delete_collection("fake-id")
        assert result is False

    # =========================================================================
    # Document-Collection Association Tests
    # =========================================================================

    def test_add_document_to_collection(self, manager):
        """Test adding document to collection."""
        collection_id = manager.create_collection(name="Test")

        result = manager.add_document_to_collection("doc-123", collection_id)

        assert result is True
        docs = manager.get_documents_in_collection(collection_id)
        assert "doc-123" in docs

    def test_get_document_collection(self, manager):
        """Test getting document's collection."""
        collection_id = manager.create_collection(name="Test")
        manager.add_document_to_collection("doc-123", collection_id)

        collection = manager.get_document_collection("doc-123")

        assert collection is not None
        assert collection.name == "Test"

    def test_get_document_collection_none(self, manager):
        """Test getting collection for unassigned document."""
        collection = manager.get_document_collection("orphan-doc")
        assert collection is None

    def test_remove_document_from_collection(self, manager):
        """Test removing document from collection."""
        collection_id = manager.create_collection(name="Test")
        manager.add_document_to_collection("doc-123", collection_id)

        result = manager.remove_document_from_collection("doc-123", collection_id)

        assert result is True
        docs = manager.get_documents_in_collection(collection_id)
        assert "doc-123" not in docs

    def test_set_document_collection(self, manager):
        """Test setting document collection (replaces existing)."""
        coll1 = manager.create_collection(name="Collection 1")
        coll2 = manager.create_collection(name="Collection 2")

        manager.add_document_to_collection("doc-123", coll1)
        manager.set_document_collection("doc-123", coll2)

        collection = manager.get_document_collection("doc-123")
        assert collection.collection_id == coll2

    def test_document_count_updates(self, manager):
        """Test that document count updates correctly."""
        collection_id = manager.create_collection(name="Test")

        manager.add_document_to_collection("doc-1", collection_id)
        manager.add_document_to_collection("doc-2", collection_id)

        collection = manager.get_collection(collection_id)
        assert collection.document_count == 2

    # =========================================================================
    # Tag Management Tests
    # =========================================================================

    def test_add_tags(self, manager):
        """Test adding tags to document."""
        added = manager.add_tags("doc-123", ["project:alpha", "type:contract"])

        assert added == 2
        tags = manager.get_document_tags("doc-123")
        assert "project:alpha" in tags
        assert "type:contract" in tags

    def test_add_tags_normalization(self, manager):
        """Test that tags are normalized to lowercase."""
        manager.add_tags("doc-123", ["PROJECT:ALPHA", "Type:Contract"])

        tags = manager.get_document_tags("doc-123")
        assert "project:alpha" in tags
        assert "type:contract" in tags

    def test_add_duplicate_tags(self, manager):
        """Test adding duplicate tags doesn't create duplicates."""
        manager.add_tags("doc-123", ["tag1", "tag2"])
        manager.add_tags("doc-123", ["tag2", "tag3"])

        tags = manager.get_document_tags("doc-123")
        assert len(tags) == 3
        assert "tag1" in tags
        assert "tag2" in tags
        assert "tag3" in tags

    def test_remove_tags(self, manager):
        """Test removing tags from document."""
        manager.add_tags("doc-123", ["tag1", "tag2", "tag3"])

        removed = manager.remove_tags("doc-123", ["tag2"])

        assert removed == 1
        tags = manager.get_document_tags("doc-123")
        assert "tag1" in tags
        assert "tag2" not in tags
        assert "tag3" in tags

    def test_set_document_tags(self, manager):
        """Test setting (replacing) all tags."""
        manager.add_tags("doc-123", ["old1", "old2"])

        manager.set_document_tags("doc-123", ["new1", "new2", "new3"])

        tags = manager.get_document_tags("doc-123")
        assert len(tags) == 3
        assert "old1" not in tags
        assert "new1" in tags

    def test_get_documents_by_tags_any(self, manager):
        """Test getting documents by tags (any match)."""
        manager.add_tags("doc-1", ["project:alpha", "type:contract"])
        manager.add_tags("doc-2", ["project:beta", "type:contract"])
        manager.add_tags("doc-3", ["project:alpha", "type:report"])

        docs = manager.get_documents_by_tags(["project:alpha"], match_all=False)

        assert len(docs) == 2
        assert "doc-1" in docs
        assert "doc-3" in docs

    def test_get_documents_by_tags_all(self, manager):
        """Test getting documents by tags (all must match)."""
        manager.add_tags("doc-1", ["project:alpha", "type:contract"])
        manager.add_tags("doc-2", ["project:alpha"])
        manager.add_tags("doc-3", ["type:contract"])

        docs = manager.get_documents_by_tags(
            ["project:alpha", "type:contract"], match_all=True
        )

        assert len(docs) == 1
        assert "doc-1" in docs

    def test_list_all_tags(self, manager):
        """Test listing all unique tags with counts."""
        manager.add_tags("doc-1", ["tag1", "tag2"])
        manager.add_tags("doc-2", ["tag1", "tag3"])
        manager.add_tags("doc-3", ["tag1"])

        tags = manager.list_all_tags()

        # Find tag1 - should have count 3
        tag1 = next(t for t in tags if t["tag"] == "tag1")
        assert tag1["document_count"] == 3

    # =========================================================================
    # Document Metadata Tests
    # =========================================================================

    def test_get_document_metadata(self, manager):
        """Test getting full document metadata."""
        collection_id = manager.create_collection(name="Test")
        manager.add_document_to_collection("doc-123", collection_id)
        manager.add_tags("doc-123", ["tag1", "tag2"])

        metadata = manager.get_document_metadata("doc-123")

        assert metadata.doc_id == "doc-123"
        assert metadata.collection_id == collection_id
        assert metadata.collection_name == "Test"
        assert "tag1" in metadata.tags
        assert "tag2" in metadata.tags

    def test_delete_document_metadata(self, manager):
        """Test deleting all document metadata."""
        collection_id = manager.create_collection(name="Test")
        manager.add_document_to_collection("doc-123", collection_id)
        manager.add_tags("doc-123", ["tag1", "tag2"])

        result = manager.delete_document_metadata("doc-123")

        assert result is True
        assert manager.get_document_collection("doc-123") is None
        assert manager.get_document_tags("doc-123") == []

    # =========================================================================
    # Statistics Tests
    # =========================================================================

    def test_get_stats(self, manager):
        """Test getting collection statistics."""
        manager.create_collection(name="Coll1")
        manager.create_collection(name="Coll2")
        manager.add_tags("doc-1", ["tag1", "tag2"])
        manager.add_tags("doc-2", ["tag1"])

        stats = manager.get_stats()

        assert stats["collections_count"] == 2
        assert stats["unique_tags"] == 2
        assert stats["documents_with_tags"] == 2

    def test_empty_stats(self, manager):
        """Test stats when nothing exists."""
        stats = manager.get_stats()

        assert stats["collections_count"] == 0
        assert stats["unique_tags"] == 0
        assert stats["documents_with_tags"] == 0


class TestCollectionIntegration:
    """Integration tests for collections with vector store."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_collection_filtering_in_vector_store(self, temp_dirs):
        """Test that collection metadata is stored in vector store."""
        from storage.vector_store import VectorStore
        from core.chunker import Chunk
        import numpy as np

        # Create vector store
        vs = VectorStore(
            persist_directory=str(temp_dirs / "chroma"),
            collection_name="test",
        )

        # Create test chunks with collection
        chunks = [
            Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Test content 1",
                token_count=10,
                metadata={"filename": "test.pdf"},
            ),
            Chunk(
                chunk_id="chunk-2",
                doc_id="doc-1",
                content="Test content 2",
                token_count=10,
                metadata={"filename": "test.pdf"},
            ),
        ]
        embeddings = [np.random.rand(768) for _ in chunks]

        # Add with collection and tags
        vs.add_chunks(
            chunks,
            embeddings,
            collection_name="Legal",
            tags=["project:alpha", "type:contract"],
        )

        # Verify collection is stored
        docs = vs.list_documents_by_collection("Legal")
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc-1"
        assert docs[0]["collection"] == "Legal"

        # Verify tags are stored
        docs_by_tag = vs.list_documents_by_tag("project:alpha")
        assert len(docs_by_tag) == 1

    def test_update_document_collection(self, temp_dirs):
        """Test updating document collection in vector store."""
        from storage.vector_store import VectorStore
        from core.chunker import Chunk
        import numpy as np

        vs = VectorStore(
            persist_directory=str(temp_dirs / "chroma"),
            collection_name="test",
        )

        chunks = [
            Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Test",
                token_count=5,
                metadata={},
            ),
        ]
        embeddings = [np.random.rand(768)]

        vs.add_chunks(chunks, embeddings, collection_name="Old")

        # Update collection
        updated = vs.update_document_collection("doc-1", "New")
        assert updated == 1

        # Verify update
        docs = vs.list_documents_by_collection("New")
        assert len(docs) == 1

        old_docs = vs.list_documents_by_collection("Old")
        assert len(old_docs) == 0

    def test_update_document_tags(self, temp_dirs):
        """Test updating document tags in vector store."""
        from storage.vector_store import VectorStore
        from core.chunker import Chunk
        import numpy as np

        vs = VectorStore(
            persist_directory=str(temp_dirs / "chroma"),
            collection_name="test",
        )

        chunks = [
            Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Test",
                token_count=5,
                metadata={},
            ),
        ]
        embeddings = [np.random.rand(768)]

        vs.add_chunks(chunks, embeddings, tags=["old-tag"])

        # Update tags
        updated = vs.update_document_tags("doc-1", ["new-tag1", "new-tag2"])
        assert updated == 1

        # Verify update
        docs = vs.list_documents_by_tag("new-tag1")
        assert len(docs) == 1

        old_docs = vs.list_documents_by_tag("old-tag")
        assert len(old_docs) == 0
