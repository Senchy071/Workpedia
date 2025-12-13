"""Tests for Phase 3: Chunking and Embedding."""

import shutil
import tempfile

import numpy as np
import pytest

from core.chunker import Chunk, SemanticChunker, chunk_document
from core.embedder import Embedder
from storage.vector_store import DocumentIndexer, VectorStore

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_parsed_doc():
    """Create a sample parsed document for testing."""
    return {
        "doc_id": "test-doc-123",
        "metadata": {
            "filename": "test_document.pdf",
            "file_path": "/tmp/test_document.pdf",
            "pages": 5,
        },
        "raw_text": """# Introduction

This is the introduction section. It contains some introductory text about the document.
The document covers various topics related to testing.

## Background

Here is some background information. This section provides context for the reader.
We discuss the history and evolution of the subject matter.

### Subsection 1

More detailed information in this subsection.

| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

## Methods

This section describes the methods used in the study.
We employed various techniques to gather and analyze data.

## Results

The results show significant findings. Statistical analysis revealed important patterns.

## Conclusion

In conclusion, this document demonstrates the testing framework.
Future work will expand on these findings.
""",
        "structure": {
            "pages": 5,
            "has_tables": True,
            "has_figures": False,
        },
    }


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SemanticChunker Tests
# =============================================================================

class TestSemanticChunker:
    """Tests for the SemanticChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initializes with correct defaults."""
        chunker = SemanticChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 0.15
        assert chunker.overlap_tokens == 76  # 512 * 0.15

    def test_chunker_custom_settings(self):
        """Test chunker with custom settings."""
        chunker = SemanticChunker(chunk_size=256, overlap=0.2)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 0.2
        assert chunker.overlap_tokens == 51  # 256 * 0.2

    def test_chunk_document_basic(self, sample_parsed_doc):
        """Test basic document chunking."""
        chunker = SemanticChunker(chunk_size=200)
        chunks = chunker.chunk_document(sample_parsed_doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_contains_metadata(self, sample_parsed_doc):
        """Test that chunks contain proper metadata."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)

        for chunk in chunks:
            assert chunk.doc_id == "test-doc-123"
            assert chunk.chunk_id.startswith("test-doc-123_chunk_")
            assert "filename" in chunk.metadata
            assert chunk.metadata["filename"] == "test_document.pdf"

    def test_chunk_preserves_tables(self, sample_parsed_doc):
        """Test that tables are preserved as single chunks."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc, preserve_tables=True)

        # Find table chunks
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert len(table_chunks) >= 1

        # Table should contain all rows
        table_content = table_chunks[0].content
        assert "Column A" in table_content
        assert "Value 6" in table_content

    def test_chunk_empty_document(self):
        """Test handling of empty document."""
        empty_doc = {
            "doc_id": "empty-doc",
            "metadata": {},
            "raw_text": "",
            "structure": {},
        }
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(empty_doc)
        assert len(chunks) == 0

    def test_chunk_sections_tracked(self, sample_parsed_doc):
        """Test that section context is tracked in chunks."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)

        # Some chunks should have section metadata
        sections_found = set()
        for chunk in chunks:
            if chunk.metadata.get("section"):
                sections_found.add(chunk.metadata["section"])

        # Should have found some sections
        assert len(sections_found) > 0

    def test_chunk_to_dict(self, sample_parsed_doc):
        """Test Chunk.to_dict() method."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)

        chunk_dict = chunks[0].to_dict()
        assert "chunk_id" in chunk_dict
        assert "doc_id" in chunk_dict
        assert "content" in chunk_dict
        assert "token_count" in chunk_dict
        assert "metadata" in chunk_dict

    def test_convenience_function(self, sample_parsed_doc):
        """Test chunk_document convenience function."""
        chunks = chunk_document(sample_parsed_doc, chunk_size=200)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)


class TestTokenEstimation:
    """Tests for token estimation in chunker."""

    def test_token_estimate_basic(self):
        """Test basic token estimation."""
        chunker = SemanticChunker()
        # ~4 chars per token
        text = "a" * 100
        estimate = chunker._estimate_tokens(text)
        assert estimate == 25  # 100 / 4

    def test_token_estimate_empty(self):
        """Test token estimation with empty string."""
        chunker = SemanticChunker()
        assert chunker._estimate_tokens("") == 0

    def test_token_estimate_realistic(self):
        """Test token estimation with realistic text."""
        chunker = SemanticChunker()
        text = "The quick brown fox jumps over the lazy dog."
        estimate = chunker._estimate_tokens(text)
        # 44 chars / 4 = 11 tokens (actual would be ~10)
        assert estimate == 11


# =============================================================================
# Embedder Tests
# =============================================================================

class TestEmbedder:
    """Tests for the Embedder class."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance (lazy loads model)."""
        return Embedder()

    def test_embedder_initialization(self, embedder):
        """Test embedder initializes correctly."""
        assert embedder.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.normalize is True
        assert embedder.dimension == 768

    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        embedding = embedder.embed("This is a test sentence.")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        # Normalized embeddings should have unit norm
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01

    def test_embed_multiple_texts(self, embedder):
        """Test embedding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    def test_embed_empty_list(self, embedder):
        """Test embedding empty list."""
        embeddings = embedder.embed([])
        assert len(embeddings) == 0

    def test_similarity_computation(self, embedder):
        """Test similarity between embeddings."""
        emb1 = embedder.embed("The cat sat on the mat.")
        emb2 = embedder.embed("A cat was sitting on a mat.")
        emb3 = embedder.embed("Python is a programming language.")

        sim_similar = embedder.similarity(emb1, emb2)
        sim_different = embedder.similarity(emb1, emb3)

        # Similar sentences should have higher similarity
        assert sim_similar > sim_different
        # Similarity range for normalized vectors
        assert -1 <= sim_similar <= 1
        assert -1 <= sim_different <= 1

    def test_embed_chunks(self, embedder, sample_parsed_doc):
        """Test embedding Chunk objects."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)

        embeddings = embedder.embed_chunks(chunks, show_progress=False)

        assert len(embeddings) == len(chunks)
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape == (768,) for e in embeddings)


# =============================================================================
# VectorStore Tests
# =============================================================================

class TestVectorStore:
    """Tests for the VectorStore class."""

    @pytest.fixture
    def vector_store(self, temp_chroma_dir):
        """Create a temporary vector store."""
        return VectorStore(
            persist_directory=temp_chroma_dir,
            collection_name="test_collection",
        )

    @pytest.fixture
    def embedder(self):
        """Create embedder for testing."""
        return Embedder()

    def test_vector_store_initialization(self, vector_store):
        """Test vector store initializes correctly."""
        assert vector_store.count == 0
        assert vector_store.collection_name == "test_collection"

    def test_add_and_retrieve_chunks(self, vector_store, embedder, sample_parsed_doc):
        """Test adding and retrieving chunks."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)

        # Add chunks
        added = vector_store.add_chunks(chunks, embeddings)
        assert added == len(chunks)
        assert vector_store.count == len(chunks)

    def test_query_by_embedding(self, vector_store, embedder, sample_parsed_doc):
        """Test querying by embedding vector."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        # Query with a new embedding
        query_emb = embedder.embed("What are the methods used?")
        results = vector_store.query(query_emb, n_results=3)

        assert "ids" in results
        assert "documents" in results
        assert "distances" in results
        assert len(results["ids"]) <= 3

    def test_query_text(self, vector_store, embedder, sample_parsed_doc):
        """Test querying by text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        # Query with text
        results = vector_store.query_text(
            "What are the results?",
            embedder=embedder,
            n_results=2,
        )

        assert len(results["ids"]) <= 2

    def test_get_by_doc_id(self, vector_store, embedder, sample_parsed_doc):
        """Test retrieving chunks by document ID."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        # Get by doc_id
        results = vector_store.get_by_doc_id("test-doc-123")
        assert len(results["ids"]) == len(chunks)

    def test_delete_by_doc_id(self, vector_store, embedder, sample_parsed_doc):
        """Test deleting chunks by document ID."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        # Delete by doc_id
        deleted = vector_store.delete_by_doc_id("test-doc-123")
        assert deleted == len(chunks)
        assert vector_store.count == 0

    def test_list_documents(self, vector_store, embedder, sample_parsed_doc):
        """Test listing documents in store."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        docs = vector_store.list_documents()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "test-doc-123"
        assert docs[0]["chunk_count"] == len(chunks)

    def test_clear_collection(self, vector_store, embedder, sample_parsed_doc):
        """Test clearing the collection."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        cleared = vector_store.clear()
        assert cleared == len(chunks)
        assert vector_store.count == 0

    def test_stats(self, vector_store, embedder, sample_parsed_doc):
        """Test getting store statistics."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_parsed_doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        stats = vector_store.stats()
        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == len(chunks)
        assert stats["documents"] == 1


# =============================================================================
# DocumentIndexer Tests
# =============================================================================

class TestDocumentIndexer:
    """Tests for the DocumentIndexer class."""

    @pytest.fixture
    def indexer(self, temp_chroma_dir):
        """Create document indexer with temp storage."""
        vector_store = VectorStore(
            persist_directory=temp_chroma_dir,
            collection_name="test_indexer",
        )
        return DocumentIndexer(vector_store=vector_store)

    def test_indexer_initialization(self, indexer):
        """Test indexer initializes correctly."""
        assert indexer.vector_store is not None
        assert indexer.embedder is not None
        assert indexer.chunker is not None

    def test_index_document(self, indexer, sample_parsed_doc):
        """Test indexing a document."""
        result = indexer.index_document(sample_parsed_doc)

        assert result["status"] == "success"
        assert result["doc_id"] == "test-doc-123"
        assert result["chunks_added"] > 0
        assert result["total_tokens"] > 0

    def test_index_and_search(self, indexer, sample_parsed_doc):
        """Test indexing and searching."""
        indexer.index_document(sample_parsed_doc)

        # Search for content
        results = indexer.search("What methods were used?", n_results=3)

        assert len(results) > 0
        assert "content" in results[0]
        assert "similarity" in results[0]
        assert results[0]["similarity"] > 0

    def test_index_empty_document(self, indexer):
        """Test indexing empty document."""
        empty_doc = {
            "doc_id": "empty-doc",
            "metadata": {"filename": "empty.pdf"},
            "raw_text": "",
            "structure": {},
        }
        result = indexer.index_document(empty_doc)
        assert result["status"] == "empty"
        assert result["chunks_added"] == 0

    def test_replace_existing(self, indexer, sample_parsed_doc):
        """Test replacing existing document."""
        # Index once
        indexer.index_document(sample_parsed_doc)
        initial_count = indexer.vector_store.count

        # Index again with replace
        indexer.index_document(sample_parsed_doc, replace_existing=True)

        # Count should be the same (replaced, not doubled)
        assert indexer.vector_store.count == initial_count

    def test_search_by_doc_id(self, indexer, sample_parsed_doc):
        """Test searching within specific document."""
        indexer.index_document(sample_parsed_doc)

        # Search with doc_id filter
        results = indexer.search(
            "introduction",
            n_results=5,
            doc_id="test-doc-123",
        )

        # All results should be from this doc
        for result in results:
            assert result["metadata"]["doc_id"] == "test-doc-123"


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase3Integration:
    """Integration tests for the complete Phase 3 pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_pipeline(self, temp_dir, sample_parsed_doc):
        """Test the complete chunking → embedding → storage pipeline."""
        # Create components
        chunker = SemanticChunker(chunk_size=200)
        embedder = Embedder()
        vector_store = VectorStore(
            persist_directory=temp_dir,
            collection_name="integration_test",
        )

        # Chunk document
        chunks = chunker.chunk_document(sample_parsed_doc)
        assert len(chunks) > 0

        # Generate embeddings
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        assert len(embeddings) == len(chunks)

        # Store in vector store
        added = vector_store.add_chunks(chunks, embeddings)
        assert added == len(chunks)

        # Query
        query = "What are the conclusions?"
        query_emb = embedder.embed(query)
        results = vector_store.query(query_emb, n_results=3)

        assert len(results["ids"]) > 0
        # Most relevant result should contain conclusion-related content
        top_result = results["documents"][0]
        assert len(top_result) > 0

    def test_semantic_relevance(self, temp_dir):
        """Test that search returns semantically relevant results."""
        # Create a document with distinct topics
        doc = {
            "doc_id": "semantic-test",
            "metadata": {"filename": "semantic.pdf"},
            "raw_text": """# Python Programming

Python is a high-level programming language known for its simplicity.
It is widely used for web development, data science, and automation.

# Cooking Recipes

Here are some delicious recipes for beginners.
Start with a simple pasta dish using tomatoes and basil.

# Machine Learning

Machine learning is a subset of artificial intelligence.
Neural networks are commonly used in deep learning applications.
""",
            "structure": {},
        }

        # Set up pipeline
        chunker = SemanticChunker(chunk_size=150)
        embedder = Embedder()
        vector_store = VectorStore(
            persist_directory=temp_dir,
            collection_name="semantic_test",
        )

        chunks = chunker.chunk_document(doc)
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add_chunks(chunks, embeddings)

        # Query about programming
        results = vector_store.query_text(
            "What programming language is discussed?",
            embedder=embedder,
            n_results=1,
        )

        # Should return Python section, not cooking or ML
        top_doc = results["documents"][0].lower()
        assert "python" in top_doc or "programming" in top_doc

        # Query about food
        results = vector_store.query_text(
            "Tell me about food and cooking",
            embedder=embedder,
            n_results=1,
        )

        top_doc = results["documents"][0].lower()
        assert "recipe" in top_doc or "pasta" in top_doc or "cooking" in top_doc
