# Workpedia

Advanced RAG (Retrieval-Augmented Generation) system using state-of-the-art document processing, vector storage, and local LLM generation.

## Overview

Workpedia is a privacy-focused RAG application that processes complex documents (PDF, DOCX, HTML, images) and enables intelligent question-answering through:

- **Document Processing**: Docling with DocLayNet + TableFormer models for superior layout understanding
- **Intelligent Chunking**: Semantic/hierarchical chunking that preserves document structure
- **Vector Storage**: ChromaDB for efficient similarity search
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 for high-quality semantic representations
- **LLM Generation**: Ollama + Mistral 7B for local, privacy-preserving text generation
- **Performance Caching**: Intelligent caching of embeddings and LLM responses for 2-3x speedup on repeated queries
- **Vector Store Backup**: Automated backup and restore with versioning, compression, and retention policies
- **Confidence Scoring**: Answer reliability indicators (🟢 High / 🟡 Medium / 🔴 Low) based on source quality
- **Document Summaries**: Auto-generated executive summaries with 3-7 bullet points per document
- **Query Suggestions**: Auto-generated questions from document headings and key concepts
- **Hybrid Search**: Semantic + BM25 keyword search combined using Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Re-rank top 20 candidates with cross-encoder for significantly improved answer quality
- **Query History**: Persistent storage of all queries with full context and sources
- **Bookmarks**: Organize favorite Q&A pairs with notes and tags
- **Export**: Export queries and answers to Markdown, JSON, or PDF formats

## Project Structure

```
workpedia/
├── config/              # Configuration files
│   └── config.py        # Central configuration (Ollama, embeddings, chunking)
├── core/                # Core RAG components
│   ├── parser.py        # DocumentParser with Docling integration
│   ├── analyzer.py      # StructureAnalyzer for document hierarchy
│   ├── validator.py     # DocumentValidator for result validation
│   ├── progress_tracker.py  # Processing progress monitoring
│   ├── large_doc_handler.py # Split-process-merge for large PDFs
│   ├── pdf_splitter.py  # PDF splitting by page ranges
│   ├── doc_merger.py    # Merge parsed chunk results
│   ├── chunker.py       # SemanticChunker for structure-aware chunking
│   ├── embedder.py      # Embedder for semantic vector generation
│   ├── llm.py           # OllamaClient for LLM integration
│   ├── query_engine.py  # RAG query engine
│   ├── caching.py       # Performance caching for embeddings and LLM responses
│   ├── confidence.py    # Answer confidence scoring
│   ├── summarizer.py    # Document summary generation
│   ├── suggestions.py   # Query suggestion generation
│   ├── hybrid_search.py # Hybrid search (semantic + BM25)
│   ├── reranker.py      # Cross-encoder reranking
│   ├── exceptions.py    # Custom exception hierarchy
│   ├── logging_config.py # Production logging infrastructure
│   ├── validators.py    # Input validation and sanitization
│   └── resilience.py    # Connection resilience patterns
├── processors/          # Document type-specific processors
│   ├── pdf_processor.py # PDF processing with auto-fallback
│   ├── docx_processor.py
│   ├── html_processor.py
│   └── image_processor.py
├── storage/             # Vector store and metadata
│   ├── vector_store.py  # ChromaDB vector store interface
│   ├── history_store.py # Query history and bookmarks storage
│   └── backup.py        # Vector store backup and restore
├── api/                 # API endpoints and query interface
│   └── endpoints.py     # FastAPI REST API
├── app.py               # Streamlit web UI
├── tests/               # Test files (150+ tests)
├── data/                # Sample data and test documents
│   ├── input/           # Input documents for testing
│   └── output/          # Processed output
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
├── CLAUDE.md            # AI assistant instructions
├── FIXES.md             # Technical fixes documentation
└── README.md            # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Ollama installed and running
- Git
- CUDA-capable GPU (recommended for large documents)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd workpedia
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify Ollama setup**:
```bash
# Check Ollama is installed
which ollama

# Check Mistral model is available
ollama list

# If Mistral is not available, pull it
ollama pull mistral
```

4. **Run verification tests**:
```bash
pytest tests/ -v
```

All 150+ tests should pass.

## Usage

### Web UI (Recommended for Beginners)

The easiest way to use Workpedia is through the Streamlit web interface:

```bash
# Make sure Ollama is running with Mistral
ollama serve
ollama pull mistral

# Start the web UI
streamlit run app.py
```

The UI will open at `http://localhost:8501` with features:
- 📤 **Upload Documents**: Drag & drop PDF, DOCX, or HTML files
- 💬 **Chat Interface**: Ask questions about your documents
- 📊 **Statistics**: View indexed documents and system stats
- ⚙️ **Settings**: Adjust context chunks and temperature

### Process a Document

```bash
# Process a single document
python3 demo_parser.py /path/to/document.pdf

# Process all documents in data/input/
python3 demo_parser.py
```

### Programmatic Usage

```python
from processors.pdf_processor import PDFProcessor

# Process a PDF (auto-selects best backend)
processor = PDFProcessor()
result = processor.process("/path/to/document.pdf")

print(f"Pages: {result['metadata']['pages']}")
print(f"Tables: {len(result['structure'].get('tables', []))}")
print(f"Text preview: {result['raw_text'][:500]}")
```

### Large Document Handling

Documents over 100 pages are automatically split and processed in chunks:

```python
from core.large_doc_handler import LargeDocumentHandler

handler = LargeDocumentHandler()
result = handler.process("/path/to/large_document.pdf")

# Progress callback for monitoring
def on_progress(chunk_num, total_chunks, info):
    print(f"Processing chunk {chunk_num}/{total_chunks}")

result = handler.process(
    "/path/to/large_document.pdf",
    progress_callback=on_progress
)
```

### Chunking and Embedding (Phase 3)

```python
from core.chunker import SemanticChunker
from core.embedder import Embedder
from storage.vector_store import VectorStore, DocumentIndexer

# Method 1: High-level DocumentIndexer (recommended)
indexer = DocumentIndexer()

# Parse and index a document
from core.parser import DocumentParser
parser = DocumentParser()
parsed_doc = parser.parse("/path/to/document.pdf")
result = indexer.index_document(parsed_doc)
print(f"Indexed {result['chunks_added']} chunks")

# Search for relevant content
results = indexer.search("What are the main findings?", n_results=5)
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content'][:200]}...")
```

```python
# Method 2: Component-level control
from core.chunker import SemanticChunker
from core.embedder import Embedder
from storage.vector_store import VectorStore

# Chunk the document
chunker = SemanticChunker(chunk_size=512, overlap=0.15)
chunks = chunker.chunk_document(parsed_doc)

# Generate embeddings
embedder = Embedder()
embeddings = embedder.embed_chunks(chunks)

# Store in ChromaDB
vector_store = VectorStore()
vector_store.add_chunks(chunks, embeddings)

# Query
query_emb = embedder.embed("What methods were used?")
results = vector_store.query(query_emb, n_results=5)
```

### Query Interface (Phase 4)

```python
from core.query_engine import QueryEngine

# Initialize query engine
engine = QueryEngine()

# Ask a question
result = engine.query("What are the main findings of this study?")
print(f"Answer: {result.answer}")
print(f"Sources: {len(result.sources)} chunks used")

# Filter to specific document
result = engine.query("Summarize the methods", doc_id="doc-123")

# Get similar chunks without generation
chunks = engine.get_similar_chunks("machine learning applications", n_results=10)
```

### REST API

Start the API server:

```bash
# Start server (default: http://localhost:8000)
python -m api.endpoints

# Or with custom settings
python -m api.endpoints --host 0.0.0.0 --port 8080 --reload
```

API Endpoints:
- `POST /query` - Query documents and get AI-generated answer (includes confidence score)
- `POST /query/stream` - Streaming query response
- `POST /search` - Semantic search without generation
- `POST /documents/index` - Index a document from filesystem (generates summary)
- `POST /documents/upload` - Upload and index a document (generates summary)
- `GET /documents` - List all indexed documents
- `GET /documents/{doc_id}` - Get document details
- `GET /documents/{doc_id}/summary` - Get auto-generated document summary
- `GET /documents/{doc_id}/suggestions` - Get auto-generated query suggestions
- `DELETE /documents/{doc_id}` - Delete a document
- `GET /history` - List query history with filters
- `GET /history/{query_id}` - Get specific query
- `DELETE /history/{query_id}` - Delete query from history
- `GET /history/export/markdown` - Export queries as Markdown
- `GET /history/export/json` - Export queries as JSON
- `GET /history/export/pdf` - Export queries as PDF
- `GET /bookmarks` - List bookmarks with tag filtering
- `POST /bookmarks` - Create a bookmark
- `GET /bookmarks/{bookmark_id}` - Get specific bookmark
- `PUT /bookmarks/{bookmark_id}` - Update bookmark
- `DELETE /bookmarks/{bookmark_id}` - Delete bookmark
- `POST /backup/create` - Create vector store backup
- `GET /backup/list` - List all backups
- `POST /backup/restore` - Restore from backup
- `DELETE /backup/delete/{backup_name}` - Delete backup
- `GET /backup/stats` - Get backup statistics
- `GET /health` - Health check
- `GET /stats` - System statistics

Interactive docs available at `http://localhost:8000/docs`

## Development Setup

For development work, install additional tools:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest (testing)
- black (code formatting)
- ruff (linting)
- mypy (type checking)

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=.

# Run specific test file
pytest tests/test_parser.py -v
```

## Configuration

Edit `config/config.py` to customize:

**Core Settings:**
- **Ollama settings**: Model name (default: "mistral") and API endpoint (http://localhost:11434)
- **Embedding settings**: Model choice (all-mpnet-base-v2) and dimensions (768)
- **ChromaDB settings**: Storage location and collection name
- **Chunking settings**: Chunk size (512 tokens) and overlap (15%)

**Document Processing:**
- **Large document thresholds**: Page count (100 pages) and file size limits (50MB)
- **Backend selection**: Auto-detect threshold (200 pages or 20MB)
- **VLM settings**: Granite-Docling model and batch size

**Production Settings:**
- **Logging**: Log level, file rotation, structured JSON, colored console output
- **Retry logic**: Max attempts (3), exponential backoff delays, jitter
- **Circuit breaker**: Failure threshold (5), recovery timeout (60s)
- **Timeouts**: Per-operation timeouts (health check: 5s, generation: 120s, streaming: 180s)

## Development Status

### Phase 1: Environment Setup - Complete ✓

- [x] Project structure created
- [x] All dependencies installed
- [x] Ollama with Mistral model verified
- [x] Configuration file created
- [x] Verification tests passing

### Phase 2: Document Processing Foundation - Complete ✓

- [x] Docling-based document parser (V2 + PyPdfium + VLM backends)
- [x] Multi-format support (PDF, DOCX, HTML, images)
- [x] Structure analysis with cross-references
- [x] Table header extraction and multi-page detection
- [x] Large document handling (split-process-merge)
- [x] Document validation and progress tracking

### Phase 3: Chunking and Embedding - Complete ✓

- [x] Semantic/hierarchical chunking (preserves tables, sections, figures)
- [x] Embedding generation with sentence-transformers/all-mpnet-base-v2
- [x] ChromaDB vector storage integration
- [x] DocumentIndexer for high-level indexing workflow
- [x] Similarity search with metadata filtering

### Phase 4: Query Interface - Complete ✓

- [x] OllamaClient for LLM integration (streaming/non-streaming)
- [x] QueryEngine combining retrieval + generation
- [x] FastAPI REST endpoints with OpenAPI docs
- [x] Document upload and indexing API
- [x] Health checks and system statistics
- [x] Streamlit web UI for user-friendly document upload and Q&A
- [x] 150+ tests passing

### Additional Features - Complete ✓

- [x] **Query History & Bookmarks**: Persistent SQLite storage with export to Markdown/JSON/PDF
- [x] **Answer Confidence Scoring**: Reliability indicators based on source quality and agreement
- [x] **Document Summaries**: Auto-generated executive summaries during indexing
- [x] **Cross-Encoder Reranking**: Re-rank search results with cross-encoder for improved quality

### Production Improvements - Complete ✓

- [x] **Ollama Startup Validation**: Fast-fail with clear errors when Ollama unavailable
- [x] **Custom Exception Hierarchy**: Comprehensive error types with context
- [x] **Production Logging**: Structured logging, request tracking, file rotation
- [x] **Input Validation**: Security hardening against path traversal, injection attacks
- [x] **Connection Resilience**: Retry with exponential backoff, circuit breaker pattern

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Document Parsing | Docling (DocLayNet + TableFormer) | Extract structured content with layout awareness |
| Large Doc Strategy | Split-Process-Merge | Handle 100+ page PDFs efficiently |
| Structure Analysis | StructureAnalyzer | Extract hierarchy, tables, cross-references |
| Chunking | SemanticChunker | Structure-aware text splitting with overlap |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | 768-dim semantic representations |
| Vector DB | ChromaDB | Persistent vector storage with similarity search |
| LLM | Ollama + Mistral 7B | Local privacy-preserving text generation |
| API | FastAPI + Uvicorn | REST API with OpenAPI documentation |
| Web UI | Streamlit | Interactive web interface for document upload and Q&A |

## Key Features

- **Privacy-First**: All processing happens locally, no data sent to external APIs
- **Performance Caching**: Automatic caching of embeddings and LLM responses for 2-3x speedup on repeated queries
- **Vector Store Backup**: Full backup and restore with compression, versioning, and retention policies for data protection
- **Structure-Aware**: Preserves document hierarchies, tables, and cross-references
- **Automatic Table of Contents**: Synthetic TOC chunk created for each document enables queries like "List main chapters"
- **Document Summaries**: Auto-generated executive summaries (3-7 bullets) when documents are indexed
- **Confidence Scoring**: Every answer includes a reliability score (🟢 High / 🟡 Medium / 🔴 Low)
- **Query Suggestions**: Auto-generated questions from document structure and headings
- **Hybrid Search**: Combines semantic + BM25 keyword search using Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Re-ranks top candidates using cross-encoder for significantly better quality
- **Multi-Format**: Supports PDF, DOCX, HTML, and images
- **Large Document Support**: Automatic chunking for 100+ page documents
- **Rich Metadata**: Stores page numbers, bounding boxes, and document structure
- **Robust Fallback**: Automatic backend switching if processing fails
- **Cross-Reference Detection**: Identifies Table/Figure/Section/Equation references
- **Semantic Search**: High-quality 768-dim embeddings with cosine similarity
- **Persistent Storage**: ChromaDB vector store with disk persistence
- **RAG Query Engine**: Combines retrieval and generation for accurate answers
- **Query History & Bookmarks**: Persistent storage of queries with export functionality
- **REST API**: Full-featured API with streaming support and OpenAPI docs
- **Web UI**: User-friendly Streamlit interface for document upload and Q&A

## Production-Ready Features

Workpedia includes enterprise-grade features for reliability and maintainability:

1. **Ollama Startup Validation**: Fast-fail behavior with clear error messages when Ollama is unavailable or misconfigured
2. **Custom Exception Hierarchy**: Comprehensive error handling with specific exceptions for different failure modes
3. **Production Logging**: Structured logging infrastructure with configurable levels and output formats
4. **Input Validation**: Robust sanitization and validation of all user inputs and file uploads
5. **Connection Resilience**: Automatic retry logic with exponential backoff for Ollama connectivity

## License

MIT

## Contributing

This project has completed all four development phases and is production-ready.
Contributions welcome for enhancements, bug fixes, and documentation improvements.
