# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Directive: Use Web Search Proactively

**IMPORTANT**: The AI field changes daily. When encountering obstacles, bugs, or design decisions:

1. **Search first, code second** - Use WebSearch to check for:
   - Latest library versions and breaking changes
   - Known issues and their fixes (GitHub issues, Stack Overflow)
   - Current best practices and recommendations
   - New tools or approaches that solve the problem better

2. **Examples where web search would have helped**:
   - Docling "corrupted double-linked list" crash → Found in GitHub issues
   - PyPdfium stability vs V2 performance → Documented in discussions
   - Large PDF handling → Community solutions and workarounds

3. **Don't rely solely on knowledge cutoff** (January 2025):
   - Libraries like Docling, ChromaDB, sentence-transformers update frequently
   - Bug fixes and patches are released continuously
   - Best practices evolve as the field matures

**Rule**: When stuck, search before implementing workarounds. The problem may already be solved.

## Project Overview

Workpedia is a privacy-focused RAG (Retrieval-Augmented Generation) application that processes complex documents and enables intelligent question-answering. All processing happens locally - no external API calls.

**Key Technologies:**

- **Document Parsing**: Docling with DocLayNet + TableFormer models for layout-aware extraction
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768-dimensional)
- **Vector Store**: ChromaDB for similarity search
- **LLM**: Ollama + Mistral 7B (local generation)
- **Chunking**: Semantic/hierarchical chunking preserving document structure

## Architecture

The codebase follows a modular architecture organized by functional concern:

```text
config/       # Configuration constants (Ollama, embeddings, ChromaDB, chunking)
core/         # Core RAG components: parser, chunker, embedder
processors/   # Document type-specific processors (PDF, DOCX, HTML, images)
storage/      # Vector store interface and metadata management
api/          # Query interface and API endpoints
tests/        # Test files
data/         # Input documents and processed output
```

**Key Configuration** (config/config.py):

Core Settings:
- Ollama: http://localhost:11434 using "mistral" model
- ChromaDB: Persisted to `chroma_db/` in project root, collection "workpedia_docs"
- Chunking: 512 tokens with 15% overlap
- Embedding: sentence-transformers/all-mpnet-base-v2 (768 dimensions)

Document Processing:
- Large documents: >100 pages or >50MB processed in 75-page chunks
- Backend selection: Auto-detect large docs (>200 pages OR >20MB) → use PyPdfium backend
- VLM settings: Granite-Docling model with configurable batch size

Production Settings:
- Logging: INFO level, file rotation (10MB, 5 backups), structured JSON option
- Retry: Max 3 attempts, exponential backoff (1s → 2s → 4s → 8s)
- Circuit breaker: Opens after 5 failures, 60s recovery timeout
- Timeouts: Health check (5s), generation (120s), streaming (180s)

Feature Settings:
- Confidence Scoring: Enabled by default, thresholds (HIGH: 0.75, MEDIUM: 0.50)
- Document Summaries: Enabled by default, 5 bullets, 15000 char input limit
- Query Suggestions: Enabled by default, max 15 per document, min heading length 5
- Hybrid Search: Enabled by default, semantic weight 0.7, keyword weight 0.3, RRF k=60
- Query History: Auto-save enabled, session tracking enabled

## Development Commands

### Environment Setup

```bash
# Install core dependencies
pip install -r requirements.txt

# Install with development tools (pytest, black, ruff, mypy)
pip install -e ".[dev]"

# Verify complete setup (must pass before Phase 2)
python tests/test_setup.py
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_setup.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=.
```

### Code Quality

```bash
# Format code (line length: 100)
black .

# Lint code
ruff check .

# Type check
mypy .
```

### Ollama Commands

```bash
# Check Ollama is running and Mistral is available
ollama list

# Pull Mistral model (if not available)
ollama pull mistral

# Test generation (30s timeout for quick verification)
timeout 30 ollama run mistral "test prompt"
```

## Development Notes

**Project Status**: ALL PHASES COMPLETE. Production-ready RAG system.

**Document Processing Flow** (ALL IMPLEMENTED):

1. ✓ Docling parser extracts structured DoclingDocument format
2. ✓ Structure analysis with cross-references, tables, figures
3. ✓ Large document handling with split-process-merge strategy
4. ✓ Semantic/hierarchical chunker preserves structure
5. ✓ Embedder generates 768-dim vectors using all-mpnet-base-v2
6. ✓ ChromaDB stores chunks with metadata
7. ✓ QueryEngine retrieves relevant chunks and generates answers via Mistral
8. ✓ FastAPI REST endpoints for full API access

**Phase 2 Components** (ALL COMPLETE):

- `core/parser.py`: DocumentParser with Docling integration (V2 + PyPdfium + VLM backends)
- `core/large_doc_handler.py`: LargeDocumentHandler with split-process-merge for 100+ page PDFs
- `core/pdf_splitter.py`: PDFSplitter for splitting large PDFs into chunks
- `core/doc_merger.py`: DocumentMerger for combining parsed chunk results
- `core/analyzer.py`: StructureAnalyzer with full feature set:
  - Cross-reference extraction (tables, figures, sections, equations, citations)
  - Table header extraction from Docling data or markdown
  - Multi-page table detection and merging
  - Table dimensions (rows, cols) extraction
  - Document hierarchy building
- `core/validator.py`: DocumentValidator for result validation
- `core/progress_tracker.py`: ProgressTracker for processing monitoring
- `processors/`: Format-specific processors (PDF, DOCX, HTML, Image)

**Production Infrastructure** (ALL COMPLETE):

- `core/exceptions.py`: Custom exception hierarchy
  - WorkpediaError base class with context
  - Specific exceptions for document, parsing, embedding, query, validation, and LLM errors
- `core/logging_config.py`: Production logging infrastructure
  - Structured JSON logging for production
  - Colored console logging for development
  - Context-aware logging (request IDs, doc IDs)
  - Performance timing decorators
  - File rotation (10MB per file, 5 backups)
- `core/validators.py`: Input validation and sanitization
  - Query validation (length, content, security)
  - File path security (path traversal prevention)
  - Document ID validation
  - File upload validation (size, type, content)
  - Metadata validation
- `core/resilience.py`: Connection resilience patterns
  - Retry logic with exponential backoff
  - Circuit breaker pattern (fail-fast when Ollama down)
  - Per-operation timeouts
  - Statistics tracking

**Phase 3 Components** (ALL COMPLETE):

- `core/chunker.py`: SemanticChunker for structure-aware text splitting
  - Preserves tables and figures as single chunks
  - Tracks section context in metadata
  - Configurable chunk size (512 tokens) and overlap (15%)
- `core/embedder.py`: Embedder using sentence-transformers
  - all-mpnet-base-v2 model (768-dim embeddings)
  - Batch processing with GPU support
  - Normalized vectors for cosine similarity
- `storage/vector_store.py`: ChromaDB vector store
  - Persistent storage to disk
  - Similarity search with metadata filtering
  - DocumentIndexer for high-level workflow

**Phase 4 Components** (ALL COMPLETE):

- `core/llm.py`: OllamaClient for LLM integration
  - Streaming and non-streaming generation
  - Chat completion support
  - RAG prompt templates
- `core/query_engine.py`: QueryEngine for RAG queries
  - Combines retrieval + generation
  - Source citation and formatting
  - Health checks
  - Summary query detection for "what's in this document" queries
- `api/endpoints.py`: FastAPI REST API
  - Query endpoints (sync and streaming)
  - Document management (upload, index, delete)
  - Document summary endpoint
  - System stats and health checks
  - OpenAPI documentation at /docs

**Additional Features** (ALL 6 HIGH PRIORITY FEATURES COMPLETE):

1. `storage/history_store.py`: Query History & Bookmarks
   - Persistent SQLite storage of all queries
   - Bookmark favorite Q&A pairs with tags and notes
   - Export to Markdown, JSON, PDF formats
   - Session tracking and filtering

2. `core/confidence.py`: Answer Confidence Scoring
   - Calculate confidence based on similarity, agreement, and coverage
   - Configurable thresholds (HIGH >= 0.75, MEDIUM >= 0.50)
   - Detailed factors breakdown for transparency
   - Emoji indicators (🟢 HIGH, 🟡 MEDIUM, 🔴 LOW)

3. `core/summarizer.py`: Document Summarization
   - Auto-generate 3-7 bullet executive summaries during indexing
   - LLM-based summarization using Ollama
   - Stored as searchable chunks in vector store
   - Automatic detection of summary queries

4. `storage/history_store.py`: Export Functionality
   - Export query history to Markdown, JSON, PDF
   - Export bookmarks with notes and tags
   - Confidence scores included in exports
   - Proper Content-Disposition headers for downloads

5. `core/suggestions.py`: Query Suggestions
   - Auto-generate questions from section headings
   - Extract from TOC entries and key concepts
   - Priority-based ordering (h1 > h2 > h3)
   - Stored as special chunks in vector store

6. `core/hybrid_search.py`: Hybrid Search (Semantic + BM25)
   - BM25Index for keyword search with persistent storage
   - HybridSearcher combines semantic + keyword using RRF
   - Configurable weights (default: 70% semantic, 30% keyword)
   - Improves exact match queries (IDs, codes, names)

**Large Document Strategy** (Split-Process-Merge):

1. PDFSplitter splits large PDFs into 75-page chunks
2. Each chunk processed with V2 backend (fast, good structure)
3. DocumentMerger combines results with page offset handling
4. Automatic cleanup of temporary files

**PDF Backend Selection** (Automatic):

- Small docs (<200 pages AND <20MB): V2 backend (fast, good structure)
- Large docs (>200 pages OR >20MB): Split-process-merge with V2 backend
- Automatic fallback chain if processing fails

**Testing**:

```bash
# Run all tests (150+ tests)
pytest tests/

# Run parser-specific tests
pytest tests/test_parser.py -v

# Run Phase 3 tests (chunking, embedding, vector store)
pytest tests/test_phase3.py -v

# Run Phase 4 tests (LLM, query engine, API)
pytest tests/test_phase4.py -v

# Run additional feature tests
pytest tests/test_confidence.py -v        # Confidence scoring
pytest tests/test_summarizer.py -v        # Document summaries
pytest tests/test_suggestions.py -v       # Query suggestions
pytest tests/test_hybrid_search.py -v     # Hybrid search

# Run integration tests
pytest tests/test_integration.py -v

# Demo parser with your documents
python3 demo_parser.py
```

**Running the API Server**:

```bash
# Start the API server
python -m api.endpoints

# With auto-reload for development
python -m api.endpoints --reload

# Custom host/port
python -m api.endpoints --host 0.0.0.0 --port 8080
```

**Important Constraints**:

- All data must stay local (privacy-first design)
- Document structure must be preserved through chunking
- Metadata tracking includes page numbers, bounding boxes, document hierarchy
- Supports PDF, DOCX, HTML, and image formats

**Configuration Customization**:
All settings centralized in `config/config.py`. Modify these for different models, chunk sizes, or storage locations. The config uses Path objects for cross-platform compatibility.
