# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

```
config/       # Configuration constants (Ollama, embeddings, ChromaDB, chunking)
core/         # Core RAG components: parser, chunker, embedder
processors/   # Document type-specific processors (PDF, DOCX, HTML, images)
storage/      # Vector store interface and metadata management
api/          # Query interface and API endpoints
tests/        # Test files
data/         # Input documents and processed output
```

**Key Configuration** (config/config.py):
- Ollama: http://localhost:11434 using "mistral" model
- ChromaDB: Persisted to `chroma_db/` in project root, collection "workpedia_docs"
- Chunking: 512 tokens with 15% overlap
- Embedding dimension: 768
- Large documents: >100 pages or >50MB processed in 75-page chunks

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

**Project Status**: Phase 2 complete (document processing foundation). Ready for Phase 3 (chunking and embedding).

**Document Processing Flow** (implemented/planned):
1. ✓ Docling parser extracts structured DoclingDocument format
2. Semantic/hierarchical chunker preserves structure (headers, tables, cross-refs) [Phase 3]
3. Embedder generates 768-dim vectors using all-mpnet-base-v2 [Phase 3]
4. ChromaDB stores chunks with metadata (page numbers, bounding boxes, structure) [Phase 3]
5. Query interface retrieves relevant chunks and generates answers via Mistral [Phase 4]

**Phase 2 Components**:
- `core/parser.py`: DocumentParser with Docling integration
- `core/large_doc_handler.py`: LargeDocumentHandler for 100+ page PDFs
- `core/analyzer.py`: StructureAnalyzer for document hierarchy extraction
- `core/validator.py`: DocumentValidator for result validation
- `core/progress_tracker.py`: ProgressTracker for processing monitoring
- `processors/`: Format-specific processors (PDF, DOCX, HTML, Image)

**Testing Phase 2**:
```bash
# Run all tests (31 tests)
pytest tests/

# Run parser-specific tests
pytest tests/test_parser.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Demo parser with your documents
python3 demo_parser.py
```

**Important Constraints**:
- All data must stay local (privacy-first design)
- Document structure must be preserved through chunking
- Metadata tracking includes page numbers, bounding boxes, document hierarchy
- Supports PDF, DOCX, HTML, and image formats

**Configuration Customization**:
All settings centralized in `config/config.py`. Modify these for different models, chunk sizes, or storage locations. The config uses Path objects for cross-platform compatibility.
