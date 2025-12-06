# Workpedia

Advanced RAG (Retrieval-Augmented Generation) system using state-of-the-art document processing, vector storage, and local LLM generation.

## Overview

Workpedia is a privacy-focused RAG application that processes complex documents (PDF, DOCX, HTML, images) and enables intelligent question-answering through:

- **Document Processing**: Docling with DocLayNet + TableFormer models for superior layout understanding
- **Intelligent Chunking**: Semantic/hierarchical chunking that preserves document structure
- **Vector Storage**: ChromaDB for efficient similarity search
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 for high-quality semantic representations
- **LLM Generation**: Ollama + Mistral 7B for local, privacy-preserving text generation

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
│   └── doc_merger.py    # Merge parsed chunk results
├── processors/          # Document type-specific processors
│   ├── pdf_processor.py # PDF processing with auto-fallback
│   ├── docx_processor.py
│   ├── html_processor.py
│   └── image_processor.py
├── storage/             # Vector store and metadata (Phase 3)
├── api/                 # API endpoints and query interface (Phase 4)
├── tests/               # Test files (46 tests)
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

All 46 tests should pass.

## Usage

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

- **Ollama settings**: Model name and API endpoint
- **Embedding settings**: Model choice and dimensions
- **ChromaDB settings**: Storage location and collection name
- **Chunking settings**: Chunk size (512 tokens) and overlap (15%)
- **Large document thresholds**: Page count and file size limits

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
- [x] 46 tests passing

### Phase 3: Chunking and Embedding - Planned

- [ ] Semantic/hierarchical chunking
- [ ] Embedding generation with sentence-transformers
- [ ] ChromaDB vector storage integration

### Phase 4: Query Interface - Planned

- [ ] API endpoint design
- [ ] Ollama integration for response generation
- [ ] Query result formatting

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Document Parsing | Docling (DocLayNet + TableFormer) | Extract structured content with layout awareness |
| Large Doc Strategy | Split-Process-Merge | Handle 100+ page PDFs efficiently |
| Structure Analysis | StructureAnalyzer | Extract hierarchy, tables, cross-references |
| Vector DB | ChromaDB | Store and retrieve embeddings (Phase 3) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Generate semantic representations (Phase 3) |
| LLM | Ollama + Mistral 7B | Local text generation (Phase 4) |

## Key Features

- **Privacy-First**: All processing happens locally, no data sent to external APIs
- **Structure-Aware**: Preserves document hierarchies, tables, and cross-references
- **Multi-Format**: Supports PDF, DOCX, HTML, and images
- **Large Document Support**: Automatic chunking for 100+ page documents
- **Rich Metadata**: Stores page numbers, bounding boxes, and document structure
- **Robust Fallback**: Automatic backend switching if processing fails
- **Cross-Reference Detection**: Identifies Table/Figure/Section/Equation references

## License

MIT

## Contributing

This project is under active development. Phase 2 (Document Processing) is complete.
Contributions welcome for Phase 3 (Chunking and Embedding).
