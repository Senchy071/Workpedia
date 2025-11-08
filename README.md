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
├── core/                # Core functionality (parser, chunker, embedder)
├── processors/          # Document type-specific processors
├── storage/             # Vector store and metadata management
├── api/                 # API endpoints and query interface
├── tests/               # Test files
├── data/                # Sample data and test documents
│   ├── input/          # Input documents for testing
│   └── output/         # Processed output
├── requirements.txt     # Python dependencies
├── pyproject.toml      # Project metadata
└── README.md           # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Ollama installed and running
- Git

### Installation

1. **Clone the repository** (if not already done):
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
python tests/test_setup.py
```

All tests should pass before proceeding to Phase 2.

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

## Configuration

Edit `config/config.py` to customize:

- **Ollama settings**: Model name and API endpoint
- **Embedding settings**: Model choice and dimensions
- **ChromaDB settings**: Storage location and collection name
- **Chunking settings**: Chunk size and overlap percentage

## Phase 1: Environment Setup - Complete ✓

- [x] Project structure created
- [x] All dependencies installed
- [x] Ollama with Mistral model verified
- [x] Configuration file created
- [x] Verification tests passing

## Next Steps

**Phase 2: Document Processing Foundation**
- Implement Docling-based document parser
- Support PDF, DOCX, HTML, and image formats
- Extract structured DoclingDocument format

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Document Parsing | Docling (DocLayNet + TableFormer) | Extract structured content with layout awareness |
| Chunking | Semantic/hierarchical | Preserve document structure and context |
| Vector DB | ChromaDB | Store and retrieve embeddings |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Generate semantic representations |
| LLM | Ollama + Mistral 7B | Local text generation |
| API Framework | (TBD in Phase 4) | Query interface |

## Key Features

- **Privacy-First**: All processing happens locally, no data sent to external APIs
- **Structure-Aware**: Preserves document hierarchies, tables, and cross-references
- **Multi-Format**: Supports PDF, DOCX, HTML, and images
- **Rich Metadata**: Stores page numbers, bounding boxes, and document structure
- **Semantic Search**: Uses advanced embeddings for accurate retrieval

## License

MIT

## Contributing

This project is under active development. Phase 1 (Environment Setup) is complete.
