# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Workpedia is a RAG (Retrieval-Augmented Generation) application for property accounting and supply documentation. It uses ChromaDB for vector storage, sentence-transformers for embeddings, and Ollama for LLM responses.

## Development Environment

**Python Environment Setup:**
```bash
source /media/development/projects/rag-workspace/venv/bin/activate
cd /media/development/projects/workpedia
```

**Running the Application:**
```bash
python app.py
```

The app will:
1. Initialize ChromaDB and load the embedding model
2. Prompt to index documents from `documents/` directory (if not already indexed)
3. Enter interactive query mode

**Testing Document Processor:**
```bash
python document_processor.py path/to/document.pdf
```

Shows chunking results with hierarchy information for debugging.

**Database Management:**
```bash
python db-info.py        # View indexed documents and chunk counts
python manage-db.py      # Delete specific documents or entire collection
```

## Architecture

### Core Components

**RAGApp (app.py:13-185)**
- Main application class orchestrating the RAG pipeline
- Manages ChromaDB client, embedding model, and collection
- Handles document loading, indexing, and querying

**StructuredDocumentProcessor (document_processor.py:10-244)**
- Specialized processor for structured documents (NATO/military style)
- Extracts hierarchical structure (chapters, sections, subsections)
- Performs structure-aware chunking that respects document organization
- Detects and removes boilerplate (headers, footers, TOC entries, references)
- Adds section breadcrumbs to chunks for better context

### Document Processing Pipeline

1. **Multi-format Support (app.py:39-102)**:
   - PDF files: Uses `StructuredDocumentProcessor` for intelligent chunking
   - DOCX files: Simple paragraph-based chunking
   - TXT files: Double-newline splitting

2. **Structure-Aware Processing (document_processor.py:111-204)**:
   - Maintains hierarchy tracker: chapters → sections → subsections
   - Breaks chunks at logical boundaries (section headers)
   - Max chunk size: ~1000 characters
   - Preserves context through hierarchy breadcrumbs

3. **Section Detection (document_processor.py:66-109)**:
   Priority order for detecting structure:
   - Level 1: CHAPTER headers and top-level paragraphs (1., 2.)
   - Level 2: Numbered sections (3-8., 4-2.)
   - Level 3: Letter subsections (a., b., c.)
   - Level 4: Numbered subsections ((1), (2))

4. **RAG Formatting (document_processor.py:206-244)**:
   - Prepends `[SECTION: hierarchy]` to each chunk
   - Stores metadata: source, section, subsection, hierarchy depth
   - Generates unique IDs for each chunk

### Vector Storage

**ChromaDB Setup (app.py:29-35)**:
- Persistent storage in `vector_db/` directory
- Collection: "workpedia_knowledge"
- Stores: embeddings, documents, metadata, IDs

**Metadata Schema**:
```python
{
    'source': 'filename.pdf',           # Source file
    'section': 'CHAPTER 3 - TITLE',     # Top-level section
    'subsection': '3-8. Subsection',    # Second-level (optional)
    'hierarchy_depth': 2,                # Nesting level
    'type': 'content',                   # Chunk type
    'chunk_id': 42,                      # Sequential ID
    'file_type': 'pdf'                   # For txt/docx only
}
```

### Query Pipeline

**Retrieval (app.py:141-150)**:
1. Embed query using same model as documents
2. ChromaDB similarity search
3. Return top N chunks (default: 3)

**Generation (app.py:162-185)**:
1. Concatenate retrieved chunks as context
2. Build prompt with context and question
3. Send to Ollama LLM (model: mistral)
4. Return generated answer

## Configuration

Edit `config.py` to modify:

- `embedding_model`: Sentence transformer model (default: all-mpnet-base-v2)
- `llm_model`: Ollama model name (default: mistral)
- `chunk_size`: Max characters per chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `n_results`: Number of chunks to retrieve (default: 3)
- `collection_name`: ChromaDB collection name

## Key Implementation Details

**Indexing Strategy (app.py:104-139)**:
- Checks for existing collection before indexing
- Prompts for re-indexing confirmation
- Deletes old collection if re-indexing
- Shows progress bar during embedding generation
- Batch stores embeddings, documents, metadata, and IDs

**Structure Detection Patterns (document_processor.py:53-109)**:
- Filters out NATO headers: "NATO UNCLASSIFIED", "Releasable to PfP"
- Removes page numbers and boilerplate
- Detects reference sections and TOC entries
- Uses regex patterns to identify section hierarchy

**Chunk Boundaries (document_processor.py:159-194)**:
- Breaks at section headers for clean boundaries
- Minimum chunk size: 50 characters
- Maximum chunk size: ~1000 characters
- Looks ahead to avoid mid-section breaks

## Directory Structure

```
workpedia/
├── documents/          # Source documents (PDF, DOCX, TXT)
├── docs_for_processing/  # Staging area for new documents
├── vector_db/          # ChromaDB persistent storage
├── logs/               # Application logs
├── configs/            # Additional configurations
├── app.py              # Main application
├── document_processor.py  # Structure-aware PDF processor
├── config.py           # Configuration settings
├── manage-db.py        # Database management CLI
└── db-info.py          # Database inspection tool
```

## Common Modifications

**Adding Support for New Document Types**:
- Add new file type handler in `RAGApp.load_documents()` (app.py:39-102)
- Follow pattern: read file → chunk content → append to documents/metadatas/ids lists

**Customizing Chunking Strategy**:
- Modify `StructuredDocumentProcessor.chunk_document()` (document_processor.py:111-204)
- Adjust section detection patterns in `detect_section_type()` (document_processor.py:66-109)
- Change max chunk size at line 184

**Changing Query Prompt**:
- Edit prompt template in `RAGApp.query()` (app.py:165-173)
- Modify how context is formatted (currently numbered Context 1, 2, 3)

**Adjusting Metadata**:
- Modify metadata dictionaries in `load_documents()` for each file type
- Update `format_for_rag()` in document_processor.py:206-244 for PDF metadata
