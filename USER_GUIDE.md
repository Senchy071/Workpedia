# Workpedia User Guide

## Privacy-First RAG Document Question-Answering System

This guide will help you get started with Workpedia and make the most of its features.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Running Workpedia](#running-workpedia)
3. [Using the Web Interface](#using-the-web-interface)
4. [Indexing Documents](#indexing-documents)
5. [Querying Documents](#querying-documents)
6. [Understanding Confidence Scores](#understanding-confidence-scores)
7. [Document Summaries](#document-summaries)
8. [Query Suggestions](#query-suggestions)
9. [Hybrid Search](#hybrid-search)
10. [Configuration](#configuration)
11. [Advanced Features](#advanced-features)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

Before using Workpedia, ensure you have:

1. **Python 3.10 or higher** installed
2. **Ollama** installed and running
3. **Mistral model** downloaded in Ollama

### Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify Ollama is running
ollama list

# 3. If Mistral is not installed, download it
ollama pull mistral

# 4. Start Workpedia
streamlit run app.py
```

The web interface will open automatically at `http://localhost:8501`

---

## Running Workpedia

### Web Interface (Recommended for Most Users)

The easiest way to use Workpedia:

```bash
streamlit run app.py
```

**What you get:**

- User-friendly interface
- Drag-and-drop file upload
- Interactive chat
- Document statistics
- Real-time processing feedback

### REST API (For Developers)

If you need programmatic access:

```bash
python -m api.endpoints
```

**Access at:** `http://localhost:8000`
**API Documentation:** `http://localhost:8000/docs`

### Command Line (For Batch Processing)

Process documents without the UI:

```bash
# Process a single file
python3 demo_parser.py /path/to/document.pdf

# Process all files in data/input/
python3 demo_parser.py
```

---

## Using the Web Interface

### Main Features

#### 1. **Sidebar - Document Upload**

**Supported Formats:**

- PDF (`.pdf`)
- Word Documents (`.docx`)
- Excel Spreadsheets (`.xlsx`, `.xls`)
- CSV/TSV (`.csv`, `.tsv`)
- HTML (`.html`, `.htm`)

**How to Upload:**

1. Click "Choose files" or drag-and-drop
2. Select one or more documents
3. Click "Index Documents"
4. Wait for processing to complete

**Processing Indicators:**

- Progress bar shows current progress
- Status messages indicate what's happening
- Success/error notifications appear when done

#### 2. **Main Chat Interface**

**How It Works:**

- Ask questions about your documents
- Get AI-generated answers with sources
- See similarity scores for retrieved chunks
- Automatic TOC retrieval for structural questions

**Structural Questions:**

Workpedia automatically creates a Table of Contents chunk when indexing documents. When you ask structural questions, this TOC chunk will be automatically retrieved:
- "List all chapters"
- "What sections are in this document?"
- "Show me the table of contents"
- "What is the document structure?"

**How to Ask Questions:**

1. Type your question in the chat input
2. Press Enter or click Send
3. Wait for the AI to process and respond
4. Review the answer and sources

**Example Questions:**

- "What are the main findings?"
- "Summarize the methodology section"
- "List all the chapters in the book"
- "What does the document say about climate change?"

#### 3. **Settings Panel**

**Adjustable Parameters:**

**Context Chunks (1-10):**
- How many relevant chunks to retrieve
- **Lower (1-3):** Faster, very focused answers
- **Medium (4-6):** Balanced (default: 5)
- **Higher (7-10):** More comprehensive context
- **Default:** 5 (good balance)

**Temperature (0.0-1.0):**

- Controls AI creativity/randomness
- **0.0:** Deterministic, factual responses
- **0.3-0.5:** Balanced (recommended)
- **0.8-1.0:** More creative, less predictable

#### 4. **Statistics Tab**

View system information:

- Total documents indexed
- Total chunks stored
- Documents list with metadata
- Collection statistics

---

## Indexing Documents

### Understanding Document Processing

When you index a document, Workpedia:

1. **Parses** the document to extract text and structure
2. **Analyzes** tables, figures, sections, and hierarchy
3. **Chunks** the content into meaningful pieces
4. **Generates** semantic embeddings (vector representations)
5. **Stores** chunks in the vector database
6. **Creates** a Table of Contents chunk for structural queries

### Indexing Options

#### Using the Web UI

**Standard Indexing:**

1. Upload file(s) via sidebar
2. Click "Index Documents"
3. Wait for completion

**What Happens:**

- Small files (<200 pages, <20MB): Fast processing with V2 backend
- Large files (>200 pages or >20MB): Automatic split-process-merge
- Tables and figures are preserved
- Document structure is maintained
- Automatic TOC generation

#### Using the API

**Index from File Path:**

```bash
curl -X POST "http://localhost:8000/documents/index" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf"
  }'
```

**Upload and Index:**

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

#### Using Python Code

```python
from storage.vector_store import DocumentIndexer
from core.parser import DocumentParser

# Initialize
parser = DocumentParser()
indexer = DocumentIndexer()

# Parse document
parsed_doc = parser.parse("/path/to/document.pdf")

# Index it
result = indexer.index_document(parsed_doc)
print(f"Indexed {result['chunks_added']} chunks")
```

### Document Processing Details

#### Small Documents

- **Size:** <200 pages AND <20MB
- **Backend:** V2 (fast, excellent structure extraction)
- **Time:** ~1-2 seconds per page
- **Features:** Full table detection, figure extraction, hierarchy

#### Large Documents

- **Size:** >200 pages OR >20MB
- **Strategy:** Split into 75-page chunks
- **Backend:** V2 for each chunk
- **Time:** ~2-3 minutes per 100 pages
- **Features:** All structure preserved, automatic merging


- Automatic chunking prevents memory issues
- Progress tracking available
- Metadata preserved across chunks
- Page offsets correctly handled

---

## Querying Documents

### Query Types

#### 1. **Factual Questions**

Best settings:

- Context chunks: 3-7
- Temperature: 0.0-0.3

Examples:

```text
What is the definition of X?
When was Y published?
Who are the authors?
What are the main conclusions?
```

#### 2. **Analytical Questions**

Best settings:

- Context chunks: 7-10
- Temperature: 0.3-0.5

Examples:

```text
Compare method A and method B
What are the strengths and weaknesses?
How does this relate to that?
Explain the methodology
```

#### 3. **Summarization**

Best settings:

- Context chunks: 8-10
- Temperature: 0.3-0.5

Examples:

```text
Summarize the introduction
What are the key findings?
Give me an overview of chapter 3
List the main arguments
```

#### 4. **Structural/TOC Questions**

**Note:** TOC retrieval is automatic - no special mode needed!

Best settings:

- Context chunks: 3-5
- Temperature: 0.0

Examples:
```
List all chapters
What sections does this document have?
Show me the table of contents
What is the document structure?
```

### Understanding Results

**Answer Section:**
- AI-generated response based on retrieved context
- Uses ONLY information from your documents
- If information isn't found, the AI will tell you

**Sources Section:**
- Shows which document chunks were used
- Displays similarity scores (higher = more relevant)
- Includes page numbers and section information
- Preview of chunk content

**Source Interpretation:**
- **95-100% similarity:** Nearly exact match
- **80-95% similarity:** Highly relevant
- **70-80% similarity:** Somewhat relevant
- **<70% similarity:** May be tangentially related

---

## Understanding Confidence Scores

Every answer from Workpedia includes a **confidence score** that indicates how reliable the answer is based on the quality of retrieved sources.

### Confidence Levels

| Level | Indicator | Score Range | What It Means |
|-------|-----------|-------------|---------------|
| **High** | 🟢 | 75-100% | Strong evidence from multiple relevant sources |
| **Medium** | 🟡 | 50-74% | Moderate evidence, some uncertainty |
| **Low** | 🔴 | 0-49% | Limited or weak evidence, use with caution |

### How Confidence Is Calculated

The confidence score combines three factors:

1. **Similarity Score (50% weight)**
   - How closely do retrieved chunks match your query?
   - Higher similarity = more relevant sources

2. **Agreement Score (30% weight)**
   - Do multiple sources say the same thing?
   - Sources from different documents agreeing = higher confidence

3. **Coverage Score (20% weight)**
   - Were enough relevant sources found?
   - Finding all requested chunks = better coverage

### Interpreting Results

**High Confidence (🟢)**
- Answer is well-supported by multiple relevant sources
- Sources agree with each other
- You can trust this answer

**Medium Confidence (🟡)**
- Answer has some support but may be incomplete
- Sources might not fully agree
- Consider asking follow-up questions

**Low Confidence (🔴)**
- Limited relevant information found
- Sources may be tangentially related
- The answer might be speculative
- Consider rephrasing your question or adding more documents

### API Response

Query responses include a confidence object:

```json
{
  "answer": "...",
  "confidence": {
    "overall_score": 0.85,
    "level": "high",
    "similarity_score": 0.92,
    "agreement_score": 0.78,
    "coverage_score": 0.80,
    "factors": {
      "chunk_count": 5,
      "unique_documents": 3,
      "top_similarities": [0.95, 0.89, 0.85]
    }
  }
}
```

---

## Document Summaries

Workpedia automatically generates **executive summaries** when you index documents, giving you a quick overview before you start querying.

### How It Works

When a document is indexed:
1. Workpedia extracts key content (sections, text)
2. The LLM generates 3-7 bullet point summaries
3. Summary is stored as a searchable chunk
4. Summary is automatically retrieved for overview queries

### Accessing Summaries

#### Via Query

Simply ask about the document's content:

```text
What is in this document?
What's this document about?
Give me a document summary
What are the main topics?
Tell me about this document
```

Workpedia automatically detects these queries and returns the summary.

#### Via API

```bash
# Get summary for a specific document
curl http://localhost:8000/documents/{doc_id}/summary
```

Response:
```json
{
  "doc_id": "report_2024_abc123",
  "summary": "# DOCUMENT SUMMARY\n\n1. First key point...",
  "bullets": [
    "First key point about the document",
    "Second important finding",
    "Third main topic covered",
    "Fourth relevant detail",
    "Fifth concluding point"
  ],
  "metadata": {
    "filename": "annual_report_2024.pdf",
    "num_bullets": 5,
    "summary_model": "mistral"
  }
}
```

#### Via Python

```python
from storage.vector_store import VectorStore

store = VectorStore()
summary = store.get_document_summary("doc_id_here")
print(summary["content"])
```

### Configuration

Edit `config/config.py` to customize summaries:

```python
SUMMARY_ENABLED = True       # Enable/disable auto-summarization
SUMMARY_MAX_BULLETS = 5      # Number of bullet points (3-7)
SUMMARY_MAX_INPUT_CHARS = 15000  # Max chars sent to LLM
SUMMARY_TEMPERATURE = 0.3    # Lower = more focused summaries
```

### When Summaries Are Generated

- **Automatically**: During document indexing (if enabled)
- **Indexing options**: `generate_summary=True/False` to override default

### Tips

- Summaries work best with well-structured documents
- Very short documents may not generate useful summaries
- Summaries are searchable - they help with overview queries
- Re-indexing a document regenerates its summary

---

## Query Suggestions

Workpedia automatically generates **suggested questions** from your documents, helping you discover what questions to ask.

### How It Works

When a document is indexed:
1. Extracts section headings (## Methods, ## Results)
2. Identifies key concepts from content
3. Converts to natural questions
4. Stores suggestions with priority ranking

### Accessing Suggestions

#### Via API

```bash
# Get suggestions for a specific document
curl http://localhost:8000/documents/{doc_id}/suggestions
```

Response:
```json
{
  "doc_id": "report_2024_abc123",
  "suggestions": [
    {
      "suggestion_id": "report_2024_abc123_heading_1",
      "text": "What is covered in the Methods section?",
      "source_type": "heading",
      "source_text": "Methods",
      "priority": 8,
      "metadata": {"heading_level": 2}
    },
    {
      "suggestion_id": "report_2024_abc123_concept_0",
      "text": "What is Machine Learning?",
      "source_type": "concept",
      "source_text": "Machine Learning",
      "priority": 5,
      "metadata": {"frequency": 12}
    }
  ],
  "count": 15
}
```

### Suggestion Types

1. **Heading-based**: From document sections
   - "What is covered in the Introduction?"
   - "Tell me about Chapter 3"

2. **Concept-based**: From frequent capitalized phrases
   - "What is Neural Network Architecture?"
   - "What is Deep Learning?"

3. **Default**: For documents without clear structure
   - "What is this document about?"
   - "What are the main topics?"

### Configuration

```python
# config/config.py
SUGGESTIONS_ENABLED = True           # Enable/disable auto-suggestions
SUGGESTIONS_MAX_PER_DOCUMENT = 15    # Maximum suggestions per document
SUGGESTIONS_MIN_HEADING_LENGTH = 5   # Minimum heading length to process
```

### Tips

- Suggestions are generated during indexing
- Higher priority suggestions appear first
- Great for discovering what's in unfamiliar documents
- Suggestions are stored as special chunks

---

## Hybrid Search

Workpedia uses **hybrid search** combining semantic similarity and keyword matching for best results.

### How It Works

For every query, Workpedia:
1. **Semantic Search**: Finds conceptually similar chunks (ChromaDB)
2. **Keyword Search**: Finds exact term matches (BM25)
3. **RRF Fusion**: Combines rankings using Reciprocal Rank Fusion
4. Returns unified, optimally ranked results

### Why Hybrid Search?

**Semantic-only limitations:**
- Misses exact matches like "invoice #12345"
- Poor with codes, IDs, and specific names
- Struggles with rare or technical terms

**Hybrid search advantages:**
- Finds exact matches: "Find report ABC-2024" ✅
- Better with technical terms and IDs
- More robust across query types
- Best of both worlds

### Example Improvements

#### Before (Semantic Only):
```
Query: "Find invoice 12345"
❌ Returns: General payment info (misses exact ID)
```

#### After (Hybrid Search):
```
Query: "Find invoice 12345"
✅ Returns: Exact invoice with ID 12345 (top result)
```

### How Results Are Combined

**Reciprocal Rank Fusion (RRF):**
```
score = (semantic_weight × 1/(k + semantic_rank)) +
        (keyword_weight × 1/(k + keyword_rank))
```

Default weights:
- **Semantic**: 70% (better for concepts)
- **Keyword**: 30% (better for exact matches)
- **k constant**: 60 (standard RRF value)

### Configuration

```python
# config/config.py
HYBRID_SEARCH_ENABLED = True           # Enable/disable hybrid search
HYBRID_SEARCH_K = 60                   # RRF constant k
HYBRID_SEARCH_SEMANTIC_WEIGHT = 0.7    # Semantic importance (0.0-1.0)
HYBRID_SEARCH_KEYWORD_WEIGHT = 0.3     # Keyword importance (0.0-1.0)
HYBRID_SEARCH_INDEX_PATH = "data/bm25_index.json"  # BM25 index location
```

### Tuning Weights

Adjust based on your use case:

**More conceptual queries** (research, analysis):
```python
SEMANTIC_WEIGHT = 0.8
KEYWORD_WEIGHT = 0.2
```

**More exact lookups** (invoices, codes, IDs):
```python
SEMANTIC_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5
```

**Pure semantic** (disable hybrid):
```python
HYBRID_SEARCH_ENABLED = False
```

### BM25 Index

The keyword search index is automatically:
- Built during document indexing
- Saved to disk (`data/bm25_index.json`)
- Loaded on startup for fast queries
- Updated when documents are added/removed

### Tips

- Hybrid search is automatic - no special queries needed
- Best for mixed query types (concepts + exact matches)
- BM25 index grows with your documents
- Restart not needed - index updates in real-time

---

## Configuration

### Basic Configuration

Edit `config/config.py` to customize Workpedia:

#### Ollama Settings

```python
OLLAMA_MODEL = "mistral"  # Change to use different model
OLLAMA_BASE_URL = "http://localhost:11434"  # If Ollama runs elsewhere
```

**Available Models:**
```bash
# List what you have
ollama list

# Popular alternatives
ollama pull llama2
ollama pull codellama
ollama pull neural-chat
```

Then update `OLLAMA_MODEL` in config.

#### Embedding Settings

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
```

**Alternative embedding models:**
- `all-MiniLM-L6-v2` (faster, smaller, 384 dim)
- `all-mpnet-base-v2` (default, balanced, 768 dim)
- `paraphrase-multilingual-mpnet-base-v2` (multilingual)

⚠️ **Warning:** Changing embedding model requires re-indexing all documents!

#### Chunking Settings

```python
CHUNK_SIZE = 512  # tokens per chunk
CHUNK_OVERLAP = 0.15  # 15% overlap between chunks
```

**Chunk Size Guidelines:**
- **256-384 tokens:** Precise retrieval, more chunks
- **512 tokens:** Balanced (default)
- **768-1024 tokens:** Broader context, fewer chunks

**Overlap Guidelines:**
- **0.0 (0%):** No overlap, maximum efficiency
- **0.15 (15%):** Balanced (default)
- **0.25-0.30 (25-30%):** More context, better continuity

#### Large Document Settings

```python
MAX_PAGES_SINGLE_PASS = 100  # Process in chunks if more
MAX_FILE_SIZE_MB = 50  # Process in chunks if larger
CHUNK_SIZE_PAGES = 75  # Pages per chunk

LARGE_DOC_PAGE_THRESHOLD = 200  # Use stable backend if >200 pages
LARGE_DOC_SIZE_MB_THRESHOLD = 20  # Use stable backend if >20MB
```

**Tuning for Your System:**

If you have lots of RAM/GPU:
```python
MAX_PAGES_SINGLE_PASS = 200
CHUNK_SIZE_PAGES = 100
```

If you have limited resources:
```python
MAX_PAGES_SINGLE_PASS = 50
CHUNK_SIZE_PAGES = 50
```

#### Vector Store Settings

```python
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_NAME = "workpedia_docs"
```

**To use a different database location:**
```python
CHROMA_PERSIST_DIR = "/path/to/your/database"
```

**To create separate collections for different projects:**
```python
CHROMA_COLLECTION_NAME = "project_a_docs"
# or
CHROMA_COLLECTION_NAME = "research_papers"
```

#### Logging Settings

```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR = PROJECT_ROOT / "logs"
LOG_STRUCTURED = False  # Set True for JSON logging
LOG_CONSOLE_COLORS = True  # Colored output for development
```

**For production:**
```python
LOG_LEVEL = "WARNING"
LOG_STRUCTURED = True
LOG_CONSOLE_COLORS = False
```

**For debugging:**
```python
LOG_LEVEL = "DEBUG"
LOG_MODULE_LEVELS = {
    "core.parser": "DEBUG",
    "core.chunker": "DEBUG",
}
```

#### Performance Settings

```python
# Retry configuration
RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_DELAY = 1.0  # seconds

# Circuit breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0  # seconds

# Timeouts
TIMEOUT_GENERATE = 120.0  # seconds
TIMEOUT_GENERATE_STREAM = 180.0  # seconds
```

**Increase timeouts for complex queries:**
```python
TIMEOUT_GENERATE = 300.0  # 5 minutes
```

**Decrease for faster failure:**
```python
TIMEOUT_GENERATE = 60.0  # 1 minute
```

### Advanced Configuration

#### Custom Ports

**Web UI (Streamlit):**
```bash
streamlit run app.py --server.port 8080
```

**REST API:**
```bash
python -m api.endpoints --host 0.0.0.0 --port 9000
```

#### Environment Variables

Create a `.env` file:
```bash
OLLAMA_BASE_URL=http://192.168.1.100:11434
OLLAMA_MODEL=llama2
LOG_LEVEL=DEBUG
```

Load in code:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Advanced Features

### 1. Filtering by Document

**Web UI:**
- Currently retrieves from all indexed documents
- Future feature: document selection dropdown

**API:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the methodology?",
    "doc_id": "document-123",
    "n_results": 10
  }'
```

**Python:**
```python
from core.query_engine import QueryEngine

engine = QueryEngine()
result = engine.query(
    "What is the methodology?",
    doc_id="document-123",
    n_results=10
)
```

### 2. Document Management

**List Documents:**
```python
from storage.vector_store import VectorStore

store = VectorStore()
docs = store.list_documents()
for doc in docs:
    print(f"{doc['doc_id']}: {doc['filename']} ({doc['chunk_count']} chunks)")
```

**Delete Document:**
```python
store.delete_by_doc_id("document-123")
```

**Via API:**
```bash
# List documents
curl http://localhost:8000/documents

# Delete document
curl -X DELETE http://localhost:8000/documents/document-123
```

### 3. Batch Processing

**Process Multiple Files:**

Create `process_batch.py`:
```python
from pathlib import Path
from core.parser import DocumentParser
from storage.vector_store import DocumentIndexer

parser = DocumentParser()
indexer = DocumentIndexer()

docs_dir = Path("data/input")
for pdf_file in docs_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")

    # Parse
    parsed = parser.parse(str(pdf_file))

    # Index
    result = indexer.index_document(parsed)
    print(f"  Indexed {result['chunks_added']} chunks")
```

Run it:
```bash
python3 process_batch.py
```

### 4. Similarity Search (Without AI Generation)

Get relevant chunks without generating an answer:

```python
from core.query_engine import QueryEngine

engine = QueryEngine()
chunks = engine.get_similar_chunks(
    "machine learning",
    n_results=10
)

for chunk in chunks:
    print(f"Similarity: {chunk['similarity']:.2%}")
    print(f"Content: {chunk['content'][:200]}...")
    print()
```

### 5. Custom Prompts

Modify `core/llm.py` to customize the RAG system prompt:

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant specializing in [YOUR DOMAIN].

Instructions:
- Answer questions using ONLY the provided context
- Be precise and cite specific sections when possible
- If unsure, acknowledge limitations
- Use technical terminology when appropriate
"""
```

---

## Troubleshooting

### Common Issues

#### "Ollama connection failed"

**Symptoms:**
- Error on startup
- "Failed to connect to Ollama" message

**Solutions:**
1. Check Ollama is running:
   ```bash
   ollama list
   ```
2. Start Ollama if needed:
   ```bash
   ollama serve
   ```
3. Verify model is installed:
   ```bash
   ollama pull mistral
   ```
4. Check base URL in `config/config.py`

#### "Model not available"

**Symptoms:**
- Startup error about missing model

**Solution:**
```bash
ollama pull mistral
# or for the specific model you configured
ollama pull your-model-name
```

#### Document indexing fails

**Symptoms:**
- Upload succeeds but indexing fails
- Error messages in UI

**Solutions:**

1. **Check file format:**
   - Supported: PDF, DOCX, XLSX, XLS, CSV, TSV, HTML
   - File must not be corrupted

2. **Check file size:**
   - Very large files (1000+ pages) may timeout
   - Try splitting into smaller files

3. **Check logs:**
   ```bash
   tail -f logs/workpedia.log
   ```

4. **Retry with different settings:**
   - Edit `config/config.py`
   - Reduce `CHUNK_SIZE_PAGES` for large files

#### Slow query responses

**Symptoms:**
- Queries take >30 seconds
- Timeouts occur

**Solutions:**

1. **Reduce context chunks:**
   - Lower from 10 to 5 in settings

2. **Increase timeout:**
   - Edit `config/config.py`:
   ```python
   TIMEOUT_GENERATE = 300.0  # 5 minutes
   ```

3. **Use faster model:**
   ```bash
   ollama pull phi  # Smaller, faster model
   ```
   Update config:
   ```python
   OLLAMA_MODEL = "phi"
   ```

#### ChromaDB errors

**Symptoms:**
- "Cannot access collection"
- Database errors

**Solutions:**

1. **Clear and rebuild database:**
   ```bash
   rm -rf chroma_db/
   # Then re-index all documents
   ```

2. **Check permissions:**
   ```bash
   chmod -R 755 chroma_db/
   ```

3. **Change database location:**
   - Edit `config/config.py`:
   ```python
   CHROMA_PERSIST_DIR = "/new/path/chroma_db"
   ```

#### Memory issues

**Symptoms:**
- Out of memory errors
- System slowdown during indexing

**Solutions:**

1. **Process smaller chunks:**
   ```python
   MAX_PAGES_SINGLE_PASS = 50
   CHUNK_SIZE_PAGES = 50
   ```

2. **Close other applications**

3. **Use CPU instead of GPU:**
   ```python
   # In core/embedder.py, modify initialization
   device = "cpu"  # Instead of "cuda"
   ```

### Getting Help

1. **Check logs:**
   ```bash
   tail -f logs/workpedia.log
   ```

2. **Enable debug logging:**
   ```python
   # config/config.py
   LOG_LEVEL = "DEBUG"
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Check GitHub issues:**
   - https://github.com/Senchy071/Workpedia/issues

---

## Best Practices

### Document Indexing

✅ **DO:**
- Index related documents together
- Use descriptive filenames
- Keep documents under 500 pages when possible
- Verify indexing completed successfully
- Regularly back up your `chroma_db/` directory

❌ **DON'T:**
- Index duplicate documents (wastes space)
- Mix unrelated topics in same collection
- Delete `chroma_db/` without backup
- Index password-protected PDFs (won't work)

### Querying

✅ **DO:**
- Be specific in your questions
- Use TOC mode for structural queries
- Check source chunks for accuracy
- Adjust context chunks based on query complexity
- Review similarity scores

❌ **DON'T:**
- Ask questions about information not in documents
- Expect the AI to "remember" previous conversations (stateless)
- Use very high temperature (>0.7) for factual queries
- Ignore low similarity scores in sources

### Performance

✅ **DO:**
- Index during off-hours for large batches
- Monitor system resources
- Use appropriate chunk sizes for your documents
- Keep Ollama model warm (run a test query first)
- Close UI when not in use (saves RAM)

❌ **DON'T:**
- Index 100+ documents simultaneously
- Use maximum context chunks (20) for every query
- Run on underpowered hardware (min 8GB RAM recommended)
- Keep debug logging enabled in production

### Data Management

✅ **DO:**
- Organize documents in logical folders
- Keep original files separate from indexed data
- Back up `chroma_db/` regularly
- Document your configuration changes
- Test queries after re-indexing

❌ **DON'T:**
- Delete original files after indexing
- Modify `chroma_db/` files manually
- Change embedding model without re-indexing
- Mix different document versions

### Security & Privacy

✅ **DO:**
- Run Workpedia on local network only
- Use firewall rules if exposing API
- Keep sensitive documents on encrypted drives
- Regularly update dependencies
- Review indexed data periodically

❌ **DON'T:**
- Expose API to internet without authentication
- Index confidential data on shared systems
- Share `chroma_db/` directory publicly
- Use in untrusted environments

---

## Quick Reference

### Common Commands

```bash
# Start Web UI
streamlit run app.py

# Start API server
python -m api.endpoints

# Process single document
python3 demo_parser.py document.pdf

# Run tests
pytest tests/ -v

# Check Ollama
ollama list
ollama pull mistral

# View logs
tail -f logs/workpedia.log
```

### Default Settings

| Setting | Default Value | Range |
|---------|---------------|-------|
| Context Chunks | 5 | 1-10 |
| Temperature | 0.7 | 0.0-1.0 |
| Chunk Size | 512 tokens | 256-1024 |
| Chunk Overlap | 15% | 0-30% |
| Max Single Pass | 100 pages | 50-200 |
| Embedding Dim | 768 | Fixed* |

*Changing embedding dimension requires different model

### File Locations

| Item | Location |
|------|----------|
| Config | `config/config.py` |
| Database | `chroma_db/` |
| Logs | `logs/` |
| Input Files | `data/input/` |
| Test Files | `tests/` |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Ask question (includes confidence) |
| `/search` | POST | Similarity search |
| `/documents` | GET | List documents |
| `/documents/index` | POST | Index from path (generates summary) |
| `/documents/upload` | POST | Upload & index (generates summary) |
| `/documents/{id}` | GET | Get document details |
| `/documents/{id}/summary` | GET | Get document summary |
| `/documents/{id}` | DELETE | Delete document |
| `/history` | GET | Query history |
| `/bookmarks` | GET/POST | Manage bookmarks |
| `/health` | GET | Health check |
| `/stats` | GET | Statistics |
| `/docs` | GET | API documentation |

---

## Glossary

**Chunk:** A segment of text from a document (typically 512 tokens)

**Confidence Score:** Measure of answer reliability based on source quality (0-100%)

**Document Summary:** Auto-generated executive summary with key bullet points

**Embedding:** Vector representation of text that captures semantic meaning

**RAG (Retrieval-Augmented Generation):** AI approach that retrieves relevant context before generating answers

**Similarity Score:** Measure of how closely a chunk matches your query (0-100%)

**Temperature:** Controls randomness in AI responses (0=deterministic, 1=creative)

**Token:** Roughly 3/4 of a word (used to measure text length)

**TOC (Table of Contents):** Synthetic chunk containing document structure

**Vector Store:** Database that stores embeddings for similarity search

**Context Chunks:** Number of relevant document segments retrieved to answer a query

**Agreement Score:** Measure of how much multiple sources agree with each other

**Coverage Score:** Measure of how many relevant sources were found for a query

---

## Next Steps

Now that you're familiar with Workpedia:

1. ✅ Index your first documents
2. ✅ Try different query types
3. ✅ Experiment with settings
4. ✅ Explore the API (if needed)
5. ✅ Customize configuration for your use case

**Happy querying!**

---

*For technical details and development information, see [README.md](README.md)*

*For AI assistant instructions, see [CLAUDE.md](CLAUDE.md)*
