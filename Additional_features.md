# Additional Features for Workpedia

This document outlines proposed enhancements to the Workpedia RAG application, organized by priority and category.

---

## ✅ Implementation Status

| # |         Feature           |     Status  |             Module                              |
|---|---------------------------|-------------|-------------------------------------------------|
| 1 | Query History & Bookmarks | ✅ Complete  | `storage/history_store.py`                      |
| 2 | Answer Confidence Scoring | ✅ Complete  | `core/confidence.py`                            |
| 3 | Document Summaries        | ✅ Complete  | `core/summarizer.py`                            |
| 4 | Export Functionality      | ✅ Complete  | `storage/history_store.py`, `api/endpoints.py`  |
| 5 | Query Suggestions         | ✅ Complete  | `core/suggestions.py`                           |
| 6 | Hybrid Search             | ✅ Complete  | `core/hybrid_search.py`                         |
| 7 | Document Collections/Tags | ✅ Complete  | `storage/collections.py`                        |
| 8 | Cross-Encoder Reranking   | ✅ Complete  | `core/reranker.py`                              |
| 16| Vector Store Backup       | ✅ Complete  | `storage/backup.py`                             |
| 18| Performance Caching       | ✅ Complete  | `core/caching.py`                               |
| 14| Custom Embedding Models   | ✅ Complete  | `config/config.py`, `core/embedder.py`          |
| 19| Agent Mode (Tool Calling) | ✅ Complete  | `core/agent.py`, `core/agent_tools.py`          |

**Progress**: 6/6 HIGH PRIORITY + Collections + Reranking + Backup + Caching + Custom Embeddings + Agent Mode implemented

---

## 🎯 HIGH PRIORITY (Quick Wins, High Value)

### 1. Query History & Bookmarks ✅ IMPLEMENTED
**Why**: Users lose their queries when the session ends

**Features:**
- Persistent chat history across sessions (SQLite)
- Bookmark favorite Q&A pairs with tags and notes
- Search past queries by text
- Export conversation history to Markdown/JSON/PDF

**Implementation:**
- `storage/history_store.py`: Full history and bookmark management
- API endpoints: `/history/*` and `/bookmarks/*`
- Session tracking and date range filtering

**Effort**: Low | **Impact**: High

---

### 2. Answer Confidence Scoring ✅ IMPLEMENTED
**Why**: Users need to know how reliable the answer is

**Features:**
- Calculate confidence based on:
  - Source similarity scores (average of top chunks)
  - Source agreement (multiple sources saying same thing)
  - Coverage score (how many relevant sources found)
- Display confidence meter (🟢 High / 🟡 Medium / 🔴 Low)
- Configurable thresholds and weights in `config/config.py`
- Full API support with detailed factor breakdown

**Implementation:**
- `core/confidence.py`: ConfidenceScorer module
- Integrated into QueryEngine with `enable_confidence` option
- API response includes confidence object with score, level, and factors

**Effort**: Medium | **Impact**: High

---

### 3. Document Summaries ✅ IMPLEMENTED
**Why**: Users want quick overview before querying

**Features:**
- Auto-generate summary when document is indexed
- Use Ollama to create 3-7 bullet executive summary
- Store as searchable chunk in vector store
- Query: "What's in this document?" → Return summary
- API endpoint: `GET /documents/{doc_id}/summary`

**Implementation:**
- `core/summarizer.py`: DocumentSummarizer with LLM-based summarization
- Integrated into DocumentIndexer (generates summary during indexing)
- Summary detection in QueryEngine for "what's in this document" queries
- Configurable settings in `config/config.py`

**Effort**: Medium | **Impact**: High

---

### 4. Export Functionality ✅ IMPLEMENTED
**Why**: Users want to save and share results

**Features:**
- Export query history to Markdown/JSON/PDF
- Export bookmarks to Markdown/JSON
- Includes confidence scores in exports
- Proper Content-Disposition headers for file downloads
- Date-stamped filenames

**API Endpoints:**
- `GET /history/export/markdown` - Export queries as Markdown
- `GET /history/export/json` - Export queries as JSON
- `GET /history/export/pdf` - Export queries as PDF
- `GET /bookmarks/export/markdown` - Export bookmarks as Markdown
- `GET /bookmarks/export/json` - Export bookmarks as JSON

**Implementation:**
- Enhanced `storage/history_store.py` with confidence scores in exports
- Added bookmark export methods (`export_bookmarks_markdown`, `export_bookmarks_json`)
- Updated API endpoints with download headers

**Effort**: Low | **Impact**: Medium

---

### 5. Query Suggestions / Auto-complete ✅ IMPLEMENTED
**Why**: Help users ask better questions

**Features:**
- Extract common questions from indexed content
  - Section headings → Questions ("What is X?")
  - TOC entries → "Tell me about [topic]"
  - Key concepts → "What is [concept]?"
- Auto-generate during document indexing
- Priority-based ordering (h1 > h2 > h3)
- API endpoint: `GET /documents/{doc_id}/suggestions`

**Implementation:**
- `core/suggestions.py`: QuerySuggestionGenerator module
- Integrated into DocumentIndexer (generates during indexing)
- Suggestions stored as special chunks in vector store
- Configurable in `config/config.py` (max suggestions, min heading length)

**Effort**: Medium | **Impact**: Medium

---

## 🚀 MEDIUM PRIORITY (Enhanced Capabilities)

### 6. Hybrid Search (Semantic + Keyword) ✅ IMPLEMENTED
**Why**: Semantic search misses exact matches (names, codes, IDs)

**Features:**
- Combine ChromaDB vector search with BM25 keyword ranking
- Use Reciprocal Rank Fusion (RRF) to merge results
- Example: "Find invoice #12345" (exact match) vs "payment issues" (semantic)
- Configurable weights for semantic vs keyword search
- Persistent BM25 index with disk storage
- Automatic indexing during document ingestion

**Implementation:**
- `core/hybrid_search.py`: BM25Index, HybridSearcher, RRF algorithm
- Integrated into DocumentIndexer (indexes chunks in BM25 during ingestion)
- Integrated into QueryEngine (hybrid search during retrieval)
- Configurable in `config/config.py` (weights, RRF k constant)

**Configuration:**
- `HYBRID_SEARCH_ENABLED`: Enable/disable hybrid search
- `HYBRID_SEARCH_K`: RRF constant (default 60)
- `HYBRID_SEARCH_SEMANTIC_WEIGHT`: Semantic weight (default 0.7)
- `HYBRID_SEARCH_KEYWORD_WEIGHT`: Keyword weight (default 0.3)

**Effort**: Medium | **Impact**: High

---

### 7. Document Collections & Tags ✅ IMPLEMENTED
**Why**: Organize documents by project, topic, category

**Features:**
- User-defined tags: `project:alpha`, `type:contract`, `year:2024`
- Collections: Group related documents
- Filter queries by collection: "Search only in legal docs"
- Assign collection and tags during document indexing
- Update collection and tags for existing documents
- Query filtering by collection name or tags

**Implementation:**
- `storage/collections.py`: CollectionManager with SQLite persistence
- `storage/vector_store.py`: Collection/tag metadata in chunks
- `core/query_engine.py`: Filtering by collection in queries
- API endpoints: `/collections/*` and `/tags/*`
- 36 comprehensive tests (all passing)

**API Endpoints:**
- `GET /collections` - List all collections
- `POST /collections` - Create a collection
- `GET /collections/{id}` - Get collection details
- `PUT /collections/{id}` - Update collection
- `DELETE /collections/{id}` - Delete collection
- `GET /collections/{id}/documents` - List documents in collection
- `POST /documents/{doc_id}/collection` - Set document collection
- `GET /tags` - List all tags
- `GET /documents/{doc_id}/tags` - Get document tags
- `POST /documents/{doc_id}/tags` - Add tags
- `PUT /documents/{doc_id}/tags` - Set (replace) tags
- `DELETE /documents/{doc_id}/tags` - Remove tags
- `GET /tags/{tag}/documents` - List documents by tag

**Effort**: Medium | **Impact**: Medium

---

### 8. Re-ranking with Cross-Encoder ✅ IMPLEMENTED
**Why**: Initial retrieval may miss the most relevant chunks

**Features:**
- After vector search, re-rank top 20 chunks using cross-encoder
- Models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, local)
- Return top 5 re-ranked chunks to LLM
- Improves answer quality significantly
- Lazy model loading (loads on first use)
- Configurable top-N retrieval and top-K return counts
- Detailed rank change logging for debugging

**Implementation:**
- `core/reranker.py`: CrossEncoderReranker with lazy loading
- Integrated into QueryEngine (automatic reranking during retrieval)
- Configurable in `config/config.py`
- 27 comprehensive tests (all passing)

**Configuration:**
- `RERANKER_ENABLED`: Enable/disable reranking (default: True)
- `RERANKER_MODEL`: Cross-encoder model (default: ms-marco-MiniLM-L-6-v2)
- `RERANKER_TOP_N`: Candidates to retrieve for reranking (default: 20)
- `RERANKER_TOP_K`: Results to return after reranking (default: 5)

**Effort**: Medium | **Impact**: High

---

### 19. Agent Mode (Agentic Query Layer) ✅ IMPLEMENTED
**Why**: Transform from fixed pipeline to autonomous reasoning agent

**Features:**
- Autonomous agent that uses tools to reason and search iteratively
- Tool calling via Ollama with mistral-nemo model (native tools support)
- 6 available tools:
  - `search_documents`: Semantic search across documents
  - `get_document_summary`: Get document executive summary
  - `list_documents`: List all indexed documents
  - `read_chunk`: Read full chunk content by ID
  - `get_document_info`: Get document metadata and structure
  - `complete`: Signal task completion with answer
- Agent loop: Reason → Act → Observe → Repeat until complete
- Real-time streaming of agent activity
- Configurable max iterations and temperature

**Implementation:**
- `core/agent.py`: WorkpediaAgent orchestrator with run loop
- `core/agent_tools.py`: Tool definitions and execution
- `core/llm.py`: Extended with `chat_with_tools()` for tool calling
- `api/endpoints.py`: Agent endpoints (`/agent/query`, `/agent/query/stream`, `/agent/status`)
- `app.py`: Agent Mode toggle in Streamlit UI
- `tests/test_agent.py`: 26 comprehensive tests (all passing)

**Configuration:**
- `AGENT_ENABLED`: Enable agent functionality (default: True)
- `AGENT_MODEL`: Model with tool support (default: mistral-nemo)
- `AGENT_MAX_ITERATIONS`: Maximum tool call iterations (default: 10)
- `AGENT_TEMPERATURE`: Temperature for reasoning (default: 0.1)
- `AGENT_TIMEOUT`: Timeout in seconds (default: 300.0)

**API Endpoints:**
- `POST /agent/query` - Synchronous agentic query
- `POST /agent/query/stream` - Streaming with real-time tool call updates
- `GET /agent/status` - Check agent availability and configuration

**Effort**: Medium | **Impact**: High

---

### 9. Table & Figure-Specific Queries
**Why**: Tables and figures need special handling

**Features:**
- Detect when query is about tabular data
- Use specialized prompts: "Analyze this table and answer..."
- Support structured queries: "Show rows where X > 5"
- Return table in formatted markdown or CSV

**Effort**: High | **Impact**: Medium

---

### 10. Document Comparison
**Why**: Users often need to compare versions or related docs

**Features:**
- Compare two documents side-by-side
- Highlight differences in structure and content
- Answer: "What changed between doc A and doc B?"
- Use cases: Contract revisions, report versions

**Effort**: High | **Impact**: Medium

---

## 💡 ADVANCED FEATURES (Power User)

### 11. Query Analytics Dashboard
**Why**: Understand usage patterns and improve system

**Features:**
- Track queries, response times, document hit rates
- Most queried documents and topics
- Failed/low-confidence queries for improvement
- Visualize with Plotly in Streamlit
- Export metrics for analysis

**Effort**: Medium | **Impact**: Low (for power users)

---

### 12. Feedback Loop & Active Learning
**Why**: Improve results based on user feedback

**Features:**
- 👍/👎 on each answer
- "Was this helpful?" after each response
- Store feedback with query/response pair
- Use to:
  - Identify problematic documents/chunks
  - Tune retrieval parameters
  - Fine-tune custom embeddings (advanced)

**Effort**: Medium | **Impact**: Medium

---

### 13. Multi-hop Reasoning
**Why**: Complex questions need multiple retrieval passes

**Features:**
- Detect multi-part questions: "What is X and how does it relate to Y?"
- First retrieval: Find info about X
- Second retrieval: Find info about Y using context from X
- Synthesize final answer
- Chain-of-thought prompting with Ollama

**Effort**: High | **Impact**: Medium

---

### 14. Custom Embedding Models ✅ IMPLEMENTED
**Why**: Domain-specific models may work better

**Features:**
- Support swappable embedding models via EMBEDDING_MODELS registry
- Pre-configured models with automatic dimension detection:
  - `sentence-transformers/all-mpnet-base-v2` (768-dim, default)
  - `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
  - `sentence-transformers/all-MiniLM-L12-v2` (384-dim)
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim, multilingual)
  - `BAAI/bge-small-en-v1.5` (384-dim)
  - `BAAI/bge-base-en-v1.5` (768-dim)
  - `BAAI/bge-large-en-v1.5` (1024-dim)
- Unknown models supported with runtime dimension detection
- Helper functions: `list_available_models()`, `get_model_info()`

**Implementation:**
- `config/config.py`: EMBEDDING_MODELS registry with dimensions
- `core/embedder.py`: Updated for swappable models
- `tests/test_embedding_models.py`: 27 comprehensive tests

**Configuration:**
```python
# Change in config/config.py to switch models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Faster, 384-dim
# WARNING: Changing models requires re-indexing all documents!
```

**Effort**: Low | **Impact**: Medium

---

### 15. Batch Query Processing
**Why**: Programmatic analysis of documents

**Features:**
- Upload CSV of questions, get CSV of answers
- Example: 100 questions about compliance doc
- API endpoint: `POST /batch-query`
- Background job with progress tracking
- Email notification when complete (optional)

**Effort**: Medium | **Impact**: Low

---

## 🔧 OPERATIONAL IMPROVEMENTS

### 16. Vector Store Backup & Restore ✅ IMPLEMENTED
**Why**: ChromaDB data is critical

**Features:**
- Full vector store backups with versioning
- Compressed (tar.gz) and uncompressed backup formats
- One-click restore from backup
- Backup retention policies (configurable max backups)
- Backup validation and integrity checking
- Export metadata to JSON
- Disaster recovery support

**Implementation:**
- `storage/backup.py`: BackupManager with full backup/restore lifecycle
- Integrated into API with `/backup/*` endpoints
- Configurable in `config/config.py`
- 24 comprehensive tests (all passing)

**Configuration:**
- `BACKUP_ENABLED`: Enable backup functionality (default: True)
- `BACKUP_DIR`: Backup storage directory (default: `data/backups/`)
- `BACKUP_MAX_BACKUPS`: Maximum backups to retain (default: 10, 0=unlimited)
- `BACKUP_COMPRESS`: Compress backups using tar.gz (default: True)
- `BACKUP_AUTO_BACKUP_ON_INDEX`: Auto-backup before indexing (default: False)

**API Endpoints:**
- `POST /backup/create` - Create a full backup
- `GET /backup/list` - List all backups with metadata
- `POST /backup/restore` - Restore from backup
- `DELETE /backup/delete/{backup_name}` - Delete a backup
- `GET /backup/stats` - Get backup statistics

**Effort**: Low | **Impact**: High (risk mitigation)

---

### 17. Document Health Monitoring
**Why**: Detect indexing issues proactively

**Features:**
- Check for:
  - Documents with 0 chunks (parsing failed)
  - Orphaned chunks (no parent document)
  - Embedding anomalies (all zeros, NaN)
- Auto-repair or flag for reprocessing
- Health dashboard in stats tab

**Effort**: Medium | **Impact**: Medium

---

### 18. Performance Caching ✅ IMPLEMENTED
**Why**: Repeated queries are slow

**Features:**
- Cache query embeddings (same query = same embedding)
- Cache LLM responses for identical context+question
- TTL-based cache eviction (1 hour default)
- Save ~2-3 seconds per cached query
- Use `diskcache` library (local, no Redis needed)

**Implementation:**
- `core/caching.py`: EmbeddingCache and LLMCache with TTL support
- Integrated into Embedder (query embeddings cached automatically)
- Integrated into QueryEngine (LLM responses cached with context)
- Configurable in `config/config.py` (enabled by default, 1 hour TTL)
- Cache statistics and clearing methods available
- Cache directory automatically ignored in `.gitignore`

**Configuration:**
- `CACHE_ENABLED`: Enable/disable caching (default: True)
- `CACHE_DIR`: Cache storage directory (default: `cache/`)
- `CACHE_EMBEDDING_TTL`: Embedding cache TTL in seconds (default: 3600)
- `CACHE_LLM_TTL`: LLM response cache TTL in seconds (default: 3600)

**Effort**: Low | **Impact**: Medium

---

## 📊 RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 1: Quick Wins (1-2 weeks)
1. Query history & bookmarks
2. Answer confidence scoring
3. Export functionality
4. Performance caching
5. Vector store backup

### Phase 2: Enhanced Search (2-3 weeks)
6. Hybrid search (semantic + keyword)
7. Document summaries
8. Query suggestions
9. Re-ranking with cross-encoder

### Phase 3: Power Features (3-4 weeks)
10. Document collections & tags
11. Table/figure-specific queries
12. Feedback loop & analytics dashboard
13. Document comparison

### Phase 4: Advanced (Future)
14. Multi-hop reasoning
15. Custom embedding models
16. Batch query processing
17. Document health monitoring

---

## 🎯 TOP 5 RECOMMENDATIONS

If you can only implement a few, start with these:

1. **Query History & Bookmarks** - Essential UX improvement
2. **Answer Confidence Scoring** - Helps users trust results
3. **Hybrid Search** - Dramatically improves retrieval accuracy
4. **Document Summaries** - Great for document discovery
5. **Performance Caching** - 2x speed improvement for free

---

## 📈 Impact vs. Effort Matrix

### High Impact, Low Effort (Do First)
- Query history & bookmarks
- Performance caching
- Vector store backup
- Custom embedding models

### High Impact, Medium Effort (Do Second)
- Answer confidence scoring
- Document summaries
- Hybrid search
- Re-ranking with cross-encoder

### Medium Impact, Low/Medium Effort (Nice to Have)
- Export functionality
- Query suggestions
- Document collections & tags
- Document health monitoring

### High Effort (Consider Carefully)
- Table & figure-specific queries
- Document comparison
- Multi-hop reasoning

---

## 🔍 Gap Analysis Summary

Based on comprehensive review of the Workpedia codebase, the system is **production-ready and feature-complete** for core RAG functionality. Enhancement opportunities exist primarily in:

1. **User Experience**: History, bookmarks, exports, suggestions
2. **Search Quality**: Hybrid search, re-ranking, multi-hop reasoning
3. **Content Discovery**: Summaries, tags, collections, analytics
4. **Reliability**: Caching, backups, health monitoring, feedback loops
5. **Advanced Queries**: Table-specific, comparison, batch processing

No critical TODOs or incomplete core features were found. All proposed features are additive enhancements rather than gap-filling requirements.

---

*Document updated: 2026-01-11*
*Workpedia Version: 1.4 (Core + Additional Features + XLSX/CSV + Custom Embeddings + Agent Mode)*
