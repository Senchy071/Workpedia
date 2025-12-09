# Critical Fixes and VLM Integration

## Synthetic Table of Contents for Better Chapter Retrieval (December 2025)

**Problem**: When users ask "List main chapters" or "What are the sections?", the system was returning glossary/acronym sections instead of the actual chapter structure.

**Root Cause**: Pure semantic similarity search matched queries containing "list" to content that literally contains lists (glossaries, acronyms), rather than understanding the user wants document structure information.

**Solution**: Automatically create a synthetic "Table of Contents" chunk during document indexing.

### How It Works

1. **During Indexing**: When a document is indexed, the system:
   - Runs `StructureAnalyzer` on the parsed document
   - Extracts all sections/chapters with their levels and page numbers
   - Creates a synthetic TOC chunk with formatted chapter listing
   - Marks it with `chunk_type: "table_of_contents"` metadata
   - Adds it to the vector store alongside regular content chunks

2. **During Retrieval**: When users ask structural questions:
   - Query: "List main chapters of indexed book"
   - The TOC chunk has high semantic similarity to such queries
   - System retrieves TOC chunk instead of glossary
   - LLM generates answer based on actual chapter structure

### Example TOC Chunk

```markdown
# Table of Contents

This document contains the following sections and chapters:

- Introduction (page 1)
- Chapter 1: Getting Started (page 5)
  - Section 1.1: Installation (page 6)
  - Section 1.2: Configuration (page 10)
- Chapter 2: Advanced Topics (page 15)
  - Section 2.1: Performance Tuning (page 16)
- Glossary (page 200)
```

### Benefits

- ✓ Users can now query document structure effectively
- ✓ Works automatically for all newly indexed documents
- ✓ No manual intervention required
- ✓ Preserves hierarchical structure (indented subsections)
- ✓ Includes page numbers for navigation

### Implementation

Location: `storage/vector_store.py:403-489` (DocumentIndexer._create_toc_chunk)

The TOC chunk is created with special metadata:
```python
metadata = {
    "chunk_type": "table_of_contents",
    "chunk_index": -1,
    "sections_count": len(sections),
}
```

# Critical Fixes and VLM Integration

## Granite-Docling VLM for Structure Extraction ✗ TOO SLOW (Abandoning for Option 1)

**Released**: September 17, 2025 by IBM
**Purpose**: Document structure preservation for RAG systems

### The Structure Problem

**PyPdfium Backend**:
- ✓ Stable (no crashes)
- ✗ **NO structure extraction** (0 sections, 0 tables, 0 figures)
- ✗ Useless for RAG that needs document understanding

**V2 Backend**:
- ✓ Excellent structure extraction
- ✗ Crashes on large docs ("corrupted double-linked list")

**Solution**: Granite-Docling VLM
- ✓ Full structure extraction (tables, sections, figures, equations)
- ✓ Stable (no crashes)
- ✓ 258M parameters (efficient)
- ✓ GPU accelerated
- ✓ Purpose-built for RAG systems

### Implementation

The system now automatically uses **Granite-Docling VLM** for large documents (>200 pages OR >20MB):

```python
# Automatic backend selection (default)
processor = PDFProcessor()  # backend="auto"

# For 873-page PDF:
# → Auto-detects large document
# → Selects VLM backend
# → Uses Granite-Docling model
# → Extracts full structure
```

### Configuration

```python
# config/config.py
USE_VLM_FOR_LARGE_DOCS = True  # Use VLM for large docs
VLM_MODEL = "granite_docling"   # Model to use
VLM_BATCH_SIZE = 1              # Batch size

LARGE_DOC_PAGE_THRESHOLD = 200  # >200 pages = large
LARGE_DOC_SIZE_MB_THRESHOLD = 20  # >20MB = large
```

### Expected Log Output

```
INFO: Processing PDF: CRCStandardMathTablesFormulasZwillinger.pdf
INFO: Large document detected: 873 pages > 200 pages
INFO: Auto-selected backend: vlm (873 pages > 200 pages)
INFO: Creating VLM pipeline with granite_docling model
INFO: DocumentConverter created with VlmPipeline (granite_docling)
[VLM model downloads/loads - may take a few minutes first time]
INFO: Processing document...
```

### Test Results and Performance Issues

**VLM Testing (242-page NATO handbook):**
- Initial test with `batch_size=64`: CUDA OOM (22.55GB/24GB used)
- Fixed with `batch_size=8`: Avoided OOM (8.5GB/24GB used)
- **Performance**: 40+ minutes for 242 pages (~10 sec/page) - **UNACCEPTABLE**

**Root Cause - CPU Bottleneck:**
- GPU utilization drops to 0-10% intermittently
- Pattern: GPU processes batch → 100% usage → CPU prepares next batch → 0% GPU
- **Issues identified:**
  - Image tiling: Granite breaks pages into tiles → massive token counts
  - No prefetching: Single-threaded - can't load next batch while GPU works
  - PDF I/O: Reading/rendering pages to images is CPU-bound
  - Transformers backend: Inefficient compared to vLLM server

**VLM Conclusion:**
- Works correctly (extracts structure)
- Too slow for production use (~10 sec/page)
- Setting up vLLM server would help but adds complexity
- **Decision**: Switch to Option 1 (Split-Process-Merge with V2)

### Manual Backend Selection

```python
# Force VLM for any document
processor = PDFProcessor(backend="vlm")

# Force PyPdfium (stable, no structure)
processor = PDFProcessor(backend="pypdfium")

# Force V2 (fast, may crash on large docs)
processor = PDFProcessor(backend="v2")
```

---

## Option 1: Split-Process-Merge with V2 Backend (IMPLEMENTED ✓)

**Approach:**
1. **Split**: Use pypdf to split large PDF into smaller files (75 pages each)
2. **Process**: Process each split with V2 backend in sequence
3. **Merge**: Combine parsed results back together with page offsets
4. **Cleanup**: Delete temporary split files automatically

**Advantages:**
- V2 backend: Fast (~1-2 sec/page) with excellent structure extraction
- Avoids V2 crashes by keeping chunks small
- No GPU memory issues (V2 uses ~5-8GB VRAM)
- Simple, reliable, maintainable
- **Expected time**: 5-10 minutes for 242-page document (vs 40+ min with VLM)

**Implementation (Phase 2 Complete):**
1. ✓ `core/pdf_splitter.py`: PDFSplitter class to split PDFs by page ranges
2. ✓ `core/doc_merger.py`: DocumentMerger class to combine parsed chunk results
3. ✓ `core/large_doc_handler.py`: Updated to use split-process-merge strategy
4. ✓ Full test coverage with 46 passing tests

**Components:**
- `PDFSplitter`: Splits PDFs into configurable chunk sizes (default 75 pages)
- `DocumentMerger`: Merges text, structure, and metadata with page offset handling
- `ChunkInfo`: Dataclass for tracking chunk processing results
- `MergeResult`: Dataclass for unified merge results

**Status**: COMPLETE - Ready for production use

---

# Critical Fixes for Docling Crash Issues

## Issue 1: Demo Script Ignoring CLI Arguments ✓ FIXED

**Problem**: `demo_parser.py` was processing entire directory instead of single file specified as command-line argument.

**Solution**: Updated `main()` to check `sys.argv` and prioritize single file processing:

```python
if len(sys.argv) > 1:
    input_path = Path(sys.argv[1])
    # Process single file
else:
    # Process directory
```

**Usage**:
```bash
# Process single file
python3 demo_parser.py /path/to/document.pdf

# Process all files in data/input/
python3 demo_parser.py
```

## Issue 2: Docling "Corrupted Double-Linked List" Crash ✓ FIXED

**Problem**: Docling's DoclingParseV2Backend crashes on large/complex PDFs (e.g., 873-page reference books) with memory corruption error.

**Root Cause**: Known bug in Docling's V2 backend under load (see: docling-project/docling-serve #389)

**Solution**: Intelligent backend auto-selection + multi-level fallback

### Auto-Selection Logic (NEW)

The processor now **automatically detects large documents** and selects the appropriate backend:

- **Large Documents** (>200 pages OR >20MB): Use **PyPdfium** backend (stable)
- **Small Documents** (<200 pages AND <20MB): Use **V2** backend (fast)

**Thresholds** (config/config.py):
```python
LARGE_DOC_PAGE_THRESHOLD = 200  # pages
LARGE_DOC_SIZE_MB_THRESHOLD = 20  # MB
```

**Why This Works**:
- 873-page math reference → Automatically uses PyPdfium → No crash
- 10-page report → Uses V2 → Fast processing
- No manual configuration needed!

### Backend Selection Behavior

```python
# Default behavior - automatic selection (RECOMMENDED)
processor = PDFProcessor()  # backend="auto" by default

# Processing flow:
# 1. Check document size (pages and MB)
# 2. If large (>200 pages OR >20MB): Use PyPdfium
# 3. If small: Use V2 backend
# 4. If V2 crashes: Automatic fallback to PyPdfium
```

**Example Logs**:
```
# For 873-page PDF:
INFO: Large document detected: 873 pages > 200 pages → using pypdfium backend
INFO: Auto-selected backend: pypdfium (873 pages > 200 pages)

# For 50-page PDF:
INFO: Small document: 50 pages, 5.2MB → using v2 backend
INFO: Auto-selected backend: v2 (50 pages, 5.2MB)
```

### Fallback Strategies (if V2 fails on small docs)

Even for small documents, if V2 crashes, the processor automatically tries:

1. **PyPdfium Backend** - More stable, slower
2. **Disable Table Structure** - V2 without table recognition
3. **PyPdfium without tables** - Most stable combination

### Implementation Details

**Parser Backend Selection**:
```python
# core/parser.py
DocumentParser(backend="v2")       # Fast, may crash on large docs
DocumentParser(backend="pypdfium")  # Stable, slower
```

**PDF Processor with Auto-Fallback**:
```python
# processors/pdf_processor.py
processor = PDFProcessor(
    enable_table_structure=True,
    backend="v2",
    auto_fallback=True  # Automatically tries fallback strategies
)
```

**Fallback Execution Flow**:
1. Try V2 backend with table structure
2. On crash → Try PyPdfium with table structure
3. On crash → Try V2 without table structure
4. On crash → Try PyPdfium without table structure
5. If all fail → Raise error with details

**Metadata Tracking**:
Results include `fallback_used` field when fallback succeeds:
- `"pypdfium"` - PyPdfium backend used
- `"v2_no_tables"` - V2 without table structure
- `"pypdfium_no_tables"` - PyPdfium without table structure

### Configuration Options

**Default (Recommended)**:
```python
# Automatic backend selection based on document size
processor = PDFProcessor()  # backend="auto", auto_fallback=True
```

**Manual Backend Selection**:
```python
# Force V2 backend (fast, may crash on large docs)
processor = PDFProcessor(backend="v2")

# Force PyPdfium backend (stable, slower)
processor = PDFProcessor(backend="pypdfium")

# Disable table structure (reduces memory usage)
processor = PDFProcessor(enable_table_structure=False)

# Disable auto-fallback (not recommended)
processor = PDFProcessor(auto_fallback=False)
```

### Testing

All 31 tests pass with new fallback logic:
```bash
pytest tests/ -v
```

### Known Limitations

1. **PyPdfium is slower** - May take 2-3x longer than V2 backend
2. **No multi-page table support** - PyPdfium doesn't preserve complex table structures
3. **Table structure quality** - Disabling table structure loses table layout information

### Recommendations

1. **For production**: Use `auto_fallback=True` (default)
2. **For speed**: Use `backend="v2"` with fallback enabled
3. **For stability**: Use `backend="pypdfium"` directly
4. **For large PDFs (500+ pages)**: Consider disabling table structure to reduce memory usage

## Testing the Fixes

```bash
# Test with your problematic PDF
python3 demo_parser.py /path/to/large-document.pdf

# The processor will automatically:
# 1. Try V2 backend first
# 2. Fall back to PyPdfium if crash detected
# 3. Try without table structure if needed
# 4. Report which strategy succeeded
```

## Logging

Enable verbose logging to see fallback attempts:
```bash
export PYTHONUNBUFFERED=1
python3 demo_parser.py document.pdf 2>&1 | tee processing.log
```

Look for log messages like:
- `"Processing failed with v2 backend"`
- `"Attempting fallback strategies..."`
- `"Fallback 1: Trying PyPdfium backend"`
- `"✓ PyPdfium backend succeeded"`
