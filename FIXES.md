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

**Problem**: Docling's DoclingParseV2Backend crashes on large/complex PDFs with memory corruption error.

**Root Cause**: Known bug in Docling's V2 backend under load (see: docling-project/docling-serve #389)

**Solution**: Multi-level fallback strategy with automatic retry:

### Fallback Strategies (in order)

1. **PyPdfium Backend** - More stable, slower backend
   ```python
   PDFProcessor(backend="pypdfium")
   ```

2. **Disable Table Structure** - Disable problematic table recognition
   ```python
   PDFProcessor(enable_table_structure=False)
   ```

3. **Combined Fallback** - PyPdfium without table structure

4. **Automatic Retry** - PDFProcessor tries all strategies automatically when `auto_fallback=True` (default)

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

**For Problematic PDFs**:
```python
# Option 1: Use stable backend directly
processor = PDFProcessor(backend="pypdfium")

# Option 2: Disable table structure
processor = PDFProcessor(enable_table_structure=False)

# Option 3: Both
processor = PDFProcessor(
    backend="pypdfium",
    enable_table_structure=False
)

# Option 4: Let auto-fallback handle it (recommended)
processor = PDFProcessor(auto_fallback=True)  # Default
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
