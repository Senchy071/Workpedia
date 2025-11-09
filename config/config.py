from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Ollama settings
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

# ChromaDB settings
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_NAME = "workpedia_docs"

# Chunking settings
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 0.15  # 15% overlap

# Large document handling settings
MAX_PAGES_SINGLE_PASS = 100  # Process in chunks if more pages
MAX_FILE_SIZE_MB = 50  # Process in chunks if larger
CHUNK_SIZE_PAGES = 75  # Pages per chunk for large documents

# Large document detection for backend selection
LARGE_DOC_PAGE_THRESHOLD = 200  # Use stable backend if >200 pages
LARGE_DOC_SIZE_MB_THRESHOLD = 20  # Use stable backend if >20MB

# VLM (Vision-Language Model) settings for document structure extraction
USE_VLM_FOR_LARGE_DOCS = True  # Use Granite-Docling VLM for large documents
VLM_MODEL = "granite_docling"  # Options: "granite_docling", "smol_docling"
VLM_BATCH_SIZE = 1  # Process pages in batches (1 for large docs)
