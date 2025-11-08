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
