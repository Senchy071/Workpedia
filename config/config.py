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
VLM_PAGE_BATCH_SIZE = 8  # Pages processed in parallel (default: 4, RTX 3090: 8-16 max)

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR = PROJECT_ROOT / "logs"  # Directory for log files
LOG_STRUCTURED = False  # Use JSON structured logging (for production)
LOG_CONSOLE_COLORS = True  # Use colored console output (for development)
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB per log file
LOG_BACKUP_COUNT = 5  # Number of backup log files to keep

# Per-module log levels (optional, override LOG_LEVEL for specific modules)
LOG_MODULE_LEVELS = {
    # "core.parser": "DEBUG",  # Enable debug logging for parser
    # "core.llm": "WARNING",   # Reduce LLM logging
}

# Performance logging
LOG_PERFORMANCE = True  # Log timing information for operations
LOG_SLOW_OPERATION_THRESHOLD = 1.0  # Log warning if operation takes >1s

# Resilience settings (retry, circuit breaker, timeouts)
# Retry configuration
RETRY_ENABLED = True  # Enable retry logic with exponential backoff
RETRY_MAX_ATTEMPTS = 3  # Maximum number of retry attempts
RETRY_INITIAL_DELAY = 1.0  # Initial delay in seconds
RETRY_MAX_DELAY = 30.0  # Maximum delay in seconds
RETRY_EXPONENTIAL_BASE = 2.0  # Base for exponential backoff
RETRY_JITTER = True  # Add randomness to prevent thundering herd

# Circuit breaker configuration
CIRCUIT_BREAKER_ENABLED = True  # Enable circuit breaker pattern
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Open circuit after N consecutive failures
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0  # Seconds to wait before testing recovery
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2  # Successful calls to close from half-open
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3  # Max concurrent calls in half-open state

# Timeout configuration (seconds)
TIMEOUT_DEFAULT = 120.0  # Default timeout for all operations
TIMEOUT_HEALTH_CHECK = 5.0  # Timeout for health checks
TIMEOUT_LIST_MODELS = 10.0  # Timeout for listing models
TIMEOUT_GENERATE = 120.0  # Timeout for non-streaming generation
TIMEOUT_GENERATE_STREAM = 180.0  # Timeout for streaming generation

# History and bookmarks settings
HISTORY_DB_DIR = DATA_DIR / "history_db"
HISTORY_DB_PATH = str(HISTORY_DB_DIR / "history.sqlite3")
HISTORY_AUTO_SAVE = True  # Auto-save all queries
HISTORY_SESSION_ENABLED = True  # Track session IDs

# Confidence scoring settings
CONFIDENCE_ENABLED = True  # Enable confidence scoring for queries
CONFIDENCE_HIGH_THRESHOLD = 0.75  # Score >= this is HIGH confidence
CONFIDENCE_MEDIUM_THRESHOLD = 0.50  # Score >= this is MEDIUM confidence (below is LOW)
CONFIDENCE_SIMILARITY_WEIGHT = 0.5  # Weight for similarity score component
CONFIDENCE_AGREEMENT_WEIGHT = 0.3  # Weight for source agreement component
CONFIDENCE_COVERAGE_WEIGHT = 0.2  # Weight for coverage score component
CONFIDENCE_MIN_SOURCES = 3  # Minimum sources for full coverage score

# Document summary settings
SUMMARY_ENABLED = True  # Auto-generate summaries when indexing documents
SUMMARY_MAX_BULLETS = 5  # Number of bullet points in summary (3-7)
SUMMARY_MAX_INPUT_CHARS = 15000  # Max chars to send to LLM for summarization
SUMMARY_TEMPERATURE = 0.3  # LLM temperature (lower = more focused/consistent)

# Query suggestion settings
SUGGESTIONS_ENABLED = True  # Auto-generate query suggestions when indexing documents
SUGGESTIONS_MAX_PER_DOCUMENT = 15  # Maximum suggestions per document
SUGGESTIONS_MIN_HEADING_LENGTH = 5  # Minimum heading length to process
SUGGESTIONS_QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Tell me about {topic}",
    "Explain {topic}",
    "What are the key points about {topic}?",
]
