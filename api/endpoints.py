"""FastAPI endpoints for Workpedia RAG system."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from config.config import INPUT_DIR
from core.exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    IndexingError,
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaTimeoutError,
    UnsupportedFormatError,
    ValidationError,
    VectorStoreQueryError,
    WorkpediaError,
    QueryNotFoundError,
    BookmarkNotFoundError,
    format_exception_chain,
)
from core.llm import OllamaClient
from core.logging_config import set_request_id
from core.parser import DocumentParser
from core.query_engine import QueryEngine
from core.resilience import CircuitBreakerError
from core.validators import (
    sanitize_filename,
    validate_document_id,
    validate_file_path,
    validate_query,
)
from storage.vector_store import DocumentIndexer
from storage.history_store import HistoryStore

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
query_engine: Optional[QueryEngine] = None
document_indexer: Optional[DocumentIndexer] = None
history_store: Optional[HistoryStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global query_engine, document_indexer, history_store

    logger.info("Initializing Workpedia API...")

    try:
        # Step 1: Validate Ollama is available
        logger.info("Checking Ollama connectivity...")
        ollama_client = OllamaClient()
        health = ollama_client.health_check()

        if not health["server_reachable"]:
            error_msg = (
                f"STARTUP FAILED: {health['message']}\n"
                f"Please ensure Ollama is running:\n"
                f"  1. Start Ollama: 'ollama serve'\n"
                f"  2. Verify it's running: 'ollama list'\n"
                f"  3. Check the URL: {health['base_url']}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not health["model_available"]:
            available = health["available_models"]
            models_str = ", ".join(available) if available else "none"
            error_msg = (
                f"STARTUP FAILED: {health['message']}\n"
                f"To fix this:\n"
                f"  1. Pull the model: 'ollama pull {health['model_name']}'\n"
                f"  2. Or use a different model in config/config.py\n"
                f"  Available models: {models_str}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"✓ Ollama connection validated: {health['message']}")

        # Step 2: Initialize history store
        history_store = HistoryStore()
        logger.info("✓ History store initialized")

        # Step 3: Initialize components
        query_engine = QueryEngine(
            history_store=history_store,
            auto_save_history=True,
        )
        document_indexer = DocumentIndexer(
            vector_store=query_engine.vector_store,
            embedder=query_engine.embedder,
        )

        logger.info("✓ Workpedia API initialized successfully")
        logger.info(f"  - Vector Store: {query_engine.vector_store.count} chunks indexed")
        logger.info(f"  - LLM: {health['model_name']}")
        logger.info(f"  - Embedder: {query_engine.embedder.model_name}")
        logger.info(f"  - History: Auto-save enabled")

    except Exception as e:
        logger.error(f"Failed to initialize Workpedia API: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Workpedia API")


# Create FastAPI app
app = FastAPI(
    title="Workpedia RAG API",
    description="Privacy-focused RAG system for document question-answering",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Middleware - Request ID tracking and performance logging
# =============================================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID and log request/response."""

    async def dispatch(self, request: Request, call_next):
        """Process request with ID tracking and logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        set_request_id(request_id)

        # Add to request state for access in endpoints
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )

        # Process request
        try:
            response = await call_next(request)

            # Log response
            elapsed = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_time": elapsed,
                },
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {e}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "elapsed_time": elapsed,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise


# Add middleware
app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# Exception Handlers - Map custom exceptions to HTTP status codes
# =============================================================================


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    """Handle document not found errors - 404."""
    logger.warning(f"Document not found: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "DocumentNotFoundError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(UnsupportedFormatError)
async def unsupported_format_handler(request: Request, exc: UnsupportedFormatError):
    """Handle unsupported format errors - 415."""
    logger.warning(f"Unsupported format: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=415,
        content={
            "error": "UnsupportedFormatError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(DocumentParsingError)
async def document_parsing_handler(request: Request, exc: DocumentParsingError):
    """Handle document parsing errors - 422."""
    logger.error(f"Document parsing failed: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "DocumentParsingError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(OllamaConnectionError)
async def ollama_connection_handler(request: Request, exc: OllamaConnectionError):
    """Handle Ollama connection errors - 503."""
    logger.error(f"Ollama connection failed: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "OllamaConnectionError",
            "message": exc.message,
            "context": exc.context,
            "suggestion": "Check if Ollama server is running: ollama serve",
        },
    )


@app.exception_handler(OllamaTimeoutError)
async def ollama_timeout_handler(request: Request, exc: OllamaTimeoutError):
    """Handle Ollama timeout errors - 504."""
    logger.error(f"Ollama timeout: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=504,
        content={
            "error": "OllamaTimeoutError",
            "message": exc.message,
            "context": exc.context,
            "suggestion": "The request took too long. Try a simpler query or increase timeout.",
        },
    )


@app.exception_handler(OllamaGenerationError)
async def ollama_generation_handler(request: Request, exc: OllamaGenerationError):
    """Handle Ollama generation errors - 500."""
    logger.error(f"Ollama generation failed: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "OllamaGenerationError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(VectorStoreQueryError)
async def vector_store_query_handler(request: Request, exc: VectorStoreQueryError):
    """Handle vector store query errors - 500."""
    logger.error(f"Vector store query failed: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "VectorStoreQueryError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(IndexingError)
async def indexing_error_handler(request: Request, exc: IndexingError):
    """Handle indexing errors - 500."""
    logger.error(f"Indexing failed: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "IndexingError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors - 400."""
    logger.warning(f"Validation error: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "ValidationError",
            "message": exc.message,
            "context": exc.context,
        },
    )


@app.exception_handler(CircuitBreakerError)
async def circuit_breaker_handler(request: Request, exc: CircuitBreakerError):
    """Handle circuit breaker open errors - 503 with graceful degradation."""
    logger.error(f"Circuit breaker is open: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "CircuitBreakerError",
            "message": str(exc),
            "suggestion": (
                "The LLM service is temporarily unavailable due to repeated failures. "
                "The system is protecting itself from cascading failures. "
                "Please try again in a few moments."
            ),
            "retry_after": 60,  # seconds
        },
    )


@app.exception_handler(WorkpediaError)
async def workpedia_error_handler(request: Request, exc: WorkpediaError):
    """Handle generic Workpedia errors - 500."""
    logger.error(f"Workpedia error: {format_exception_chain(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "context": exc.context,
        },
    )


# =============================================================================
# Pydantic Models
# =============================================================================


class QueryRequest(BaseModel):
    """Request model for queries."""

    question: str = Field(
        ..., description="Question to ask about documents", min_length=1, max_length=5000
    )
    n_results: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    doc_id: Optional[str] = Field(None, description="Filter to specific document")
    temperature: float = Field(0.7, ge=0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum response tokens")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        """Validate and sanitize query string."""
        try:
            return validate_query(v, min_length=1, max_length=5000)
        except ValueError as e:
            raise ValueError(f"Invalid question: {e}")

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, v):
        """Validate document ID format."""
        if v is None:
            return v
        try:
            return validate_document_id(v)
        except ValueError as e:
            raise ValueError(f"Invalid doc_id: {e}")


class QueryResponse(BaseModel):
    """Response model for queries."""

    question: str
    answer: str
    sources: List[dict]
    metadata: dict


class DocumentInfo(BaseModel):
    """Document information model."""

    doc_id: str
    filename: str
    file_path: str
    chunk_count: int


class IndexRequest(BaseModel):
    """Request model for indexing."""

    file_path: str = Field(..., description="Path to document to index")
    replace_existing: bool = Field(True, description="Replace if already indexed")

    @field_validator("file_path")
    @classmethod
    def validate_file_path_field(cls, v):
        """Validate file path for security."""
        try:
            # Validate path with security checks
            validated_path = validate_file_path(
                v,
                must_exist=True,
                allowed_extensions={".pdf", ".docx", ".html", ".htm", ".txt", ".md"},
            )
            # Convert back to string for consistency
            return str(validated_path)
        except ValueError as e:
            raise ValueError(f"Invalid file_path: {e}")


class IndexResponse(BaseModel):
    """Response model for indexing."""

    doc_id: str
    filename: str
    chunks_added: int
    total_tokens: int
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_store: dict
    llm: dict
    embedder: dict


class SearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., description="Search query", min_length=1, max_length=5000)
    n_results: int = Field(5, ge=1, le=50, description="Number of results")
    doc_id: Optional[str] = Field(None, description="Filter to specific document")

    @field_validator("query")
    @classmethod
    def validate_query_field(cls, v):
        """Validate and sanitize search query."""
        try:
            return validate_query(v, min_length=1, max_length=5000)
        except ValueError as e:
            raise ValueError(f"Invalid query: {e}")

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, v):
        """Validate document ID format."""
        if v is None:
            return v
        try:
            return validate_document_id(v)
        except ValueError as e:
            raise ValueError(f"Invalid doc_id: {e}")


class SearchResult(BaseModel):
    """Single search result."""

    chunk_id: str
    content: str
    metadata: dict
    similarity: float


class HistoryQueryResponse(BaseModel):
    """History query response."""

    query_id: str
    session_id: Optional[str]
    timestamp: float
    question: str
    answer: str
    sources: List[dict]
    metadata: dict


class BookmarkResponse(BaseModel):
    """Bookmark response."""

    bookmark_id: str
    query_id: str
    timestamp: float
    notes: Optional[str]
    tags: List[str]
    query: Optional[HistoryQueryResponse] = None


class CreateBookmarkRequest(BaseModel):
    """Create bookmark request."""

    query_id: str = Field(..., description="Query ID to bookmark")
    notes: Optional[str] = Field(None, description="Notes about this bookmark")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")


class UpdateBookmarkRequest(BaseModel):
    """Update bookmark request."""

    notes: Optional[str] = None
    tags: Optional[List[str]] = None


# =============================================================================
# Query Endpoints
# =============================================================================


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query documents and get an AI-generated answer.

    Returns relevant context and sources along with the generated answer.
    """
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = query_engine.query(
            question=request.question,
            n_results=request.n_results,
            doc_id=request.doc_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return QueryResponse(**result.to_dict())
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["Query"])
async def query_documents_stream(request: QueryRequest):
    """
    Query documents with streaming response.

    Returns answer tokens as they're generated.
    """
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    def generate():
        try:
            for token in query_engine.query_stream(
                question=request.question,
                n_results=request.n_results,
                doc_id=request.doc_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                yield token
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"\n\nError: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/search", response_model=List[SearchResult], tags=["Query"])
async def search_documents(request: SearchRequest):
    """
    Semantic search without answer generation.

    Returns relevant chunks ranked by similarity.
    """
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        chunks = query_engine.get_similar_chunks(
            question=request.query,
            n_results=request.n_results,
            doc_id=request.doc_id,
        )
        return [
            SearchResult(
                chunk_id=c["chunk_id"],
                content=c["content"],
                metadata=c["metadata"],
                similarity=c["similarity"],
            )
            for c in chunks
        ]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Document Management Endpoints
# =============================================================================


@app.post("/documents/index", response_model=IndexResponse, tags=["Documents"])
async def index_document(request: IndexRequest):
    """
    Index a document from the filesystem.

    Parses, chunks, embeds, and stores the document.
    """
    if document_indexer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        # Parse document
        parser = DocumentParser()
        parsed_doc = parser.parse(file_path)

        # Index
        result = document_indexer.index_document(
            parsed_doc,
            replace_existing=request.replace_existing,
        )

        return IndexResponse(**result)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=IndexResponse, tags=["Documents"])
async def upload_and_index(
    file: UploadFile = File(...),
    replace_existing: bool = Query(True),
):
    """
    Upload and index a document.

    Accepts PDF, DOCX, HTML files.
    """
    if document_indexer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    # Sanitize filename to prevent path traversal
    try:
        safe_filename = sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {e}")

    try:
        # Save uploaded file
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        file_path = INPUT_DIR / safe_filename

        # Read file content
        content = await file.read()

        # Validate file size (100MB max)
        max_size_mb = 100
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.2f}MB (max {max_size_mb}MB)",
            )

        # Validate file is not empty
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Write file
        with open(file_path, "wb") as f:
            f.write(content)

        # Parse and index
        parser = DocumentParser()
        parsed_doc = parser.parse(file_path)
        result = document_indexer.index_document(
            parsed_doc,
            replace_existing=replace_existing,
        )

        return IndexResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        docs = query_engine.vector_store.list_documents()
        return [DocumentInfo(**d) for d in docs]
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Get document details by ID."""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate document ID
    try:
        doc_id = validate_document_id(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {e}")

    try:
        result = query_engine.vector_store.get_by_doc_id(doc_id)
        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        return {
            "doc_id": doc_id,
            "chunk_count": len(result["ids"]),
            "chunks": [
                {"id": id_, "metadata": meta}
                for id_, meta in zip(result["ids"], result["metadatas"])
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document by ID."""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate document ID
    try:
        doc_id = validate_document_id(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {e}")

    try:
        deleted = query_engine.vector_store.delete_by_doc_id(doc_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        return {"status": "deleted", "doc_id": doc_id, "chunks_deleted": deleted}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# System Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and component status."""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        health = query_engine.health_check()
        return HealthResponse(
            status="healthy",
            vector_store=health["vector_store"],
            llm=health["llm"],
            embedder=health["embedder"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            vector_store={"status": "unknown"},
            llm={"status": "unknown"},
            embedder={"status": "unknown"},
        )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get system statistics."""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        store_stats = query_engine.vector_store.stats()
        return {
            "vector_store": store_stats,
            "llm_model": query_engine.llm.model,
            "embedding_model": query_engine.embedder.model_name,
            "embedding_dimension": query_engine.embedder.dimension,
        }
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resilience", tags=["System"])
async def get_resilience_stats():
    """Get resilience statistics (circuit breaker, retry)."""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        circuit_breaker_stats = query_engine.llm.get_circuit_breaker_stats()

        return {
            "circuit_breaker": (
                circuit_breaker_stats
                if circuit_breaker_stats
                else {"enabled": False, "message": "Circuit breaker is disabled"}
            ),
            "retry": {
                "enabled": query_engine.llm.enable_retry,
                "max_attempts": (
                    query_engine.llm.retry_config.max_retries
                    if query_engine.llm.enable_retry
                    else None
                ),
                "initial_delay": (
                    query_engine.llm.retry_config.initial_delay
                    if query_engine.llm.enable_retry
                    else None
                ),
            },
        }
    except Exception as e:
        logger.error(f"Resilience stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# History Endpoints
# =============================================================================


@app.get("/history", response_model=List[HistoryQueryResponse], tags=["History"])
async def list_history(
    limit: int = Query(50, ge=1, le=200, description="Maximum queries to return"),
    offset: int = Query(0, ge=0, description="Number of queries to skip"),
    search: Optional[str] = Query(None, description="Search text in questions/answers"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    start_date: Optional[float] = Query(None, description="Unix timestamp - start of date range"),
    end_date: Optional[float] = Query(None, description="Unix timestamp - end of date range"),
):
    """
    List query history with filters.

    Supports:
    - Pagination (limit, offset)
    - Text search (question/answer)
    - Session filtering
    - Date range filtering
    """
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        if search:
            queries = history_store.search_queries(
                search_text=search,
                limit=limit,
                offset=offset,
            )
        else:
            queries = history_store.list_queries(
                limit=limit,
                offset=offset,
                session_id=session_id,
                start_date=start_date,
                end_date=end_date,
            )

        return [HistoryQueryResponse(**q.to_dict()) for q in queries]

    except Exception as e:
        logger.error(f"List history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{query_id}", response_model=HistoryQueryResponse, tags=["History"])
async def get_history_query(query_id: str):
    """Get specific query from history."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        query = history_store.get_query(query_id)
        if not query:
            raise HTTPException(status_code=404, detail=f"Query not found: {query_id}")

        return HistoryQueryResponse(**query.to_dict())

    except HTTPException:
        raise
    except QueryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get history query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{query_id}", tags=["History"])
async def delete_history_query(query_id: str):
    """Delete query from history."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        deleted = history_store.delete_query(query_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Query not found: {query_id}")

        return {"status": "deleted", "query_id": query_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete history query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/export/markdown", tags=["History"])
async def export_history_markdown(
    query_ids: Optional[List[str]] = Query(None, description="Query IDs to export (or all recent if not specified)"),
    title: str = Query("Workpedia Query History", description="Document title"),
):
    """Export query history as Markdown."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        if query_ids:
            markdown = history_store.export_queries_markdown(query_ids, title)
        else:
            # Export all recent queries
            queries = history_store.list_queries(limit=100)
            query_ids = [q.query_id for q in queries]
            markdown = history_store.export_queries_markdown(query_ids, title)

        return Response(content=markdown, media_type="text/markdown")

    except Exception as e:
        logger.error(f"Export markdown failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/export/json", tags=["History"])
async def export_history_json(
    query_ids: Optional[List[str]] = Query(None, description="Query IDs to export (or all recent if not specified)"),
    include_sources: bool = Query(True, description="Include source data in export"),
):
    """Export query history as JSON."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        json_data = history_store.export_queries_json(query_ids, include_sources)
        return Response(content=json_data, media_type="application/json")

    except Exception as e:
        logger.error(f"Export JSON failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/export/pdf", tags=["History"])
async def export_history_pdf(
    query_ids: Optional[List[str]] = Query(None, description="Query IDs to export (or all recent if not specified)"),
    title: str = Query("Workpedia Query History", description="Document title"),
):
    """Export query history as PDF."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        if query_ids:
            pdf_bytes = history_store.export_queries_pdf(query_ids, title)
        else:
            # Export all recent queries
            queries = history_store.list_queries(limit=100)
            query_ids = [q.query_id for q in queries]
            pdf_bytes = history_store.export_queries_pdf(query_ids, title)

        return Response(content=pdf_bytes, media_type="application/pdf")

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF export requires reportlab. Install with: pip install reportlab",
        )
    except Exception as e:
        logger.error(f"Export PDF failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Bookmark Endpoints
# =============================================================================


@app.get("/bookmarks", response_model=List[BookmarkResponse], tags=["Bookmarks"])
async def list_bookmarks(
    tags: Optional[List[str]] = Query(None, description="Filter by tags (OR condition)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum bookmarks to return"),
    offset: int = Query(0, ge=0, description="Number of bookmarks to skip"),
):
    """List bookmarks with optional tag filtering."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        bookmarks = history_store.list_bookmarks(tags=tags, limit=limit, offset=offset)
        return [BookmarkResponse(**b.to_dict()) for b in bookmarks]

    except Exception as e:
        logger.error(f"List bookmarks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bookmarks", response_model=BookmarkResponse, tags=["Bookmarks"])
async def create_bookmark(request: CreateBookmarkRequest):
    """Create a bookmark for a query."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        bookmark_id = history_store.add_bookmark(
            query_id=request.query_id,
            notes=request.notes,
            tags=request.tags,
        )

        bookmark = history_store.get_bookmark(bookmark_id)
        if not bookmark:
            raise HTTPException(status_code=500, detail="Failed to retrieve created bookmark")

        return BookmarkResponse(**bookmark.to_dict())

    except QueryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Create bookmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bookmarks/{bookmark_id}", response_model=BookmarkResponse, tags=["Bookmarks"])
async def get_bookmark(bookmark_id: str):
    """Get specific bookmark."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        bookmark = history_store.get_bookmark(bookmark_id)
        if not bookmark:
            raise HTTPException(status_code=404, detail=f"Bookmark not found: {bookmark_id}")

        return BookmarkResponse(**bookmark.to_dict())

    except HTTPException:
        raise
    except BookmarkNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get bookmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/bookmarks/{bookmark_id}", response_model=BookmarkResponse, tags=["Bookmarks"])
async def update_bookmark(bookmark_id: str, request: UpdateBookmarkRequest):
    """Update bookmark notes or tags."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        updated = history_store.update_bookmark(
            bookmark_id=bookmark_id,
            notes=request.notes,
            tags=request.tags,
        )

        if not updated:
            raise HTTPException(status_code=404, detail=f"Bookmark not found: {bookmark_id}")

        bookmark = history_store.get_bookmark(bookmark_id)
        return BookmarkResponse(**bookmark.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update bookmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/bookmarks/{bookmark_id}", tags=["Bookmarks"])
async def delete_bookmark(bookmark_id: str):
    """Delete a bookmark."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store not initialized")

    try:
        deleted = history_store.delete_bookmark(bookmark_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Bookmark not found: {bookmark_id}")

        return {"status": "deleted", "bookmark_id": bookmark_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete bookmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# System Endpoints
# =============================================================================


@app.get("/", tags=["System"])
async def root():
    """API root - basic info."""
    return {
        "name": "Workpedia RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "resilience": "/resilience",
        "history": "/history",
        "bookmarks": "/bookmarks",
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "api.endpoints:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Workpedia RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)
