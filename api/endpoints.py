"""FastAPI endpoints for Workpedia RAG system."""

import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.query_engine import QueryEngine, QueryResult
from core.parser import DocumentParser
from core.llm import OllamaClient
from storage.vector_store import VectorStore, DocumentIndexer
from config.config import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
query_engine: Optional[QueryEngine] = None
document_indexer: Optional[DocumentIndexer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global query_engine, document_indexer

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
            error_msg = (
                f"STARTUP FAILED: {health['message']}\n"
                f"To fix this:\n"
                f"  1. Pull the model: 'ollama pull {health['model_name']}'\n"
                f"  2. Or use a different model in config/config.py\n"
                f"  Available models: {', '.join(health['available_models']) if health['available_models'] else 'none'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"✓ Ollama connection validated: {health['message']}")

        # Step 2: Initialize components
        query_engine = QueryEngine()
        document_indexer = DocumentIndexer(
            vector_store=query_engine.vector_store,
            embedder=query_engine.embedder,
        )

        logger.info("✓ Workpedia API initialized successfully")
        logger.info(f"  - Vector Store: {query_engine.vector_store.count} chunks indexed")
        logger.info(f"  - LLM: {health['model_name']}")
        logger.info(f"  - Embedder: {query_engine.embedder.model_name}")

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
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str = Field(..., description="Question to ask about documents")
    n_results: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    doc_id: Optional[str] = Field(None, description="Filter to specific document")
    temperature: float = Field(0.7, ge=0, le=1, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum response tokens")


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
    query: str = Field(..., description="Search query")
    n_results: int = Field(5, ge=1, le=50, description="Number of results")
    doc_id: Optional[str] = Field(None, description="Filter to specific document")


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    content: str
    metadata: dict
    similarity: float


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
    allowed_extensions = {".pdf", ".docx", ".html", ".htm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    try:
        # Save uploaded file
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        file_path = INPUT_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Parse and index
        parser = DocumentParser()
        parsed_doc = parser.parse(file_path)
        result = document_indexer.index_document(
            parsed_doc,
            replace_existing=replace_existing,
        )

        return IndexResponse(**result)
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


@app.get("/", tags=["System"])
async def root():
    """API root - basic info."""
    return {
        "name": "Workpedia RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
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
