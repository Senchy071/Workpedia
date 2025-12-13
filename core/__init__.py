"""Core RAG components for Workpedia."""

from core.analyzer import (
    CrossReference,
    DocumentElement,
    DocumentStructure,
    StructureAnalyzer,
    TableInfo,
)
from core.chunker import Chunk, SemanticChunker, chunk_document
from core.doc_merger import ChunkInfo, DocumentMerger, MergeResult, create_chunk_info
from core.embedder import Embedder, embed_text
from core.large_doc_handler import LargeDocumentHandler
from core.llm import RAG_SYSTEM_PROMPT, OllamaClient, format_rag_prompt
from core.parser import DocumentParser
from core.pdf_splitter import PDFSplitter, SplitResult
from core.progress_tracker import ProcessingStage, ProgressTracker
from core.query_engine import QueryEngine, QueryResult, ask
from core.validator import DocumentValidator, ValidationIssue, ValidationReport

__all__ = [
    # Parser
    "DocumentParser",
    # Large document handling
    "LargeDocumentHandler",
    "PDFSplitter",
    "SplitResult",
    "DocumentMerger",
    "ChunkInfo",
    "MergeResult",
    "create_chunk_info",
    # Chunking (Phase 3)
    "SemanticChunker",
    "Chunk",
    "chunk_document",
    # Embedding (Phase 3)
    "Embedder",
    "embed_text",
    # LLM (Phase 4)
    "OllamaClient",
    "RAG_SYSTEM_PROMPT",
    "format_rag_prompt",
    # Query Engine (Phase 4)
    "QueryEngine",
    "QueryResult",
    "ask",
    # Structure analysis
    "StructureAnalyzer",
    "DocumentStructure",
    "DocumentElement",
    "TableInfo",
    "CrossReference",
    # Validation
    "DocumentValidator",
    "ValidationReport",
    "ValidationIssue",
    # Progress tracking
    "ProgressTracker",
    "ProcessingStage",
]
