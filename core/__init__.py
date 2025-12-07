"""Core RAG components for Workpedia."""

from core.parser import DocumentParser
from core.large_doc_handler import LargeDocumentHandler
from core.analyzer import (
    StructureAnalyzer,
    DocumentStructure,
    DocumentElement,
    TableInfo,
    CrossReference,
)
from core.validator import DocumentValidator, ValidationReport, ValidationIssue
from core.progress_tracker import ProgressTracker, ProcessingStage
from core.pdf_splitter import PDFSplitter, SplitResult
from core.doc_merger import DocumentMerger, ChunkInfo, MergeResult, create_chunk_info
from core.chunker import SemanticChunker, Chunk, chunk_document
from core.embedder import Embedder, embed_text
from core.llm import OllamaClient, RAG_SYSTEM_PROMPT, format_rag_prompt
from core.query_engine import QueryEngine, QueryResult, ask

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
