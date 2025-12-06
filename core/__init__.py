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
