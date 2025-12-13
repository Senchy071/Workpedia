"""Custom exceptions for Workpedia RAG system.

This module defines a comprehensive exception hierarchy for better error handling
and debugging throughout the application. Each exception type includes context
and actionable error messages.
"""

from typing import Any, Dict, List, Optional


class WorkpediaError(Exception):
    """
    Base exception for all Workpedia errors.

    All custom exceptions inherit from this class, making it easy to catch
    any Workpedia-specific error.

    Attributes:
        message: Human-readable error message
        context: Optional dictionary with additional context
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional context.

        Args:
            message: Error message describing what went wrong
            context: Optional dict with additional context (doc_id, filename, etc.)
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        msg = self.message

        # Add context if available
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg = f"{msg} [{context_str}]"

        return msg


# =============================================================================
# Document Processing Errors
# =============================================================================

class DocumentProcessingError(WorkpediaError):
    """Base class for document processing errors."""
    pass


class DocumentNotFoundError(DocumentProcessingError):
    """Document file not found."""

    def __init__(self, file_path: str):
        super().__init__(
            f"Document not found: {file_path}",
            context={"file_path": file_path}
        )


class DocumentParsingError(DocumentProcessingError):
    """Failed to parse document."""

    def __init__(self, file_path: str, reason: Optional[str] = None):
        message = f"Failed to parse document: {file_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"file_path": file_path, "reason": reason}
        )


class UnsupportedFormatError(DocumentProcessingError):
    """Document format not supported."""

    def __init__(self, file_path: str, format: Optional[str] = None):
        message = f"Unsupported document format: {file_path}"
        if format:
            message += f" (format: {format})"
        super().__init__(
            message,
            context={"file_path": file_path, "format": format}
        )


class ChunkingError(DocumentProcessingError):
    """Failed to chunk document."""

    def __init__(self, doc_id: str, reason: Optional[str] = None):
        message = f"Failed to chunk document: {doc_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"doc_id": doc_id, "reason": reason}
        )


# =============================================================================
# LLM Errors
# =============================================================================

class LLMError(WorkpediaError):
    """Base class for LLM-related errors."""
    pass


class OllamaConnectionError(LLMError):
    """Ollama server is not reachable."""

    def __init__(self, base_url: str, reason: Optional[str] = None):
        message = f"Cannot connect to Ollama server at {base_url}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"base_url": base_url, "reason": reason}
        )


class OllamaModelNotFoundError(LLMError):
    """Requested model not available in Ollama."""

    def __init__(self, model_name: str, available_models: Optional[List[str]] = None):
        message = f"Model '{model_name}' not found in Ollama"
        if available_models:
            message += f". Available models: {', '.join(available_models[:5])}"
            if len(available_models) > 5:
                message += f" (and {len(available_models) - 5} more)"
        super().__init__(
            message,
            context={"model_name": model_name, "available_models": available_models}
        )


class OllamaGenerationError(LLMError):
    """LLM text generation failed."""

    def __init__(self, reason: Optional[str] = None, model: Optional[str] = None):
        message = "LLM generation failed"
        if model:
            message += f" (model: {model})"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"model": model, "reason": reason}
        )


class OllamaTimeoutError(LLMError):
    """LLM request timed out."""

    def __init__(self, timeout: int, model: Optional[str] = None):
        message = f"LLM request timed out after {timeout}s"
        if model:
            message += f" (model: {model})"
        super().__init__(
            message,
            context={"timeout": timeout, "model": model}
        )


# =============================================================================
# Embedding Errors
# =============================================================================

class EmbeddingError(WorkpediaError):
    """Base class for embedding-related errors."""
    pass


class EmbeddingGenerationError(EmbeddingError):
    """Failed to generate embeddings."""

    def __init__(self, reason: Optional[str] = None, model: Optional[str] = None, text_length: Optional[int] = None):
        message = "Failed to generate embeddings"
        if model:
            message += f" (model: {model})"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"model": model, "reason": reason, "text_length": text_length}
        )


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Embedding dimension doesn't match expected size."""

    def __init__(self, expected: int, actual: int):
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {actual}",
            context={"expected": expected, "actual": actual}
        )


# =============================================================================
# Vector Store Errors
# =============================================================================

class VectorStoreError(WorkpediaError):
    """Base class for vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Cannot connect to vector store."""

    def __init__(self, store_path: str, reason: Optional[str] = None):
        message = f"Cannot connect to vector store at {store_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"store_path": store_path, "reason": reason}
        )


class DocumentNotIndexedError(VectorStoreError):
    """Requested document not found in vector store."""

    def __init__(self, doc_id: str):
        super().__init__(
            f"Document not indexed: {doc_id}",
            context={"doc_id": doc_id}
        )


class IndexingError(VectorStoreError):
    """Failed to index document."""

    def __init__(self, doc_id: str, reason: Optional[str] = None):
        message = f"Failed to index document: {doc_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"doc_id": doc_id, "reason": reason}
        )


class VectorStoreQueryError(VectorStoreError):
    """Query to vector store failed."""

    def __init__(self, reason: Optional[str] = None, query_length: Optional[int] = None):
        message = "Vector store query failed"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"reason": reason, "query_length": query_length}
        )


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(WorkpediaError):
    """Base class for input validation errors."""
    pass


class InvalidQueryError(ValidationError):
    """Invalid query input."""

    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Invalid query: {reason}",
            context={"query": query[:100], "reason": reason}
        )


class InvalidDocumentIdError(ValidationError):
    """Invalid document ID format."""

    def __init__(self, doc_id: str, expected_format: Optional[str] = None):
        message = f"Invalid document ID: {doc_id}"
        if expected_format:
            message += f" (expected format: {expected_format})"
        super().__init__(
            message,
            context={"doc_id": doc_id, "expected_format": expected_format}
        )


class InvalidFilePathError(ValidationError):
    """Invalid or unsafe file path."""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"Invalid file path: {file_path} - {reason}",
            context={"file_path": file_path, "reason": reason}
        )


class InvalidParameterError(ValidationError):
    """Invalid parameter value."""

    def __init__(self, param_name: str, param_value, reason: str):
        super().__init__(
            f"Invalid parameter '{param_name}': {reason}",
            context={"param_name": param_name, "param_value": str(param_value), "reason": reason}
        )


# =============================================================================
# Query Engine Errors
# =============================================================================

class QueryError(WorkpediaError):
    """Base class for query execution errors."""
    pass


class NoResultsError(QueryError):
    """Query returned no results."""

    def __init__(self, query: str):
        super().__init__(
            "No results found for query",
            context={"query": query[:100]}
        )


class QueryExecutionError(QueryError):
    """Failed to execute query."""

    def __init__(self, query: str, reason: Optional[str] = None):
        message = "Query execution failed"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            context={"query": query[:100], "reason": reason}
        )


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(WorkpediaError):
    """Configuration is invalid or missing."""

    def __init__(self, config_name: str, reason: str):
        super().__init__(
            f"Configuration error for '{config_name}': {reason}",
            context={"config_name": config_name, "reason": reason}
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_exception_context(exc: Exception) -> dict:
    """
    Extract context from exception if available.

    Args:
        exc: Exception instance

    Returns:
        Context dictionary or empty dict if not a WorkpediaError
    """
    if isinstance(exc, WorkpediaError):
        return exc.context
    return {}


def format_exception_chain(exc: Exception) -> str:
    """
    Format exception chain for logging.

    Args:
        exc: Exception instance

    Returns:
        Formatted string with exception chain
    """
    parts = [str(exc)]

    # Add cause if available
    if exc.__cause__:
        parts.append(f"Caused by: {exc.__cause__}")

    # Add context if WorkpediaError
    context = get_exception_context(exc)
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        parts.append(f"Context: {context_str}")

    return " | ".join(parts)
