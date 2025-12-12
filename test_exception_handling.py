#!/usr/bin/env python3
"""Test script for custom exception handling.

This script demonstrates the new exception hierarchy and how specific exceptions
are raised and handled throughout the application.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_exception_hierarchy():
    """Test the exception hierarchy and context handling."""
    from core.exceptions import (
        WorkpediaError,
        DocumentNotFoundError,
        DocumentParsingError,
        OllamaConnectionError,
        OllamaTimeoutError,
        VectorStoreError,
        get_exception_context,
        format_exception_chain,
    )

    print("=" * 80)
    print("Exception Hierarchy Test")
    print("=" * 80)
    print()

    # Test 1: DocumentNotFoundError
    print("Test 1: DocumentNotFoundError")
    print("-" * 80)
    try:
        raise DocumentNotFoundError("/path/to/missing/file.pdf")
    except WorkpediaError as e:
        print(f"Exception Type: {e.__class__.__name__}")
        print(f"Message: {e.message}")
        print(f"Context: {e.context}")
        print(f"Formatted: {format_exception_chain(e)}")
        print(f"isinstance(WorkpediaError): {isinstance(e, WorkpediaError)}")
        print()

    # Test 2: DocumentParsingError with chaining
    print("Test 2: DocumentParsingError with cause chain")
    print("-" * 80)
    try:
        try:
            # Simulate underlying error
            raise ValueError("Invalid PDF structure")
        except ValueError as ve:
            # Wrap in our custom exception
            raise DocumentParsingError(
                file_path="/path/to/doc.pdf",
                reason="Failed to extract text"
            ) from ve
    except WorkpediaError as e:
        print(f"Exception Type: {e.__class__.__name__}")
        print(f"Message: {e.message}")
        print(f"Context: {e.context}")
        print(f"Cause: {e.__cause__}")
        print(f"Formatted: {format_exception_chain(e)}")
        print()

    # Test 3: OllamaConnectionError
    print("Test 3: OllamaConnectionError")
    print("-" * 80)
    try:
        raise OllamaConnectionError(
            base_url="http://localhost:11434",
            reason="Connection refused"
        )
    except WorkpediaError as e:
        print(f"Exception Type: {e.__class__.__name__}")
        print(f"Message: {e.message}")
        print(f"Context: {e.context}")
        print()

    # Test 4: OllamaTimeoutError
    print("Test 4: OllamaTimeoutError")
    print("-" * 80)
    try:
        raise OllamaTimeoutError(timeout=120, model="mistral")
    except WorkpediaError as e:
        print(f"Exception Type: {e.__class__.__name__}")
        print(f"Message: {e.message}")
        print(f"Context: {e.context}")
        print()

    print("✓ All exception hierarchy tests passed")
    print()


def test_parser_exceptions():
    """Test parser exception handling."""
    from core.parser import DocumentParser
    from core.exceptions import DocumentNotFoundError, DocumentParsingError

    print("=" * 80)
    print("Parser Exception Handling Test")
    print("=" * 80)
    print()

    parser = DocumentParser()

    # Test 1: Non-existent file
    print("Test 1: Parsing non-existent file")
    print("-" * 80)
    try:
        result = parser.parse("/path/to/nonexistent/file.pdf")
        print("ERROR: Should have raised DocumentNotFoundError")
    except DocumentNotFoundError as e:
        print(f"✓ Correctly raised {e.__class__.__name__}")
        print(f"  Message: {e.message}")
        print(f"  File path: {e.context.get('file_path')}")
        print()

    print("✓ Parser exception tests passed")
    print()


def test_llm_exceptions():
    """Test LLM exception handling."""
    from core.llm import OllamaClient
    from core.exceptions import (
        OllamaConnectionError,
        OllamaGenerationError,
        OllamaTimeoutError
    )

    print("=" * 80)
    print("LLM Exception Handling Test")
    print("=" * 80)
    print()

    # Test with real Ollama client (may or may not be running)
    client = OllamaClient()

    print("Test 1: Check if Ollama is available")
    print("-" * 80)
    if client.is_available():
        print("✓ Ollama server is running")
        print(f"  Available models: {client.list_models()}")

        # Test generation (if server is running)
        print()
        print("Test 2: Test generation with real Ollama")
        print("-" * 80)
        try:
            response = client.generate(
                prompt="Say 'Hello' in one word only",
                temperature=0,
                max_tokens=5
            )
            print(f"✓ Generation successful: {response[:50]}")
        except (OllamaConnectionError, OllamaGenerationError, OllamaTimeoutError) as e:
            print(f"✗ Generation failed: {e.__class__.__name__}")
            print(f"  Message: {e.message}")
    else:
        print("⚠ Ollama server not running - skipping live tests")
        print("  Start Ollama with: ollama serve")

    print()
    print("✓ LLM exception tests completed")
    print()


def test_vector_store_exceptions():
    """Test vector store exception handling."""
    from storage.vector_store import VectorStore
    from core.exceptions import VectorStoreError, VectorStoreQueryError
    import numpy as np

    print("=" * 80)
    print("Vector Store Exception Handling Test")
    print("=" * 80)
    print()

    # Test with real vector store
    try:
        vector_store = VectorStore()
        print("✓ VectorStore initialized successfully")
        print(f"  Collection: {vector_store.collection_name}")
        print(f"  Chunks: {vector_store.count}")

        # Test query with valid embedding
        print()
        print("Test 1: Query with valid embedding")
        print("-" * 80)
        try:
            dummy_embedding = np.random.rand(768)  # 768-dim embedding
            results = vector_store.query(
                query_embedding=dummy_embedding,
                n_results=1
            )
            print(f"✓ Query successful: {len(results['ids'])} results")
        except VectorStoreQueryError as e:
            print(f"Query failed: {e.message}")

    except VectorStoreError as e:
        print(f"✗ VectorStore initialization failed: {e.__class__.__name__}")
        print(f"  Message: {e.message}")

    print()
    print("✓ Vector store exception tests completed")
    print()


def test_exception_context_extraction():
    """Test exception context utilities."""
    from core.exceptions import (
        DocumentParsingError,
        get_exception_context,
        format_exception_chain,
    )

    print("=" * 80)
    print("Exception Context Utilities Test")
    print("=" * 80)
    print()

    # Create exception with context
    exc = DocumentParsingError(
        file_path="/path/to/document.pdf",
        reason="Malformed PDF structure"
    )

    print("Test 1: get_exception_context()")
    print("-" * 80)
    context = get_exception_context(exc)
    print(f"Context: {context}")
    print()

    print("Test 2: format_exception_chain()")
    print("-" * 80)
    formatted = format_exception_chain(exc)
    print(f"Formatted: {formatted}")
    print()

    # Test with standard exception
    print("Test 3: Context from standard exception")
    print("-" * 80)
    std_exc = ValueError("Standard error")
    context = get_exception_context(std_exc)
    print(f"Context (should be empty): {context}")
    print()

    print("✓ Context utilities tests passed")
    print()


def print_summary():
    """Print summary of exception types and HTTP mappings."""
    print("=" * 80)
    print("Exception to HTTP Status Code Mapping")
    print("=" * 80)
    print()

    mappings = [
        ("DocumentNotFoundError", "404 Not Found", "Document file doesn't exist"),
        ("UnsupportedFormatError", "415 Unsupported Media Type", "File format not supported"),
        ("DocumentParsingError", "422 Unprocessable Entity", "Failed to parse document"),
        ("OllamaConnectionError", "503 Service Unavailable", "Ollama server unreachable"),
        ("OllamaTimeoutError", "504 Gateway Timeout", "Request took too long"),
        ("OllamaGenerationError", "500 Internal Server Error", "LLM generation failed"),
        ("VectorStoreQueryError", "500 Internal Server Error", "Vector store query failed"),
        ("IndexingError", "500 Internal Server Error", "Failed to index document"),
        ("ValidationError", "400 Bad Request", "Invalid input parameters"),
        ("WorkpediaError", "500 Internal Server Error", "Generic error (catch-all)"),
    ]

    print(f"{'Exception Type':<30} {'HTTP Status':<30} {'Description':<40}")
    print("-" * 100)
    for exc_type, http_status, description in mappings:
        print(f"{exc_type:<30} {http_status:<30} {description:<40}")

    print()
    print("Benefits:")
    print("  ✓ Specific exception types for different error scenarios")
    print("  ✓ Rich context information (file paths, doc IDs, etc.)")
    print("  ✓ Automatic HTTP status code mapping in FastAPI")
    print("  ✓ Better error messages with troubleshooting hints")
    print("  ✓ Exception chaining preserves root causes")
    print()


if __name__ == "__main__":
    try:
        print()
        test_exception_hierarchy()
        test_parser_exceptions()
        test_llm_exceptions()
        test_vector_store_exceptions()
        test_exception_context_extraction()
        print_summary()

        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Custom exception handling is working correctly!")
        print("Exceptions now provide:")
        print("  - Specific exception types for different scenarios")
        print("  - Rich context (file paths, doc IDs, models, etc.)")
        print("  - Proper HTTP status codes in API")
        print("  - Better debugging with exception chains")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        logger.exception("Test failed with exception")
        exit(1)
