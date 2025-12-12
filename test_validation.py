#!/usr/bin/env python3
"""Test script for input validation and sanitization (Improvement #4).

This script tests the comprehensive validation system including:
- Query string validation
- File path validation and security
- Document ID validation
- API parameter validation
- File upload validation
- Metadata validation
- Sanitization utilities
"""

import os
import tempfile
from pathlib import Path
import uuid

from core.validators import (
    validate_query,
    validate_query_params,
    validate_file_path,
    validate_directory_path,
    validate_document_id,
    validate_collection_name,
    validate_file_upload,
    validate_metadata,
    sanitize_filename,
    sanitize_log_message,
)


def test_query_validation():
    """Test query string validation."""
    print("=" * 80)
    print("Test 1: Query String Validation")
    print("=" * 80)
    print()

    # Valid queries
    print("Testing valid queries...")
    assert validate_query("What is the capital of France?") == "What is the capital of France?"
    assert validate_query("  Leading and trailing spaces  ") == "Leading and trailing spaces"
    assert validate_query("A" * 5000) == "A" * 5000  # Max length
    print("✓ Valid queries passed")
    print()

    # Invalid queries
    print("Testing invalid queries...")
    errors_caught = 0

    try:
        validate_query("")
    except ValueError as e:
        print(f"✓ Empty query rejected: {e}")
        errors_caught += 1

    try:
        validate_query("   ")
    except ValueError as e:
        print(f"✓ Whitespace-only query rejected: {e}")
        errors_caught += 1

    try:
        validate_query("A" * 10000)
    except ValueError as e:
        print(f"✓ Too long query rejected: {e}")
        errors_caught += 1

    try:
        validate_query("Query with null\x00byte")
    except ValueError as e:
        print(f"✓ Null byte rejected: {e}")
        errors_caught += 1

    try:
        validate_query(123)
    except ValueError as e:
        print(f"✓ Non-string rejected: {e}")
        errors_caught += 1

    assert errors_caught == 5
    print()
    print("✓ Query validation test passed")
    print()


def test_query_params_validation():
    """Test query parameter validation."""
    print("=" * 80)
    print("Test 2: Query Parameters Validation")
    print("=" * 80)
    print()

    # Valid parameters
    print("Testing valid parameters...")
    params = validate_query_params(n_results=10, temperature=0.5, max_tokens=1000)
    assert params == {'n_results': 10, 'temperature': 0.5, 'max_tokens': 1000}
    print("✓ Valid parameters passed")
    print()

    # Edge cases
    params = validate_query_params(n_results=1, temperature=0.0, max_tokens=1)
    assert params == {'n_results': 1, 'temperature': 0.0, 'max_tokens': 1}
    print("✓ Edge case parameters (min values) passed")

    params = validate_query_params(n_results=50, temperature=2.0, max_tokens=4096)
    assert params == {'n_results': 50, 'temperature': 2.0, 'max_tokens': 4096}
    print("✓ Edge case parameters (max values) passed")
    print()

    # Invalid parameters
    print("Testing invalid parameters...")
    errors_caught = 0

    try:
        validate_query_params(n_results=0)
    except ValueError as e:
        print(f"✓ n_results too small rejected: {e}")
        errors_caught += 1

    try:
        validate_query_params(n_results=100)
    except ValueError as e:
        print(f"✓ n_results too large rejected: {e}")
        errors_caught += 1

    try:
        validate_query_params(temperature=-0.1)
    except ValueError as e:
        print(f"✓ temperature too small rejected: {e}")
        errors_caught += 1

    try:
        validate_query_params(temperature=2.5)
    except ValueError as e:
        print(f"✓ temperature too large rejected: {e}")
        errors_caught += 1

    try:
        validate_query_params(max_tokens=0)
    except ValueError as e:
        print(f"✓ max_tokens too small rejected: {e}")
        errors_caught += 1

    try:
        validate_query_params(max_tokens=10000)
    except ValueError as e:
        print(f"✓ max_tokens too large rejected: {e}")
        errors_caught += 1

    assert errors_caught == 6
    print()
    print("✓ Query parameters validation test passed")
    print()


def test_file_path_validation():
    """Test file path validation and security."""
    print("=" * 80)
    print("Test 3: File Path Validation and Security")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        test_file = tmpdir_path / "test.pdf"
        test_file.write_text("test content")

        test_txt = tmpdir_path / "document.txt"
        test_txt.write_text("text content")

        # Valid file paths
        print("Testing valid file paths...")
        validated = validate_file_path(str(test_file), must_exist=True)
        assert validated == test_file.resolve()
        print(f"✓ Valid file path: {test_file}")

        validated = validate_file_path(
            str(test_file),
            must_exist=True,
            allowed_extensions={'.pdf', '.docx'}
        )
        assert validated == test_file.resolve()
        print("✓ Valid file with allowed extension")
        print()

        # Invalid file paths
        print("Testing invalid file paths...")
        errors_caught = 0

        try:
            validate_file_path("/nonexistent/file.pdf", must_exist=True)
        except ValueError as e:
            print(f"✓ Nonexistent file rejected: {e}")
            errors_caught += 1

        try:
            validate_file_path("../../etc/passwd", must_exist=False)
        except ValueError as e:
            print(f"✓ Path traversal rejected: {e}")
            errors_caught += 1

        try:
            validate_file_path(
                str(test_txt),
                must_exist=True,
                allowed_extensions={'.pdf'}
            )
        except ValueError as e:
            print(f"✓ Invalid extension rejected: {e}")
            errors_caught += 1

        # Test base directory restriction
        other_dir = tmpdir_path / "other"
        other_dir.mkdir()
        other_file = other_dir / "file.pdf"
        other_file.write_text("content")

        try:
            validate_file_path(
                str(other_file),
                must_exist=True,
                base_directory=tmpdir_path / "restricted"
            )
        except ValueError as e:
            print(f"✓ Outside base directory rejected: {e}")
            errors_caught += 1

        assert errors_caught == 4
        print()
        print("✓ File path validation test passed")
        print()


def test_directory_path_validation():
    """Test directory path validation."""
    print("=" * 80)
    print("Test 4: Directory Path Validation")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Valid directory
        print("Testing valid directory...")
        validated = validate_directory_path(str(tmpdir_path), must_exist=True)
        assert validated == tmpdir_path.resolve()
        print(f"✓ Valid directory: {tmpdir_path}")
        print()

        # Create if missing
        new_dir = tmpdir_path / "newdir"
        validated = validate_directory_path(str(new_dir), create_if_missing=True)
        assert new_dir.exists()
        print(f"✓ Created missing directory: {new_dir}")
        print()

        # Invalid directories
        print("Testing invalid directories...")
        errors_caught = 0

        try:
            validate_directory_path("/nonexistent/dir", must_exist=True)
        except ValueError as e:
            print(f"✓ Nonexistent directory rejected: {e}")
            errors_caught += 1

        try:
            validate_directory_path("../../etc", must_exist=False)
        except ValueError as e:
            print(f"✓ Path traversal rejected: {e}")
            errors_caught += 1

        # Test file passed as directory
        test_file = tmpdir_path / "file.txt"
        test_file.write_text("content")

        try:
            validate_directory_path(str(test_file), must_exist=True)
        except ValueError as e:
            print(f"✓ File instead of directory rejected: {e}")
            errors_caught += 1

        assert errors_caught == 3
        print()
        print("✓ Directory path validation test passed")
        print()


def test_document_id_validation():
    """Test document ID validation."""
    print("=" * 80)
    print("Test 5: Document ID Validation")
    print("=" * 80)
    print()

    # Valid document IDs
    print("Testing valid document IDs...")

    # UUID format
    doc_id = str(uuid.uuid4())
    assert validate_document_id(doc_id) == doc_id
    print(f"✓ Valid UUID: {doc_id}")

    # Alphanumeric format
    assert validate_document_id("my_document_123") == "my_document_123"
    print("✓ Valid alphanumeric ID: my_document_123")

    assert validate_document_id("doc-2024-01-15") == "doc-2024-01-15"
    print("✓ Valid ID with hyphens: doc-2024-01-15")
    print()

    # Invalid document IDs
    print("Testing invalid document IDs...")
    errors_caught = 0

    try:
        validate_document_id("")
    except ValueError as e:
        print(f"✓ Empty ID rejected: {e}")
        errors_caught += 1

    try:
        validate_document_id("../../../etc/passwd")
    except ValueError as e:
        print(f"✓ Path separators rejected: {e}")
        errors_caught += 1

    try:
        validate_document_id("doc/id/with/slashes")
    except ValueError as e:
        print(f"✓ Forward slashes rejected: {e}")
        errors_caught += 1

    try:
        validate_document_id("doc\\id\\with\\backslashes")
    except ValueError as e:
        print(f"✓ Backslashes rejected: {e}")
        errors_caught += 1

    try:
        validate_document_id("doc with spaces")
    except ValueError as e:
        print(f"✓ Spaces rejected: {e}")
        errors_caught += 1

    try:
        validate_document_id("doc\x00null")
    except ValueError as e:
        print(f"✓ Null bytes rejected: {e}")
        errors_caught += 1

    try:
        validate_document_id("A" * 300)
    except ValueError as e:
        print(f"✓ Too long ID rejected: {e}")
        errors_caught += 1

    assert errors_caught == 7
    print()
    print("✓ Document ID validation test passed")
    print()


def test_collection_name_validation():
    """Test collection name validation."""
    print("=" * 80)
    print("Test 6: Collection Name Validation")
    print("=" * 80)
    print()

    # Valid collection names
    print("Testing valid collection names...")
    assert validate_collection_name("workpedia_docs") == "workpedia_docs"
    assert validate_collection_name("collection-2024") == "collection-2024"
    assert validate_collection_name("my_collection_123") == "my_collection_123"
    print("✓ Valid collection names passed")
    print()

    # Invalid collection names
    print("Testing invalid collection names...")
    errors_caught = 0

    try:
        validate_collection_name("")
    except ValueError as e:
        print(f"✓ Empty name rejected: {e}")
        errors_caught += 1

    try:
        validate_collection_name("collection with spaces")
    except ValueError as e:
        print(f"✓ Spaces rejected: {e}")
        errors_caught += 1

    try:
        validate_collection_name("A" * 100)
    except ValueError as e:
        print(f"✓ Too long name rejected: {e}")
        errors_caught += 1

    assert errors_caught == 3
    print()
    print("✓ Collection name validation test passed")
    print()


def test_file_upload_validation():
    """Test file upload validation."""
    print("=" * 80)
    print("Test 7: File Upload Validation")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test file
        test_file = tmpdir_path / "document.pdf"
        test_file.write_bytes(b"PDF content here" * 100)

        # Valid upload
        print("Testing valid file upload...")
        validated_path, info = validate_file_upload(test_file, max_size_mb=1)
        assert validated_path == test_file
        assert info['name'] == 'document.pdf'
        assert info['extension'] == '.pdf'
        print(f"✓ Valid upload: {info}")
        print()

        # Test with extension filter
        validated_path, info = validate_file_upload(
            test_file,
            max_size_mb=1,
            allowed_extensions={'.pdf', '.docx'}
        )
        assert validated_path == test_file
        print("✓ Valid upload with extension filter")
        print()

        # Invalid uploads
        print("Testing invalid file uploads...")
        errors_caught = 0

        # Too large
        large_file = tmpdir_path / "large.pdf"
        large_file.write_bytes(b"X" * (2 * 1024 * 1024))  # 2MB
        try:
            validate_file_upload(large_file, max_size_mb=1)
        except ValueError as e:
            print(f"✓ Too large file rejected: {e}")
            errors_caught += 1

        # Empty file
        empty_file = tmpdir_path / "empty.pdf"
        empty_file.write_bytes(b"")
        try:
            validate_file_upload(empty_file)
        except ValueError as e:
            print(f"✓ Empty file rejected: {e}")
            errors_caught += 1

        # Wrong extension
        try:
            validate_file_upload(test_file, allowed_extensions={'.docx'})
        except ValueError as e:
            print(f"✓ Wrong extension rejected: {e}")
            errors_caught += 1

        # Nonexistent file
        try:
            validate_file_upload(tmpdir_path / "nonexistent.pdf")
        except ValueError as e:
            print(f"✓ Nonexistent file rejected: {e}")
            errors_caught += 1

        assert errors_caught == 4
        print()
        print("✓ File upload validation test passed")
        print()


def test_metadata_validation():
    """Test metadata validation."""
    print("=" * 80)
    print("Test 8: Metadata Validation")
    print("=" * 80)
    print()

    # Valid metadata
    print("Testing valid metadata...")
    metadata = {
        'filename': 'doc.pdf',
        'page': 1,
        'score': 0.95,
        'tags': ['important', 'review'],
        'nested': {'key': 'value'},
    }
    validated = validate_metadata(metadata)
    assert validated == metadata
    print(f"✓ Valid metadata: {metadata}")
    print()

    # Invalid metadata
    print("Testing invalid metadata...")
    errors_caught = 0

    try:
        validate_metadata("not a dict")
    except ValueError as e:
        print(f"✓ Non-dict rejected: {e}")
        errors_caught += 1

    try:
        validate_metadata({123: "numeric key"})
    except ValueError as e:
        print(f"✓ Non-string key rejected: {e}")
        errors_caught += 1

    try:
        validate_metadata({"key\x00null": "value"})
    except ValueError as e:
        print(f"✓ Null byte in key rejected: {e}")
        errors_caught += 1

    # Too deeply nested
    deeply_nested = {'a': {'b': {'c': {'d': {'e': 'too deep'}}}}}
    try:
        validate_metadata(deeply_nested, max_depth=3)
    except ValueError as e:
        print(f"✓ Too deep nesting rejected: {e}")
        errors_caught += 1

    assert errors_caught == 4
    print()
    print("✓ Metadata validation test passed")
    print()


def test_filename_sanitization():
    """Test filename sanitization."""
    print("=" * 80)
    print("Test 9: Filename Sanitization")
    print("=" * 80)
    print()

    print("Testing filename sanitization...")

    # Basic sanitization
    assert sanitize_filename("My Document.pdf") == "My_Document.pdf"
    print("✓ Spaces replaced with underscores")

    assert sanitize_filename("Document (v2).pdf") == "Document_v2_.pdf"
    print("✓ Special characters replaced with underscores")

    assert sanitize_filename("../../../etc/passwd") == ".._.._.._etc_passwd"
    print("✓ Path separators replaced with underscores")

    # Test long filename truncation
    long_name = "file" * 100 + ".pdf"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 255
    assert sanitized.endswith(".pdf")
    print("✓ Long filename truncated while preserving extension")

    assert sanitize_filename("multiple___spaces___file.pdf") == "multiple_spaces_file.pdf"
    print("✓ Multiple underscores collapsed")

    print()
    print("✓ Filename sanitization test passed")
    print()


def test_log_message_sanitization():
    """Test log message sanitization."""
    print("=" * 80)
    print("Test 10: Log Message Sanitization")
    print("=" * 80)
    print()

    print("Testing log message sanitization...")

    # Remove newlines
    assert sanitize_log_message("Line 1\nLine 2\nLine 3") == "Line 1 Line 2 Line 3"
    print("✓ Newlines removed")

    # Remove carriage returns
    assert sanitize_log_message("Line 1\r\nLine 2") == "Line 1 Line 2"
    print("✓ Carriage returns removed")

    # Remove null bytes
    assert sanitize_log_message("Text\x00null") == "Textnull"
    print("✓ Null bytes removed")

    # Truncate long messages
    long_msg = "A" * 20000
    sanitized = sanitize_log_message(long_msg)
    assert len(sanitized) == 10003  # 10000 + "..."
    assert sanitized.endswith("...")
    print("✓ Long messages truncated")

    # Collapse multiple spaces
    assert sanitize_log_message("Multiple    spaces    here") == "Multiple spaces here"
    print("✓ Multiple spaces collapsed")

    print()
    print("✓ Log message sanitization test passed")
    print()


def print_summary():
    """Print summary of validation features."""
    print("=" * 80)
    print("Validation Features Summary")
    print("=" * 80)
    print()

    print("✓ Query string validation (length, content, null bytes)")
    print("✓ Query parameter validation (ranges, types)")
    print("✓ File path validation (security, traversal prevention)")
    print("✓ Directory path validation (existence, creation)")
    print("✓ Document ID validation (format, security)")
    print("✓ Collection name validation (ChromaDB requirements)")
    print("✓ File upload validation (size, type, content)")
    print("✓ Metadata validation (structure, depth, serialization)")
    print("✓ Filename sanitization (safety, length)")
    print("✓ Log message sanitization (injection prevention)")
    print()

    print("Security Benefits:")
    print("  - Path traversal attack prevention")
    print("  - Null byte injection prevention")
    print("  - SQL/NoSQL injection prevention via parameterization")
    print("  - Log injection prevention")
    print("  - File upload attack prevention")
    print("  - Input length validation (DoS prevention)")
    print()


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("Workpedia Input Validation Test Suite")
    print("=" * 80)
    print()

    try:
        test_query_validation()
        test_query_params_validation()
        test_file_path_validation()
        test_directory_path_validation()
        test_document_id_validation()
        test_collection_name_validation()
        test_file_upload_validation()
        test_metadata_validation()
        test_filename_sanitization()
        test_log_message_sanitization()

        print_summary()

        print("=" * 80)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("=" * 80)
        print()
        print("Input validation and sanitization is working correctly!")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
