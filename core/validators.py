"""Input validation and sanitization for Workpedia.

This module provides comprehensive validation for:
- Query strings (length, content)
- File paths (security, path traversal prevention)
- Document IDs (format validation)
- API parameters (range validation)
- File uploads (size, type validation)

All validation functions raise ValueError with descriptive messages on validation failure.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Optional, Set

# ============================================================================
# Query Validation
# ============================================================================

def validate_query(query: str, min_length: int = 1, max_length: int = 5000) -> str:
    """
    Validate and sanitize user query strings.

    Args:
        query: User query string
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Sanitized query string

    Raises:
        ValueError: If query is invalid

    Example:
        >>> validate_query("What is the capital of France?")
        "What is the capital of France?"
        >>> validate_query("")  # Raises ValueError
        >>> validate_query("x" * 10000)  # Raises ValueError
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}")

    # Strip whitespace
    query = query.strip()

    # Check length
    if len(query) < min_length:
        raise ValueError(f"Query is too short (minimum {min_length} characters)")

    if len(query) > max_length:
        raise ValueError(f"Query is too long (maximum {max_length} characters)")

    # Check for null bytes (security)
    if '\x00' in query:
        raise ValueError("Query contains invalid null bytes")

    return query


def validate_query_params(
    n_results: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Validate query engine parameters.

    Args:
        n_results: Number of results to retrieve (1-50)
        temperature: LLM temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate (1-4096)

    Returns:
        Dictionary of validated parameters

    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}

    if n_results is not None:
        if not isinstance(n_results, int):
            raise ValueError(f"n_results must be an integer, got {type(n_results).__name__}")
        if n_results < 1 or n_results > 50:
            raise ValueError(f"n_results must be between 1 and 50, got {n_results}")
        validated['n_results'] = n_results

    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise ValueError(f"temperature must be a number, got {type(temperature).__name__}")
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        validated['temperature'] = float(temperature)

    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            raise ValueError(f"max_tokens must be an integer, got {type(max_tokens).__name__}")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError(f"max_tokens must be between 1 and 4096, got {max_tokens}")
        validated['max_tokens'] = max_tokens

    return validated


# ============================================================================
# File Path Validation
# ============================================================================

def validate_file_path(
    file_path: str,
    must_exist: bool = True,
    allowed_extensions: Optional[Set[str]] = None,
    base_directory: Optional[Path] = None,
) -> Path:
    """
    Validate file paths with security checks.

    Prevents:
    - Path traversal attacks (../)
    - Symlink attacks
    - Access outside allowed directories

    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        allowed_extensions: Set of allowed file extensions (e.g., {'.pdf', '.docx'})
        base_directory: Base directory to restrict access to

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or insecure

    Example:
        >>> validate_file_path("/data/input/doc.pdf", must_exist=True)
        PosixPath('/data/input/doc.pdf')
        >>> validate_file_path("../../etc/passwd")  # Raises ValueError
    """
    if not isinstance(file_path, (str, Path)):
        raise ValueError(f"file_path must be a string or Path, got {type(file_path).__name__}")

    # Convert to Path object
    path = Path(file_path)

    # Security: Resolve to absolute path to prevent traversal
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}")

    # Security: Check for path traversal attempts
    if ".." in path.parts:
        raise ValueError(f"Path traversal detected: {file_path}")

    # Security: Prevent symlink attacks
    if resolved_path.is_symlink():
        raise ValueError(f"Symlinks are not allowed: {file_path}")

    # Security: Restrict to base directory if specified
    if base_directory is not None:
        base_directory = Path(base_directory).resolve()
        try:
            resolved_path.relative_to(base_directory)
        except ValueError:
            raise ValueError(
                f"Path is outside allowed directory: {file_path} "
                f"(allowed: {base_directory})"
            )

    # Check existence
    if must_exist and not resolved_path.exists():
        raise ValueError(f"File not found: {file_path}")

    # Check file extension
    if allowed_extensions is not None:
        if resolved_path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Invalid file extension: {resolved_path.suffix} "
                f"(allowed: {', '.join(sorted(allowed_extensions))})"
            )

    return resolved_path


def validate_directory_path(
    dir_path: str,
    must_exist: bool = True,
    create_if_missing: bool = False,
    base_directory: Optional[Path] = None,
) -> Path:
    """
    Validate directory paths with security checks.

    Args:
        dir_path: Directory path to validate
        must_exist: Whether directory must exist
        create_if_missing: Create directory if it doesn't exist
        base_directory: Base directory to restrict access to

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or insecure
    """
    if not isinstance(dir_path, (str, Path)):
        raise ValueError(f"dir_path must be a string or Path, got {type(dir_path).__name__}")

    # Convert to Path object
    path = Path(dir_path)

    # Security: Resolve to absolute path
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid directory path: {e}")

    # Security: Check for path traversal
    if ".." in path.parts:
        raise ValueError(f"Path traversal detected: {dir_path}")

    # Security: Restrict to base directory if specified
    if base_directory is not None:
        base_directory = Path(base_directory).resolve()
        try:
            resolved_path.relative_to(base_directory)
        except ValueError:
            raise ValueError(
                f"Path is outside allowed directory: {dir_path} "
                f"(allowed: {base_directory})"
            )

    # Check existence
    if resolved_path.exists():
        if not resolved_path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {dir_path}")
    elif must_exist and not create_if_missing:
        raise ValueError(f"Directory not found: {dir_path}")
    elif create_if_missing:
        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Failed to create directory: {e}")

    return resolved_path


# ============================================================================
# Document ID Validation
# ============================================================================

def validate_document_id(doc_id: str) -> str:
    """
    Validate document ID format.

    Document IDs should be:
    - UUID format (for security and consistency)
    - Or alphanumeric with hyphens/underscores only

    Args:
        doc_id: Document ID to validate

    Returns:
        Validated document ID

    Raises:
        ValueError: If document ID is invalid

    Example:
        >>> validate_document_id("550e8400-e29b-41d4-a716-446655440000")
        "550e8400-e29b-41d4-a716-446655440000"
        >>> validate_document_id("my_document_123")
        "my_document_123"
        >>> validate_document_id("../../../etc/passwd")  # Raises ValueError
    """
    if not isinstance(doc_id, str):
        raise ValueError(f"Document ID must be a string, got {type(doc_id).__name__}")

    doc_id = doc_id.strip()

    if not doc_id:
        raise ValueError("Document ID cannot be empty")

    # Security: Check for null bytes
    if '\x00' in doc_id:
        raise ValueError("Document ID contains invalid null bytes")

    # Security: Check for path separators
    if '/' in doc_id or '\\' in doc_id:
        raise ValueError("Document ID cannot contain path separators")

    # Try to parse as UUID first
    try:
        uuid.UUID(doc_id)
        return doc_id
    except ValueError:
        pass

    # Otherwise, check for safe alphanumeric format
    if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
        raise ValueError(
            "Document ID must be UUID format or alphanumeric with hyphens/underscores only"
        )

    # Check length
    if len(doc_id) > 255:
        raise ValueError("Document ID is too long (maximum 255 characters)")

    return doc_id


def validate_collection_name(collection_name: str) -> str:
    """
    Validate ChromaDB collection name.

    Args:
        collection_name: Collection name to validate

    Returns:
        Validated collection name

    Raises:
        ValueError: If collection name is invalid
    """
    if not isinstance(collection_name, str):
        raise ValueError(
            f"Collection name must be a string, got {type(collection_name).__name__}"
        )

    collection_name = collection_name.strip()

    if not collection_name:
        raise ValueError("Collection name cannot be empty")

    # ChromaDB naming requirements
    if not re.match(r'^[a-zA-Z0-9_-]+$', collection_name):
        raise ValueError(
            "Collection name must contain only alphanumeric characters, "
            "hyphens, and underscores"
        )

    if len(collection_name) > 63:
        raise ValueError("Collection name is too long (maximum 63 characters)")

    return collection_name


# ============================================================================
# File Upload Validation
# ============================================================================

def validate_file_upload(
    file_path: Path,
    max_size_mb: int = 100,
    allowed_extensions: Optional[Set[str]] = None,
) -> tuple[Path, dict]:
    """
    Validate uploaded files.

    Args:
        file_path: Path to uploaded file
        max_size_mb: Maximum file size in megabytes
        allowed_extensions: Set of allowed file extensions

    Returns:
        Tuple of (validated_path, file_info_dict)

    Raises:
        ValueError: If file is invalid

    Example:
        >>> path, info = validate_file_upload(
        ...     Path("document.pdf"),
        ...     max_size_mb=50,
        ...     allowed_extensions={'.pdf', '.docx'}
        ... )
    """
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file size
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File is too large: {file_size_mb:.2f}MB (maximum {max_size_mb}MB)"
        )

    # Check file extension
    if allowed_extensions is not None:
        if file_path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Invalid file type: {file_path.suffix} "
                f"(allowed: {', '.join(sorted(allowed_extensions))})"
            )

    # Security: Check for empty files
    if file_size_bytes == 0:
        raise ValueError("File is empty")

    # Build file info
    file_info = {
        'name': file_path.name,
        'size_bytes': file_size_bytes,
        'size_mb': file_size_mb,
        'extension': file_path.suffix.lower(),
    }

    return file_path, file_info


# ============================================================================
# Metadata Validation
# ============================================================================

def validate_metadata(metadata: dict, max_depth: int = 3) -> dict:
    """
    Validate metadata dictionaries.

    Ensures metadata is JSON-serializable and doesn't contain dangerous content.

    Args:
        metadata: Metadata dictionary to validate
        max_depth: Maximum nesting depth allowed

    Returns:
        Validated metadata dictionary

    Raises:
        ValueError: If metadata is invalid
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dictionary, got {type(metadata).__name__}")

    def check_depth(obj, current_depth=0):
        """Recursively check nesting depth."""
        if current_depth > max_depth:
            raise ValueError(f"Metadata nesting depth exceeds maximum ({max_depth})")

        if isinstance(obj, dict):
            for key, value in obj.items():
                # Check key type
                if not isinstance(key, str):
                    raise ValueError(f"Metadata keys must be strings, got {type(key).__name__}")

                # Check for null bytes in keys
                if '\x00' in key:
                    raise ValueError("Metadata keys cannot contain null bytes")

                # Recursively check values
                check_depth(value, current_depth + 1)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                check_depth(item, current_depth + 1)

        elif not isinstance(obj, (str, int, float, bool, type(None))):
            raise ValueError(
                f"Metadata values must be JSON-serializable, got {type(obj).__name__}"
            )

    check_depth(metadata)

    return metadata


# ============================================================================
# Sanitization Utilities
# ============================================================================

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe storage.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename

    Example:
        >>> sanitize_filename("My Document (v2).pdf")
        "My_Document_v2.pdf"
        >>> sanitize_filename("../../etc/passwd")
        "etc_passwd"
    """
    if not isinstance(filename, str):
        raise ValueError(f"Filename must be a string, got {type(filename).__name__}")

    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Replace unsafe characters with underscores
    filename = re.sub(r'[^\w\s.-]', '_', filename)

    # Collapse multiple underscores/spaces
    filename = re.sub(r'[_\s]+', '_', filename)

    # Remove leading/trailing underscores
    filename = filename.strip('_')

    # Truncate to max length
    if len(filename) > max_length:
        # Preserve extension if present
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    if not filename:
        raise ValueError("Sanitized filename is empty")

    return filename


def sanitize_log_message(message: str, max_length: int = 10000) -> str:
    """
    Sanitize log messages to prevent log injection.

    Args:
        message: Log message to sanitize
        max_length: Maximum message length

    Returns:
        Sanitized log message
    """
    if not isinstance(message, str):
        message = str(message)

    # Remove null bytes
    message = message.replace('\x00', '')

    # Replace newlines to prevent log injection
    message = message.replace('\n', ' ').replace('\r', ' ')

    # Collapse multiple spaces
    message = re.sub(r'\s+', ' ', message)

    # Truncate to max length
    if len(message) > max_length:
        message = message[:max_length] + '...'

    return message.strip()
