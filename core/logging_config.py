"""Centralized logging configuration for Workpedia.

This module provides production-ready logging with:
- Structured logging with context (request IDs, doc IDs, etc.)
- File rotation for production deployments
- Per-module log level configuration
- Performance timing decorators
- JSON formatting option for log aggregation
"""

import logging
import logging.handlers
import sys
import time
import json
import functools
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import contextvars

# Context variables for request tracking
request_id_var = contextvars.ContextVar('request_id', default=None)
doc_id_var = contextvars.ContextVar('doc_id', default=None)


class ContextFilter(logging.Filter):
    """Add context variables to log records."""

    def filter(self, record):
        """Add request_id and doc_id to log record if available."""
        record.request_id = request_id_var.get() or '-'
        record.doc_id = doc_id_var.get() or '-'
        return True


class StructuredFormatter(logging.Formatter):
    """
    Structured log formatter that outputs JSON for log aggregation systems.

    Includes timestamp, level, logger name, message, and context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add context if available
        if hasattr(record, 'request_id') and record.request_id != '-':
            log_data['request_id'] = record.request_id
        if hasattr(record, 'doc_id') and record.doc_id != '-':
            log_data['doc_id'] = record.doc_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields from record.__dict__
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                           'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
                           'pathname', 'process', 'processName', 'relativeCreated',
                           'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                           'request_id', 'doc_id']:
                try:
                    # Only include JSON-serializable values
                    json.dumps(value)
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)

        if extra_fields:
            log_data['extra'] = extra_fields

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.

    Makes logs easier to read in development with color-coded levels.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
            record.levelname = colored_levelname

        # Add context info if available
        context_parts = []
        if hasattr(record, 'request_id') and record.request_id != '-':
            context_parts.append(f"req={record.request_id[:8]}")
        if hasattr(record, 'doc_id') and record.doc_id != '-':
            context_parts.append(f"doc={record.doc_id[:8]}")

        if context_parts:
            context_str = f" [{', '.join(context_parts)}]"
            # Insert context before message
            original_msg = record.getMessage()
            record.msg = f"{context_str} {original_msg}"
            record.args = ()

        formatted = super().format(record)

        # Reset levelname for next use
        record.levelname = levelname

        return formatted


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    structured: bool = False,
    console_colors: bool = True,
    module_levels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Configure logging for Workpedia.

    Args:
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path (overrides log_dir)
        log_dir: Directory for log files (creates app.log and error.log)
        max_bytes: Maximum size per log file before rotation
        backup_count: Number of backup files to keep
        structured: Use JSON structured logging (for production)
        console_colors: Use colored output in console (for development)
        module_levels: Per-module log levels (e.g., {"core.parser": "DEBUG"})

    Example:
        # Development setup
        setup_logging(level="DEBUG", console_colors=True)

        # Production setup
        setup_logging(
            level="INFO",
            log_dir=Path("/var/log/workpedia"),
            structured=True,
            console_colors=False,
            module_levels={
                "core.parser": "DEBUG",
                "core.llm": "WARNING",
            }
        )
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Add context filter to all handlers
    context_filter = ContextFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.addFilter(context_filter)

    if structured:
        console_formatter = StructuredFormatter()
    elif console_colors:
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handlers (if log_file or log_dir specified)
    if log_file:
        _add_file_handler(
            root_logger, log_file, level, max_bytes, backup_count,
            structured, context_filter
        )
    elif log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main application log
        _add_file_handler(
            root_logger, log_dir / "app.log", level, max_bytes, backup_count,
            structured, context_filter
        )

        # Error log (only ERROR and above)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(context_filter)
        if structured:
            error_handler.setFormatter(StructuredFormatter())
        else:
            error_handler.setFormatter(logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        root_logger.addHandler(error_handler)

    # Configure per-module log levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(getattr(logging, module_level.upper()))

    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

    # Log configuration success
    root_logger.info(
        f"Logging configured: level={level}, structured={structured}, "
        f"file_logging={'enabled' if (log_file or log_dir) else 'disabled'}"
    )


def _add_file_handler(
    logger: logging.Logger,
    log_file: Path,
    level: str,
    max_bytes: int,
    backup_count: int,
    structured: bool,
    context_filter: ContextFilter,
) -> None:
    """Add rotating file handler to logger."""
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.addFilter(context_filter)

    if structured:
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    logger.addHandler(file_handler)


# Context managers for setting context variables

class LogContext:
    """
    Context manager for setting log context variables.

    Usage:
        with LogContext(request_id="abc123", doc_id="doc456"):
            logger.info("Processing document")
            # Logs will include request_id and doc_id
    """

    def __init__(self, request_id: Optional[str] = None, doc_id: Optional[str] = None):
        self.request_id = request_id
        self.doc_id = doc_id
        self.request_id_token = None
        self.doc_id_token = None

    def __enter__(self):
        if self.request_id:
            self.request_id_token = request_id_var.set(self.request_id)
        if self.doc_id:
            self.doc_id_token = doc_id_var.set(self.doc_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.request_id_token:
            request_id_var.reset(self.request_id_token)
        if self.doc_id_token:
            doc_id_var.reset(self.doc_id_token)


def set_request_id(request_id: str) -> None:
    """Set request ID for current context."""
    request_id_var.set(request_id)


def set_doc_id(doc_id: str) -> None:
    """Set document ID for current context."""
    doc_id_var.set(doc_id)


def get_request_id() -> Optional[str]:
    """Get request ID from current context."""
    return request_id_var.get()


def get_doc_id() -> Optional[str]:
    """Get document ID from current context."""
    return doc_id_var.get()


# Performance timing decorator

def log_timing(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    message_template: str = "{func_name} completed in {elapsed:.3f}s",
) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level for timing message
        message_template: Template for log message (supports {func_name}, {elapsed})

    Usage:
        @log_timing()
        def process_document(doc_path):
            # Processing...
            pass

        # Logs: "process_document completed in 2.345s"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                message = message_template.format(
                    func_name=func.__name__,
                    elapsed=elapsed,
                )
                logger.log(level, message, extra={'elapsed_time': elapsed})

                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}",
                    extra={'elapsed_time': elapsed, 'error': str(e)},
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


# Utility functions for structured logging

def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any
) -> None:
    """
    Log message with additional context fields.

    Args:
        logger: Logger instance
        level: Log level (logging.INFO, logging.ERROR, etc.)
        message: Log message
        **context: Additional context fields to include

    Usage:
        log_with_context(
            logger, logging.INFO,
            "Document processed successfully",
            doc_id="abc123",
            chunk_count=42,
            processing_time=1.23
        )
    """
    logger.log(level, message, extra=context)


def log_performance(
    logger: logging.Logger,
    operation: str,
    elapsed: float,
    **metrics: Any
) -> None:
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Name of the operation
        elapsed: Elapsed time in seconds
        **metrics: Additional metrics to log

    Usage:
        start = time.time()
        # ... do work ...
        log_performance(
            logger, "document_parsing",
            time.time() - start,
            pages=150,
            file_size_mb=12.5
        )
    """
    log_with_context(
        logger, logging.INFO,
        f"Performance: {operation} completed in {elapsed:.3f}s",
        operation=operation,
        elapsed_time=elapsed,
        **metrics
    )
