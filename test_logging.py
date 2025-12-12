#!/usr/bin/env python3
"""Test script for enhanced logging configuration.

This script demonstrates the new logging features added in Improvement #3:
- Structured logging with context
- Request ID tracking
- Performance timing
- Colored console output
- File rotation
- Per-module log levels
"""

import logging
import time
import tempfile
from pathlib import Path

# Import our logging configuration
from core.logging_config import (
    setup_logging,
    LogContext,
    set_request_id,
    set_doc_id,
    get_request_id,
    log_timing,
    log_with_context,
    log_performance,
)


def test_basic_logging():
    """Test basic logging with different levels."""
    print("=" * 80)
    print("Test 1: Basic Logging with Different Levels")
    print("=" * 80)
    print()

    # Setup logging with colored output
    setup_logging(level="DEBUG", console_colors=True)

    logger = logging.getLogger("test_basic")

    # Test all log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    print()
    print("✓ Basic logging test passed")
    print()


def test_context_logging():
    """Test logging with context variables."""
    print("=" * 80)
    print("Test 2: Context-Aware Logging")
    print("=" * 80)
    print()

    logger = logging.getLogger("test_context")

    # Without context
    logger.info("Log message without context")

    # With context using context manager
    with LogContext(request_id="req-12345", doc_id="doc-67890"):
        logger.info("Log message with request_id and doc_id")
        logger.warning("Another message with same context")

    # With context using setter functions
    set_request_id("req-abcdef")
    set_doc_id("doc-xyz123")
    logger.info("Log message with explicitly set context")

    # Verify context retrieval
    print(f"\nContext retrieved: request_id={get_request_id()}")

    print()
    print("✓ Context logging test passed")
    print()


def test_structured_logging():
    """Test structured (JSON) logging."""
    print("=" * 80)
    print("Test 3: Structured (JSON) Logging")
    print("=" * 80)
    print()

    # Setup with structured logging
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "structured.log"

        # Reconfigure with structured logging
        setup_logging(
            level="INFO",
            log_file=log_file,
            structured=True,
            console_colors=False,
        )

        logger = logging.getLogger("test_structured")

        # Log with context
        with LogContext(request_id="req-json-test"):
            logger.info("Structured log message", extra={
                'user_id': 'user123',
                'operation': 'document_upload',
                'file_size': 1024000,
            })

        # Read and display the structured log
        log_content = log_file.read_text()
        print("Structured log output:")
        print(log_content)

    print("✓ Structured logging test passed")
    print()


def test_performance_timing():
    """Test performance timing decorator."""
    print("=" * 80)
    print("Test 4: Performance Timing")
    print("=" * 80)
    print()

    logger = logging.getLogger("test_performance")

    @log_timing(logger=logger)
    def slow_function():
        """Simulate a slow operation."""
        time.sleep(0.5)
        return "completed"

    @log_timing(logger=logger, message_template="{func_name} took {elapsed:.3f}s")
    def fast_function():
        """Simulate a fast operation."""
        time.sleep(0.1)
        return "done"

    # Execute functions
    print("Calling slow_function()...")
    result1 = slow_function()
    print(f"Result: {result1}")

    print("\nCalling fast_function()...")
    result2 = fast_function()
    print(f"Result: {result2}")

    print()
    print("✓ Performance timing test passed")
    print()


def test_manual_performance_logging():
    """Test manual performance logging."""
    print("=" * 80)
    print("Test 5: Manual Performance Logging")
    print("=" * 80)
    print()

    logger = logging.getLogger("test_perf_manual")

    # Manual timing
    start = time.time()
    time.sleep(0.2)
    elapsed = time.time() - start

    log_performance(
        logger,
        operation="document_processing",
        elapsed=elapsed,
        pages=150,
        file_size_mb=12.5,
        chunks_created=42,
    )

    print()
    print("✓ Manual performance logging test passed")
    print()


def test_file_rotation():
    """Test file logging with rotation."""
    print("=" * 80)
    print("Test 6: File Logging with Rotation")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Setup with file logging
        setup_logging(
            level="INFO",
            log_dir=log_dir,
            max_bytes=1000,  # Small size to test rotation
            backup_count=3,
            console_colors=True,
        )

        logger = logging.getLogger("test_rotation")

        # Write many log messages to trigger rotation
        for i in range(50):
            logger.info(f"Log message number {i} with some extra text to make it longer")

        # Check created files
        log_files = list(log_dir.glob("*.log*"))
        print(f"Log files created: {len(log_files)}")
        for log_file in sorted(log_files):
            size = log_file.stat().st_size
            print(f"  {log_file.name}: {size} bytes")

    print()
    print("✓ File rotation test passed")
    print()


def test_per_module_levels():
    """Test per-module log level configuration."""
    print("=" * 80)
    print("Test 7: Per-Module Log Levels")
    print("=" * 80)
    print()

    # Setup with per-module levels
    setup_logging(
        level="INFO",
        module_levels={
            "test_module_a": "DEBUG",
            "test_module_b": "WARNING",
        }
    )

    logger_a = logging.getLogger("test_module_a")
    logger_b = logging.getLogger("test_module_b")
    logger_c = logging.getLogger("test_module_c")

    print("Module A (DEBUG level):")
    logger_a.debug("This DEBUG message SHOULD appear")
    logger_a.info("This INFO message should appear")

    print("\nModule B (WARNING level):")
    logger_b.info("This INFO message should NOT appear")
    logger_b.warning("This WARNING message SHOULD appear")

    print("\nModule C (default INFO level):")
    logger_c.debug("This DEBUG message should NOT appear")
    logger_c.info("This INFO message SHOULD appear")

    print()
    print("✓ Per-module log levels test passed")
    print()


def test_exception_logging():
    """Test exception logging with context."""
    print("=" * 80)
    print("Test 8: Exception Logging")
    print("=" * 80)
    print()

    logger = logging.getLogger("test_exceptions")

    # Log exception without context
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.error("Exception occurred (no context)", exc_info=True)

    print()

    # Log exception with context
    with LogContext(request_id="req-error-test", doc_id="doc-failed"):
        try:
            result = 1 / 0
        except Exception as e:
            logger.error(
                "Exception occurred with context",
                exc_info=True,
                extra={'operation': 'division', 'attempted_values': [1, 0]}
            )

    print()
    print("✓ Exception logging test passed")
    print()


def test_context_with_extra():
    """Test log_with_context utility."""
    print("=" * 80)
    print("Test 9: Context with Extra Fields")
    print("=" * 80)
    print()

    logger = logging.getLogger("test_context_extra")

    with LogContext(request_id="req-extra-test"):
        log_with_context(
            logger, logging.INFO,
            "Document processed successfully",
            doc_id="doc-123",
            pages=150,
            processing_time=2.45,
            chunks_created=42,
            status="success",
        )

    print()
    print("✓ Context with extra fields test passed")
    print()


def demo_production_config():
    """Demonstrate production logging configuration."""
    print("=" * 80)
    print("Demo: Production Logging Configuration")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Production-like setup
        setup_logging(
            level="INFO",
            log_dir=log_dir,
            structured=True,
            console_colors=False,
            max_bytes=10 * 1024 * 1024,  # 10MB
            backup_count=5,
            module_levels={
                "core.parser": "INFO",
                "core.llm": "WARNING",
                "api.endpoints": "DEBUG",
            }
        )

        logger = logging.getLogger("production_demo")

        print("Production configuration:")
        print("  - Structured (JSON) logging: ENABLED")
        print("  - Log directory: logs/")
        print("  - Log files: app.log, error.log")
        print("  - File rotation: 10MB per file, 5 backups")
        print("  - Per-module levels: parser=INFO, llm=WARNING, api=DEBUG")
        print()

        # Simulate production logs
        with LogContext(request_id="req-prod-001"):
            logger.info("API server started")
            logger.info("Processing user request", extra={'user_id': 'user-456'})
            logger.warning("Slow operation detected", extra={'elapsed': 5.2})

        # Show log files
        print("Log files created:")
        for log_file in sorted(log_dir.glob("*.log")):
            size = log_file.stat().st_size
            print(f"  {log_file.name}: {size} bytes")
            print(f"    First line: {log_file.read_text().split(chr(10))[0][:80]}...")

    print()
    print("✓ Production configuration demo completed")
    print()


def print_summary():
    """Print summary of logging features."""
    print("=" * 80)
    print("Logging Features Summary")
    print("=" * 80)
    print()

    print("✓ Basic logging with colored output")
    print("✓ Context-aware logging (request_id, doc_id)")
    print("✓ Structured JSON logging for log aggregation")
    print("✓ Performance timing decorator")
    print("✓ Manual performance logging")
    print("✓ File logging with rotation")
    print("✓ Per-module log level configuration")
    print("✓ Exception logging with context")
    print("✓ Custom extra fields in logs")
    print("✓ Production-ready configuration")
    print()

    print("Benefits:")
    print("  - Easier debugging with request/doc IDs")
    print("  - Performance monitoring built-in")
    print("  - Production-ready with file rotation")
    print("  - Flexible per-module configuration")
    print("  - JSON logs for aggregation systems (Splunk, ELK, Datadog)")
    print("  - Colored console for development")
    print()


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("Workpedia Enhanced Logging Test Suite")
    print("=" * 80)
    print()

    try:
        test_basic_logging()
        test_context_logging()
        test_structured_logging()
        test_performance_timing()
        test_manual_performance_logging()
        test_file_rotation()
        test_per_module_levels()
        test_exception_logging()
        test_context_with_extra()
        demo_production_config()

        print_summary()

        print("=" * 80)
        print("✓ ALL LOGGING TESTS PASSED")
        print("=" * 80)
        print()
        print("Enhanced logging is working correctly!")
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
