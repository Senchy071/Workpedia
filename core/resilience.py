"""Resilience patterns for Ollama connection handling.

This module provides production-ready resilience patterns:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Timeout management
- Graceful degradation

These patterns ensure the system remains responsive even when Ollama is slow or unavailable.
"""

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ============================================================================
# Retry Logic with Exponential Backoff
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


def exponential_backoff(
    attempt: int,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential growth
        max_delay: Maximum delay in seconds
        jitter: Add random jitter to prevent thundering herd

    Returns:
        Delay in seconds before next retry

    Example:
        >>> exponential_backoff(0)  # First retry
        1.0
        >>> exponential_backoff(1)  # Second retry
        2.0
        >>> exponential_backoff(2)  # Third retry
        4.0
    """
    delay = min(initial_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        import random
        # Add ±25% jitter
        jitter_amount = delay * 0.25
        delay = delay + random.uniform(-jitter_amount, jitter_amount)

    return max(0, delay)


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        config: Retry configuration (uses defaults if None)
        retryable_exceptions: Tuple of exceptions to retry on
        on_retry: Callback function called on each retry (exc, attempt, delay)

    Usage:
        @retry_with_backoff(
            config=RetryConfig(max_retries=3, initial_delay=1.0),
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
        def call_ollama():
            return ollama_client.generate("test")
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt >= config.max_retries:
                        logger.error(
                            f"{func.__name__} failed after {config.max_retries} retries: {e}"
                        )
                        raise

                    # Calculate backoff delay
                    delay = exponential_backoff(
                        attempt,
                        initial_delay=config.initial_delay,
                        exponential_base=config.exponential_base,
                        max_delay=config.max_delay,
                        jitter=config.jitter,
                    )

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{config.max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt, delay)

                    # Wait before retrying
                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Open circuit after N failures
    recovery_timeout: float = 60.0  # seconds to wait before testing recovery
    success_threshold: int = 2  # Successful calls to close circuit from half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests immediately fail
    - HALF_OPEN: Testing recovery, limited requests allowed

    Usage:
        breaker = CircuitBreaker(
            name="ollama",
            config=CircuitBreakerConfig(failure_threshold=5)
        )

        @breaker.call
        def call_ollama():
            return ollama_client.generate("test")
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name (for logging)
            config: Circuit breaker configuration
            on_state_change: Callback for state changes (old_state, new_state)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        self.stats = CircuitBreakerStats()
        self._lock = threading.Lock()
        self._half_open_calls = 0

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout}s"
        )

    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit breaker state."""
        old_state = self.stats.state
        if old_state != new_state:
            self.stats.state = new_state
            self.stats.last_state_change = datetime.now()

            logger.warning(
                f"CircuitBreaker '{self.name}' state change: {old_state.value} → {new_state.value}"
            )

            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN."""
        if self.stats.last_failure_time is None:
            return False

        elapsed = (datetime.now() - self.stats.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout

    def call(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.

        Args:
            func: Function to protect

        Returns:
            Wrapped function

        Raises:
            CircuitBreakerError: If circuit is open
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with self._lock:
                self.stats.total_calls += 1

                # Check circuit state
                if self.stats.state == CircuitState.OPEN:
                    if self._should_attempt_reset():
                        logger.info(f"CircuitBreaker '{self.name}' attempting recovery")
                        self._change_state(CircuitState.HALF_OPEN)
                        self._half_open_calls = 0
                    else:
                        logger.warning(
                            f"CircuitBreaker '{self.name}' is OPEN, rejecting call"
                        )
                        raise CircuitBreakerError(
                            f"Circuit breaker '{self.name}' is open. "
                            f"Service is unavailable. Try again later."
                        )

                # Limit concurrent calls in HALF_OPEN state
                if self.stats.state == CircuitState.HALF_OPEN:
                    if self._half_open_calls >= self.config.half_open_max_calls:
                        raise CircuitBreakerError(
                            f"Circuit breaker '{self.name}' is testing recovery. "
                            f"Too many concurrent requests."
                        )
                    self._half_open_calls += 1

            # Execute the function
            try:
                result = func(*args, **kwargs)

                # Success - update state
                with self._lock:
                    self._on_success()

                return result

            except Exception:
                # Failure - update state
                with self._lock:
                    self._on_failure()
                raise

        return wrapper

    def _on_success(self) -> None:
        """Handle successful call."""
        self.stats.total_successes += 1
        self.stats.success_count += 1
        self.stats.failure_count = 0

        if self.stats.state == CircuitState.HALF_OPEN:
            if self.stats.success_count >= self.config.success_threshold:
                logger.info(
                    f"CircuitBreaker '{self.name}' recovered after "
                    f"{self.stats.success_count} successful calls"
                )
                self._change_state(CircuitState.CLOSED)
                self.stats.success_count = 0
                self._half_open_calls = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.stats.total_failures += 1
        self.stats.failure_count += 1
        self.stats.last_failure_time = datetime.now()

        if self.stats.state == CircuitState.HALF_OPEN:
            # Failed during recovery - back to OPEN
            logger.warning(
                f"CircuitBreaker '{self.name}' failed during recovery, "
                f"returning to OPEN state"
            )
            self._change_state(CircuitState.OPEN)
            self.stats.success_count = 0
            self._half_open_calls = 0

        elif self.stats.state == CircuitState.CLOSED:
            if self.stats.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"CircuitBreaker '{self.name}' threshold reached "
                    f"({self.stats.failure_count} failures), opening circuit"
                )
                self._change_state(CircuitState.OPEN)
                self.stats.success_count = 0

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"CircuitBreaker '{self.name}' manually reset")
            self._change_state(CircuitState.CLOSED)
            self.stats.failure_count = 0
            self.stats.success_count = 0
            self._half_open_calls = 0

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.stats.state.value,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "total_calls": self.stats.total_calls,
                "total_failures": self.stats.total_failures,
                "total_successes": self.stats.total_successes,
                "last_failure_time": (
                    self.stats.last_failure_time.isoformat()
                    if self.stats.last_failure_time
                    else None
                ),
                "last_state_change": self.stats.last_state_change.isoformat(),
            }


# ============================================================================
# Timeout Management
# ============================================================================

class TimeoutConfig:
    """Timeout configuration for different operations."""

    def __init__(
        self,
        default: float = 120.0,
        health_check: float = 5.0,
        list_models: float = 10.0,
        generate: float = 120.0,
        generate_stream: float = 180.0,
    ):
        """
        Initialize timeout configuration.

        Args:
            default: Default timeout for all operations
            health_check: Timeout for health check operations
            list_models: Timeout for listing models
            generate: Timeout for non-streaming generation
            generate_stream: Timeout for streaming generation
        """
        self.default = default
        self.health_check = health_check
        self.list_models = list_models
        self.generate = generate
        self.generate_stream = generate_stream

    def get_timeout(self, operation: str) -> float:
        """
        Get timeout for specific operation.

        Args:
            operation: Operation name (health_check, list_models, generate, etc.)

        Returns:
            Timeout in seconds
        """
        return getattr(self, operation, self.default)


# ============================================================================
# Helper Functions
# ============================================================================

def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable.

    Args:
        exception: Exception to check

    Returns:
        True if exception is retryable (transient failure)
    """
    # Import here to avoid circular dependency
    from core.exceptions import (
        OllamaConnectionError,
        OllamaTimeoutError,
    )

    retryable_types = (
        OllamaConnectionError,
        OllamaTimeoutError,
        ConnectionError,
        TimeoutError,
    )

    return isinstance(exception, retryable_types)


def should_use_circuit_breaker(operation: str) -> bool:
    """
    Determine if circuit breaker should be used for operation.

    Args:
        operation: Operation name

    Returns:
        True if circuit breaker should protect this operation
    """
    # Use circuit breaker for all external calls to Ollama
    protected_operations = [
        "generate",
        "generate_stream",
        "list_models",
        "chat",
    ]

    return operation in protected_operations
