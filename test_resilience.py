#!/usr/bin/env python3
"""Test script for Ollama connection resilience (Improvement #5).

This script tests the resilience patterns:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Timeout management
- Graceful degradation
"""

import time
import pytest
from unittest.mock import Mock, patch
from core.resilience import (
    exponential_backoff,
    RetryConfig,
    retry_with_backoff,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerError,
    TimeoutConfig,
)


def test_exponential_backoff():
    """Test exponential backoff calculation."""
    print("=" * 80)
    print("Test 1: Exponential Backoff")
    print("=" * 80)
    print()

    # Test without jitter
    delay0 = exponential_backoff(0, initial_delay=1.0, jitter=False)
    delay1 = exponential_backoff(1, initial_delay=1.0, jitter=False)
    delay2 = exponential_backoff(2, initial_delay=1.0, jitter=False)

    assert delay0 == 1.0
    assert delay1 == 2.0
    assert delay2 == 4.0
    print(f"✓ Exponential backoff without jitter: {delay0}, {delay1}, {delay2}")

    # Test with max delay
    delay5 = exponential_backoff(5, initial_delay=1.0, max_delay=10.0, jitter=False)
    assert delay5 == 10.0  # Capped at max_delay
    print(f"✓ Max delay cap works: {delay5}s")

    # Test with jitter
    delay_jitter = exponential_backoff(1, initial_delay=1.0, jitter=True)
    assert 1.5 <= delay_jitter <= 2.5  # 2.0 ± 25%
    print(f"✓ Jitter applied: {delay_jitter:.3f}s (expected 1.5-2.5s)")

    print()
    print("✓ Exponential backoff test passed")
    print()


def test_retry_decorator():
    """Test retry decorator with backoff."""
    print("=" * 80)
    print("Test 2: Retry Decorator")
    print("=" * 80)
    print()

    # Test successful retry after failures
    call_count = 0

    @retry_with_backoff(
        config=RetryConfig(max_retries=3, initial_delay=0.1, jitter=False),
        retryable_exceptions=(ValueError,)
    )
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"Attempt {call_count} failed")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count == 3
    print(f"✓ Function succeeded after {call_count} attempts")

    # Test max retries exceeded
    call_count2 = 0

    @retry_with_backoff(
        config=RetryConfig(max_retries=2, initial_delay=0.1, jitter=False),
        retryable_exceptions=(ValueError,)
    )
    def always_fails():
        nonlocal call_count2
        call_count2 += 1
        raise ValueError(f"Attempt {call_count2} failed")

    try:
        always_fails()
        assert False, "Should have raised ValueError"
    except ValueError:
        assert call_count2 == 3  # Initial + 2 retries
        print(f"✓ Max retries exceeded after {call_count2} attempts")

    print()
    print("✓ Retry decorator test passed")
    print()


def test_circuit_breaker_states():
    """Test circuit breaker state transitions."""
    print("=" * 80)
    print("Test 3: Circuit Breaker State Transitions")
    print("=" * 80)
    print()

    breaker = CircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2,
        )
    )

    assert breaker.stats.state == CircuitState.CLOSED
    print("✓ Initial state: CLOSED")

    # Simulate failures to open circuit
    @breaker.call
    def failing_function():
        raise ConnectionError("Service unavailable")

    for i in range(3):
        try:
            failing_function()
        except ConnectionError:
            pass

    assert breaker.stats.state == CircuitState.OPEN
    print("✓ Circuit opened after 3 failures")

    # Verify circuit rejects calls
    try:
        failing_function()
        assert False, "Should have raised CircuitBreakerError"
    except CircuitBreakerError:
        print("✓ Circuit breaker rejects calls when OPEN")

    # Wait for recovery timeout
    time.sleep(1.1)

    # Simulate successful recovery
    @breaker.call
    def successful_function():
        return "success"

    # Circuit should transition to HALF_OPEN
    result1 = successful_function()
    assert result1 == "success"
    assert breaker.stats.state == CircuitState.HALF_OPEN
    print("✓ Circuit transitioned to HALF_OPEN after recovery timeout")

    # Second success should close circuit
    result2 = successful_function()
    assert result2 == "success"
    assert breaker.stats.state == CircuitState.CLOSED
    print("✓ Circuit closed after 2 successful calls in HALF_OPEN")

    print()
    print("✓ Circuit breaker state transitions test passed")
    print()


def test_circuit_breaker_half_open_failure():
    """Test circuit breaker returns to OPEN on HALF_OPEN failure."""
    print("=" * 80)
    print("Test 4: Circuit Breaker HALF_OPEN Failure")
    print("=" * 80)
    print()

    breaker = CircuitBreaker(
        name="test-half-open",
        config=CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            success_threshold=2,
        )
    )

    # Open the circuit
    @breaker.call
    def failing_function():
        raise ConnectionError("Failed")

    for _ in range(2):
        try:
            failing_function()
        except ConnectionError:
            pass

    assert breaker.stats.state == CircuitState.OPEN
    print("✓ Circuit opened")

    # Wait for recovery
    time.sleep(0.6)

    # Attempt recovery but fail
    try:
        failing_function()
        assert False, "Should have raised ConnectionError"
    except ConnectionError:
        pass

    assert breaker.stats.state == CircuitState.OPEN
    print("✓ Circuit returned to OPEN after failure in HALF_OPEN")

    print()
    print("✓ Circuit breaker HALF_OPEN failure test passed")
    print()


def test_circuit_breaker_stats():
    """Test circuit breaker statistics."""
    print("=" * 80)
    print("Test 5: Circuit Breaker Statistics")
    print("=" * 80)
    print()

    breaker = CircuitBreaker(
        name="test-stats",
        config=CircuitBreakerConfig(failure_threshold=5)
    )

    @breaker.call
    def test_function(should_fail=False):
        if should_fail:
            raise ValueError("Failed")
        return "success"

    # Execute some calls
    test_function(should_fail=False)
    test_function(should_fail=False)
    try:
        test_function(should_fail=True)
    except ValueError:
        pass

    stats = breaker.get_stats()
    assert stats['name'] == "test-stats"
    assert stats['state'] == CircuitState.CLOSED.value
    assert stats['total_calls'] == 3
    assert stats['total_successes'] == 2
    assert stats['total_failures'] == 1
    print(f"✓ Circuit breaker stats: {stats}")

    print()
    print("✓ Circuit breaker statistics test passed")
    print()


def test_timeout_config():
    """Test timeout configuration."""
    print("=" * 80)
    print("Test 6: Timeout Configuration")
    print("=" * 80)
    print()

    config = TimeoutConfig(
        default=120.0,
        health_check=5.0,
        list_models=10.0,
        generate=120.0,
        generate_stream=180.0,
    )

    assert config.get_timeout("health_check") == 5.0
    assert config.get_timeout("list_models") == 10.0
    assert config.get_timeout("generate") == 120.0
    assert config.get_timeout("generate_stream") == 180.0
    assert config.get_timeout("unknown_operation") == 120.0  # default
    print("✓ Timeout configuration working correctly")

    print()
    print("✓ Timeout configuration test passed")
    print()


def test_circuit_breaker_reset():
    """Test manual circuit breaker reset."""
    print("=" * 80)
    print("Test 7: Circuit Breaker Manual Reset")
    print("=" * 80)
    print()

    breaker = CircuitBreaker(
        name="test-reset",
        config=CircuitBreakerConfig(failure_threshold=2)
    )

    # Open the circuit
    @breaker.call
    def failing_function():
        raise ConnectionError("Failed")

    for _ in range(2):
        try:
            failing_function()
        except ConnectionError:
            pass

    assert breaker.stats.state == CircuitState.OPEN
    print("✓ Circuit opened")

    # Manually reset
    breaker.reset()
    assert breaker.stats.state == CircuitState.CLOSED
    assert breaker.stats.failure_count == 0
    print("✓ Circuit manually reset to CLOSED")

    print()
    print("✓ Circuit breaker manual reset test passed")
    print()


def test_retry_config():
    """Test retry configuration."""
    print("=" * 80)
    print("Test 8: Retry Configuration")
    print("=" * 80)
    print()

    config = RetryConfig(
        max_retries=5,
        initial_delay=2.0,
        max_delay=60.0,
        exponential_base=3.0,
        jitter=False,
    )

    assert config.max_retries == 5
    assert config.initial_delay == 2.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 3.0
    assert config.jitter == False
    print("✓ Retry configuration created successfully")

    print()
    print("✓ Retry configuration test passed")
    print()


def print_summary():
    """Print summary of resilience features."""
    print("=" * 80)
    print("Resilience Features Summary")
    print("=" * 80)
    print()

    print("✓ Exponential backoff calculation")
    print("✓ Retry decorator with configurable attempts")
    print("✓ Circuit breaker state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)")
    print("✓ Circuit breaker rejects calls when OPEN")
    print("✓ Circuit breaker recovery testing")
    print("✓ Circuit breaker statistics tracking")
    print("✓ Timeout configuration per operation")
    print("✓ Manual circuit breaker reset")
    print()

    print("Production Benefits:")
    print("  - Automatic retry on transient failures")
    print("  - Prevent cascading failures with circuit breaker")
    print("  - Graceful degradation when service unavailable")
    print("  - Configurable timeouts prevent hung requests")
    print("  - Statistics for monitoring and alerting")
    print()


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("Workpedia Resilience Test Suite")
    print("=" * 80)
    print()

    try:
        test_exponential_backoff()
        test_retry_decorator()
        test_circuit_breaker_states()
        test_circuit_breaker_half_open_failure()
        test_circuit_breaker_stats()
        test_timeout_config()
        test_circuit_breaker_reset()
        test_retry_config()

        print_summary()

        print("=" * 80)
        print("✓ ALL RESILIENCE TESTS PASSED")
        print("=" * 80)
        print()
        print("Ollama connection resilience is working correctly!")
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
