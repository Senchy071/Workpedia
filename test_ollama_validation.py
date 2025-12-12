#!/usr/bin/env python3
"""Test script for Ollama startup validation.

This script demonstrates the new startup validation features added in improvement #1.
Run this to verify that Ollama connectivity checks work correctly.
"""

import logging
import sys
from core.llm import OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_health_check():
    """Test the health check functionality."""
    print("=" * 80)
    print("Ollama Startup Validation Test")
    print("=" * 80)
    print()

    # Test 1: Default configuration (from config.py)
    print("Test 1: Checking default Ollama configuration...")
    print("-" * 80)
    client = OllamaClient()
    health = client.health_check()

    print(f"Server URL:        {health['base_url']}")
    print(f"Server Reachable:  {health['server_reachable']}")
    print(f"Model Name:        {health['model_name']}")
    print(f"Model Available:   {health['model_available']}")
    print(f"Status Message:    {health['message']}")
    print()

    if health['available_models']:
        print(f"Available Models ({len(health['available_models'])}):")
        for model in sorted(health['available_models']):
            marker = "✓" if model == health['model_name'] or model.startswith(f"{health['model_name']}:") else " "
            print(f"  {marker} {model}")
    print()

    # Test 2: Check specific model
    print("Test 2: Testing model availability check...")
    print("-" * 80)
    is_available, message = client.check_model_available()
    print(f"Model Check Result: {is_available}")
    print(f"Message: {message}")
    print()

    # Test 3: Check non-existent model
    print("Test 3: Testing with non-existent model...")
    print("-" * 80)
    fake_client = OllamaClient(model="nonexistent-model")
    is_available, message = fake_client.check_model_available()
    print(f"Model Check Result: {is_available}")
    print(f"Message: {message[:200]}...")
    print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    if health['server_reachable'] and health['model_available']:
        print("✓ SUCCESS: Ollama is properly configured and ready to use")
        print(f"✓ Server: {health['base_url']}")
        print(f"✓ Model:  {health['model_name']}")
        return 0
    elif health['server_reachable'] and not health['model_available']:
        print("⚠ WARNING: Ollama server is running but model is not available")
        print(f"⚠ Run: ollama pull {health['model_name']}")
        return 1
    else:
        print("✗ ERROR: Ollama server is not reachable")
        print("✗ Ensure Ollama is running: ollama serve")
        return 2


def test_startup_scenario():
    """Simulate API/Streamlit startup scenario."""
    print()
    print("=" * 80)
    print("Simulating Application Startup")
    print("=" * 80)
    print()

    try:
        logger.info("Checking Ollama connectivity...")
        ollama_client = OllamaClient()
        health = ollama_client.health_check()

        if not health["server_reachable"]:
            error_msg = (
                f"STARTUP FAILED: {health['message']}\n"
                f"Please ensure Ollama is running:\n"
                f"  1. Start Ollama: 'ollama serve'\n"
                f"  2. Verify it's running: 'ollama list'\n"
                f"  3. Check the URL: {health['base_url']}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not health["model_available"]:
            error_msg = (
                f"STARTUP FAILED: {health['message']}\n"
                f"To fix this:\n"
                f"  1. Pull the model: 'ollama pull {health['model_name']}'\n"
                f"  2. Or use a different model in config/config.py\n"
                f"  Available models: {', '.join(health['available_models']) if health['available_models'] else 'none'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"✓ Ollama connection validated: {health['message']}")
        logger.info(f"✓ Application would start successfully")

        print()
        print("✓ STARTUP VALIDATION PASSED")
        print(f"  - Server: {health['base_url']}")
        print(f"  - Model:  {health['model_name']}")
        print(f"  - Status: {health['message']}")
        return 0

    except Exception as e:
        print()
        print("✗ STARTUP VALIDATION FAILED")
        print(f"  Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = test_health_check()
    exit_code = max(exit_code, test_startup_scenario())
    sys.exit(exit_code)
