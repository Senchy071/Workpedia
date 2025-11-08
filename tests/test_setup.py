"""Verify Phase 1 setup is complete and working."""
import subprocess
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import docling
        import chromadb
        import sentence_transformers
        import torch
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_ollama():
    """Test Ollama availability."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "mistral" in result.stdout:
            print("✓ Ollama with Mistral model is available")
            return True
        else:
            print("✗ Mistral model not found in Ollama")
            return False
    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        return False

def test_project_structure():
    """Verify project structure exists."""
    required_dirs = [
        "config", "core", "processors",
        "storage", "api", "tests", "data/input", "data/output"
    ]
    project_root = Path(__file__).parent.parent

    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"✗ {dir_name}/ missing")
            all_exist = False
    return all_exist

def test_config():
    """Test configuration file."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.config import (
            OLLAMA_MODEL, EMBEDDING_MODEL,
            CHROMA_COLLECTION_NAME
        )
        print("✓ Configuration file loaded successfully")
        print(f"  - Ollama model: {OLLAMA_MODEL}")
        print(f"  - Embedding model: {EMBEDDING_MODEL}")
        print(f"  - ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("WORKPEDIA PHASE 1 VERIFICATION")
    print("=" * 60)

    tests = [
        ("Project Structure", test_project_structure),
        ("Package Imports", test_imports),
        ("Ollama Setup", test_ollama),
        ("Configuration", test_config),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL PHASE 1 TESTS PASSED")
        print("Ready to proceed to Phase 2")
    else:
        print("✗ SOME TESTS FAILED")
        print("Fix issues before proceeding")
    print("=" * 60)
