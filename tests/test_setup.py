"""Verify Phase 1 setup is complete and working."""
import subprocess
import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    try:
        import chromadb
        import docling
        import sentence_transformers
        import torch
        print("✓ All core packages imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        assert False, f"Import failed: {e}"

def test_ollama():
    """Test Ollama availability."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert "mistral" in result.stdout, "Mistral model not found in Ollama"
        print("✓ Ollama with Mistral model is available")
    except subprocess.TimeoutExpired:
        assert False, "Ollama command timed out"
    except FileNotFoundError:
        assert False, "Ollama command not found - is Ollama installed?"
    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        assert False, f"Ollama test failed: {e}"

def test_project_structure():
    """Verify project structure exists."""
    required_dirs = [
        "config", "core", "processors",
        "storage", "api", "tests", "data/input", "data/output"
    ]
    project_root = Path(__file__).parent.parent

    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"✗ {dir_name}/ missing")
            missing_dirs.append(dir_name)

    assert not missing_dirs, f"Missing directories: {', '.join(missing_dirs)}"

def test_config():
    """Test configuration file."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_MODEL
        print("✓ Configuration file loaded successfully")
        print(f"  - Ollama model: {OLLAMA_MODEL}")
        print(f"  - Embedding model: {EMBEDDING_MODEL}")
        print(f"  - ChromaDB collection: {CHROMA_COLLECTION_NAME}")

        # Verify config values are set
        assert OLLAMA_MODEL, "OLLAMA_MODEL not configured"
        assert EMBEDDING_MODEL, "EMBEDDING_MODEL not configured"
        assert CHROMA_COLLECTION_NAME, "CHROMA_COLLECTION_NAME not configured"
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        assert False, f"Configuration test failed: {e}"

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

    passed = 0
    failed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"✓ ALL {passed} PHASE 1 TESTS PASSED")
        print("Ready to proceed to Phase 2")
    else:
        print(f"✗ {failed} TEST(S) FAILED, {passed} PASSED")
        print("Fix issues before proceeding")
    print("=" * 60)
