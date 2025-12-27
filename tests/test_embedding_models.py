"""Tests for custom embedding models feature."""

from config.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    get_embedding_dimension,
)
from core.embedder import Embedder, get_model_info, list_available_models


class TestEmbeddingModelRegistry:
    """Tests for the embedding model registry in config."""

    def test_registry_has_default_model(self):
        """Test that registry contains the default model."""
        assert EMBEDDING_MODEL in EMBEDDING_MODELS

    def test_registry_has_expected_models(self):
        """Test that registry contains commonly used models."""
        expected_models = [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-base-en-v1.5",
        ]
        for model in expected_models:
            assert model in EMBEDDING_MODELS, f"Missing model: {model}"

    def test_registry_has_short_aliases(self):
        """Test that short aliases are available."""
        assert "all-mpnet-base-v2" in EMBEDDING_MODELS
        assert "all-MiniLM-L6-v2" in EMBEDDING_MODELS

    def test_registry_dimensions_are_valid(self):
        """Test that all dimensions are positive integers."""
        for model, dim in EMBEDDING_MODELS.items():
            assert isinstance(dim, int), f"Dimension for {model} should be int"
            assert dim > 0, f"Dimension for {model} should be positive"

    def test_default_dimension_matches_model(self):
        """Test that EMBEDDING_DIM matches the default model."""
        expected_dim = EMBEDDING_MODELS[EMBEDDING_MODEL]
        assert EMBEDDING_DIM == expected_dim


class TestGetEmbeddingDimension:
    """Tests for the get_embedding_dimension function."""

    def test_known_model_returns_dimension(self):
        """Test that known models return their dimension."""
        dim = get_embedding_dimension("sentence-transformers/all-mpnet-base-v2")
        assert dim == 768

    def test_minilm_model_dimension(self):
        """Test MiniLM model returns correct dimension."""
        dim = get_embedding_dimension("sentence-transformers/all-MiniLM-L6-v2")
        assert dim == 384

    def test_bge_large_model_dimension(self):
        """Test BGE large model returns correct dimension."""
        dim = get_embedding_dimension("BAAI/bge-large-en-v1.5")
        assert dim == 1024

    def test_unknown_model_returns_none(self):
        """Test that unknown models return None."""
        dim = get_embedding_dimension("unknown/fake-model")
        assert dim is None

    def test_short_alias_returns_dimension(self):
        """Test that short aliases work."""
        dim = get_embedding_dimension("all-mpnet-base-v2")
        assert dim == 768


class TestListAvailableModels:
    """Tests for the list_available_models function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        models = list_available_models()
        assert isinstance(models, dict)

    def test_returns_copy_not_original(self):
        """Test that function returns a copy, not the original."""
        models = list_available_models()
        models["fake/model"] = 123
        # Original should not be modified
        assert "fake/model" not in EMBEDDING_MODELS

    def test_contains_all_models(self):
        """Test that all models are included."""
        models = list_available_models()
        assert len(models) == len(EMBEDDING_MODELS)
        for model, dim in EMBEDDING_MODELS.items():
            assert model in models
            assert models[model] == dim


class TestGetModelInfo:
    """Tests for the get_model_info function."""

    def test_known_model_info(self):
        """Test info for a known model."""
        info = get_model_info("sentence-transformers/all-mpnet-base-v2")
        assert info["name"] == "sentence-transformers/all-mpnet-base-v2"
        assert info["dimension"] == 768
        assert info["known"] is True

    def test_default_model_is_default(self):
        """Test that default model shows is_default=True."""
        info = get_model_info(EMBEDDING_MODEL)
        assert info["is_default"] is True

    def test_non_default_model(self):
        """Test that non-default model shows is_default=False."""
        # Find a non-default model
        non_default = None
        for model in EMBEDDING_MODELS:
            if model != EMBEDDING_MODEL:
                non_default = model
                break
        if non_default:
            info = get_model_info(non_default)
            assert info["is_default"] is False

    def test_unknown_model_info(self):
        """Test info for an unknown model."""
        info = get_model_info("unknown/fake-model")
        assert info["name"] == "unknown/fake-model"
        assert info["dimension"] is None
        assert info["known"] is False
        assert info["is_default"] is False


class TestEmbedderWithDifferentModels:
    """Tests for Embedder with different models."""

    def test_default_model_initialization(self):
        """Test embedder with default model."""
        embedder = Embedder(enable_cache=False)
        assert embedder.model_name == EMBEDDING_MODEL
        assert embedder.dimension == 768

    def test_dimension_from_registry(self):
        """Test that dimension comes from registry before model load."""
        embedder = Embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            enable_cache=False,
        )
        # Dimension should be available before model is loaded
        assert embedder._model is None
        assert embedder.dimension == 384

    def test_embedder_with_short_alias(self):
        """Test embedder with short model alias."""
        embedder = Embedder(model_name="all-mpnet-base-v2", enable_cache=False)
        assert embedder.dimension == 768

    def test_unknown_model_dimension_none_initially(self):
        """Test that unknown model has no dimension until loaded."""
        embedder = Embedder(
            model_name="unknown/test-model",
            enable_cache=False,
        )
        # Dimension should be None initially
        assert embedder._dimension is None

    def test_embedder_embed_returns_correct_shape(self):
        """Test that embedding has correct shape for default model."""
        embedder = Embedder(enable_cache=False)
        embedding = embedder.embed("Test sentence")
        assert embedding.shape == (768,)

    def test_embedder_cleanup(self):
        """Test that cleanup releases resources."""
        embedder = Embedder(enable_cache=False)
        _ = embedder.embed("Load the model")
        assert embedder._model is not None

        embedder.cleanup()
        assert embedder._model is None


class TestEmbedderDimensionConsistency:
    """Tests for dimension consistency in embedder."""

    def test_dimension_matches_embedding_size(self):
        """Test that dimension property matches actual embedding size."""
        embedder = Embedder(enable_cache=False)
        embedding = embedder.embed("Test text")

        assert embedder.dimension == embedding.shape[0]

    def test_batch_embeddings_have_correct_dimension(self):
        """Test that batch embeddings have correct dimension."""
        embedder = Embedder(enable_cache=False)
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (3, embedder.dimension)


class TestModelInfoIntegration:
    """Integration tests for model info and embedder."""

    def test_model_info_matches_embedder(self):
        """Test that model info matches actual embedder behavior."""
        model_name = "sentence-transformers/all-mpnet-base-v2"
        info = get_model_info(model_name)

        embedder = Embedder(model_name=model_name, enable_cache=False)
        embedding = embedder.embed("Test")

        assert embedding.shape[0] == info["dimension"]

    def test_all_registry_models_have_valid_dimensions(self):
        """Test that all models in registry have valid dimension info."""
        for model_name in EMBEDDING_MODELS:
            info = get_model_info(model_name)
            assert info["known"] is True
            assert info["dimension"] is not None
            assert info["dimension"] > 0
