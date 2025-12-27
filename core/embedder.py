"""Embedder using sentence-transformers for semantic vector generation."""

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from config.config import (
    CACHE_DIR,
    CACHE_EMBEDDING_TTL,
    CACHE_ENABLED,
    EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    get_embedding_dimension,
)

if TYPE_CHECKING:
    from core.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedder using sentence-transformers for semantic representations.

    Features:
    - Swappable embedding models (see config.EMBEDDING_MODELS)
    - Automatic dimension detection for known models
    - Batch processing for efficiency
    - GPU acceleration when available
    - Normalized vectors for cosine similarity

    Supported models (configure in config/config.py):
    - sentence-transformers/all-mpnet-base-v2 (768-dim, balanced, default)
    - sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)
    - sentence-transformers/all-MiniLM-L12-v2 (384-dim, quality)
    - sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-dim, multilingual)
    - BAAI/bge-small-en-v1.5 (384-dim)
    - BAAI/bge-base-en-v1.5 (768-dim)
    - BAAI/bge-large-en-v1.5 (1024-dim)
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
        enable_cache: bool = CACHE_ENABLED,
        cache_ttl: int = CACHE_EMBEDDING_TTL,
    ):
        """
        Initialize embedder with sentence-transformers model.

        Args:
            model_name: HuggingFace model name (default from config.EMBEDDING_MODEL)
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            enable_cache: Whether to enable caching for query embeddings
            cache_ttl: Cache TTL in seconds (default: 1 hour)

        Note:
            Changing embedding models requires re-indexing all documents!
            Different models have different dimensions.
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._device = device
        self._dimension = None  # Will be set on first model load or from registry

        # Get dimension from registry if available
        known_dim = get_embedding_dimension(model_name)
        if known_dim:
            self._dimension = known_dim
            logger.debug(f"Model {model_name} dimension from registry: {known_dim}")
        else:
            logger.warning(
                f"Model '{model_name}' not in EMBEDDING_MODELS registry. "
                f"Dimension will be detected at runtime."
            )

        # Initialize cache
        self._cache = None
        if enable_cache:
            from core.caching import EmbeddingCache

            cache_dir = CACHE_DIR / "embeddings"
            self._cache = EmbeddingCache(
                cache_dir=cache_dir,
                ttl=cache_ttl,
                enabled=True,
            )

        logger.info(
            f"Embedder initialized: model={model_name}, "
            f"dim={self._dimension or 'auto'}, normalize={normalize}, cache={enable_cache}"
        )

    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)

            # Detect dimension from loaded model if not known
            if self._dimension is None:
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Detected dimension {self._dimension} for model {self.model_name}"
                )

            logger.info(
                f"Model loaded: {self.model_name} on {self._model.device}, "
                f"dimension={self._dimension}"
            )
        return self._model

    @property
    def dimension(self) -> int:
        """
        Get embedding dimension.

        Returns dimension from registry if known, otherwise loads model to detect.
        """
        if self._dimension is not None:
            return self._dimension

        # Force model load to detect dimension
        _ = self.model
        return self._dimension

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar for large batches

        Returns:
            numpy array of embeddings, shape (n_texts, embedding_dim)
            For single text input, returns shape (embedding_dim,)
        """
        single_input = isinstance(texts, str)
        original_text = texts if single_input else None

        # Check cache for single text queries (typical for user queries)
        if single_input and self._cache is not None:
            cached = self._cache.get(texts)
            if cached is not None:
                return cached

        if single_input:
            texts = [texts]

        if not texts:
            return np.array([])

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        # Cache single text query embeddings
        if single_input and self._cache is not None and original_text:
            self._cache.set(original_text, embeddings[0])

        if single_input:
            return embeddings[0]

        return embeddings

    def embed_chunks(
        self,
        chunks: List["Chunk"],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for Chunk objects.

        Args:
            chunks: List of Chunk objects from SemanticChunker
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of embedding vectors (one per chunk)
        """
        if not chunks:
            return []

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        logger.info(f"Embedding {len(texts)} chunks...")

        # Generate embeddings
        embeddings = self.embed(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logger.info(f"Generated {len(embeddings)} embeddings")

        return list(embeddings)

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        # If embeddings are normalized, dot product = cosine similarity
        if self.normalize:
            return float(np.dot(embedding1, embedding2))

        # Otherwise compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hits, misses, etc.)
        """
        if self._cache is None:
            return {"enabled": False}
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Embedding cache cleared")

    def cleanup(self):
        """Release model resources."""
        if self._model is not None:
            logger.info("Cleaning up embedder resources")
            del self._model
            self._model = None


# Convenience functions
def embed_text(
    text: Union[str, List[str]],
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Quick function to embed text without managing Embedder instance.

    Args:
        text: Text or list of texts to embed
        model_name: Model to use

    Returns:
        Embedding vector(s)
    """
    embedder = Embedder(model_name=model_name)
    return embedder.embed(text)


def list_available_models() -> dict:
    """
    List all available embedding models with their dimensions.

    Returns:
        Dictionary mapping model name to dimension
    """
    return EMBEDDING_MODELS.copy()


def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific embedding model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model info (name, dimension, known)
    """
    dim = get_embedding_dimension(model_name)
    return {
        "name": model_name,
        "dimension": dim,
        "known": dim is not None,
        "is_default": model_name == EMBEDDING_MODEL,
    }
