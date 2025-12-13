"""RAG Query Engine combining retrieval and generation."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from core.embedder import Embedder
from core.exceptions import InvalidParameterError, InvalidQueryError
from core.llm import RAG_SYSTEM_PROMPT, OllamaClient, format_rag_prompt
from core.validators import validate_document_id, validate_query, validate_query_params
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """
    Result of a RAG query.

    Attributes:
        question: Original question
        answer: Generated answer
        sources: List of source chunks used
        metadata: Additional query metadata
    """

    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class QueryEngine:
    """
    RAG Query Engine for document question-answering.

    Combines:
    - Vector similarity search for retrieval
    - Ollama LLM for answer generation
    - Source citation and formatting

    Usage:
        engine = QueryEngine()
        result = engine.query("What is the main finding?")
        print(result.answer)
        for source in result.sources:
            print(f"- {source['metadata']['filename']}")
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        llm: Optional[OllamaClient] = None,
        n_results: int = 5,
        temperature: float = 0.7,
    ):
        """
        Initialize query engine.

        Args:
            vector_store: VectorStore instance (creates default if None)
            embedder: Embedder instance (creates default if None)
            llm: OllamaClient instance (creates default if None)
            n_results: Number of chunks to retrieve
            temperature: LLM sampling temperature
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.llm = llm or OllamaClient()
        self.n_results = n_results
        self.temperature = temperature

        logger.info(
            f"QueryEngine initialized: n_results={n_results}, " f"temperature={temperature}"
        )

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        doc_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> QueryResult:
        """
        Query the document collection.

        Args:
            question: User's question
            n_results: Number of chunks to retrieve (overrides default)
            doc_id: Filter to specific document
            temperature: LLM temperature (overrides default)
            max_tokens: Maximum tokens for response

        Returns:
            QueryResult with answer and sources

        Raises:
            InvalidQueryError: If query is invalid
            InvalidParameterError: If parameters are invalid
        """
        # Validate question
        try:
            question = validate_query(question, min_length=1, max_length=5000)
        except ValueError as e:
            raise InvalidQueryError(str(e)) from e

        # Validate parameters
        try:
            params = validate_query_params(
                n_results=n_results,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as e:
            raise InvalidParameterError(str(e)) from e

        # Validate doc_id if provided
        if doc_id is not None:
            try:
                doc_id = validate_document_id(doc_id)
            except ValueError as e:
                raise InvalidParameterError(f"Invalid doc_id: {e}") from e

        # Use validated or default values
        n_results = params.get("n_results", n_results or self.n_results)
        temperature = params.get("temperature", temperature or self.temperature)
        max_tokens = params.get("max_tokens", max_tokens)

        logger.info(f"Query: '{question[:50]}...' (n_results={n_results})")

        # Step 1: Retrieve relevant chunks
        chunks = self._retrieve(question, n_results, doc_id)

        if not chunks:
            logger.warning("No relevant chunks found")
            no_info_msg = (
                "I couldn't find any relevant information in the "
                "documents to answer your question."
            )
            return QueryResult(
                question=question,
                answer=no_info_msg,
                sources=[],
                metadata={"chunks_retrieved": 0},
            )

        # Step 2: Generate answer
        answer = self._generate(question, chunks, temperature, max_tokens)

        # Step 3: Format result
        sources = [
            {
                "content": c["content"][:500] + "..." if len(c["content"]) > 500 else c["content"],
                "metadata": c["metadata"],
                "similarity": c["similarity"],
            }
            for c in chunks
        ]

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            metadata={
                "chunks_retrieved": len(chunks),
                "temperature": temperature,
                "model": self.llm.model,
            },
        )

    def query_stream(
        self,
        question: str,
        n_results: Optional[int] = None,
        doc_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, QueryResult]:
        """
        Query with streaming response.

        Yields answer tokens as they're generated, then returns full QueryResult.

        Args:
            question: User's question
            n_results: Number of chunks to retrieve
            doc_id: Filter to specific document
            temperature: LLM temperature
            max_tokens: Maximum tokens for response

        Yields:
            Answer tokens as strings

        Returns:
            Complete QueryResult (after iteration)

        Raises:
            InvalidQueryError: If query is invalid
            InvalidParameterError: If parameters are invalid
        """
        # Validate question
        try:
            question = validate_query(question, min_length=1, max_length=5000)
        except ValueError as e:
            raise InvalidQueryError(str(e)) from e

        # Validate parameters
        try:
            params = validate_query_params(
                n_results=n_results,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as e:
            raise InvalidParameterError(str(e)) from e

        # Validate doc_id if provided
        if doc_id is not None:
            try:
                doc_id = validate_document_id(doc_id)
            except ValueError as e:
                raise InvalidParameterError(f"Invalid doc_id: {e}") from e

        # Use validated or default values
        n_results = params.get("n_results", n_results or self.n_results)
        temperature = params.get("temperature", temperature or self.temperature)
        max_tokens = params.get("max_tokens", max_tokens)

        # Retrieve chunks
        chunks = self._retrieve(question, n_results, doc_id)

        if not chunks:
            no_info_msg = (
                "I couldn't find any relevant information in the "
                "documents to answer your question."
            )
            yield no_info_msg
            return QueryResult(
                question=question,
                answer=no_info_msg,
                sources=[],
                metadata={"chunks_retrieved": 0},
            )

        # Stream generation
        prompt = format_rag_prompt(question, chunks)
        answer_parts = []

        for token in self.llm.generate(
            prompt=prompt,
            system=RAG_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            answer_parts.append(token)
            yield token

        answer = "".join(answer_parts)

        sources = [
            {
                "content": c["content"][:500] + "..." if len(c["content"]) > 500 else c["content"],
                "metadata": c["metadata"],
                "similarity": c["similarity"],
            }
            for c in chunks
        ]

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            metadata={
                "chunks_retrieved": len(chunks),
                "temperature": temperature,
                "model": self.llm.model,
            },
        )

    def _retrieve(
        self,
        question: str,
        n_results: int,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks."""
        # Generate query embedding
        query_embedding = self.embedder.embed(question)

        # Search vector store
        where = {"doc_id": doc_id} if doc_id else None
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        # Format results
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append(
                {
                    "chunk_id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "similarity": 1 - results["distances"][i],  # Convert distance to similarity
                }
            )

        return chunks

    def _generate(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate answer from chunks."""
        prompt = format_rag_prompt(question, chunks)

        answer = self.llm.generate(
            prompt=prompt,
            system=RAG_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        return answer

    def get_similar_chunks(
        self,
        question: str,
        n_results: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get similar chunks without generating an answer.

        Useful for debugging or showing sources before generation.

        Args:
            question: Query text
            n_results: Number of chunks to retrieve
            doc_id: Filter to specific document

        Returns:
            List of similar chunks with similarity scores

        Raises:
            InvalidQueryError: If query is invalid
            InvalidParameterError: If parameters are invalid
        """
        # Validate question
        try:
            question = validate_query(question, min_length=1, max_length=5000)
        except ValueError as e:
            raise InvalidQueryError(str(e)) from e

        # Validate parameters
        try:
            params = validate_query_params(n_results=n_results)
        except ValueError as e:
            raise InvalidParameterError(str(e)) from e

        # Validate doc_id if provided
        if doc_id is not None:
            try:
                doc_id = validate_document_id(doc_id)
            except ValueError as e:
                raise InvalidParameterError(f"Invalid doc_id: {e}") from e

        # Use validated values
        n_results = params.get("n_results", n_results)

        return self._retrieve(question, n_results, doc_id)

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all components.

        Returns:
            Dictionary with component status
        """
        return {
            "vector_store": {
                "status": "ok",
                "documents": len(self.vector_store.list_documents()),
                "chunks": self.vector_store.count,
            },
            "llm": {
                "status": "ok" if self.llm.is_available() else "unavailable",
                "model": self.llm.model,
                "available_models": self.llm.list_models(),
            },
            "embedder": {
                "status": "ok",
                "model": self.embedder.model_name,
                "dimension": self.embedder.dimension,
            },
        }


# Convenience function for quick queries
def ask(
    question: str,
    n_results: int = 5,
    doc_id: Optional[str] = None,
) -> QueryResult:
    """
    Quick function to query documents.

    Args:
        question: Your question
        n_results: Number of chunks to retrieve
        doc_id: Optional document filter

    Returns:
        QueryResult with answer and sources
    """
    engine = QueryEngine(n_results=n_results)
    return engine.query(question, doc_id=doc_id)
