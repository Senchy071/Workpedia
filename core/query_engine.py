"""RAG Query Engine combining retrieval and generation."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from config.config import HISTORY_AUTO_SAVE
from core.embedder import Embedder
from core.exceptions import InvalidParameterError, InvalidQueryError
from core.llm import RAG_SYSTEM_PROMPT, OllamaClient, format_rag_prompt
from core.validators import validate_document_id, validate_query, validate_query_params
from storage.vector_store import VectorStore

if TYPE_CHECKING:
    from storage.history_store import HistoryStore

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
        history_store: Optional["HistoryStore"] = None,
        auto_save_history: bool = HISTORY_AUTO_SAVE,
    ):
        """
        Initialize query engine.

        Args:
            vector_store: VectorStore instance (creates default if None)
            embedder: Embedder instance (creates default if None)
            llm: OllamaClient instance (creates default if None)
            n_results: Number of chunks to retrieve
            temperature: LLM sampling temperature
            history_store: HistoryStore instance for saving queries (optional)
            auto_save_history: Automatically save queries to history
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.llm = llm or OllamaClient()
        self.n_results = n_results
        self.temperature = temperature
        self.history_store = history_store
        self.auto_save_history = auto_save_history

        logger.info(
            f"QueryEngine initialized: n_results={n_results}, temperature={temperature}, "
            f"auto_save_history={auto_save_history}"
        )

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        doc_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        save_to_history: Optional[bool] = None,
    ) -> QueryResult:
        """
        Query the document collection.

        Args:
            question: User's question
            n_results: Number of chunks to retrieve (overrides default)
            doc_id: Filter to specific document
            temperature: LLM temperature (overrides default)
            max_tokens: Maximum tokens for response
            session_id: Session identifier for grouping queries
            save_to_history: Save this query to history (None = use auto_save_history setting)

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
            raise InvalidQueryError(query=question, reason=str(e)) from e

        # Validate parameters
        try:
            params = validate_query_params(
                n_results=n_results,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as e:
            raise InvalidParameterError(param_name="params", param_value="", reason=str(e)) from e

        # Validate doc_id if provided
        if doc_id is not None:
            try:
                doc_id = validate_document_id(doc_id)
            except ValueError as e:
                raise InvalidParameterError(param_name="doc_id", param_value=str(doc_id), reason=str(e)) from e

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

        result = QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            metadata={
                "chunks_retrieved": len(chunks),
                "temperature": temperature,
                "model": self.llm.model,
            },
        )

        # Step 4: Save to history if enabled
        should_save = save_to_history if save_to_history is not None else self.auto_save_history
        if should_save and self.history_store:
            try:
                query_id = self.history_store.add_query(
                    question=result.question,
                    answer=result.answer,
                    sources=result.sources,
                    metadata=result.metadata,
                    session_id=session_id,
                )
                result.metadata["query_id"] = query_id
                logger.debug(f"Saved query to history: {query_id}")
            except Exception as e:
                logger.warning(f"Failed to save query to history: {e}")

        return result

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
            raise InvalidQueryError(query=question, reason=str(e)) from e

        # Validate parameters
        try:
            params = validate_query_params(
                n_results=n_results,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as e:
            raise InvalidParameterError(param_name="params", param_value="", reason=str(e)) from e

        # Validate doc_id if provided
        if doc_id is not None:
            try:
                doc_id = validate_document_id(doc_id)
            except ValueError as e:
                raise InvalidParameterError(param_name="doc_id", param_value=str(doc_id), reason=str(e)) from e

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
        chunks = []

        # Check if this is a TOC-related query
        # Semantic search fails for TOC because the detailed TOC chunk embedding
        # is diluted across 88+ chapter names, while short chunks with just
        # "TABLE OF CONTENTS" rank higher. We detect TOC intent and fetch directly.
        toc_keywords = [
            "table of contents",
            "toc",
            "list of chapters",
            "all chapters",
            "main chapters",
            "chapter list",
            "what chapters",
            "show chapters",
            "document outline",
            "document structure",
            "all sections",
            "main sections",
        ]
        question_lower = question.lower()
        is_toc_query = any(kw in question_lower for kw in toc_keywords)

        if is_toc_query:
            # Fetch TOC chunk directly by metadata
            toc_filter = {"chunk_type": "table_of_contents"}
            if doc_id:
                toc_filter["doc_id"] = doc_id

            toc_results = self.vector_store._collection.get(
                where=toc_filter,
                include=["documents", "metadatas"],
            )

            if toc_results["ids"]:
                for chunk_id, content, metadata in zip(
                    toc_results["ids"],
                    toc_results["documents"],
                    toc_results["metadatas"],
                ):
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "content": content,
                            "metadata": metadata,
                            "similarity": 1.0,  # Perfect match for explicit TOC request
                        }
                    )
                logger.info(f"TOC query detected, retrieved {len(chunks)} TOC chunk(s)")

        # Fill remaining slots with semantic search
        remaining_slots = n_results - len(chunks)
        if remaining_slots > 0:
            # Generate query embedding
            query_embedding = self.embedder.embed(question)

            # Search vector store
            where = {"doc_id": doc_id} if doc_id else None
            results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=remaining_slots + len(chunks),  # Get extra to filter duplicates
                where=where,
            )

            # Add semantic results, skipping any already added (TOC chunks)
            existing_ids = {c["chunk_id"] for c in chunks}
            for i in range(len(results["ids"])):
                if results["ids"][i] not in existing_ids and len(chunks) < n_results:
                    chunks.append(
                        {
                            "chunk_id": results["ids"][i],
                            "content": results["documents"][i],
                            "metadata": results["metadatas"][i],
                            "similarity": 1 - results["distances"][i],
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
            raise InvalidQueryError(query=question, reason=str(e)) from e

        # Validate parameters
        try:
            params = validate_query_params(n_results=n_results)
        except ValueError as e:
            raise InvalidParameterError(param_name="params", param_value="", reason=str(e)) from e

        # Validate doc_id if provided
        if doc_id is not None:
            try:
                doc_id = validate_document_id(doc_id)
            except ValueError as e:
                raise InvalidParameterError(param_name="doc_id", param_value=str(doc_id), reason=str(e)) from e

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
