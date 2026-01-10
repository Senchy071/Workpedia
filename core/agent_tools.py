"""
Agent tools for Workpedia.

Each tool wraps existing Workpedia functionality and exposes it to the agent.
Tools are defined with JSON schemas for parameter validation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.embedder import Embedder
    from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Definition of a tool available to the agent."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    function: Callable[..., Dict[str, Any]]

    def to_schema(self) -> Dict[str, Any]:
        """Convert to Ollama tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class WorkpediaTools:
    """
    Tools available to the Workpedia agent.

    Wraps VectorStore and other components to provide search,
    document info, and completion signaling capabilities.
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        embedder: "Embedder",
    ):
        """
        Initialize tools with required components.

        Args:
            vector_store: VectorStore instance for document operations
            embedder: Embedder instance for query embedding
        """
        self.vector_store = vector_store
        self.embedder = embedder

        # Build tool registry
        self._tools = self._build_tools()
        self._tool_map = {t.name: t for t in self._tools}

    def _build_tools(self) -> List[Tool]:
        """Build list of available tools."""
        return [
            self._search_documents_tool(),
            self._get_document_summary_tool(),
            self._list_documents_tool(),
            self._read_chunk_tool(),
            self._get_document_info_tool(),
            self._complete_tool(),
        ]

    def get_tools(self) -> List[Tool]:
        """Get list of all available tools."""
        return self._tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM."""
        return [t.to_schema() for t in self._tools]

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tool_map.get(name)

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        try:
            result = tool.function(**arguments)
            logger.debug(f"Tool {name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return {"error": str(e)}

    # Tool definitions

    def _search_documents_tool(self) -> Tool:
        """Search indexed documents for relevant content."""
        return Tool(
            name="search_documents",
            description=(
                "Search indexed documents using semantic similarity. "
                "Returns chunks of text relevant to the query. "
                "Use this to find information in the document collection. "
                "You can search all documents or limit to a specific document by doc_id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Be specific and use relevant keywords. "
                            "Try different phrasings if initial results are poor."
                        ),
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-20). Default 5.",
                        "default": 5,
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Optional. Limit search to a specific document ID.",
                    },
                },
                "required": ["query"],
            },
            function=self._execute_search,
        )

    def _execute_search(
        self,
        query: str,
        n_results: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute document search."""
        # Clamp n_results
        n_results = max(1, min(20, n_results))

        # Build filter
        where = {"doc_id": doc_id} if doc_id else None

        # Search using embedder
        results = self.vector_store.query_text(
            query_text=query,
            embedder=self.embedder,
            n_results=n_results,
            where=where,
        )

        # Format results
        formatted = []
        for i, (chunk_id, content, metadata, distance) in enumerate(
            zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("distances", []),
            )
        ):
            formatted.append({
                "rank": i + 1,
                "chunk_id": chunk_id,
                "content": content[:2000],  # Truncate long content
                "doc_id": metadata.get("doc_id", ""),
                "filename": metadata.get("filename", ""),
                "section": metadata.get("section", ""),
                "page": metadata.get("page_number"),
                "similarity": round(1 - distance, 3),
            })

        return {
            "query": query,
            "results": formatted,
            "total_found": len(formatted),
        }

    def _get_document_summary_tool(self) -> Tool:
        """Get auto-generated summary of a document."""
        return Tool(
            name="get_document_summary",
            description=(
                "Get the auto-generated executive summary of a specific document. "
                "Use this to understand what a document is about before searching within it. "
                "Requires the document ID."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document ID to get summary for.",
                    },
                },
                "required": ["doc_id"],
            },
            function=self._execute_get_summary,
        )

    def _execute_get_summary(self, doc_id: str) -> Dict[str, Any]:
        """Get document summary."""
        summary = self.vector_store.get_document_summary(doc_id)

        if not summary:
            return {
                "error": f"No summary found for document {doc_id}",
                "doc_id": doc_id,
            }

        return {
            "doc_id": doc_id,
            "summary": summary.get("content", ""),
            "metadata": summary.get("metadata", {}),
        }

    def _list_documents_tool(self) -> Tool:
        """List all indexed documents."""
        return Tool(
            name="list_documents",
            description=(
                "List all documents currently indexed in Workpedia. "
                "Use this to see what documents are available to search. "
                "Returns document IDs, filenames, and chunk counts."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            function=self._execute_list_documents,
        )

    def _execute_list_documents(self) -> Dict[str, Any]:
        """List all documents."""
        docs = self.vector_store.list_documents()

        return {
            "documents": [
                {
                    "doc_id": d.get("doc_id", ""),
                    "filename": d.get("filename", "Unknown"),
                    "chunk_count": d.get("chunk_count", 0),
                }
                for d in docs
            ],
            "total": len(docs),
        }

    def _read_chunk_tool(self) -> Tool:
        """Read a specific chunk by ID."""
        return Tool(
            name="read_chunk",
            description=(
                "Read the full content of a specific chunk by its ID. "
                "Use this to get more context from a chunk found in search results. "
                "Chunk IDs are returned by the search_documents tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID to read.",
                    },
                },
                "required": ["chunk_id"],
            },
            function=self._execute_read_chunk,
        )

    def _execute_read_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Read specific chunk by ID."""
        try:
            # Get chunk by ID
            result = self.vector_store._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )

            if not result["ids"]:
                return {"error": f"Chunk {chunk_id} not found"}

            return {
                "chunk_id": chunk_id,
                "content": result["documents"][0] if result.get("documents") else "",
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
            }
        except Exception as e:
            return {"error": f"Failed to read chunk: {e}"}

    def _get_document_info_tool(self) -> Tool:
        """Get detailed information about a document."""
        return Tool(
            name="get_document_info",
            description=(
                "Get detailed information about a specific document including "
                "all its chunks and metadata. Use this to understand document structure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document ID.",
                    },
                },
                "required": ["doc_id"],
            },
            function=self._execute_get_document_info,
        )

    def _execute_get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get document info."""
        doc_data = self.vector_store.get_by_doc_id(doc_id)

        if not doc_data.get("ids"):
            return {"error": f"Document {doc_id} not found"}

        # Gather unique sections
        sections = set()
        chunk_types = {}
        for metadata in doc_data.get("metadatas", []):
            if metadata.get("section"):
                sections.add(metadata["section"])
            chunk_type = metadata.get("chunk_type", "text")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        # Get filename from first chunk
        filename = ""
        if doc_data.get("metadatas"):
            filename = doc_data["metadatas"][0].get("filename", "")

        return {
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": len(doc_data.get("ids", [])),
            "sections": list(sections)[:20],  # Limit to 20 sections
            "chunk_types": chunk_types,
        }

    def _complete_tool(self) -> Tool:
        """Signal task completion."""
        return Tool(
            name="complete",
            description=(
                "Signal that you have completed the task. "
                "Call this when you have gathered enough information to provide a final answer, "
                "or when you have determined the task cannot be completed. "
                "You MUST call this tool to finish - do not just respond with text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer to the user's question.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "Your confidence in the answer. "
                            "high: multiple relevant sources agree. "
                            "medium: some relevant information found. "
                            "low: limited or no relevant information found."
                        ),
                    },
                    "sources_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of chunk IDs used to form the answer.",
                    },
                },
                "required": ["answer", "confidence"],
            },
            function=self._execute_complete,
        )

    def _execute_complete(
        self,
        answer: str,
        confidence: str,
        sources_used: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Signal completion - this is handled specially by the agent."""
        return {
            "status": "complete",
            "answer": answer,
            "confidence": confidence,
            "sources_used": sources_used or [],
        }
