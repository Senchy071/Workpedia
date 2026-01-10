"""Tests for the Workpedia agent layer."""

import json
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from core.agent import AgentResult, AgentStatus, WorkpediaAgent, create_agent
from core.agent_tools import Tool, WorkpediaTools
from core.llm import ChatResponse, ToolCall


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store with test data."""
    store = MagicMock()

    # Mock list_documents
    store.list_documents.return_value = [
        {"doc_id": "doc1", "filename": "test.pdf", "chunk_count": 10},
        {"doc_id": "doc2", "filename": "report.docx", "chunk_count": 5},
    ]

    # Mock query_text
    store.query_text.return_value = {
        "ids": ["chunk1", "chunk2"],
        "documents": ["This is test content about AI.", "More content about machine learning."],
        "metadatas": [
            {"doc_id": "doc1", "filename": "test.pdf", "section": "Introduction"},
            {"doc_id": "doc1", "filename": "test.pdf", "section": "Methods"},
        ],
        "distances": [0.1, 0.2],
    }

    # Mock get_document_summary
    store.get_document_summary.return_value = {
        "content": "This document covers AI and ML topics.",
        "metadata": {"filename": "test.pdf"},
    }

    # Mock get_by_doc_id
    store.get_by_doc_id.return_value = {
        "ids": ["chunk1", "chunk2", "chunk3"],
        "metadatas": [
            {"doc_id": "doc1", "section": "Intro"},
            {"doc_id": "doc1", "section": "Body"},
            {"doc_id": "doc1", "section": "Conclusion"},
        ],
    }

    # Mock _collection.get for read_chunk
    store._collection = MagicMock()
    store._collection.get.return_value = {
        "ids": ["chunk1"],
        "documents": ["Full chunk content here."],
        "metadatas": [{"doc_id": "doc1", "section": "Test"}],
    }

    return store


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    return embedder


@pytest.fixture
def workpedia_tools(mock_vector_store, mock_embedder):
    """Create WorkpediaTools with mocks."""
    return WorkpediaTools(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    return client


# =============================================================================
# Tool Tests
# =============================================================================


class TestWorkpediaTools:
    """Tests for WorkpediaTools."""

    def test_get_tools_returns_all_tools(self, workpedia_tools):
        """Test that get_tools returns all expected tools."""
        tools = workpedia_tools.get_tools()

        assert len(tools) == 6
        tool_names = [t.name for t in tools]
        assert "search_documents" in tool_names
        assert "get_document_summary" in tool_names
        assert "list_documents" in tool_names
        assert "read_chunk" in tool_names
        assert "get_document_info" in tool_names
        assert "complete" in tool_names

    def test_get_tool_schemas(self, workpedia_tools):
        """Test that get_tool_schemas returns valid schemas."""
        schemas = workpedia_tools.get_tool_schemas()

        assert len(schemas) == 6
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_get_tool_by_name(self, workpedia_tools):
        """Test getting a tool by name."""
        tool = workpedia_tools.get_tool("search_documents")
        assert tool is not None
        assert tool.name == "search_documents"

        unknown = workpedia_tools.get_tool("unknown_tool")
        assert unknown is None

    def test_execute_search_documents(self, workpedia_tools):
        """Test search_documents tool execution."""
        result = workpedia_tools.execute_tool(
            "search_documents",
            {"query": "AI concepts", "n_results": 5},
        )

        assert "results" in result
        assert "total_found" in result
        assert result["total_found"] == 2
        assert len(result["results"]) == 2

    def test_execute_list_documents(self, workpedia_tools):
        """Test list_documents tool execution."""
        result = workpedia_tools.execute_tool("list_documents", {})

        assert "documents" in result
        assert "total" in result
        assert result["total"] == 2
        assert result["documents"][0]["doc_id"] == "doc1"

    def test_execute_get_document_summary(self, workpedia_tools):
        """Test get_document_summary tool execution."""
        result = workpedia_tools.execute_tool(
            "get_document_summary",
            {"doc_id": "doc1"},
        )

        assert "doc_id" in result
        assert "summary" in result
        assert result["doc_id"] == "doc1"

    def test_execute_get_document_summary_not_found(self, workpedia_tools, mock_vector_store):
        """Test get_document_summary when document not found."""
        mock_vector_store.get_document_summary.return_value = None

        result = workpedia_tools.execute_tool(
            "get_document_summary",
            {"doc_id": "nonexistent"},
        )

        assert "error" in result

    def test_execute_read_chunk(self, workpedia_tools):
        """Test read_chunk tool execution."""
        result = workpedia_tools.execute_tool(
            "read_chunk",
            {"chunk_id": "chunk1"},
        )

        assert "chunk_id" in result
        assert "content" in result
        assert result["chunk_id"] == "chunk1"

    def test_execute_read_chunk_not_found(self, workpedia_tools, mock_vector_store):
        """Test read_chunk when chunk not found."""
        mock_vector_store._collection.get.return_value = {"ids": []}

        result = workpedia_tools.execute_tool(
            "read_chunk",
            {"chunk_id": "nonexistent"},
        )

        assert "error" in result

    def test_execute_get_document_info(self, workpedia_tools):
        """Test get_document_info tool execution."""
        result = workpedia_tools.execute_tool(
            "get_document_info",
            {"doc_id": "doc1"},
        )

        assert "doc_id" in result
        assert "total_chunks" in result
        assert result["total_chunks"] == 3

    def test_execute_complete(self, workpedia_tools):
        """Test complete tool execution."""
        result = workpedia_tools.execute_tool(
            "complete",
            {
                "answer": "The answer is 42.",
                "confidence": "high",
                "sources_used": ["chunk1", "chunk2"],
            },
        )

        assert result["status"] == "complete"
        assert result["answer"] == "The answer is 42."
        assert result["confidence"] == "high"
        assert result["sources_used"] == ["chunk1", "chunk2"]

    def test_execute_unknown_tool(self, workpedia_tools):
        """Test executing unknown tool raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            workpedia_tools.execute_tool("unknown_tool", {})


class TestToolSchema:
    """Tests for Tool dataclass."""

    def test_tool_to_schema(self):
        """Test Tool.to_schema() format."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                },
                "required": ["param1"],
            },
            function=lambda **kwargs: {},
        )

        schema = tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"]["type"] == "object"


# =============================================================================
# Agent Tests
# =============================================================================


class TestWorkpediaAgent:
    """Tests for WorkpediaAgent."""

    def test_agent_initialization(self, mock_llm_client, mock_vector_store, mock_embedder):
        """Test agent initializes correctly."""
        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        assert agent.llm == mock_llm_client
        assert agent.tools is not None
        assert len(agent.tools.get_tools()) == 6

    def test_agent_completes_on_first_search(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent completes when it finds good results immediately."""
        # Mock LLM to first search, then complete
        search_response = ChatResponse(
            content="I'll search for relevant information.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_documents",
                    arguments={"query": "test query", "n_results": 5},
                )
            ],
        )

        complete_response = ChatResponse(
            content="Based on the search results...",
            tool_calls=[
                ToolCall(
                    id="call_2",
                    name="complete",
                    arguments={
                        "answer": "The answer based on documents.",
                        "confidence": "high",
                        "sources_used": ["chunk1"],
                    },
                )
            ],
        )

        mock_llm_client.chat_with_tools.side_effect = [search_response, complete_response]

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = agent.run("What is in the documents?")

        assert result.status == AgentStatus.COMPLETE
        assert result.answer == "The answer based on documents."
        assert result.confidence == "high"
        assert result.iterations == 2
        assert len(result.tool_calls) == 2

    def test_agent_retries_with_different_query(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent tries different queries when initial results are poor."""
        # First search
        search1_response = ChatResponse(
            content="First search attempt.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_documents",
                    arguments={"query": "initial query"},
                )
            ],
        )

        # Second search with different query
        search2_response = ChatResponse(
            content="Trying a different query.",
            tool_calls=[
                ToolCall(
                    id="call_2",
                    name="search_documents",
                    arguments={"query": "refined query"},
                )
            ],
        )

        # Complete
        complete_response = ChatResponse(
            content="Found the answer.",
            tool_calls=[
                ToolCall(
                    id="call_3",
                    name="complete",
                    arguments={
                        "answer": "Found after refinement.",
                        "confidence": "medium",
                    },
                )
            ],
        )

        mock_llm_client.chat_with_tools.side_effect = [
            search1_response,
            search2_response,
            complete_response,
        ]

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = agent.run("Complex question")

        assert result.status == AgentStatus.COMPLETE
        assert result.iterations == 3
        assert len(result.tool_calls) == 3

    def test_agent_respects_max_iterations(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent stops at max_iterations."""
        # Always return a search (never complete)
        search_response = ChatResponse(
            content="Searching...",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_documents",
                    arguments={"query": "endless search"},
                )
            ],
        )

        mock_llm_client.chat_with_tools.return_value = search_response

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            max_iterations=3,
        )

        result = agent.run("Unanswerable question")

        assert result.status == AgentStatus.MAX_ITERATIONS
        assert result.iterations == 3
        assert result.confidence == "low"

    def test_agent_handles_llm_error(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent handles LLM errors gracefully."""
        mock_llm_client.chat_with_tools.side_effect = Exception("LLM connection failed")

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = agent.run("Test question")

        assert result.status == AgentStatus.FAILED
        assert result.error is not None
        assert "LLM connection failed" in result.error

    def test_agent_handles_tool_errors(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent handles tool execution errors."""
        # Mock search that triggers an error
        search_response = ChatResponse(
            content="Searching...",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_documents",
                    arguments={"query": "test"},
                )
            ],
        )

        complete_response = ChatResponse(
            content="Completing despite error.",
            tool_calls=[
                ToolCall(
                    id="call_2",
                    name="complete",
                    arguments={
                        "answer": "Completed with limited info.",
                        "confidence": "low",
                    },
                )
            ],
        )

        mock_llm_client.chat_with_tools.side_effect = [search_response, complete_response]

        # Make vector store throw an error
        mock_vector_store.query_text.side_effect = Exception("Database error")

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = agent.run("Test question")

        # Agent should still complete (tool error is passed back to LLM)
        assert result.status == AgentStatus.COMPLETE

    def test_agent_handles_no_tool_call_response(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent handles LLM responding without tool calls."""
        # First response has no tool call
        no_tool_response = ChatResponse(
            content="Let me think about this...",
            tool_calls=[],
        )

        # Second response completes properly
        complete_response = ChatResponse(
            content="Now I'll complete.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="complete",
                    arguments={
                        "answer": "Final answer.",
                        "confidence": "medium",
                    },
                )
            ],
        )

        mock_llm_client.chat_with_tools.side_effect = [no_tool_response, complete_response]

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = agent.run("Test question")

        assert result.status == AgentStatus.COMPLETE

    def test_agent_uses_multiple_tools(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test agent can use multiple different tools."""
        # List documents first
        list_response = ChatResponse(
            content="Let me see what documents are available.",
            tool_calls=[
                ToolCall(id="call_1", name="list_documents", arguments={})
            ],
        )

        # Then search
        search_response = ChatResponse(
            content="Now I'll search.",
            tool_calls=[
                ToolCall(
                    id="call_2",
                    name="search_documents",
                    arguments={"query": "test"},
                )
            ],
        )

        # Complete
        complete_response = ChatResponse(
            content="Found the answer.",
            tool_calls=[
                ToolCall(
                    id="call_3",
                    name="complete",
                    arguments={
                        "answer": "Answer after listing and searching.",
                        "confidence": "high",
                        "sources_used": ["chunk1"],
                    },
                )
            ],
        )

        mock_llm_client.chat_with_tools.side_effect = [
            list_response,
            search_response,
            complete_response,
        ]

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = agent.run("What documents do we have?")

        assert result.status == AgentStatus.COMPLETE
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0]["tool"] == "list_documents"
        assert result.tool_calls[1]["tool"] == "search_documents"
        assert result.tool_calls[2]["tool"] == "complete"


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_to_dict(self):
        """Test AgentResult.to_dict() method."""
        result = AgentResult(
            status=AgentStatus.COMPLETE,
            answer="Test answer",
            confidence="high",
            sources=["chunk1", "chunk2"],
            iterations=3,
            tool_calls=[
                {"iteration": 1, "tool": "search", "arguments": {}},
            ],
        )

        d = result.to_dict()

        assert d["status"] == "complete"
        assert d["answer"] == "Test answer"
        assert d["confidence"] == "high"
        assert d["sources"] == ["chunk1", "chunk2"]
        assert d["iterations"] == 3


class TestAgentStreaming:
    """Tests for agent streaming functionality."""

    def test_run_stream_yields_events(
        self, mock_llm_client, mock_vector_store, mock_embedder
    ):
        """Test run_stream yields correct event types."""
        search_response = ChatResponse(
            content="Searching...",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_documents",
                    arguments={"query": "test"},
                )
            ],
        )

        complete_response = ChatResponse(
            content="Done.",
            tool_calls=[
                ToolCall(
                    id="call_2",
                    name="complete",
                    arguments={
                        "answer": "Stream answer.",
                        "confidence": "high",
                    },
                )
            ],
        )

        mock_llm_client.chat_with_tools.side_effect = [search_response, complete_response]

        agent = WorkpediaAgent(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        events = list(agent.run_stream("Test question"))

        # Check event types
        event_types = [e["type"] for e in events]
        assert "iteration_start" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "complete" in event_types


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_agent_with_defaults(self):
        """Test create_agent with default parameters."""
        with patch("storage.vector_store.VectorStore") as MockVS, \
             patch("core.embedder.Embedder") as MockEmb:

            mock_vs = MagicMock()
            mock_emb = MagicMock()
            mock_llm = MagicMock()

            MockVS.return_value = mock_vs
            MockEmb.return_value = mock_emb

            agent = create_agent(
                vector_store=mock_vs,
                embedder=mock_emb,
                llm_client=mock_llm,
            )

            assert agent is not None
            assert isinstance(agent, WorkpediaAgent)


# =============================================================================
# Integration Tests (require actual services)
# =============================================================================


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests that require Ollama and ChromaDB."""

    @pytest.fixture
    def real_agent(self):
        """Create agent with real services."""
        try:
            from core.agent import create_agent
            from core.llm import OllamaClient

            # Check if Ollama is available
            client = OllamaClient()
            if not client.is_available():
                pytest.skip("Ollama not available")

            return create_agent()
        except Exception as e:
            pytest.skip(f"Could not create agent: {e}")

    def test_agent_can_query_documents(self, real_agent):
        """Test agent can query real documents."""
        result = real_agent.run("What documents are available?")

        # Agent may complete, hit max iterations, or fail due to LLM issues
        assert result.status in (
            AgentStatus.COMPLETE,
            AgentStatus.MAX_ITERATIONS,
            AgentStatus.FAILED,
        )
        # If completed, should have an answer
        if result.status == AgentStatus.COMPLETE:
            assert result.answer is not None

    def test_agent_stream_works(self, real_agent):
        """Test streaming works with real agent."""
        events = list(real_agent.run_stream("List available documents"))

        assert len(events) > 0
        assert any(e["type"] == "iteration_start" for e in events)
