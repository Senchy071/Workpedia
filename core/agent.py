"""
Agent orchestrator for Workpedia.

Runs the agent loop: reason -> act -> observe -> repeat until complete.
The agent uses tools to search documents and gather information,
then signals completion when it has enough to answer the question.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from config.config import AGENT_MAX_ITERATIONS, AGENT_MODEL, AGENT_TEMPERATURE
from core.agent_tools import WorkpediaTools
from core.llm import OllamaClient, ChatResponse

if TYPE_CHECKING:
    from core.embedder import Embedder
    from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent run."""

    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AgentResult:
    """Result of an agent query."""

    status: AgentStatus
    answer: Optional[str] = None
    confidence: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    iterations: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }


class WorkpediaAgent:
    """
    Agent that uses tools to answer questions about indexed documents.

    The agent operates in a loop:
    1. Send context to LLM with available tools
    2. LLM decides which tool to call (or complete)
    3. Execute tool and return result
    4. Repeat until completion or max iterations
    """

    DEFAULT_SYSTEM_PROMPT = """You are a research assistant with access to a document collection. Your task is to answer questions by searching and analyzing the indexed documents.

Available tools:
{tool_descriptions}

Instructions:
1. Start by understanding what the user is asking
2. Use search_documents to find relevant information
3. If results are insufficient, try different search queries or check document summaries
4. Use list_documents if you need to see what's available
5. When you have enough information, call the complete tool with your answer
6. If you cannot find the answer after trying multiple searches, call complete with what you found and low confidence

Important:
- Always explain your reasoning before making tool calls
- Be thorough but efficient - don't make unnecessary searches
- The complete tool is REQUIRED to finish - you must call it with your final answer
- Include relevant source chunk IDs in your answer"""

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        vector_store: Optional["VectorStore"] = None,
        embedder: Optional["Embedder"] = None,
        max_iterations: int = AGENT_MAX_ITERATIONS,
        model: str = AGENT_MODEL,
        temperature: float = AGENT_TEMPERATURE,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm_client: OllamaClient instance (creates default if None)
            vector_store: VectorStore instance (creates default if None)
            embedder: Embedder instance (creates default if None)
            max_iterations: Maximum tool call iterations
            model: Model to use for agent reasoning
            temperature: Temperature for generation
            system_prompt: Custom system prompt (uses default if None)
        """
        # Lazy initialization
        if llm_client is None:
            llm_client = OllamaClient(model=model)
        if vector_store is None:
            from storage.vector_store import VectorStore

            vector_store = VectorStore()
        if embedder is None:
            from core.embedder import Embedder

            embedder = Embedder()

        self.llm = llm_client
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Initialize tools
        self.tools = WorkpediaTools(
            vector_store=vector_store,
            embedder=embedder,
        )

        logger.info(
            f"WorkpediaAgent initialized: model={model}, "
            f"max_iterations={max_iterations}, "
            f"tools={[t.name for t in self.tools.get_tools()]}"
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools.get_tools()
        ])
        return self.system_prompt.format(tool_descriptions=tool_descriptions)

    def _format_tool_result_message(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format a tool result as a message."""
        return {
            "role": "tool",
            "content": json.dumps(result, default=str),
        }

    def run(self, user_query: str) -> AgentResult:
        """
        Run the agent loop until completion or max iterations.

        Args:
            user_query: The user's question

        Returns:
            AgentResult with answer and metadata
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]

        tool_schemas = self.tools.get_tool_schemas()
        tool_call_log = []

        for iteration in range(self.max_iterations):
            logger.debug(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            try:
                # Get LLM response with tools
                response = self.llm.chat_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    model=self.model,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return AgentResult(
                    status=AgentStatus.FAILED,
                    error=str(e),
                    iterations=iteration + 1,
                    tool_calls=tool_call_log,
                )

            # Check if LLM made tool calls
            if response.has_tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments

                    logger.info(f"Agent calling tool: {tool_name}")
                    logger.debug(f"Tool arguments: {tool_args}")

                    # Log the call
                    tool_call_log.append({
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "arguments": tool_args,
                    })

                    # Execute tool
                    try:
                        result = self.tools.execute_tool(tool_name, tool_args)
                    except ValueError as e:
                        result = {"error": str(e)}
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        result = {"error": f"Tool execution failed: {e}"}

                    # Check for completion
                    if tool_name == "complete":
                        logger.info(
                            f"Agent completed after {iteration + 1} iterations "
                            f"with {result.get('confidence', 'unknown')} confidence"
                        )
                        return AgentResult(
                            status=AgentStatus.COMPLETE,
                            answer=result.get("answer"),
                            confidence=result.get("confidence"),
                            sources=result.get("sources_used", []),
                            iterations=iteration + 1,
                            tool_calls=tool_call_log,
                        )

                    # Add assistant message with tool call
                    # Note: Ollama expects arguments as object, not JSON string
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [{
                            "id": tool_call.id,
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args,  # Object, not string
                            },
                        }],
                    })

                    # Add tool result
                    messages.append(
                        self._format_tool_result_message(tool_call.id, result)
                    )

            else:
                # LLM responded without tool call
                # This shouldn't happen with good prompting, but handle it
                logger.warning("LLM responded without tool call")

                # Add the response and prompt to use tools
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Please use the available tools to search for information, "
                        "then call the 'complete' tool with your answer. "
                        "You must use the complete tool to finish."
                    ),
                })

        # Max iterations reached
        logger.warning(f"Agent reached max iterations ({self.max_iterations})")
        return AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            answer=(
                "I was unable to complete the task within the allowed iterations. "
                "Please try rephrasing your question or being more specific."
            ),
            confidence="low",
            iterations=self.max_iterations,
            tool_calls=tool_call_log,
        )

    def run_stream(
        self,
        user_query: str,
    ) -> Generator[Dict[str, Any], None, AgentResult]:
        """
        Run agent with streaming updates.

        Yields status updates as the agent works, then returns the final result.

        Args:
            user_query: The user's question

        Yields:
            Status updates as dicts with 'type' and content

        Returns:
            Final AgentResult
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]

        tool_schemas = self.tools.get_tool_schemas()
        tool_call_log = []

        for iteration in range(self.max_iterations):
            yield {
                "type": "iteration_start",
                "iteration": iteration + 1,
                "max_iterations": self.max_iterations,
            }

            try:
                # Get LLM response (non-streaming for simplicity in tool handling)
                response = self.llm.chat_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    model=self.model,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                yield {"type": "error", "error": str(e)}
                return AgentResult(
                    status=AgentStatus.FAILED,
                    error=str(e),
                    iterations=iteration + 1,
                    tool_calls=tool_call_log,
                )

            # Yield thinking content
            if response.content:
                yield {
                    "type": "thinking",
                    "content": response.content,
                }

            # Process tool calls
            if response.has_tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments

                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "arguments": tool_args,
                    }

                    tool_call_log.append({
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "arguments": tool_args,
                    })

                    # Execute tool
                    try:
                        result = self.tools.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = {"error": str(e)}

                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": result,
                    }

                    # Check for completion
                    if tool_name == "complete":
                        final_result = AgentResult(
                            status=AgentStatus.COMPLETE,
                            answer=result.get("answer"),
                            confidence=result.get("confidence"),
                            sources=result.get("sources_used", []),
                            iterations=iteration + 1,
                            tool_calls=tool_call_log,
                        )
                        yield {"type": "complete", "result": final_result.to_dict()}
                        return final_result

                    # Add messages for next iteration
                    # Note: Ollama expects arguments as object, not JSON string
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [{
                            "id": tool_call.id,
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args,  # Object, not string
                            },
                        }],
                    })
                    messages.append(
                        self._format_tool_result_message(tool_call.id, result)
                    )

            else:
                # No tool call - prompt to use tools
                yield {"type": "no_tool_call", "content": response.content}

                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Please use the available tools to search for information, "
                        "then call the 'complete' tool with your answer."
                    ),
                })

        # Max iterations
        final_result = AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            answer=(
                "I was unable to complete the task within the allowed iterations. "
                "Please try rephrasing your question."
            ),
            confidence="low",
            iterations=self.max_iterations,
            tool_calls=tool_call_log,
        )
        yield {"type": "max_iterations", "result": final_result.to_dict()}
        return final_result


def create_agent(
    vector_store: Optional["VectorStore"] = None,
    embedder: Optional["Embedder"] = None,
    llm_client: Optional[OllamaClient] = None,
) -> WorkpediaAgent:
    """
    Factory function to create a configured agent.

    Args:
        vector_store: Optional VectorStore instance
        embedder: Optional Embedder instance
        llm_client: Optional OllamaClient instance

    Returns:
        Configured WorkpediaAgent
    """
    return WorkpediaAgent(
        llm_client=llm_client,
        vector_store=vector_store,
        embedder=embedder,
    )
