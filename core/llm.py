"""Ollama LLM client for local text generation."""

import logging
import json
from typing import Optional, Generator, Dict, Any, List
import requests

from config.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    RETRY_ENABLED,
    RETRY_MAX_ATTEMPTS,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    RETRY_EXPONENTIAL_BASE,
    RETRY_JITTER,
    CIRCUIT_BREAKER_ENABLED,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
    CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    TIMEOUT_DEFAULT,
    TIMEOUT_HEALTH_CHECK,
    TIMEOUT_LIST_MODELS,
    TIMEOUT_GENERATE,
    TIMEOUT_GENERATE_STREAM,
)
from core.exceptions import (
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaTimeoutError,
)
from core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    RetryConfig,
    retry_with_backoff,
    is_retryable_error,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama local LLM inference.

    Features:
    - Streaming and non-streaming generation
    - Configurable model and parameters
    - Context window management
    - Health checking
    - Retry logic with exponential backoff
    - Circuit breaker pattern
    - Configurable timeouts per operation
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = None,
        enable_retry: bool = RETRY_ENABLED,
        enable_circuit_breaker: bool = CIRCUIT_BREAKER_ENABLED,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (default: mistral)
            base_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: from config)
            enable_retry: Enable retry logic with exponential backoff
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout if timeout is not None else TIMEOUT_DEFAULT
        self.enable_retry = enable_retry
        self.enable_circuit_breaker = enable_circuit_breaker

        # Initialize retry configuration
        self.retry_config = RetryConfig(
            max_retries=RETRY_MAX_ATTEMPTS,
            initial_delay=RETRY_INITIAL_DELAY,
            max_delay=RETRY_MAX_DELAY,
            exponential_base=RETRY_EXPONENTIAL_BASE,
            jitter=RETRY_JITTER,
        )

        # Initialize circuit breaker
        if self.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                name=f"ollama-{model}",
                config=CircuitBreakerConfig(
                    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                    recovery_timeout=CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                    success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
                    half_open_max_calls=CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
                ),
                on_state_change=self._on_circuit_state_change,
            )
        else:
            self.circuit_breaker = None

        logger.info(
            f"OllamaClient initialized: model={model}, url={base_url}, "
            f"retry={enable_retry}, circuit_breaker={enable_circuit_breaker}"
        )

    def _on_circuit_state_change(self, old_state, new_state):
        """Callback for circuit breaker state changes."""
        logger.warning(
            f"Ollama circuit breaker state changed: {old_state.value} → {new_state.value}"
        )

    def get_circuit_breaker_stats(self) -> Optional[dict]:
        """Get circuit breaker statistics."""
        if self.circuit_breaker:
            return self.circuit_breaker.get_stats()
        return None

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def check_model_available(self, model_name: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if a specific model is available.

        Args:
            model_name: Model name to check (uses self.model if None)

        Returns:
            Tuple of (is_available, message)
        """
        model_name = model_name or self.model

        # First check if server is reachable
        if not self.is_available():
            return False, f"Ollama server is not reachable at {self.base_url}"

        # Check if model exists
        available_models = self.list_models()
        if not available_models:
            return False, f"Could not retrieve model list from Ollama server at {self.base_url}"

        # Check for exact match first
        if model_name in available_models:
            return True, f"Model '{model_name}' is available"

        # Check for partial match (e.g., "mistral" matches "mistral:latest")
        # This handles cases where user specifies "mistral" but Ollama has "mistral:latest"
        matching_models = [m for m in available_models if m.startswith(f"{model_name}:")]
        if matching_models:
            matched_model = matching_models[0]
            logger.info(f"Model '{model_name}' matched to '{matched_model}'")
            return True, f"Model '{model_name}' is available (using {matched_model})"

        # No match found
        return False, (
            f"Model '{model_name}' is not available. "
            f"Available models: {', '.join(available_models) if available_models else 'none'}. "
            f"Run 'ollama pull {model_name}' to download it."
        )


    def health_check(self) -> dict:
        """
        Perform comprehensive health check.

        Returns:
            Dictionary with health status and details
        """
        result = {
            "server_reachable": False,
            "model_available": False,
            "model_name": self.model,
            "base_url": self.base_url,
            "available_models": [],
            "message": "",
        }

        # Check server
        if not self.is_available():
            result["message"] = f"Ollama server is not reachable at {self.base_url}"
            return result

        result["server_reachable"] = True

        # Get available models
        available_models = self.list_models()
        result["available_models"] = available_models

        # Check model
        is_available, message = self.check_model_available()
        result["model_available"] = is_available
        result["message"] = message

        return result

    def list_models(self) -> List[str]:
        """List available models with retry logic."""
        def _list_models_internal():
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=TIMEOUT_LIST_MODELS
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]

        try:
            if self.enable_retry:
                # Apply retry logic
                @retry_with_backoff(
                    config=self.retry_config,
                    retryable_exceptions=(requests.Timeout, requests.ConnectionError)
                )
                def _with_retry():
                    return _list_models_internal()
                return _with_retry()
            else:
                return _list_models_internal()

        except requests.Timeout as e:
            logger.error(f"Timeout listing models: {e}")
            # Don't raise, return empty list for graceful degradation
            return []
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            # Don't raise, return empty list for graceful degradation
            return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text (or generator if streaming)
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if stream:
            return self._stream_generate(payload)
        else:
            return self._sync_generate(payload)

    def _sync_generate(self, payload: Dict[str, Any]) -> str:
        """Synchronous generation with retry and circuit breaker."""
        def _generate_internal():
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=TIMEOUT_GENERATE,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except requests.Timeout as e:
                logger.error(f"Generation timed out after {TIMEOUT_GENERATE}s: {e}")
                raise OllamaTimeoutError(timeout=TIMEOUT_GENERATE, model=self.model) from e
            except requests.ConnectionError as e:
                logger.error(f"Connection failed: {e}")
                raise OllamaConnectionError(
                    base_url=self.base_url,
                    reason="Connection failed during generation"
                ) from e
            except requests.RequestException as e:
                logger.error(f"Generation failed: {e}")
                raise OllamaGenerationError(reason=str(e), model=self.model) from e

        # Apply circuit breaker if enabled
        if self.enable_circuit_breaker and self.circuit_breaker:
            _generate_internal = self.circuit_breaker.call(_generate_internal)

        # Apply retry logic if enabled
        if self.enable_retry:
            @retry_with_backoff(
                config=self.retry_config,
                retryable_exceptions=(
                    OllamaConnectionError,
                    OllamaTimeoutError,
                    requests.Timeout,
                    requests.ConnectionError,
                )
            )
            def _with_retry():
                return _generate_internal()
            return _with_retry()
        else:
            return _generate_internal()

    def _stream_generate(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Streaming generation."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
        except requests.Timeout as e:
            logger.error(f"Streaming generation timed out after {self.timeout}s: {e}")
            raise OllamaTimeoutError(timeout=self.timeout, model=self.model) from e
        except requests.ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            raise OllamaConnectionError(
                base_url=self.base_url,
                reason="Connection failed during streaming"
            ) from e
        except requests.RequestException as e:
            logger.error(f"Streaming generation failed: {e}")
            raise OllamaGenerationError(reason=str(e), model=self.model) from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Chat completion with message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated response (or generator if streaming)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if stream:
            return self._stream_chat(payload)
        else:
            return self._sync_chat(payload)

    def _sync_chat(self, payload: Dict[str, Any]) -> str:
        """Synchronous chat completion with retry and circuit breaker."""
        def _chat_internal():
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=TIMEOUT_GENERATE,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
            except requests.Timeout as e:
                logger.error(f"Chat timed out after {TIMEOUT_GENERATE}s: {e}")
                raise OllamaTimeoutError(timeout=TIMEOUT_GENERATE, model=self.model) from e
            except requests.ConnectionError as e:
                logger.error(f"Connection failed: {e}")
                raise OllamaConnectionError(
                    base_url=self.base_url,
                    reason="Connection failed during chat"
                ) from e
            except requests.RequestException as e:
                logger.error(f"Chat completion failed: {e}")
                raise OllamaGenerationError(reason=str(e), model=self.model) from e

        # Apply circuit breaker if enabled
        if self.enable_circuit_breaker and self.circuit_breaker:
            _chat_internal = self.circuit_breaker.call(_chat_internal)

        # Apply retry logic if enabled
        if self.enable_retry:
            @retry_with_backoff(
                config=self.retry_config,
                retryable_exceptions=(
                    OllamaConnectionError,
                    OllamaTimeoutError,
                    requests.Timeout,
                    requests.ConnectionError,
                )
            )
            def _with_retry():
                return _chat_internal()
            return _with_retry()
        else:
            return _chat_internal()

    def _stream_chat(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Streaming chat completion."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        content = data["message"].get("content", "")
                        if content:
                            yield content
                    if data.get("done", False):
                        break
        except requests.Timeout as e:
            logger.error(f"Streaming chat timed out after {self.timeout}s: {e}")
            raise OllamaTimeoutError(timeout=self.timeout, model=self.model) from e
        except requests.ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            raise OllamaConnectionError(
                base_url=self.base_url,
                reason="Connection failed during streaming chat"
            ) from e
        except requests.RequestException as e:
            logger.error(f"Streaming chat failed: {e}")
            raise OllamaGenerationError(reason=str(e), model=self.model) from e


# Default prompts for RAG
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer questions using ONLY the information from the provided context
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct in your responses
- Cite specific sections or quotes when relevant
- Do not make up information not present in the context"""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Please answer the question based on the context provided above."""


def format_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    max_context_chars: int = 8000,
) -> str:
    """
    Format a RAG prompt with context chunks.

    Args:
        question: User's question
        context_chunks: List of retrieved chunks with 'content' and 'metadata'
        max_context_chars: Maximum characters for context

    Returns:
        Formatted prompt string
    """
    # Build context from chunks
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(context_chunks, 1):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})

        # Add source info
        source = metadata.get("filename", "Unknown")
        section = metadata.get("section", "")

        chunk_text = f"[Source {i}: {source}"
        if section:
            chunk_text += f", Section: {section}"
        chunk_text += f"]\n{content}\n"

        if total_chars + len(chunk_text) > max_context_chars:
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    context = "\n".join(context_parts)

    return RAG_USER_TEMPLATE.format(context=context, question=question)
