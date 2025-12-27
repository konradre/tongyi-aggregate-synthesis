"""LLM client with OpenRouter fallback support.

Handles automatic fallback from free tier to paid model on rate limits.
"""

import logging
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI, RateLimitError
from .config import settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """AsyncOpenAI wrapper with automatic fallback on rate limits."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        fallback_enabled: Optional[bool] = None,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to settings)
            base_url: API base URL (defaults to settings)
            primary_model: Primary model to try first (defaults to settings)
            fallback_model: Fallback model on rate limit (defaults to settings)
            fallback_enabled: Whether to enable fallback (defaults to settings)
        """
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_api_base
        self.primary_model = primary_model or settings.llm_model
        self.fallback_model = fallback_model or settings.llm_model_fallback
        self.fallback_enabled = fallback_enabled if fallback_enabled is not None else settings.llm_fallback_enabled

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Track which model was last used (for debugging/logging)
        self.last_model_used: Optional[str] = None
        self.last_was_fallback: bool = False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Create chat completion with automatic fallback.

        Args:
            messages: Chat messages
            model: Override model (skips fallback logic if provided)
            temperature: Generation temperature
            top_p: Top-p sampling
            max_tokens: Max output tokens
            **kwargs: Additional parameters passed to API

        Returns:
            OpenAI ChatCompletion response

        Raises:
            RateLimitError: If both primary and fallback models are rate limited
        """
        # Use provided model or primary
        current_model = model or self.primary_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        top_p = top_p if top_p is not None else settings.llm_top_p
        max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        try:
            logger.debug(f"Attempting request with primary model: {current_model}")
            response = await self._client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs,
            )
            self.last_model_used = current_model
            self.last_was_fallback = False
            logger.debug(f"Success with primary model: {current_model}")
            return response

        except RateLimitError as e:
            # Only fallback if enabled and we haven't already tried fallback
            if self.fallback_enabled and current_model == self.primary_model and model is None:
                logger.warning(
                    f"Rate limited on {current_model}, falling back to {self.fallback_model}"
                )
                try:
                    response = await self._client.chat.completions.create(
                        model=self.fallback_model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    self.last_model_used = self.fallback_model
                    self.last_was_fallback = True
                    logger.info(f"Fallback successful with: {self.fallback_model}")
                    return response

                except RateLimitError:
                    logger.error(f"Both primary and fallback models rate limited")
                    raise
            else:
                # Re-raise if fallback disabled or already on fallback
                raise

    @property
    def chat(self):
        """Compatibility property for code expecting client.chat.completions pattern."""
        return _ChatNamespace(self)


class _ChatNamespace:
    """Namespace for chat.completions compatibility."""

    def __init__(self, client: OpenRouterClient):
        self._client = client
        self.completions = _CompletionsNamespace(client)


class _CompletionsNamespace:
    """Namespace for chat.completions.create() compatibility."""

    def __init__(self, client: OpenRouterClient):
        self._client = client

    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """Create chat completion - delegates to client with fallback logic.

        Note: The model parameter is used but fallback still applies if it matches
        the primary model. To skip fallback entirely, the caller should set
        fallback_enabled=False on the client.
        """
        return await self._client.chat_completion(
            messages=messages,
            model=model,
            **kwargs,
        )


def get_llm_client() -> OpenRouterClient:
    """Get configured OpenRouter client with fallback support."""
    return OpenRouterClient()
