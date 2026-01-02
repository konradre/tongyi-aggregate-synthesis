"""Configuration for the research tool."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # SearXNG Configuration
    searxng_host: str = Field(default="http://192.168.1.3:8888", description="SearXNG instance URL")
    searxng_engines: str = Field(default="google,bing,duckduckgo,brave,startpage", description="Comma-separated search engines")
    searxng_categories: str = Field(default="general", description="Search categories")
    searxng_language: str = Field(default="en", description="Search language")
    searxng_safesearch: int = Field(default=0, description="Safe search level (0=off, 1=moderate, 2=strict)")

    # Tavily Configuration
    tavily_api_key: str = Field(default="", description="Tavily API key")
    tavily_search_depth: str = Field(default="advanced", description="Search depth: basic or advanced")

    # LinkUp Configuration
    linkup_api_key: str = Field(default="", description="LinkUp API key")
    linkup_depth: str = Field(default="standard", description="Search depth: standard or deep")

    # OpenRouter LLM Configuration
    llm_api_base: str = Field(default="https://openrouter.ai/api/v1", description="LLM API base URL")
    llm_api_key: str = Field(default="", description="OpenRouter API key")
    llm_model: str = Field(default="alibaba/tongyi-deepresearch-30b-a3b:free", description="Primary model (free tier)")
    llm_model_fallback: str = Field(default="alibaba/tongyi-deepresearch-30b-a3b", description="Fallback model (paid)")
    llm_fallback_enabled: bool = Field(default=True, description="Enable fallback to paid model on rate limit")
    llm_temperature: float = Field(default=0.85, description="Generation temperature")
    llm_top_p: float = Field(default=0.95, description="Top-p sampling")
    llm_max_tokens: int = Field(default=8192, description="Max output tokens")
    llm_timeout: int = Field(default=120, description="LLM request timeout in seconds")

    # Search Configuration
    default_top_k: int = Field(default=10, description="Default number of results per source")
    rrf_k: int = Field(default=60, description="RRF fusion constant")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    model_config = {"env_prefix": "RESEARCH_"}


settings = Settings()
