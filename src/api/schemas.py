"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Literal


class SearchRequest(BaseModel):
    """Request for multi-source search."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Results per source")
    connectors: list[str] | None = Field(
        default=None,
        description="Specific connectors to use (searxng, tavily, linkup)"
    )


class SourceSchema(BaseModel):
    """Source document schema."""

    id: str
    title: str
    url: str
    content: str
    score: float
    connector: str


class SearchResponse(BaseModel):
    """Response from search endpoint."""

    query: str
    sources: list[SourceSchema]
    connectors_used: list[str]
    total_results: int


class ResearchRequest(BaseModel):
    """Request for full research with synthesis."""

    query: str = Field(..., description="Research query")
    top_k: int = Field(default=10, ge=1, le=50, description="Results per source")
    connectors: list[str] | None = Field(
        default=None,
        description="Specific connectors to use"
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Depth of analysis"
    )


class CitationSchema(BaseModel):
    """Citation reference."""

    id: str
    title: str
    url: str


class ResearchResponse(BaseModel):
    """Response from research endpoint."""

    query: str
    content: str
    citations: list[CitationSchema]
    sources: list[SourceSchema]
    connectors_used: list[str]
    model: str | None = None
    usage: dict | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    connectors: list[str]
    llm_configured: bool
