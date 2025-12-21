"""FastAPI routes for research tool."""

from fastapi import APIRouter, HTTPException
from .schemas import (
    SearchRequest,
    SearchResponse,
    ResearchRequest,
    ResearchResponse,
    HealthResponse,
    SourceSchema,
    CitationSchema,
)
from ..search import SearchAggregator
from ..synthesis import SynthesisEngine
from ..config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and configuration."""
    aggregator = SearchAggregator()
    return HealthResponse(
        status="healthy",
        connectors=aggregator.get_active_connectors(),
        llm_configured=bool(settings.llm_api_base),
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Execute multi-source search with RRF fusion.

    Returns aggregated and ranked results from configured connectors.
    """
    aggregator = SearchAggregator()

    if not aggregator.connectors:
        raise HTTPException(
            status_code=503,
            detail="No search connectors configured"
        )

    sources, raw_results = await aggregator.search(
        query=request.query,
        top_k=request.top_k,
        connectors=request.connectors,
    )

    return SearchResponse(
        query=request.query,
        sources=[
            SourceSchema(
                id=s.id,
                title=s.title,
                url=s.url,
                content=s.content,
                score=s.score,
                connector=s.connector,
            )
            for s in sources
        ],
        connectors_used=list(raw_results.keys()),
        total_results=len(sources),
    )


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Perform full research: search + synthesis with citations.

    This replicates Perplexity's deep research capability using
    local connectors and Tongyi DeepResearch model.
    """
    aggregator = SearchAggregator()
    engine = SynthesisEngine()

    if not aggregator.connectors:
        raise HTTPException(
            status_code=503,
            detail="No search connectors configured"
        )

    # Step 1: Aggregate search results
    sources, raw_results = await aggregator.search(
        query=request.query,
        top_k=request.top_k,
        connectors=request.connectors,
    )

    if not sources:
        raise HTTPException(
            status_code=404,
            detail="No search results found"
        )

    # Step 2: Synthesize with LLM
    result = await engine.research(
        query=request.query,
        sources=sources,
        reasoning_effort=request.reasoning_effort,
    )

    if "error" in result:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis error: {result['error']}"
        )

    return ResearchResponse(
        query=request.query,
        content=result["content"],
        citations=[
            CitationSchema(id=c["id"], title=c["title"], url=c["url"])
            for c in result["citations"]
        ],
        sources=[
            SourceSchema(
                id=s.id,
                title=s.title,
                url=s.url,
                content=s.content,
                score=s.score,
                connector=s.connector,
            )
            for s in sources
        ],
        connectors_used=list(raw_results.keys()),
        model=result.get("model"),
        usage=result.get("usage"),
    )


@router.post("/ask")
async def ask(request: ResearchRequest):
    """
    Conversational research endpoint (Perplexity ask-style).

    Same as /research but optimized for quick answers.
    """
    request.reasoning_effort = "low"
    return await research(request)
