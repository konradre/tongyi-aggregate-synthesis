"""FastAPI routes for research tool."""

import re
from openai import AsyncOpenAI
from fastapi import APIRouter, HTTPException
from .schemas import (
    # Existing
    SearchRequest,
    SearchResponse,
    ResearchRequest,
    ResearchResponse,
    HealthResponse,
    SourceSchema,
    CitationSchema,
    # Discovery (perplexity_search role)
    DiscoverRequest,
    DiscoverResponse,
    KnowledgeGapSchema,
    KnowledgeLandscapeSchema,
    ScoredSourceSchema,
    # Synthesis (perplexity_research role)
    SynthesizeRequest,
    SynthesizeResponse,
    PreGatheredSourceSchema,
    SynthesisAttributionSchema,
    # Reasoning (perplexity_reason role)
    ReasonRequest,
    ReasonResponse,
    # Conversation (perplexity_ask role)
    AskRequest,
    AskResponse,
)
from ..search import SearchAggregator
from ..synthesis import (
    SynthesisEngine,
    SynthesisAggregator,
    SynthesisStyle,
    PreGatheredSource,
)
from ..discovery import Explorer
from ..config import settings

router = APIRouter()


def _get_llm_client():
    """Get OpenAI-compatible LLM client."""
    return AsyncOpenAI(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
    )


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


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Quick conversational answer (mirrors perplexity_ask).

    Optimized for fast, concise responses.
    """
    aggregator = SearchAggregator()
    engine = SynthesisEngine()

    if not aggregator.connectors:
        raise HTTPException(
            status_code=503,
            detail="No search connectors configured"
        )

    # Quick search with fewer sources
    sources, raw_results = await aggregator.search(
        query=request.query,
        top_k=request.top_k,
        connectors=request.connectors,
    )

    if not sources:
        return AskResponse(
            query=request.query,
            content="I couldn't find relevant information to answer this question.",
            citations=[],
            sources=[],
            model=None,
        )

    # Quick synthesis with low effort
    result = await engine.research(
        query=request.query,
        sources=sources,
        reasoning_effort="low",
    )

    return AskResponse(
        query=request.query,
        content=result.get("content", ""),
        citations=[
            CitationSchema(id=c["id"], title=c["title"], url=c["url"])
            for c in result.get("citations", [])
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
            for s in sources[:5]  # Limit sources in response
        ],
        model=result.get("model"),
    )


# =============================================================================
# Enhanced Endpoints (Mirroring Perplexity Tools)
# =============================================================================


@router.post("/discover", response_model=DiscoverResponse)
async def discover(request: DiscoverRequest):
    """
    Exploratory discovery with breadth expansion (mirrors perplexity_search).

    Optimized for the EXPLORATORY workflow:
    1. Expand knowledge landscape (explicit, implicit, related, contrasting)
    2. Identify knowledge gaps in the query
    3. Score sources by gap coverage
    4. Recommend URLs for deep dives (Jina parallel_read)

    This sets the table for targeted research expansion.
    """
    aggregator = SearchAggregator()
    llm_client = _get_llm_client()

    if not aggregator.connectors:
        raise HTTPException(
            status_code=503,
            detail="No search connectors configured"
        )

    explorer = Explorer(
        llm_client=llm_client,
        search_aggregator=aggregator,
        model=settings.llm_model,
    )

    try:
        result = await explorer.discover(
            query=request.query,
            top_k=request.top_k,
            expand_searches=request.expand_searches,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Discovery error: {e}"
        )

    # Convert to response schemas
    return DiscoverResponse(
        query=result.query,
        landscape=KnowledgeLandscapeSchema(
            explicit_topics=result.landscape.explicit_topics,
            implicit_topics=result.landscape.implicit_topics,
            related_concepts=result.landscape.related_concepts,
            contrasting_views=result.landscape.contrasting_views,
        ),
        knowledge_gaps=[
            KnowledgeGapSchema(
                gap=g.gap,
                description=g.description,
                importance=g.importance,
                suggested_search=g.suggested_search,
            )
            for g in result.knowledge_gaps
        ],
        sources=[
            ScoredSourceSchema(
                id=s.source.id,
                title=s.source.title,
                url=s.source.url,
                content=s.source.content or "",
                score=s.source.score,
                connector=s.source.connector,
                relevance_score=s.relevance_score,
                gaps_addressed=s.gaps_addressed,
                unique_value=s.unique_value,
                recommended_priority=s.recommended_priority,
            )
            for s in result.sources
        ],
        synthesis_preview=result.synthesis_preview,
        recommended_deep_dives=result.recommended_deep_dives,
        connectors_used=aggregator.get_active_connectors(),
    )


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Pure synthesis of pre-gathered content (mirrors perplexity_research).

    Optimized for the SYNTHESIS workflow:
    - Takes content already fetched by Ref/Exa/Jina
    - Weaves into coherent narrative with attribution
    - NO additional searching - pure aggregation

    This is the final step after Triple Stack research.
    """
    llm_client = _get_llm_client()
    aggregator = SynthesisAggregator(
        llm_client=llm_client,
        model=settings.llm_model,
    )

    # Convert request sources to internal format
    sources = [
        PreGatheredSource(
            origin=s.origin,
            url=s.url,
            title=s.title,
            content=s.content,
            source_type=s.source_type,
            metadata=s.metadata,
        )
        for s in request.sources
    ]

    # Map style string to enum
    style_map = {
        "comprehensive": SynthesisStyle.COMPREHENSIVE,
        "concise": SynthesisStyle.CONCISE,
        "comparative": SynthesisStyle.COMPARATIVE,
        "tutorial": SynthesisStyle.TUTORIAL,
        "academic": SynthesisStyle.ACADEMIC,
    }
    style = style_map.get(request.style, SynthesisStyle.COMPREHENSIVE)

    try:
        result = await aggregator.synthesize(
            query=request.query,
            sources=sources,
            style=style,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis error: {e}"
        )

    return SynthesizeResponse(
        query=request.query,
        content=result.content,
        citations=[
            CitationSchema(
                id=str(c.get("number", "")),
                title=c.get("title", ""),
                url=c.get("url", ""),
            )
            for c in result.citations
        ],
        source_attribution=[
            SynthesisAttributionSchema(origin=origin, contribution=contrib)
            for origin, contrib in result.source_attribution.items()
        ],
        confidence=result.confidence,
        style_used=result.style_used.value,
        word_count=result.word_count,
        model=settings.llm_model,
    )


@router.post("/reason", response_model=ReasonResponse)
async def reason(request: ReasonRequest):
    """
    Advanced reasoning with chain-of-thought (mirrors perplexity_reason).

    Same as /synthesize but with explicit reasoning process:
    1. Analyze key aspects of the query
    2. Map source relevance to each aspect
    3. Identify contradictions between sources
    4. Determine confident vs uncertain claims
    5. Synthesize with reasoning trace
    """
    llm_client = _get_llm_client()
    aggregator = SynthesisAggregator(
        llm_client=llm_client,
        model=settings.llm_model,
    )

    # Convert request sources to internal format
    sources = [
        PreGatheredSource(
            origin=s.origin,
            url=s.url,
            title=s.title,
            content=s.content,
            source_type=s.source_type,
            metadata=s.metadata,
        )
        for s in request.sources
    ]

    # Map style
    style_map = {
        "comprehensive": SynthesisStyle.COMPREHENSIVE,
        "concise": SynthesisStyle.CONCISE,
        "comparative": SynthesisStyle.COMPARATIVE,
        "tutorial": SynthesisStyle.TUTORIAL,
        "academic": SynthesisStyle.ACADEMIC,
    }
    style = style_map.get(request.style, SynthesisStyle.COMPREHENSIVE)

    try:
        result = await aggregator.synthesize_with_reasoning(
            query=request.query,
            sources=sources,
            style=style,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning error: {e}"
        )

    # Try to extract reasoning from full response if present
    # The synthesize_with_reasoning returns just the synthesis portion,
    # but we might want to expose reasoning in the future
    reasoning = None  # Reasoning trace not exposed in current implementation

    return ReasonResponse(
        query=request.query,
        content=result.content,
        reasoning=reasoning,
        citations=[
            CitationSchema(
                id=str(c.get("number", "")),
                title=c.get("title", ""),
                url=c.get("url", ""),
            )
            for c in result.citations
        ],
        source_attribution=[
            SynthesisAttributionSchema(origin=origin, contribution=contrib)
            for origin, contrib in result.source_attribution.items()
        ],
        confidence=result.confidence,
        word_count=result.word_count,
        model=settings.llm_model,
    )
