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
    # P0 Enhancement schemas
    ContradictionSchema,
    QualityGateSchema,
    VerifiedClaimSchema,
    DiscoverRequestEnhanced,
    SynthesizeRequestEnhanced,
    SynthesizeResponseEnhanced,
)
from ..search import SearchAggregator
from ..synthesis import (
    SynthesisEngine,
    SynthesisAggregator,
    SynthesisStyle,
    PreGatheredSource,
    # P0 Enhancements
    SourceQualityGate,
    QualityDecision,
    ContradictionDetector,
    CitationVerifier,
    extract_claims_with_citations,
)
from ..discovery import (
    Explorer,
    # P0 Enhancements
    ConnectorRouter,
    QueryExpander,
    GapFiller,
)
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
async def discover(request: DiscoverRequest | DiscoverRequestEnhanced):
    """
    Exploratory discovery with breadth expansion (mirrors perplexity_search).

    Optimized for the EXPLORATORY workflow:
    1. Expand knowledge landscape (explicit, implicit, related, contrasting)
    2. Identify knowledge gaps in the query
    3. Score sources by gap coverage
    4. Recommend URLs for deep dives (Jina parallel_read)

    P0 Enhancements (when using DiscoverRequestEnhanced):
    - Query expansion via HyDE-style variant generation
    - Adaptive connector routing based on query type
    - Iterative gap-filling for coverage

    This sets the table for targeted research expansion.
    """
    aggregator = SearchAggregator()
    llm_client = _get_llm_client()

    if not aggregator.connectors:
        raise HTTPException(
            status_code=503,
            detail="No search connectors configured"
        )

    # Initialize P0 Enhancement components based on request type
    router_component = None
    expander_component = None
    gap_filler_component = None

    # Check if using enhanced request with P0 options
    use_routing = getattr(request, 'use_adaptive_routing', True)
    fill_gaps = getattr(request, 'fill_gaps', True)

    if use_routing:
        router_component = ConnectorRouter()

    if request.expand_searches:
        expander_component = QueryExpander(
            llm_client=llm_client,
            model=settings.llm_model,
        )

    if fill_gaps:
        gap_filler_component = GapFiller(
            search_aggregator=aggregator,
        )

    explorer = Explorer(
        llm_client=llm_client,
        search_aggregator=aggregator,
        model=settings.llm_model,
        router=router_component,
        expander=expander_component,
        gap_filler=gap_filler_component,
    )

    try:
        result = await explorer.discover(
            query=request.query,
            top_k=request.top_k,
            expand_searches=request.expand_searches,
            fill_gaps=fill_gaps,
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


# =============================================================================
# P0 Enhanced Endpoint
# =============================================================================


@router.post("/synthesize/enhanced", response_model=SynthesizeResponseEnhanced)
async def synthesize_enhanced(request: SynthesizeRequestEnhanced):
    """
    Enhanced synthesis with P0 reliability features.

    Adds to standard /synthesize:
    1. Source Quality Gate - Evaluate sources BEFORE synthesis (CRAG)
    2. Contradiction Detection - Surface source disagreements (PaperQA2)
    3. Citation Verification - NLI-verify claims against evidence (VeriCite)

    Use this endpoint when citation reliability is critical.
    """
    llm_client = _get_llm_client()

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

    quality_gate_result = None
    contradictions_list = []
    verified_claims_list = []
    sources_for_synthesis = sources

    # Step 1: Quality Gate (CRAG-style)
    if request.run_quality_gate:
        quality_gate = SourceQualityGate(
            llm_client=llm_client,
            model=settings.llm_model,
        )
        gate_result = await quality_gate.evaluate(request.query, sources)

        quality_gate_result = QualityGateSchema(
            decision=gate_result.decision.value,
            avg_quality=gate_result.avg_quality,
            passed_count=len(gate_result.good_sources),
            rejected_count=len(gate_result.rejected_sources),
            suggestion=gate_result.suggestion,
        )

        if gate_result.decision == QualityDecision.REJECT:
            # Return early with rejection
            return SynthesizeResponseEnhanced(
                query=request.query,
                content=f"Source quality insufficient. {gate_result.suggestion or 'Try gathering more relevant sources.'}",
                citations=[],
                source_attribution=[],
                confidence=0.0,
                style_used=request.style,
                word_count=0,
                model=settings.llm_model,
                quality_gate=quality_gate_result,
                contradictions=[],
                verified_claims=[],
            )
        elif gate_result.decision == QualityDecision.PARTIAL:
            # Use only good sources
            sources_for_synthesis = gate_result.good_sources

    # Step 2: Contradiction Detection (PaperQA2-style)
    if request.detect_contradictions and len(sources_for_synthesis) >= 2:
        detector = ContradictionDetector(
            llm_client=llm_client,
            model=settings.llm_model,
        )
        contradictions = await detector.detect(request.query, sources_for_synthesis)

        contradictions_list = [
            ContradictionSchema(
                topic=c.topic,
                position_a=c.position_a,
                source_a=c.source_a,
                position_b=c.position_b,
                source_b=c.source_b,
                severity=c.severity.value,
                resolution_hint=c.resolution_hint,
            )
            for c in contradictions
        ]

        # Inject contradiction awareness into synthesis
        contradiction_context = detector.format_for_synthesis(contradictions)
    else:
        contradiction_context = ""

    # Step 3: Synthesize with contradiction awareness
    aggregator = SynthesisAggregator(
        llm_client=llm_client,
        model=settings.llm_model,
    )

    style_map = {
        "comprehensive": SynthesisStyle.COMPREHENSIVE,
        "concise": SynthesisStyle.CONCISE,
        "comparative": SynthesisStyle.COMPARATIVE,
        "tutorial": SynthesisStyle.TUTORIAL,
        "academic": SynthesisStyle.ACADEMIC,
    }
    style = style_map.get(request.style, SynthesisStyle.COMPREHENSIVE)

    try:
        # If contradictions found, append to synthesis context
        if contradiction_context:
            # Inject contradiction awareness by temporarily modifying sources
            enhanced_sources = sources_for_synthesis.copy()
            if enhanced_sources:
                # Add contradiction context to first source for LLM awareness
                first = enhanced_sources[0]
                enhanced_sources[0] = PreGatheredSource(
                    origin=first.origin,
                    url=first.url,
                    title=first.title,
                    content=f"{first.content}\n\n{contradiction_context}",
                    source_type=first.source_type,
                    metadata=first.metadata,
                )
            result = await aggregator.synthesize(
                query=request.query,
                sources=enhanced_sources,
                style=style,
                max_tokens=request.max_tokens,
            )
        else:
            result = await aggregator.synthesize(
                query=request.query,
                sources=sources_for_synthesis,
                style=style,
                max_tokens=request.max_tokens,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced synthesis error: {e}"
        )

    # Step 4: Citation Verification (VeriCite-style, optional)
    if request.verify_citations and result.citations:
        verifier = CitationVerifier()
        claims_with_citations = extract_claims_with_citations(
            result.content,
            sources_for_synthesis,
        )

        for claim, evidence, source_num in claims_with_citations[:10]:  # Limit to first 10
            verified = verifier.verify(claim, evidence, source_num)
            verified_claims_list.append(VerifiedClaimSchema(
                claim=verified.claim,
                source_number=verified.source_number,
                label=verified.label,
                confidence=verified.confidence,
            ))

    return SynthesizeResponseEnhanced(
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
        quality_gate=quality_gate_result,
        contradictions=contradictions_list,
        verified_claims=verified_claims_list,
    )
