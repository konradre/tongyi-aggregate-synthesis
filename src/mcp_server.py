"""FastMCP server for local inference research tools.

Exposes research tools via Model Context Protocol using stdio transport.
Refactored to use FastMCP for reliable stdio handling.

Usage:
    python -m src.mcp_server
"""

import hashlib
import json
import os
from typing import Literal

# Suppress FastMCP logging before import to avoid polluting stdio transport
os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"

from fastmcp import FastMCP
import fastmcp
fastmcp.settings.log_level = "ERROR"
from openai import AsyncOpenAI

from .config import settings
from .search import SearchAggregator
from .synthesis import (
    SynthesisEngine,
    SynthesisAggregator,
    SynthesisStyle,
    PreGatheredSource,
    SourceQualityGate,
    QualityDecision,
    ContradictionDetector,
    OutlineGuidedSynthesizer,
    RCSPreprocessor,
    get_preset,
)
from .discovery import (
    Explorer,
    FocusModeType,
    FocusModeSelector,
    get_focus_mode,
    get_search_params,
)
from .cache import cache, cached


# Initialize FastMCP server
mcp = FastMCP("deepresearch")


def _get_llm_client() -> AsyncOpenAI:
    """Get OpenAI-compatible LLM client for local inference."""
    return AsyncOpenAI(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
    )


@mcp.tool()
async def search(query: str, top_k: int = 10) -> str:
    """Multi-source search with RRF (Reciprocal Rank Fusion).

    Returns ranked results from SearXNG, Tavily, and LinkUp.
    Use for raw search results without synthesis.

    Args:
        query: Search query
        top_k: Results per source (1-50)
    """
    aggregator = SearchAggregator()
    sources, raw_results = await aggregator.search(query=query, top_k=top_k)

    lines = [f"# Search Results for: {query}\n"]
    for i, s in enumerate(sources, 1):
        lines.append(f"## [{i}] {s.title}")
        lines.append(f"**URL:** {s.url}")
        lines.append(f"**Source:** {s.connector} (score: {s.score:.3f})")
        lines.append(f"\n{s.content[:500]}{'...' if len(s.content) > 500 else ''}\n")

    lines.append(f"\n---\n*{len(sources)} results from {list(raw_results.keys())}*")
    return "\n".join(lines)


@mcp.tool()
async def research(
    query: str,
    top_k: int = 10,
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
) -> str:
    """Full research pipeline: search + LLM synthesis with citations.

    Pipeline: Multi-source search → Source aggregation → LLM synthesis → Citation formatting

    Args:
        query: Research query
        top_k: Results per source
        reasoning_effort: Depth of analysis (low=concise, medium=balanced, high=academic)
    """
    aggregator = SearchAggregator()
    sources, _ = await aggregator.search(query=query, top_k=top_k)

    if not sources:
        return "No sources found for query."

    client = _get_llm_client()
    engine = SynthesisEngine(client, model=settings.llm_model)

    effort_map = {
        "low": SynthesisStyle.CONCISE,
        "medium": SynthesisStyle.COMPREHENSIVE,
        "high": SynthesisStyle.ACADEMIC
    }
    style = effort_map.get(reasoning_effort, SynthesisStyle.COMPREHENSIVE)

    result = await engine.synthesize(query=query, sources=sources, style=style)

    lines = [f"# Research: {query}\n"]
    lines.append(result.content)
    lines.append("\n## Citations\n")
    for c in result.citations:
        lines.append(f"- [{c.id}] [{c.title}]({c.url})")

    return "\n".join(lines)


@mcp.tool()
async def ask(query: str, context: str = "") -> str:
    """Quick conversational answer using local LLM.

    No search, direct response from model knowledge.
    Use for simple factual questions or follow-ups.

    Args:
        query: Question to answer
        context: Optional context to consider
    """
    client = _get_llm_client()

    messages = []
    if context:
        messages.append({"role": "system", "content": f"Context: {context}"})
    messages.append({"role": "user", "content": query})

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    return response.choices[0].message.content


@mcp.tool()
async def discover(
    query: str,
    top_k: int = 10,
    identify_gaps: bool = True,
    focus_mode: Literal["general", "academic", "documentation", "comparison", "debugging", "tutorial", "news"] = "general"
) -> str:
    """Exploratory discovery with knowledge gap analysis.

    Identifies what's known and unknown about a topic.
    Use for cold-start exploration.

    Args:
        query: Topic to explore
        top_k: Results per source
        identify_gaps: Analyze knowledge gaps
        focus_mode: Domain-specific discovery mode
    """
    client = _get_llm_client()
    aggregator = SearchAggregator()

    try:
        focus_mode_type = FocusModeType(focus_mode.lower())
    except ValueError:
        focus_mode_type = FocusModeType.GENERAL

    focus_config = get_focus_mode(focus_mode)
    search_params = get_search_params(focus_mode_type)
    expand_searches = search_params.get("expand_searches", True)

    explorer = Explorer(client, aggregator, model=settings.llm_model)
    result = await explorer.discover(
        query=query,
        top_k=top_k,
        expand_searches=expand_searches,
        fill_gaps=identify_gaps,
    )

    lines = [f"# Discovery: {query}\n"]
    lines.append(f"*Focus Mode: {focus_config.name}* - {focus_config.description}\n")

    lines.append("## Knowledge Landscape\n")
    if hasattr(result, 'landscape') and result.landscape:
        landscape = result.landscape
        lines.append(f"**Explicit Topics:** {', '.join(landscape.explicit_topics)}")
        if landscape.implicit_topics:
            lines.append(f"**Implicit Topics:** {', '.join(landscape.implicit_topics)}")
        if landscape.related_concepts:
            lines.append(f"**Related Concepts:** {', '.join(landscape.related_concepts)}")
    else:
        lines.append("Exploration complete.")

    if identify_gaps and hasattr(result, 'knowledge_gaps') and result.knowledge_gaps:
        lines.append("\n## Knowledge Gaps\n")
        gap_categories = focus_config.gap_categories
        for gap in result.knowledge_gaps:
            gap_type = getattr(gap, 'category', None) or gap.gap.lower()
            relevance = "🎯 " if any(cat in gap_type for cat in gap_categories) else ""
            lines.append(f"- {relevance}**{gap.gap}** ({gap.importance}): {gap.description}")

    lines.append(f"\n## Sources ({len(result.sources)})\n")
    for s in result.sources[:5]:
        lines.append(f"- [{s.source.title}]({s.source.url})")

    if hasattr(result, 'recommended_deep_dives') and result.recommended_deep_dives:
        lines.append("\n## Recommended Deep Dives\n")
        for url in result.recommended_deep_dives[:5]:
            lines.append(f"- {url}")

    lines.append(f"\n---\n*Search expansion: {'enabled' if expand_searches else 'disabled'}*")
    if focus_config.gap_categories:
        lines.append(f"*Gap focus: {', '.join(focus_config.gap_categories)}*")

    return "\n".join(lines)


@mcp.tool()
async def synthesize(
    query: str,
    sources: list[dict],
    style: Literal["comprehensive", "concise", "comparative", "academic", "tutorial"] = "comprehensive",
    preset: Literal["comprehensive", "fast", "contracrow", "academic", "tutorial"] | None = None
) -> str:
    """Synthesize pre-gathered content into coherent analysis.

    Use when you already have sources from other tools.

    Args:
        query: Synthesis focus/question
        sources: Pre-gathered source documents with title, content, url, origin, source_type
        style: Output format/length
        preset: Processing pipeline preset (comprehensive, fast, contracrow, academic, tutorial)
    """
    # Source-aware cache key
    sources_hash = hashlib.md5(
        json.dumps(sorted([s.get("url", s.get("title", "")) for s in sources])).encode()
    ).hexdigest()[:8]
    cache_extra = f"preset={preset}:style={style}:src={sources_hash}"

    cached_result = cache.get(query, tier="synthesis", extra=cache_extra)
    if cached_result is not None:
        return f"*[cached]*\n\n{cached_result}"

    # Convert to PreGatheredSource
    pre_sources = [
        PreGatheredSource(
            origin=s.get("origin", "external"),
            url=s.get("url", ""),
            title=s["title"],
            content=s["content"],
            source_type=s.get("source_type", "article"),
        )
        for s in sources
    ]

    client = _get_llm_client()

    style_map = {
        "comprehensive": SynthesisStyle.COMPREHENSIVE,
        "concise": SynthesisStyle.CONCISE,
        "comparative": SynthesisStyle.COMPARATIVE,
        "academic": SynthesisStyle.ACADEMIC,
        "tutorial": SynthesisStyle.TUTORIAL,
    }
    synth_style = style_map.get(style, SynthesisStyle.COMPREHENSIVE)

    if preset:
        preset_config = get_preset(preset)
        metadata = {"preset": preset_config.name}
        processed_sources = pre_sources

        # Quality gate
        if preset_config.run_quality_gate:
            quality_gate = SourceQualityGate(client, model=settings.llm_model)
            gate_result = await quality_gate.evaluate(query=query, sources=pre_sources)
            if gate_result.decision in (QualityDecision.PARTIAL, QualityDecision.PROCEED):
                processed_sources = gate_result.good_sources if gate_result.good_sources else pre_sources
            metadata["quality_gate"] = {
                "passed": len(gate_result.good_sources),
                "filtered": len(gate_result.rejected_sources),
                "avg_quality": gate_result.avg_quality,
            }

        # RCS preprocessing
        if preset_config.use_rcs and processed_sources:
            rcs = RCSPreprocessor(client, model=settings.llm_model)
            rcs_result = await rcs.prepare(query=query, sources=processed_sources, top_k=min(len(processed_sources), 5))
            if rcs_result.summaries:
                rcs_processed = []
                for ctx_summary in rcs_result.summaries:
                    new_source = PreGatheredSource(
                        origin=ctx_summary.source.origin,
                        url=ctx_summary.source.url,
                        title=ctx_summary.source.title,
                        content=f"{ctx_summary.summary}\n\nKey Points:\n" + "\n".join(f"- {p}" for p in ctx_summary.key_points),
                        source_type=ctx_summary.source.source_type,
                    )
                    rcs_processed.append(new_source)
                processed_sources = rcs_processed
            metadata["rcs_applied"] = True
            metadata["rcs_kept"] = len(rcs_result.summaries)

        # Contradiction detection
        contradictions = []
        if preset_config.detect_contradictions and processed_sources:
            detector = ContradictionDetector(client, model=settings.llm_model)
            contradictions = await detector.detect(query=query, sources=processed_sources)
            metadata["contradictions_found"] = len(contradictions)

        # Synthesis
        if preset_config.use_outline:
            outline_synth = OutlineGuidedSynthesizer(client, model=settings.llm_model)
            result = await outline_synth.synthesize(query=query, sources=processed_sources, style=synth_style)
        else:
            aggregator = SynthesisAggregator(client, model=settings.llm_model)
            result = await aggregator.synthesize(query=query, sources=processed_sources, style=synth_style)

        lines = [f"# Synthesis: {query}\n"]
        lines.append(f"*Preset: {preset_config.name}*\n")
        lines.append(result.content)

        if contradictions:
            lines.append("\n## Contradictions Detected\n")
            for c in contradictions:
                lines.append(f"- **{c.topic}** ({c.severity.value}): {c.position_a} vs {c.position_b}")
                if c.resolution_hint:
                    lines.append(f"  - Resolution: {c.resolution_hint}")

        if hasattr(result, 'citations') and result.citations:
            lines.append("\n## Citations\n")
            for c in result.citations:
                if hasattr(c, 'title'):
                    lines.append(f"- [{c.id}] [{c.title}]({c.url})")
                else:
                    lines.append(f"- [{c.get('number', '?')}] [{c.get('title', 'Unknown')}]({c.get('url', '')})")

        if metadata.get("quality_gate"):
            qg = metadata["quality_gate"]
            lines.append(f"\n---\n*Quality gate: {qg['passed']} passed, {qg['filtered']} filtered (avg quality: {qg['avg_quality']:.2f})*")
        if metadata.get("rcs_applied"):
            lines.append(f"*RCS: {metadata.get('rcs_kept', 0)} sources processed*")

        output = "\n".join(lines)
        cache.set(query, output, tier="synthesis", extra=cache_extra)
        return output

    # Standard synthesis (no preset)
    aggregator = SynthesisAggregator(client, model=settings.llm_model)
    result = await aggregator.synthesize(query=query, sources=pre_sources, style=synth_style)

    lines = [f"# Synthesis: {query}\n"]
    lines.append(result.content)

    if result.citations:
        lines.append("\n## Citations\n")
        for c in result.citations:
            lines.append(f"- [{c.get('number', '?')}] [{c.get('title', 'Unknown')}]({c.get('url', '')})")

    output = "\n".join(lines)
    cache.set(query, output, tier="synthesis", extra=cache_extra)
    return output


@mcp.tool()
async def reason(
    query: str,
    context: str = "",
    reasoning_depth: Literal["shallow", "moderate", "deep"] = "moderate"
) -> str:
    """Deep reasoning with chain-of-thought analysis.

    For complex problems requiring step-by-step logical breakdown.

    Args:
        query: Problem or question requiring reasoning
        context: Background information or constraints
        reasoning_depth: How thorough (shallow=2-3 steps, moderate=4-6, deep=7+)
    """
    client = _get_llm_client()

    depth_prompts = {
        "shallow": "Provide a brief analysis.",
        "moderate": "Think through this step-by-step, showing your reasoning.",
        "deep": "Analyze this comprehensively. Consider multiple perspectives, potential counterarguments, edge cases, and implications. Show detailed chain-of-thought reasoning.",
    }

    system_prompt = f"""You are a reasoning assistant. {depth_prompts.get(reasoning_depth, depth_prompts['moderate'])}

Structure your response with clear sections for:
1. Understanding the problem
2. Key considerations
3. Step-by-step reasoning
4. Conclusion"""

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "user", "content": f"Context: {context}"})
    messages.append({"role": "user", "content": query})

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.7,
        max_tokens=settings.llm_max_tokens,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    mcp.run(show_banner=False)
