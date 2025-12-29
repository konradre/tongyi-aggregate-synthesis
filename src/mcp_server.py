"""Stdio MCP server for local inference.

Exposes research tools via Model Context Protocol using stdio transport.
Optimized for local 3090 inference - no network overhead.

Usage:
    python -m src.mcp_server

Claude Desktop config:
    {
        "mcpServers": {
            "research-tool": {
                "command": "python",
                "args": ["-m", "src.mcp_server"],
                "cwd": "/path/to/research/tool"
            }
        }
    }
"""

import asyncio
import hashlib
import json
from typing import Literal
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import AsyncOpenAI

from .config import settings
from .search import SearchAggregator
from .synthesis import (
    SynthesisEngine,
    SynthesisAggregator,
    SynthesisStyle,
    PreGatheredSource,
    # P0/P1 components
    SourceQualityGate,
    QualityDecision,
    ContradictionDetector,
    OutlineGuidedSynthesizer,
    RCSPreprocessor,
    get_preset,
)
from .discovery import (
    Explorer,
    # P1 focus modes
    FocusModeType,
    FocusModeSelector,
    get_focus_mode,
    get_search_params,
)
from .cache import cache, cached


# Initialize MCP server
server = Server("research-tool")


def _get_llm_client() -> AsyncOpenAI:
    """Get OpenAI-compatible LLM client for local inference."""
    return AsyncOpenAI(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
    )


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available research tools."""
    return [
        Tool(
            name="search",
            description="""Multi-source search with RRF (Reciprocal Rank Fusion). Returns ranked results from SearXNG, Tavily, and LinkUp.

Use this when you need raw search results without synthesis. Returns URLs, titles, snippets, and relevance scores.
For research with synthesis, use 'research' instead. For pre-gathered sources, use 'synthesize'.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 10, "description": "Results per source (1-50)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="research",
            description="""Full research pipeline: search + LLM synthesis with citations. Standalone end-to-end tool.

Pipeline: Multi-source search → Source aggregation → LLM synthesis → Citation formatting

Use this for quick research when you don't need fine control. For more control:
- Use 'search' to get sources, then 'synthesize' with a preset
- Use 'discover' for exploratory research with gap analysis""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                    "top_k": {"type": "integer", "default": 10, "description": "Results per source"},
                    "reasoning_effort": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium",
                        "description": "Depth of analysis: low (concise, fast), medium (balanced), high (academic, thorough)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask",
            description="""Quick conversational answer using local LLM. No search, direct response from model knowledge.

Use for:
- Simple factual questions the model likely knows
- Clarification or follow-up on previous responses
- When you already provided context and just need reasoning

Do NOT use for current events, specific documentation, or anything requiring fresh data.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question to answer"},
                    "context": {"type": "string", "description": "Optional context to consider (prior conversation, relevant facts)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="discover",
            description="""Exploratory discovery with knowledge gap analysis. Identifies what's known and unknown about a topic.

Each focus_mode configures:
- Gap categories: What types of missing information to look for
- Search expansion: Whether to broaden searches beyond the query
- Priority engines: Which search backends to prioritize

Use this for cold-start exploration when you don't know what you don't know.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to explore"},
                    "top_k": {"type": "integer", "default": 10, "description": "Results per source"},
                    "identify_gaps": {"type": "boolean", "default": True, "description": "Analyze knowledge gaps"},
                    "focus_mode": {
                        "type": "string",
                        "enum": ["general", "academic", "documentation", "comparison", "debugging", "tutorial", "news"],
                        "default": "general",
                        "description": """Domain-specific discovery mode. Each mode tailors gap analysis and search strategy:

- general: Broad technical questions. Gaps: documentation, examples, alternatives, gotchas. Search expansion ON.
- academic: Research papers, citations. Gaps: methodology, limitations, replications, critiques. Search expansion ON. Use for scientific topics.
- documentation: Official docs, API refs. Gaps: api_reference, examples, migration, changelog, configuration. Search expansion OFF (stays focused). Use for library/framework questions.
- comparison: X vs Y evaluations. Gaps: criteria, tradeoffs, edge_cases, benchmarks, community_preference. Search expansion ON. Use for "which should I use" questions.
- debugging: Error investigation. Gaps: error_context, similar_issues, root_cause, workarounds, fixes. Search expansion ON. Use for error messages, stack traces.
- tutorial: How-to guides. Gaps: prerequisites, step_by_step, common_mistakes, next_steps. Search expansion OFF. Use for learning/getting started.
- news: Recent events. Gaps: announcement, reaction, impact, timeline. Search expansion ON, time-filtered to recent. Use for "latest" or "announced" queries.""",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="synthesize",
            description="""Synthesize pre-gathered content into coherent analysis. Use when you already have sources from other tools.

Pipeline components (controlled by preset):
- Quality Gate (CRAG): Filter low-quality/irrelevant sources before synthesis
- RCS: Query-focused summarization - summarize each source specifically for the question
- Contradiction Detection: Find conflicting claims between sources
- Outline-Guided Synthesis: Plan structure before writing (better coverage)

Without preset: Direct synthesis, no preprocessing.
With preset: Runs the configured pipeline components for that preset.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Synthesis focus/question"},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "content": {"type": "string"},
                                "origin": {"type": "string", "description": "Source origin (ref, exa, jina, external)"},
                                "source_type": {"type": "string", "description": "Type (documentation, code, article)"},
                            },
                            "required": ["title", "content"],
                        },
                        "description": "Pre-gathered source documents",
                    },
                    "style": {
                        "type": "string",
                        "enum": ["comprehensive", "concise", "comparative", "academic", "tutorial"],
                        "default": "comprehensive",
                        "description": "Output format/length: comprehensive (full analysis with sections), concise (2-4 paragraphs), comparative (side-by-side for X vs Y), academic (formal with citations), tutorial (step-by-step guide)",
                    },
                    "preset": {
                        "type": "string",
                        "enum": ["comprehensive", "fast", "contracrow", "academic", "tutorial"],
                        "description": """Processing pipeline preset. Each preset enables different components:

- comprehensive: FULL PIPELINE. Quality gate → RCS summarization → Contradiction detection → Outline-guided synthesis. Best quality, slowest. Use for important research.
- fast: MINIMAL. Direct synthesis only, no preprocessing. Fastest, use when sources are already high-quality and you need speed.
- contracrow: CONTRADICTION-FOCUSED. Quality gate → Contradiction detection → Standard synthesis. Highlights conflicting claims. Use when sources may disagree (comparisons, controversial topics).
- academic: SCHOLARLY. Quality gate → RCS → Outline-guided synthesis. Structured output with proper citations. Use for formal reports, documentation.
- tutorial: STEP-BY-STEP. Quality gate → Outline-guided synthesis. Produces structured how-to format. Use for guides, tutorials, explanations.

Omit preset for direct synthesis without preprocessing (equivalent to 'fast' but explicit).""",
                    },
                },
                "required": ["query", "sources"],
            },
        ),
        Tool(
            name="reason",
            description="""Deep reasoning with chain-of-thought analysis. For complex problems requiring step-by-step logical breakdown.

Use this when you need:
- Multi-step logical deduction (A implies B, B implies C, therefore...)
- Trade-off analysis with explicit criteria weighting
- Architectural decision reasoning
- Root cause analysis for complex bugs
- Evaluating multiple approaches systematically

Do NOT use for simple questions - use 'ask' instead. Do NOT use for research - use 'discover' or 'synthesize'.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Problem or question requiring reasoning"},
                    "context": {"type": "string", "description": "Background information, constraints, or gathered sources to reason over"},
                    "reasoning_depth": {
                        "type": "string",
                        "enum": ["shallow", "moderate", "deep"],
                        "default": "moderate",
                        "description": """How thorough the reasoning chain should be:

- shallow: Quick logical check, 2-3 reasoning steps. Use for simple deductions or sanity checks.
- moderate: Standard analysis, 4-6 reasoning steps. Default for most decisions and trade-off analysis.
- deep: Exhaustive chain-of-thought, 7+ steps with backtracking. Use for critical architectural decisions, complex debugging, or when stakes are high.""",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a research tool."""

    if name == "search":
        return await _tool_search(arguments)
    elif name == "research":
        return await _tool_research(arguments)
    elif name == "ask":
        return await _tool_ask(arguments)
    elif name == "discover":
        return await _tool_discover(arguments)
    elif name == "synthesize":
        return await _tool_synthesize(arguments)
    elif name == "reason":
        return await _tool_reason(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


@cached(tier="search", key_params=["top_k"])
async def _tool_search(args: dict) -> list[TextContent]:
    """Execute search tool."""
    query = args["query"]
    top_k = args.get("top_k", 10)

    aggregator = SearchAggregator()
    sources, raw_results = await aggregator.search(query=query, top_k=top_k)

    # Format results
    lines = [f"# Search Results for: {query}\n"]
    for i, s in enumerate(sources, 1):
        lines.append(f"## [{i}] {s.title}")
        lines.append(f"**URL:** {s.url}")
        lines.append(f"**Source:** {s.connector} (score: {s.score:.3f})")
        lines.append(f"\n{s.content[:500]}{'...' if len(s.content) > 500 else ''}\n")

    lines.append(f"\n---\n*{len(sources)} results from {list(raw_results.keys())}*")

    return [TextContent(type="text", text="\n".join(lines))]


@cached(tier="research", key_params=["reasoning_effort"])
async def _tool_research(args: dict) -> list[TextContent]:
    """Execute full research pipeline."""
    query = args["query"]
    top_k = args.get("top_k", 10)
    reasoning_effort = args.get("reasoning_effort", "medium")

    # Search
    aggregator = SearchAggregator()
    sources, _ = await aggregator.search(query=query, top_k=top_k)

    if not sources:
        return [TextContent(type="text", text="No sources found for query.")]

    # Synthesize
    client = _get_llm_client()
    engine = SynthesisEngine(client, model=settings.llm_model)

    effort_map = {"low": SynthesisStyle.CONCISE, "medium": SynthesisStyle.COMPREHENSIVE, "high": SynthesisStyle.ACADEMIC}
    style = effort_map.get(reasoning_effort, SynthesisStyle.COMPREHENSIVE)

    result = await engine.synthesize(
        query=query,
        sources=sources,
        style=style,
    )

    # Format with citations
    lines = [f"# Research: {query}\n"]
    lines.append(result.content)
    lines.append("\n## Citations\n")
    for c in result.citations:
        lines.append(f"- [{c.id}] [{c.title}]({c.url})")

    return [TextContent(type="text", text="\n".join(lines))]


@cached(tier="ask")
async def _tool_ask(args: dict) -> list[TextContent]:
    """Quick conversational answer."""
    query = args["query"]
    context = args.get("context", "")

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

    return [TextContent(type="text", text=response.choices[0].message.content)]


@cached(tier="discover", key_params=["focus_mode", "identify_gaps"])
async def _tool_discover(args: dict) -> list[TextContent]:
    """Exploratory discovery with gap analysis and P1 focus modes."""
    query = args["query"]
    top_k = args.get("top_k", 10)
    identify_gaps = args.get("identify_gaps", True)
    focus_mode_name = args.get("focus_mode")

    client = _get_llm_client()
    aggregator = SearchAggregator()

    # P1: Focus mode selection
    if focus_mode_name:
        # Explicit focus mode
        try:
            focus_mode_type = FocusModeType(focus_mode_name.lower())
        except ValueError:
            focus_mode_type = FocusModeType.GENERAL
        focus_mode = get_focus_mode(focus_mode_name)
    else:
        # Auto-detect focus mode from query
        selector = FocusModeSelector(client, model=settings.llm_model)
        focus_mode_type = await selector.select(query)
        focus_mode = selector.get_mode_config(focus_mode_type)

    # Get search params from focus mode
    search_params = get_search_params(focus_mode_type)
    expand_searches = search_params.get("expand_searches", True)

    explorer = Explorer(client, aggregator, model=settings.llm_model)

    result = await explorer.discover(
        query=query,
        top_k=top_k,
        expand_searches=expand_searches,
        fill_gaps=identify_gaps,
    )

    # Format output with focus mode info
    lines = [f"# Discovery: {query}\n"]
    lines.append(f"*Focus Mode: {focus_mode.name}* - {focus_mode.description}\n")

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
        # Filter gaps by focus mode's gap_categories if available
        gap_categories = focus_mode.gap_categories
        for gap in result.knowledge_gaps:
            gap_type = getattr(gap, 'category', None) or gap.gap.lower()
            # Show all gaps, but mark focus-relevant ones
            relevance = "🎯 " if any(cat in gap_type for cat in gap_categories) else ""
            lines.append(f"- {relevance}**{gap.gap}** ({gap.importance}): {gap.description}")

    lines.append(f"\n## Sources ({len(result.sources)})\n")
    for s in result.sources[:5]:
        lines.append(f"- [{s.source.title}]({s.source.url})")

    if hasattr(result, 'recommended_deep_dives') and result.recommended_deep_dives:
        lines.append("\n## Recommended Deep Dives\n")
        for url in result.recommended_deep_dives[:5]:
            lines.append(f"- {url}")

    # Add focus mode metadata
    lines.append(f"\n---\n*Search expansion: {'enabled' if expand_searches else 'disabled'}*")
    if focus_mode.gap_categories:
        lines.append(f"*Gap focus: {', '.join(focus_mode.gap_categories)}*")

    return [TextContent(type="text", text="\n".join(lines))]


async def _tool_synthesize(args: dict) -> list[TextContent]:
    """Synthesize pre-gathered content with optional P1 pipeline."""
    query = args["query"]
    sources_data = args["sources"]
    style_name = args.get("style", "comprehensive")
    preset_name = args.get("preset")

    # Source-aware cache key (hash sources content)
    sources_hash = hashlib.md5(
        json.dumps(sorted([s.get("url", s.get("title", "")) for s in sources_data])).encode()
    ).hexdigest()[:8]
    cache_extra = f"preset={preset_name}:style={style_name}:src={sources_hash}"

    cached_result = cache.get(query, tier="synthesis", extra=cache_extra)
    if cached_result is not None:
        return [TextContent(type="text", text=f"*[cached]*\n\n{cached_result}")]

    # Convert to PreGatheredSource
    sources = [
        PreGatheredSource(
            origin=s.get("origin", "external"),
            url=s.get("url", ""),
            title=s["title"],
            content=s["content"],
            source_type=s.get("source_type", "article"),
        )
        for s in sources_data
    ]

    client = _get_llm_client()

    style_map = {
        "comprehensive": SynthesisStyle.COMPREHENSIVE,
        "concise": SynthesisStyle.CONCISE,
        "comparative": SynthesisStyle.COMPARATIVE,
        "academic": SynthesisStyle.ACADEMIC,
        "tutorial": SynthesisStyle.TUTORIAL,
    }
    style = style_map.get(style_name, SynthesisStyle.COMPREHENSIVE)

    # P1 pipeline when preset specified
    if preset_name:
        preset = get_preset(preset_name)
        metadata = {"preset": preset.name}
        processed_sources = sources

        # P0: Quality gate (CRAG-style filtering)
        if preset.run_quality_gate:
            quality_gate = SourceQualityGate(client, model=settings.llm_model)
            gate_result = await quality_gate.evaluate(
                query=query,
                sources=sources,  # Pass PreGatheredSource objects directly
            )
            # Keep only good sources that passed quality gate
            if gate_result.decision in (QualityDecision.PARTIAL, QualityDecision.PROCEED):
                processed_sources = gate_result.good_sources if gate_result.good_sources else sources
            else:
                # REJECT - still use original sources but note the warning
                processed_sources = sources
            metadata["quality_gate"] = {
                "passed": len(gate_result.good_sources),
                "filtered": len(gate_result.rejected_sources),
                "avg_quality": gate_result.avg_quality,
            }

        # P1: RCS preprocessing (query-focused summarization)
        if preset.use_rcs and processed_sources:
            rcs = RCSPreprocessor(client, model=settings.llm_model)
            rcs_result = await rcs.prepare(
                query=query,
                sources=processed_sources,  # Pass PreGatheredSource objects directly
                top_k=min(len(processed_sources), 5),
            )
            # Replace sources with RCS-processed summaries
            if rcs_result.summaries:
                # Create new sources with focused content from RCS summaries
                rcs_processed = []
                for ctx_summary in rcs_result.summaries:
                    # Use the contextual summary as the new content
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

        # P0: Contradiction detection (PaperQA2-style)
        contradictions = []
        if preset.detect_contradictions and processed_sources:
            detector = ContradictionDetector(client, model=settings.llm_model)
            contradictions = await detector.detect(
                query=query,
                sources=processed_sources,  # Pass PreGatheredSource objects directly
            )
            metadata["contradictions_found"] = len(contradictions)

        # P1: Outline-guided synthesis (SciRAG-style)
        if preset.use_outline:
            outline_synth = OutlineGuidedSynthesizer(client, model=settings.llm_model)
            result = await outline_synth.synthesize(
                query=query,
                sources=processed_sources,  # Pass PreGatheredSource objects directly
                style=style,
            )
        else:
            # Standard synthesis
            aggregator = SynthesisAggregator(client, model=settings.llm_model)
            result = await aggregator.synthesize(
                query=query,
                sources=processed_sources,
                style=style,
            )

        # Format output with P1 metadata
        lines = [f"# Synthesis: {query}\n"]
        lines.append(f"*Preset: {preset.name}*\n")
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

        # Cache result
        output = "\n".join(lines)
        cache.set(query, output, tier="synthesis", extra=cache_extra)
        return [TextContent(type="text", text=output)]

    # Standard synthesis (no preset)
    aggregator = SynthesisAggregator(client, model=settings.llm_model)
    result = await aggregator.synthesize(
        query=query,
        sources=sources,
        style=style,
    )

    lines = [f"# Synthesis: {query}\n"]
    lines.append(result.content)

    if result.citations:
        lines.append("\n## Citations\n")
        for c in result.citations:
            lines.append(f"- [{c.get('number', '?')}] [{c.get('title', 'Unknown')}]({c.get('url', '')})")

    # Cache result
    output = "\n".join(lines)
    cache.set(query, output, tier="synthesis", extra=cache_extra)
    return [TextContent(type="text", text=output)]


@cached(tier="reason", key_params=["reasoning_depth"])
async def _tool_reason(args: dict) -> list[TextContent]:
    """Deep reasoning with chain-of-thought."""
    query = args["query"]
    context = args.get("context", "")
    depth = args.get("reasoning_depth", "moderate")

    client = _get_llm_client()

    depth_prompts = {
        "shallow": "Provide a brief analysis.",
        "moderate": "Think through this step-by-step, showing your reasoning.",
        "deep": "Analyze this comprehensively. Consider multiple perspectives, potential counterarguments, edge cases, and implications. Show detailed chain-of-thought reasoning.",
    }

    system_prompt = f"""You are a reasoning assistant. {depth_prompts.get(depth, depth_prompts['moderate'])}

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
        temperature=0.7,  # Slightly lower for reasoning
        max_tokens=settings.llm_max_tokens,
    )

    return [TextContent(type="text", text=response.choices[0].message.content)]


async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
