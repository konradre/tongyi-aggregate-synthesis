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
)
from .discovery import Explorer


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
            description="Multi-source search with RRF fusion. Returns ranked results from SearXNG, Tavily, and LinkUp.",
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
            description="Full research pipeline: search + LLM synthesis with citations. Uses local Tongyi model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                    "top_k": {"type": "integer", "default": 10, "description": "Results per source"},
                    "reasoning_effort": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium",
                        "description": "Depth of analysis",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask",
            description="Quick conversational answer using local LLM. No search, direct response.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question to answer"},
                    "context": {"type": "string", "description": "Optional context to consider"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="discover",
            description="Exploratory discovery with knowledge gap analysis. Identifies what's known and unknown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to explore"},
                    "top_k": {"type": "integer", "default": 10, "description": "Results per source"},
                    "identify_gaps": {"type": "boolean", "default": True, "description": "Analyze knowledge gaps"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="synthesize",
            description="Synthesize pre-gathered content into coherent analysis. Use when you already have sources.",
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
                            },
                            "required": ["title", "content"],
                        },
                        "description": "Pre-gathered source documents",
                    },
                    "style": {
                        "type": "string",
                        "enum": ["comprehensive", "concise", "technical", "explanatory"],
                        "default": "comprehensive",
                    },
                },
                "required": ["query", "sources"],
            },
        ),
        Tool(
            name="reason",
            description="Deep reasoning with chain-of-thought. For complex analysis requiring step-by-step logic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Problem or question requiring reasoning"},
                    "context": {"type": "string", "description": "Background information"},
                    "reasoning_depth": {
                        "type": "string",
                        "enum": ["shallow", "moderate", "deep"],
                        "default": "moderate",
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

    effort_map = {"low": SynthesisStyle.CONCISE, "medium": SynthesisStyle.COMPREHENSIVE, "high": SynthesisStyle.TECHNICAL}
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


async def _tool_discover(args: dict) -> list[TextContent]:
    """Exploratory discovery with gap analysis."""
    query = args["query"]
    top_k = args.get("top_k", 10)
    identify_gaps = args.get("identify_gaps", True)

    client = _get_llm_client()
    explorer = Explorer(client, model=settings.llm_model)

    result = await explorer.explore(
        query=query,
        top_k=top_k,
        identify_gaps=identify_gaps,
    )

    lines = [f"# Discovery: {query}\n"]
    lines.append("## Knowledge Landscape\n")
    lines.append(result.landscape.summary if hasattr(result, 'landscape') else "Exploration complete.")

    if identify_gaps and hasattr(result, 'gaps') and result.gaps:
        lines.append("\n## Knowledge Gaps\n")
        for gap in result.gaps:
            lines.append(f"- **{gap.category}**: {gap.description}")

    lines.append(f"\n## Sources ({len(result.sources)})\n")
    for s in result.sources[:5]:
        lines.append(f"- [{s.title}]({s.url})")

    return [TextContent(type="text", text="\n".join(lines))]


async def _tool_synthesize(args: dict) -> list[TextContent]:
    """Synthesize pre-gathered content."""
    query = args["query"]
    sources_data = args["sources"]
    style_name = args.get("style", "comprehensive")

    # Convert to PreGatheredSource
    sources = [
        PreGatheredSource(
            title=s["title"],
            url=s.get("url", ""),
            content=s["content"],
        )
        for s in sources_data
    ]

    client = _get_llm_client()
    aggregator = SynthesisAggregator(client, model=settings.llm_model)

    style_map = {
        "comprehensive": SynthesisStyle.COMPREHENSIVE,
        "concise": SynthesisStyle.CONCISE,
        "technical": SynthesisStyle.TECHNICAL,
        "explanatory": SynthesisStyle.EXPLANATORY,
    }
    style = style_map.get(style_name, SynthesisStyle.COMPREHENSIVE)

    result = await aggregator.synthesize_pregathered(
        query=query,
        sources=sources,
        style=style,
    )

    lines = [f"# Synthesis: {query}\n"]
    lines.append(result.content)

    if result.attributions:
        lines.append("\n## Attributions\n")
        for a in result.attributions:
            lines.append(f"- {a.source_title}: {a.claim}")

    return [TextContent(type="text", text="\n".join(lines))]


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
