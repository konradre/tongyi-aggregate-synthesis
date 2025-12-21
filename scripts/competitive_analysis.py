#!/usr/bin/env python3
"""
Competitive analysis framework for comparing our tool vs Perplexity.

Run identical queries through both systems and measure:
- Latency
- Source selection patterns
- Citation accuracy
- Synthesis quality
- Coverage gaps
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx


@dataclass
class AnalysisResult:
    """Results from analyzing a single query."""
    query: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Latency
    our_latency_ms: float = 0
    perplexity_latency_ms: float = 0

    # Sources
    our_sources: list[str] = field(default_factory=list)
    perplexity_sources: list[str] = field(default_factory=list)
    source_overlap: float = 0

    # Citations
    our_citation_count: int = 0
    perplexity_citation_count: int = 0

    # Quality (manual or LLM-evaluated)
    our_quality_score: float = 0
    perplexity_quality_score: float = 0

    # Observations
    notes: str = ""


async def query_our_tool(query: str, base_url: str = "http://localhost:8000") -> dict:
    """Query our research tool."""
    async with httpx.AsyncClient(timeout=120) as client:
        start = time.perf_counter()
        response = await client.post(
            f"{base_url}/research",
            json={"query": query, "reasoning_effort": "medium"}
        )
        latency = (time.perf_counter() - start) * 1000

        data = response.json()
        return {
            "latency_ms": latency,
            "sources": [s.get("url", "") for s in data.get("sources", [])],
            "synthesis": data.get("synthesis", ""),
            "citation_count": len(data.get("sources", [])),
        }


async def analyze_query(query: str) -> AnalysisResult:
    """Run comparative analysis on a single query."""
    result = AnalysisResult(query=query)

    # Query our tool
    try:
        our_result = await query_our_tool(query)
        result.our_latency_ms = our_result["latency_ms"]
        result.our_sources = our_result["sources"]
        result.our_citation_count = our_result["citation_count"]
    except Exception as e:
        result.notes += f"Our tool error: {e}\n"

    # TODO: Query Perplexity API (requires API key)
    # For now, manual comparison

    return result


async def run_analysis(queries: list[str], output_file: str = "analysis_results.json"):
    """Run analysis on multiple queries."""
    results = []

    for query in queries:
        print(f"Analyzing: {query}")
        result = await analyze_query(query)
        results.append(result)

    # Save results
    output_path = Path(output_file)
    with output_path.open("w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    print(f"Results saved to {output_file}")
    return results


# Test queries covering different types
TEST_QUERIES = [
    # Factual
    "What is the current population of Tokyo?",

    # Academic
    "What are the latest advances in transformer architecture efficiency?",

    # Comparative
    "Compare PostgreSQL vs MySQL for high-write workloads",

    # Tutorial
    "How do I implement rate limiting in FastAPI?",

    # Current events
    "What are the latest developments in AI regulation in 2025?",

    # Complex multi-part
    "Explain the tradeoffs between RAG and fine-tuning for domain-specific LLM applications",
]


if __name__ == "__main__":
    asyncio.run(run_analysis(TEST_QUERIES))
