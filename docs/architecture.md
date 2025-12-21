# Architecture

## Overview

Tongyi Aggregate Synthesis is a modular research tool that combines multi-source web search with LLM-powered synthesis. It serves as a self-hosted replacement for Perplexity API.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   /search    │    │  /research   │    │    /ask      │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Search Aggregator                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                  │   │
│  │  │ SearXNG │  │ Tavily  │  │ LinkUp  │   (parallel)     │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘                  │   │
│  │       └───────────┬┴───────────┘                        │   │
│  │                   ▼                                      │   │
│  │            ┌─────────────┐                               │   │
│  │            │ RRF Fusion  │                               │   │
│  │            └──────┬──────┘                               │   │
│  └───────────────────┼─────────────────────────────────────┘   │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Synthesis Engine                        │   │
│  │  ┌─────────────┐    ┌─────────────┐                     │   │
│  │  │   Prompts   │───▶│ OpenAI API  │──▶ LLM Server       │   │
│  │  └─────────────┘    └─────────────┘                     │   │
│  │                            │                             │   │
│  │                            ▼                             │   │
│  │                   ┌─────────────────┐                   │   │
│  │                   │ Citation Extract│                   │   │
│  │                   └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Connectors (`src/connectors/`)

Connectors are responsible for querying external search services and normalizing results into a common format.

```python
@dataclass
class Source:
    id: str          # Unique identifier (e.g., "sx_a1b2c3d4")
    title: str       # Document title
    url: str         # Source URL
    content: str     # Content snippet
    score: float     # Relevance score
    connector: str   # Source connector name
    metadata: dict   # Additional metadata
```

**Available Connectors:**

| Connector | Type | Description |
|-----------|------|-------------|
| `SearXNGConnector` | Meta-search | Aggregates multiple search engines |
| `TavilyConnector` | AI-optimized | Tavily's AI search API |
| `LinkUpConnector` | Premium | LinkUp's deep search API |

**Adding a New Connector:**

```python
from src.connectors.base import Connector, SearchResult, Source

class MyConnector(Connector):
    name = "myconnector"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, top_k: int = 10) -> SearchResult:
        # Implement search logic
        sources = [...]
        return SearchResult(
            sources=sources,
            query=query,
            connector_name=self.name,
            total_results=len(sources),
        )
```

### 2. Search Aggregator (`src/search/`)

The aggregator executes searches across all configured connectors in parallel and combines results using RRF fusion.

**RRF (Reciprocal Rank Fusion):**

```python
def rrf_fusion(results_lists, k=60, top_k=20):
    """
    RRF score = Σ (1 / (k + rank)) for each list where item appears

    - Higher k = more balanced weighting
    - Lower k = more weight to top results
    """
```

**Flow:**
1. Execute searches in parallel via `asyncio.gather()`
2. Collect results from each connector
3. Apply RRF fusion to combine and re-rank
4. Return deduplicated, scored results

### 3. Synthesis Engine (`src/synthesis/`)

The synthesis engine uses an LLM to generate research answers with citations.

**Components:**

- **Prompts** (`prompts.py`): System prompts and citation formatting
- **Engine** (`engine.py`): LLM interaction and citation extraction

**Citation Flow:**

1. Build prompt with sources in format: `[source_id] Title\nURL: ...\nContent: ...`
2. Send to LLM with system prompt requiring `[source_id]` citations
3. Parse response to extract cited source IDs
4. Return content + citations mapping

**Reasoning Effort Levels:**

| Level | Behavior |
|-------|----------|
| `low` | Quick answers, minimal analysis |
| `medium` | Balanced analysis with key findings |
| `high` | Exhaustive analysis, multiple perspectives |

### 4. API Layer (`src/api/`)

FastAPI routes and Pydantic schemas for the REST API.

**Request Flow:**

```
Request → Validation (Pydantic) → Route Handler → Aggregator → Synthesis → Response
```

## Data Flow

### /search Endpoint

```
Query → SearchAggregator
         ├─► SearXNGConnector ─┐
         ├─► TavilyConnector  ─┼─► RRF Fusion ─► Response
         └─► LinkUpConnector  ─┘
```

### /research Endpoint

```
Query → SearchAggregator → RRF Fusion → SynthesisEngine → LLM → Citation Extract → Response
```

## Design Decisions

### 1. Parallel Search Execution

All connector searches run concurrently using `asyncio.gather()`. This minimizes latency when multiple connectors are configured.

### 2. URL-Based Deduplication

Sources are deduplicated by URL before RRF fusion. The first occurrence (highest original rank) is kept.

### 3. OpenAI-Compatible API

The synthesis engine uses the OpenAI client library, making it compatible with any OpenAI-compatible server (llama.cpp, vLLM, Ollama, etc.).

### 4. Citation by ID

Citations use short IDs (`sx_a1b2c3d4`) instead of numeric indices. This allows the LLM to reference sources consistently even if context order changes.

### 5. Environment-Based Configuration

All configuration is via environment variables with the `RESEARCH_` prefix. This follows 12-factor app principles and simplifies container deployment.

## Performance Considerations

### Latency

| Component | Typical Latency |
|-----------|----------------|
| SearXNG | 500-2000ms |
| Tavily | 1000-3000ms |
| LinkUp | 1000-3000ms |
| RRF Fusion | <10ms |
| LLM Synthesis | 5000-30000ms |

Total `/research` latency: ~6-35 seconds depending on LLM speed.

### Memory

- Search results: ~10KB per source
- LLM context: Depends on `top_k` and content length
- Recommended: 2GB+ container memory

### Scaling

For high-throughput scenarios:
1. Run multiple container replicas
2. Use connection pooling for HTTP clients
3. Consider caching frequent queries
4. Use faster LLM inference (vLLM with tensor parallelism)

## Security

- API keys are passed via environment variables
- No authentication by default (add via reverse proxy)
- CORS is open by default (restrict in production)
- Input validation via Pydantic schemas
