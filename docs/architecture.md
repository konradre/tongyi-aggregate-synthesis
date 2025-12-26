# Architecture

## Overview

Tongyi Aggregate Synthesis is a modular research tool combining multi-source web search with LLM-powered synthesis. It implements research-backed techniques for query understanding, source quality assessment, and citation-aware generation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Application                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   API Layer (src/api/)                                                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │ /search  │  │/research │  │   /ask   │  │ /presets │  │ /health  │    │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘    │
│        │             │             │             │                          │
│        └─────────────┴─────────────┴─────────────┘                          │
│                              │                                               │
│ ┌────────────────────────────┼────────────────────────────────────────────┐ │
│ │                    Discovery Module (src/discovery/)                     │ │
│ │                            │                                             │ │
│ │  ┌─────────────────────────┴─────────────────────────┐                  │ │
│ │  │                    Explorer                        │                  │ │
│ │  │  (Orchestrates discovery workflow)                │                  │ │
│ │  └─────────────────────────┬─────────────────────────┘                  │ │
│ │                            │                                             │ │
│ │  ┌──────────┐  ┌──────────┴┐  ┌──────────┐  ┌──────────┐               │ │
│ │  │ Routing  │  │ Expansion │  │Decomposer│  │Focus Mode│               │ │
│ │  │          │  │           │  │          │  │ Selector │               │ │
│ │  │ Classify │  │  HyDE-    │  │ Multi-   │  │          │               │ │
│ │  │ query →  │  │  style    │  │ aspect   │  │ Domain-  │               │ │
│ │  │ optimal  │  │  variant  │  │ breakdown│  │ specific │               │ │
│ │  │connector │  │generation │  │          │  │ configs  │               │ │
│ │  └──────────┘  └───────────┘  └──────────┘  └──────────┘               │ │
│ │                            │                                             │ │
│ │                            ▼                                             │ │
│ │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│ │  │                      Gap Filler                                  │   │ │
│ │  │  (Identifies knowledge gaps, suggests follow-up queries)        │   │ │
│ │  └─────────────────────────────────────────────────────────────────┘   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│ ┌────────────────────────────┼────────────────────────────────────────────┐ │
│ │                   Search Layer (src/search/, src/connectors/)           │ │
│ │                            │                                             │ │
│ │  ┌─────────────────────────┴─────────────────────────┐                  │ │
│ │  │              Search Aggregator                     │                  │ │
│ │  │                                                    │                  │ │
│ │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐       │                  │ │
│ │  │   │ SearXNG  │  │  Tavily  │  │  LinkUp  │       │                  │ │
│ │  │   │Connector │  │Connector │  │Connector │       │                  │ │
│ │  │   └────┬─────┘  └────┬─────┘  └────┬─────┘       │                  │ │
│ │  │        │             │             │              │                  │ │
│ │  │        └─────────────┼─────────────┘              │                  │ │
│ │  │                      ▼                            │                  │ │
│ │  │              ┌──────────────┐                     │                  │ │
│ │  │              │  RRF Fusion  │                     │                  │ │
│ │  │              │  + Dedup     │                     │                  │ │
│ │  │              └──────────────┘                     │                  │ │
│ │  └───────────────────────────────────────────────────┘                  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│ ┌────────────────────────────┼────────────────────────────────────────────┐ │
│ │                   Ranking Layer (src/ranking/)                          │ │
│ │                            │                                             │ │
│ │   ┌──────────┐  ┌──────────┴┐  ┌──────────┐                            │ │
│ │   │ Passage  │  │  Hybrid   │  │Authority │                            │ │
│ │   │ Ranker   │  │  Ranker   │  │ Scorer   │                            │ │
│ │   └──────────┘  └───────────┘  └──────────┘                            │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│ ┌────────────────────────────┼────────────────────────────────────────────┐ │
│ │                  Synthesis Module (src/synthesis/)                       │ │
│ │                            │                                             │ │
│ │  ┌─────────────────────────┴─────────────────────────┐                  │ │
│ │  │               Pre-Synthesis Pipeline               │                  │ │
│ │  │                                                    │                  │ │
│ │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │                  │ │
│ │  │  │ Quality  │  │   RCS    │  │ Outline  │        │                  │ │
│ │  │  │  Gate    │  │ Preproc  │  │Generator │        │                  │ │
│ │  │  │ (CRAG)   │  │(PaperQA2)│  │ (SciRAG) │        │                  │ │
│ │  │  └──────────┘  └──────────┘  └──────────┘        │                  │ │
│ │  └───────────────────────────────────────────────────┘                  │ │
│ │                            │                                             │ │
│ │                            ▼                                             │ │
│ │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│ │  │                   Synthesis Engine                               │   │ │
│ │  │                                                                  │   │ │
│ │  │   ┌────────────────────────────────────────────────────────┐   │   │ │
│ │  │   │              LLM Client (OpenAI-compatible)             │   │   │ │
│ │  │   │   ┌──────────────────────────────────────────────┐     │   │   │ │
│ │  │   │   │  get_llm_content() - Reasoning Model Support  │     │   │   │ │
│ │  │   │   │  (handles reasoning_content fallback)         │     │   │   │ │
│ │  │   │   └──────────────────────────────────────────────┘     │   │   │ │
│ │  │   └────────────────────────────────────────────────────────┘   │   │ │
│ │  └─────────────────────────────────────────────────────────────────┘   │ │
│ │                            │                                             │ │
│ │                            ▼                                             │ │
│ │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│ │  │               Post-Synthesis Pipeline                            │   │ │
│ │  │                                                                  │   │ │
│ │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │ │
│ │  │  │Citation  │  │Contradict│  │ Evidence │  │  Claim   │        │   │ │
│ │  │  │ Extract  │  │ Detector │  │ Binding  │  │ Verify   │        │   │ │
│ │  │  │          │  │(PaperQA2)│  │          │  │(VeriCite)│        │   │ │
│ │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │   │ │
│ │  └─────────────────────────────────────────────────────────────────┘   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Reference

### Discovery Module (`src/discovery/`)

Research-backed query understanding and source discovery.

| Component | File | Research Basis | Purpose |
|-----------|------|----------------|---------|
| **Explorer** | `explorer.py` | - | Orchestrates discovery workflow |
| **ConnectorRouter** | `routing.py` | Query classification | Routes queries to optimal connectors |
| **QueryExpander** | `expansion.py` | HyDE (arXiv:2212.10496) | Generates query variants for broader coverage |
| **QueryDecomposer** | `decomposer.py` | Multi-hop (arXiv:2507.00355) | Breaks complex queries into aspects |
| **GapFiller** | `gap_filler.py` | - | Identifies knowledge gaps |
| **FocusModeSelector** | `focus_modes.py` | Perplexica | Domain-specific configurations |

**Focus Modes:**
- `general` - Default balanced mode
- `academic` - Research papers, citations
- `documentation` - API docs, tutorials
- `comparison` - X vs Y analysis
- `debugging` - Error resolution
- `tutorial` - Step-by-step guides
- `news` - Recent events

### Connectors (`src/connectors/`)

External search service integrations.

| Connector | Type | Features |
|-----------|------|----------|
| **SearXNGConnector** | Meta-search | Multi-engine aggregation |
| **TavilyConnector** | AI-optimized | Semantic search, freshness |
| **LinkUpConnector** | Premium | Deep search, high quality |

**Source Schema:**
```python
@dataclass
class Source:
    id: str          # Unique ID (e.g., "sx_a1b2c3d4")
    title: str       # Document title
    url: str         # Source URL
    content: str     # Content snippet
    score: float     # Relevance score (0-1)
    connector: str   # Source connector name
    metadata: dict   # Additional metadata
```

### Ranking Module (`src/ranking/`)

Multi-signal relevance ranking.

| Component | File | Purpose |
|-----------|------|---------|
| **PassageRanker** | `passage.py` | Passage-level relevance scoring |
| **HybridRanker** | `hybrid.py` | Combines multiple ranking signals |
| **AuthorityScorer** | `authority.py` | Domain/source authority scoring |

### Synthesis Module (`src/synthesis/`)

LLM-powered research synthesis with quality controls.

| Component | File | Research Basis | Purpose |
|-----------|------|----------------|---------|
| **SynthesisEngine** | `engine.py` | - | Core LLM synthesis |
| **SynthesisAggregator** | `aggregator.py` | - | Multi-source aggregation |
| **SourceQualityGate** | `quality_gate.py` | CRAG (arXiv:2401.15884) | Pre-synthesis source filtering |
| **ContradictionDetector** | `contradictions.py` | PaperQA2 (arXiv:2409.13740) | Finds disagreements between sources |
| **CitationVerifier** | `verification.py` | VeriCite patterns | Verifies claim-citation support |
| **BidirectionalBinder** | `binding.py` | - | Links claims to evidence |
| **OutlineGuidedSynthesizer** | `outline.py` | SciRAG (arXiv:2511.14362) | Structured outline-first synthesis |
| **RCSPreprocessor** | `rcs.py` | PaperQA2 | Contextual source summarization |

**Synthesis Styles:**
- `comprehensive` - Full analysis with sections
- `concise` - Brief, focused answer
- `comparative` - Side-by-side analysis
- `tutorial` - Step-by-step guide
- `academic` - Scholarly with citations

### LLM Utilities (`src/llm_utils.py`)

Compatibility layer for reasoning models.

```python
def get_llm_content(message) -> str:
    """
    Extract content from LLM response.

    Reasoning models (DeepSeek-R1, Tongyi-DeepResearch, Qwen-QwQ)
    output to `reasoning_content` instead of `content`.
    This helper handles both formats.
    """
    content = getattr(message, 'content', None) or ""
    if not content:
        content = getattr(message, 'reasoning_content', None) or ""
    return content
```

**Supported Models:**
- Standard: GPT-4, Llama-3, Mistral, etc.
- Reasoning: DeepSeek-R1, Tongyi-DeepResearch, Qwen-QwQ

## Data Flow

### `/search` Endpoint

```
Query
  │
  ├─► ConnectorRouter.classify() ─► QueryType
  │
  ├─► QueryExpander.expand() ─► [variant1, variant2, ...]
  │
  ▼
SearchAggregator
  │
  ├─► SearXNGConnector.search() ─┐
  ├─► TavilyConnector.search()  ─┼─► [results...]
  └─► LinkUpConnector.search()  ─┘
                                  │
                                  ▼
                           RRF Fusion + Dedup
                                  │
                                  ▼
                           Ranked Sources
```

### `/research` Endpoint

```
Query
  │
  ├─► Discovery Module (routing, expansion, decomposition)
  │
  ▼
Search Aggregator ─► RRF Fusion ─► Sources
  │
  ├─► SourceQualityGate.evaluate() ─► Filter low-quality
  │
  ├─► RCSPreprocessor.prepare() ─► Contextual summaries
  │
  ├─► OutlineGuidedSynthesizer.generate_outline() ─► Outline
  │
  ▼
SynthesisEngine
  │
  ├─► LLM Generation (with get_llm_content())
  │
  ├─► CitationExtract ─► [cited_ids]
  │
  ├─► ContradictionDetector.detect() ─► [contradictions]
  │
  ├─► BidirectionalBinder.bind() ─► claim↔evidence links
  │
  ▼
Response (content + citations + metadata)
```

## Configuration

All configuration via environment variables with `RESEARCH_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_LLM_API_BASE` | `http://192.168.1.119:8080/v1` | LLM API endpoint |
| `RESEARCH_LLM_MODEL` | `tongyi-deepresearch-30b` | Model name |
| `RESEARCH_LLM_TEMPERATURE` | `0.85` | Generation temperature |
| `RESEARCH_LLM_MAX_TOKENS` | `8192` | Max output tokens |
| `RESEARCH_SEARXNG_HOST` | `http://192.168.1.3:8888` | SearXNG instance |
| `RESEARCH_TAVILY_API_KEY` | - | Tavily API key |
| `RESEARCH_LINKUP_API_KEY` | - | LinkUp API key |
| `RESEARCH_DEFAULT_TOP_K` | `10` | Results per source |
| `RESEARCH_RRF_K` | `60` | RRF fusion constant |

## Performance

### Latency Breakdown

| Stage | Typical Latency |
|-------|-----------------|
| Query routing/expansion | 50-200ms |
| Search (parallel) | 500-3000ms |
| RRF fusion | <10ms |
| Quality gate | 2-5s (with LLM) |
| RCS preprocessing | 3-8s (with LLM) |
| LLM synthesis | 5-30s |
| Post-processing | <500ms |
| **Total `/research`** | **10-45s** |

### LLM Inference (RTX 3090)

| Metric | Value |
|--------|-------|
| Throughput | 157-187 tokens/sec |
| Context window | 8192 tokens |
| VRAM usage | ~22GB |

## Design Decisions

### 1. Heuristic Fallbacks

Every LLM-powered component has a heuristic fallback for when LLM is unavailable:
- `ConnectorRouter` → keyword-based classification
- `QueryExpander` → synonym/pattern expansion
- `QualityGate` → URL/domain scoring
- `ContradictionDetector` → negation pattern matching

### 2. Reasoning Model Support

The `get_llm_content()` helper handles reasoning models that output to `reasoning_content` instead of `content`. This ensures compatibility with:
- DeepSeek-R1
- Tongyi-DeepResearch
- Qwen-QwQ

### 3. Citation by ID

Citations use short IDs (`sx_a1b2c3d4`) instead of numeric indices. This allows:
- Consistent references across context shuffling
- Unique identification per connector
- Easy extraction via regex

### 4. Parallel Execution

All independent operations run concurrently:
- Connector searches via `asyncio.gather()`
- Query variants searched in parallel
- Post-synthesis checks parallelized

### 5. Pre-Synthesis Quality Control

Sources are filtered BEFORE synthesis (CRAG pattern):
- Prevents hallucination from irrelevant sources
- Reduces LLM context noise
- Enables rejection with query suggestions

## Testing

```bash
# All tests
pytest

# P0 tests (core functionality)
pytest -m p0

# Integration tests (requires LLM)
pytest -m integration

# Unit tests only
pytest -m unit
```

**Test Coverage:**
- 30 P0 tests (26 unit, 4 integration)
- Heuristic fallback tests
- LLM integration tests
- API endpoint tests
