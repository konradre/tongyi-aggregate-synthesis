# Tongyi Aggregate Synthesis

A research-backed hybrid research tool combining multi-source web search with LLM-powered synthesis. Uses OpenRouter for Tongyi DeepResearch 30B with per-request API key support for multi-tenant deployments.

## Features

### Core
- **Multi-Source Search**: Parallel queries across SearXNG, Tavily, and LinkUp
- **RRF Fusion**: Reciprocal Rank Fusion combines and re-ranks results
- **Citation-Aware Synthesis**: LLM generates answers with inline citations
- **OpenRouter Integration**: Per-request API key support for multi-tenant usage

### Discovery Module (Research-Backed)
- **Adaptive Routing**: Query classification routes to optimal connectors
- **Query Expansion**: HyDE-style variant generation for broader coverage
- **Query Decomposition**: Multi-aspect breakdown for complex queries
- **Gap Detection**: Identifies knowledge gaps in retrieved sources
- **Focus Modes**: Domain-specific configurations (academic, tutorial, debugging, etc.)

### Synthesis Module (Research-Backed)
- **Quality Gate**: CRAG-style source filtering before synthesis
- **Contradiction Detection**: PaperQA2-style disagreement surfacing
- **Citation Verification**: VeriCite-style claim-to-evidence binding
- **Outline-Guided Synthesis**: SciRAG-style structured generation
- **Contextual Summarization**: RCS preprocessing for better context

### Compatibility
- **Reasoning Model Support**: Works with DeepSeek-R1, Tongyi-DeepResearch, Qwen-QwQ
- **Multi-Tenant Support**: Per-request API keys allow usage billing to individual accounts
- **Docker-Ready**: Single container deployment

## Quick Start

```bash
# Clone
git clone https://github.com/konradre/tongyi-aggregate-synthesis.git
cd tongyi-aggregate-synthesis

# Configure
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
cp .env.example .env
# Edit .env with your SearXNG host and optional Tavily/LinkUp keys

# Run
docker compose up -d

# Test
curl http://localhost:8000/api/v1/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check, lists active connectors |
| `/api/v1/search` | POST | Multi-source search with RRF fusion |
| `/api/v1/research` | POST | Full research with LLM synthesis |
| `/api/v1/ask` | POST | Quick answers (low reasoning effort) |
| `/api/v1/presets` | GET | List available synthesis presets |
| `/api/v1/focus-modes` | GET | List available focus modes |

## Example Usage

```bash
# Search only (no LLM)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "RAG retrieval augmented generation", "top_k": 5}'

# Full research with synthesis
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "reasoning_effort": "medium"}'

# With focus mode
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI authentication", "focus_mode": "documentation"}'

# With tutorial preset (outline-guided step-by-step format)
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "How to implement RAG?", "preset": "tutorial"}'
```

### Presets (OpenRouter-Optimized)

| Preset | Latency | Use Case |
|--------|---------|----------|
| `fast` | ~2-5s | Quick answers, single LLM call (recommended) |
| `tutorial` | ~5-10s | Step-by-step explanations with outline |

*Note: Heavy presets (comprehensive, contracrow, academic) are disabled in this OpenRouter branch to avoid multiple sequential LLM calls that increase latency and cost.*

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Discovery Module                         │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │  │ Routing  │ │Expansion │ │Decompose │ │Focus Mode│       │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │    │
│  │       └────────────┴────────────┴────────────┘              │    │
│  │                         │                                    │    │
│  │                         ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────┐   │    │
│  │  │              Search Aggregator                       │   │    │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │    │
│  │  │  │ SearXNG │  │ Tavily  │  │ LinkUp  │  (parallel) │   │    │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘             │   │    │
│  │  │       └───────────┬┴───────────┘                    │   │    │
│  │  │                   ▼                                  │   │    │
│  │  │            ┌─────────────┐                          │   │    │
│  │  │            │ RRF Fusion  │                          │   │    │
│  │  │            └──────┬──────┘                          │   │    │
│  │  └───────────────────┼─────────────────────────────────┘   │    │
│  └──────────────────────┼──────────────────────────────────────┘    │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Synthesis Module                          │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │  │ Quality  │ │Contradict│ │ Outline  │ │   RCS    │       │    │
│  │  │  Gate    │ │ Detector │ │ Guided   │ │ Preproc  │       │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │    │
│  │       └────────────┴────────────┴────────────┘              │    │
│  │                         │                                    │    │
│  │                         ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────┐   │    │
│  │  │              Synthesis Engine                        │   │    │
│  │  │         (OpenAI-compatible LLM API)                 │   │    │
│  │  │                      │                               │   │    │
│  │  │                      ▼                               │   │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │    │
│  │  │  │ Citation │  │ Binding  │  │ Verify   │          │   │    │
│  │  │  │ Extract  │  │ Evidence │  │ Claims   │          │   │    │
│  │  │  └──────────┘  └──────────┘  └──────────┘          │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Deployment](docs/deployment.md)

## Research Foundations

This tool implements techniques from recent research:

| Feature | Research Basis |
|---------|---------------|
| Quality Gate | CRAG (arXiv:2401.15884) |
| Contradiction Detection | PaperQA2 (arXiv:2409.13740) |
| Query Expansion | HyDE (arXiv:2212.10496) |
| Query Decomposition | Multi-hop Retrieval (arXiv:2507.00355) |
| Outline-Guided Synthesis | SciRAG (arXiv:2511.14362) |
| Focus Modes | Perplexica patterns |

## Requirements

- Docker & Docker Compose
- OpenRouter API key
- SearXNG instance (or Tavily/LinkUp API keys)

## License

MIT
