# Tongyi Aggregate Synthesis

A research-backed hybrid research tool combining multi-source web search with LLM-powered synthesis. Self-hosted replacement for Perplexity API using local Tongyi DeepResearch 30B model.

Available as both a **FastAPI REST service** and an **MCP server** for Claude Code integration.

## Features

### Core
- **Multi-Source Search**: Parallel queries across SearXNG, Tavily, and LinkUp
- **RRF Fusion**: Reciprocal Rank Fusion combines and re-ranks results
- **Citation-Aware Synthesis**: LLM generates answers with inline citations
- **OpenAI-Compatible**: Works with any OpenAI-compatible API (llama.cpp, vLLM, etc.)

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
- **Docker-Ready**: Single container deployment
- **MCP Server**: FastMCP-based stdio transport for Claude Code integration

## Quick Start

```bash
# Clone
git clone https://github.com/konradre/tongyi-aggregate-synthesis.git
cd tongyi-aggregate-synthesis

# Configure
cp .env.example .env
# Edit .env with your API keys and LLM endpoint

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

## MCP Server (Claude Code Integration)

The tool exposes an MCP server using FastMCP for direct integration with Claude Code.

### Installation

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "gigaxity-mcp": {
      "type": "stdio",
      "command": "/path/to/python",
      "args": ["/path/to/research/tool/run_mcp.py"],
      "env": {
        "RESEARCH_LLM_API_BASE": "http://localhost:8080/v1",
        "RESEARCH_LLM_MODEL": "tongyi-deepresearch-30b",
        "RESEARCH_SEARXNG_HOST": "http://localhost:8888"
      }
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `search` | Multi-source search with RRF fusion (no LLM) |
| `research` | Full research pipeline with LLM synthesis |
| `ask` | Quick conversational answers |
| `discover` | Exploratory discovery with knowledge gap analysis |
| `synthesize` | Synthesize pre-gathered content into analysis |
| `reason` | Deep reasoning with chain-of-thought |

### Running Standalone

```bash
# Test MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python run_mcp.py
```

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

# With preset (enables quality gate, RCS, contradiction detection)
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "preset": "comprehensive"}'
```

### Presets

| Preset | Latency | Use Case |
|--------|---------|----------|
| `fast` | ~4-5s | Quick answers without verification |
| `comprehensive` | ~45-55s | Full research with quality checks |
| `contracrow` | ~45-55s | Focus on finding contradictions |
| `academic` | ~40-50s | Scholarly style with outline |
| `tutorial` | ~40-50s | Step-by-step explanations |

*Latency assumes single RTX 3090 GPU running llama.cpp*

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

- Docker & Docker Compose (for REST API)
- Python 3.10+ with FastMCP (for MCP server)
- SearXNG instance (or Tavily/LinkUp API keys)
- OpenAI-compatible LLM API (llama.cpp recommended)

## License

MIT
