# Tongyi Aggregate Synthesis

A lightweight hybrid research tool that combines multi-source web search with LLM-powered synthesis. Designed as a self-hosted replacement for Perplexity API, using local Tongyi DeepResearch 30B model for citation-aware research synthesis.

## Features

- **Multi-Source Search**: Parallel queries across SearXNG, Tavily, and LinkUp
- **RRF Fusion**: Reciprocal Rank Fusion combines and re-ranks results
- **Citation-Aware Synthesis**: Tongyi DeepResearch generates answers with inline citations
- **OpenAI-Compatible**: Works with any OpenAI-compatible LLM API (llama.cpp, vLLM, etc.)
- **Docker-Ready**: Single container deployment alongside your LLM server

## Quick Start

```bash
# Clone
git clone https://github.com/konradre/tongyi-aggregate-synthesis.git
cd tongyi-aggregate-synthesis

# Configure (optional - SearXNG works out of the box)
cp .env.example .env
# Edit .env with your API keys

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
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Deployment](docs/deployment.md)

## Requirements

- Docker & Docker Compose
- SearXNG instance (or Tavily/LinkUp API keys)
- OpenAI-compatible LLM API (llama.cpp recommended)

## License

MIT
