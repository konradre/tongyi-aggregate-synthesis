# Configuration Reference

All configuration is done via environment variables with the `RESEARCH_` prefix.

## Environment Variables

### SearXNG Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_SEARXNG_HOST` | `http://192.168.1.3:8888` | SearXNG instance URL |
| `RESEARCH_SEARXNG_ENGINES` | `google,bing,duckduckgo,brave,startpage` | Comma-separated search engines |
| `RESEARCH_SEARXNG_CATEGORIES` | `general` | Search categories |
| `RESEARCH_SEARXNG_LANGUAGE` | `en` | Search language code |
| `RESEARCH_SEARXNG_SAFESEARCH` | `0` | Safe search level (0=off, 1=moderate, 2=strict) |

### Tavily Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_TAVILY_API_KEY` | `` | Tavily API key (leave empty to disable) |
| `RESEARCH_TAVILY_SEARCH_DEPTH` | `advanced` | Search depth: `basic` or `advanced` |

### LinkUp Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_LINKUP_API_KEY` | `` | LinkUp API key (leave empty to disable) |
| `RESEARCH_LINKUP_DEPTH` | `standard` | Search depth: `standard` or `deep` |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_LLM_API_BASE` | `http://172.17.0.1:8080/v1` | OpenAI-compatible API base URL |
| `RESEARCH_LLM_API_KEY` | `not-needed` | API key (dummy for local models) |
| `RESEARCH_LLM_MODEL` | `tongyi-deepresearch-30b` | Model name |
| `RESEARCH_LLM_TEMPERATURE` | `0.85` | Generation temperature |
| `RESEARCH_LLM_TOP_P` | `0.95` | Top-p sampling parameter |
| `RESEARCH_LLM_MAX_TOKENS` | `8192` | Maximum output tokens |

### Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_DEFAULT_TOP_K` | `10` | Default results per source |
| `RESEARCH_RRF_K` | `60` | RRF fusion constant |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_HOST` | `0.0.0.0` | Server bind host |
| `RESEARCH_PORT` | `8000` | Server port |

## Configuration Examples

### Minimal (SearXNG Only)

```bash
export RESEARCH_SEARXNG_HOST="http://localhost:8888"
export RESEARCH_LLM_API_BASE="http://localhost:8080/v1"
```

### Full Configuration

```bash
# SearXNG
export RESEARCH_SEARXNG_HOST="http://192.168.1.3:8888"
export RESEARCH_SEARXNG_ENGINES="google,bing,duckduckgo"
export RESEARCH_SEARXNG_LANGUAGE="en"

# Tavily (optional)
export RESEARCH_TAVILY_API_KEY="tvly-xxxxx"
export RESEARCH_TAVILY_SEARCH_DEPTH="advanced"

# LinkUp (optional)
export RESEARCH_LINKUP_API_KEY="xxxxx"
export RESEARCH_LINKUP_DEPTH="deep"

# LLM
export RESEARCH_LLM_API_BASE="http://172.17.0.1:8080/v1"
export RESEARCH_LLM_MODEL="tongyi-deepresearch-30b"
export RESEARCH_LLM_TEMPERATURE="0.85"
export RESEARCH_LLM_MAX_TOKENS="8192"

# Search
export RESEARCH_DEFAULT_TOP_K="10"
export RESEARCH_RRF_K="60"
```

### Docker Compose Configuration

In `docker-compose.yml`:

```yaml
services:
  research-tool:
    environment:
      RESEARCH_SEARXNG_HOST: "http://192.168.1.3:8888"
      RESEARCH_TAVILY_API_KEY: "${TAVILY_API_KEY:-}"
      RESEARCH_LLM_API_BASE: "http://172.17.0.1:8080/v1"
```

## Connector Priority

Connectors are used in parallel. If a connector is not configured (empty API key), it's automatically disabled.

Active connectors are shown in the `/api/v1/health` response:

```json
{
  "status": "healthy",
  "connectors": ["searxng", "tavily"],
  "llm_configured": true
}
```

## RRF Fusion Constant

The `RESEARCH_RRF_K` parameter controls result fusion behavior:

- **Lower values (e.g., 10)**: Higher weight to top-ranked results
- **Higher values (e.g., 100)**: More balanced weighting across ranks
- **Default (60)**: Standard RRF constant, good balance

Formula: `score = Σ (1 / (k + rank))` for each list where item appears.

## LLM Model Recommendations

| Model | Context | Notes |
|-------|---------|-------|
| Tongyi DeepResearch 30B | 32K | Recommended, research-optimized |
| Qwen2.5-32B | 128K | Good alternative |
| Llama 3.1 70B | 128K | Strong general purpose |
| Mistral Large | 128K | Good for synthesis |

Adjust `RESEARCH_LLM_MAX_TOKENS` based on your model's context window.
