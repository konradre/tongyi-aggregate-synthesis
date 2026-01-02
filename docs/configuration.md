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
| `RESEARCH_LLM_API_BASE` | `https://openrouter.ai/api/v1` | OpenAI-compatible API base URL |
| `RESEARCH_LLM_API_KEY` | `` | API key (OpenRouter key for hosted, or dummy for local) |
| `RESEARCH_LLM_MODEL` | `alibaba/tongyi-deepresearch-30b-a3b:free` | Primary model (free tier) |
| `RESEARCH_LLM_MODEL_FALLBACK` | `alibaba/tongyi-deepresearch-30b-a3b` | Fallback model (paid tier) |
| `RESEARCH_LLM_FALLBACK_ENABLED` | `true` | Auto-fallback on 429 rate limit |
| `RESEARCH_LLM_TEMPERATURE` | `0.85` | Generation temperature |
| `RESEARCH_LLM_TOP_P` | `0.95` | Top-p sampling parameter |
| `RESEARCH_LLM_MAX_TOKENS` | `8192` | Maximum output tokens |
| `RESEARCH_LLM_TIMEOUT` | `120` | LLM request timeout in seconds |

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

### Minimal Configuration

```bash
export RESEARCH_LLM_API_KEY="sk-or-v1-your-key-here"
export RESEARCH_SEARXNG_HOST="http://192.168.1.3:8888"
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

# OpenRouter LLM (with auto-fallback)
export RESEARCH_LLM_API_KEY="sk-or-v1-your-key-here"
export RESEARCH_LLM_API_BASE="https://openrouter.ai/api/v1"
export RESEARCH_LLM_MODEL="alibaba/tongyi-deepresearch-30b-a3b:free"
export RESEARCH_LLM_MODEL_FALLBACK="alibaba/tongyi-deepresearch-30b-a3b"
export RESEARCH_LLM_FALLBACK_ENABLED="true"
export RESEARCH_LLM_TEMPERATURE="0.85"
export RESEARCH_LLM_MAX_TOKENS="8192"
export RESEARCH_LLM_TIMEOUT="120"

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
      RESEARCH_LINKUP_API_KEY: "${LINKUP_API_KEY:-}"
      RESEARCH_LLM_API_KEY: "${OPENROUTER_API_KEY}"
      RESEARCH_LLM_API_BASE: "https://openrouter.ai/api/v1"
      RESEARCH_LLM_MODEL: "alibaba/tongyi-deepresearch-30b-a3b:free"
      RESEARCH_LLM_MODEL_FALLBACK: "alibaba/tongyi-deepresearch-30b-a3b"
      RESEARCH_LLM_FALLBACK_ENABLED: "true"
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

## OpenRouter Rate Limit Fallback

This version uses automatic fallback from free to paid tier:

1. Primary request uses free model: `alibaba/tongyi-deepresearch-30b-a3b:free`
2. If 429 rate limit returned, automatically retries with: `alibaba/tongyi-deepresearch-30b-a3b`
3. Fallback can be disabled with `RESEARCH_LLM_FALLBACK_ENABLED=false`

This ensures high availability while minimizing costs.
