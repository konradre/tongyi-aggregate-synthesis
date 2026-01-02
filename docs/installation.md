# Installation Guide

## Prerequisites

- **Docker** 20.10+ and **Docker Compose** v2
- **OpenRouter API key** for LLM access
- **SearXNG** instance (recommended) or API keys for Tavily/LinkUp

## Docker Installation (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/konradre/tongyi-aggregate-synthesis.git
cd tongyi-aggregate-synthesis
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxx
TAVILY_API_KEY=tvly-xxxxx
LINKUP_API_KEY=xxxxx
```

### 3. Configure docker-compose.yml

The default configuration uses OpenRouter. Update SearXNG host if needed:

```yaml
environment:
  # Point to your SearXNG instance
  RESEARCH_SEARXNG_HOST: "http://192.168.1.3:8888"
```

### 4. Build and Run

```bash
docker compose up -d --build
```

### 5. Verify Installation

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "connectors": ["searxng", "tavily", "linkup"],
  "llm_configured": true
}
```

## Local Development

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Set Environment Variables

```bash
export RESEARCH_LLM_API_KEY="sk-or-v1-your-key-here"
export RESEARCH_LLM_API_BASE="https://openrouter.ai/api/v1"
export RESEARCH_LLM_MODEL="alibaba/tongyi-deepresearch-30b-a3b:free"
export RESEARCH_SEARXNG_HOST="http://192.168.1.3:8888"
```

### 3. Run Development Server

```bash
python -m uvicorn src.main:app --reload --port 8000
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## OpenRouter Configuration

This version uses OpenRouter exclusively with automatic rate limit fallback:

| Variable | Value | Description |
|----------|-------|-------------|
| `RESEARCH_LLM_API_BASE` | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `RESEARCH_LLM_API_KEY` | Your key | OpenRouter API key |
| `RESEARCH_LLM_MODEL` | `alibaba/tongyi-deepresearch-30b-a3b:free` | Free tier model |
| `RESEARCH_LLM_MODEL_FALLBACK` | `alibaba/tongyi-deepresearch-30b-a3b` | Paid fallback |
| `RESEARCH_LLM_FALLBACK_ENABLED` | `true` | Auto-fallback on 429 |

When the free tier hits rate limits (429), the tool automatically retries with the paid model.

## SearXNG Setup

If you don't have SearXNG running:

```bash
docker run -d \
  --name searxng \
  -p 8888:8080 \
  -v ./searxng:/etc/searxng \
  searxng/searxng
```

Configure in `searxng/settings.yml`:
```yaml
search:
  formats:
    - html
    - json  # Required for API access
```

## Troubleshooting

### "No search connectors configured"

- Check that at least one connector is configured
- Verify SearXNG is accessible: `curl http://your-searxng:8888/search?q=test&format=json`
- Check API keys are set correctly in `.env`

### "Rate limit exceeded" (429)

- This is normal for free tier - the tool auto-falls back to paid model
- If fallback also fails, wait and retry or check your OpenRouter credits

### OpenRouter authentication error

- Verify `RESEARCH_LLM_API_KEY` is set correctly
- Check your key at https://openrouter.ai/keys
