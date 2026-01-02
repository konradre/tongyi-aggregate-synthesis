# Installation Guide

## Prerequisites

- **Docker** 20.10+ and **Docker Compose** v2
- **SearXNG** instance (recommended) or API keys for Tavily/LinkUp
- **LLM Server**: OpenAI-compatible API (llama.cpp, vLLM, Ollama, etc.)

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

Edit `.env` with your API keys (optional if using SearXNG only):

```env
TAVILY_API_KEY=tvly-xxxxx
LINKUP_API_KEY=xxxxx
```

### 3. Configure docker-compose.yml

Edit `docker-compose.yml` to match your infrastructure:

```yaml
environment:
  # Point to your SearXNG instance
  RESEARCH_SEARXNG_HOST: "http://192.168.1.3:8888"

  # Point to your LLM server (from container perspective)
  RESEARCH_LLM_API_BASE: "http://172.17.0.1:8080/v1"
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
  "connectors": ["searxng"],
  "llm_configured": true
}
```

## MCP Server Installation (Claude Code)

The tool includes a FastMCP-based MCP server for direct integration with Claude Code.

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Configure Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "gigaxity-mcp": {
      "type": "stdio",
      "command": "/path/to/python",
      "args": ["/path/to/research/tool/run_mcp.py"],
      "env": {
        "RESEARCH_LLM_API_BASE": "http://192.168.1.119:8080/v1",
        "RESEARCH_LLM_MODEL": "tongyi-deepresearch-30b",
        "RESEARCH_SEARXNG_HOST": "http://192.168.1.3:8888"
      }
    }
  }
}
```

### 3. Test MCP Server

```bash
# Test JSON-RPC handshake
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python run_mcp.py
```

### 4. Restart Claude Code

After updating `~/.claude.json`, restart Claude Code to load the new MCP server.

## Local Development (REST API)

Dependencies should already be installed from the MCP section above. If not:

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 1. Set Environment Variables

```bash
export RESEARCH_SEARXNG_HOST="http://localhost:8888"
export RESEARCH_LLM_API_BASE="http://localhost:8080/v1"
```

### 2. Run Development Server

```bash
python -m uvicorn src.main:app --reload --port 8000
```

### 3. Run Tests

```bash
python scripts/test_local.py
```

## LLM Server Setup

### llama.cpp (Recommended for Tongyi)

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j LLAMA_CUDA=1

# Run server with Tongyi model
./llama-server \
  -m /path/to/tongyi-deepresearch-30b.gguf \
  -c 32768 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

### vLLM

```bash
vllm serve Qwen/Tongyi-DeepResearch-30B \
  --host 0.0.0.0 \
  --port 8080 \
  --max-model-len 32768
```

### Ollama

```bash
ollama serve
# In another terminal:
ollama run tongyi-deepresearch
```

Update `RESEARCH_LLM_API_BASE` to match your server.

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

### "Synthesis error: Connection refused"

- Verify LLM server is running: `curl http://your-llm:8080/v1/models`
- Check `RESEARCH_LLM_API_BASE` points to correct address
- From Docker, use `172.17.0.1` to reach host services

### Container can't reach host services

Add to `docker-compose.yml`:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Then use `http://host.docker.internal:8080/v1` as API base.
