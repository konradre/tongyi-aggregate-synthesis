# API Reference

Base URL: `http://localhost:8000`

## Endpoints

### GET /api/v1/health

Health check endpoint. Returns service status and active connectors.

**Response**

```json
{
  "status": "healthy",
  "connectors": ["searxng", "tavily", "linkup"],
  "llm_configured": true
}
```

---

### POST /api/v1/search

Execute multi-source search with RRF fusion. Returns aggregated results without LLM synthesis.

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `top_k` | integer | No | 10 | Results per source (1-50) |
| `connectors` | string[] | No | all | Specific connectors to use |

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "retrieval augmented generation tutorial",
    "top_k": 5,
    "connectors": ["searxng", "tavily"]
  }'
```

**Response**

```json
{
  "query": "retrieval augmented generation tutorial",
  "sources": [
    {
      "id": "sx_a1b2c3d4",
      "title": "RAG Tutorial - LangChain",
      "url": "https://python.langchain.com/docs/tutorials/rag/",
      "content": "Build a Retrieval Augmented Generation (RAG) App...",
      "score": 0.0323,
      "connector": "searxng"
    }
  ],
  "connectors_used": ["searxng", "tavily"],
  "total_results": 15
}
```

---

### POST /api/v1/research

Full research pipeline: search + LLM synthesis with citations.

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Research query |
| `top_k` | integer | No | 10 | Results per source (1-50) |
| `connectors` | string[] | No | all | Specific connectors to use |
| `reasoning_effort` | string | No | "medium" | Analysis depth: "low", "medium", "high" |

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG and how does it work?",
    "top_k": 10,
    "reasoning_effort": "high"
  }'
```

**Response**

```json
{
  "query": "What is RAG and how does it work?",
  "content": "Retrieval Augmented Generation (RAG) is a technique that enhances large language models by incorporating external knowledge retrieval [sx_a1b2c3d4]. The process works in three main steps:\n\n1. **Retrieval**: When a query is received, relevant documents are retrieved from a knowledge base using semantic search [tv_e5f6g7h8]...\n\n## Sources\n\n[sx_a1b2c3d4] RAG Tutorial - LangChain - https://python.langchain.com/docs/tutorials/rag/\n[tv_e5f6g7h8] Understanding RAG - https://example.com/rag",
  "citations": [
    {
      "id": "sx_a1b2c3d4",
      "title": "RAG Tutorial - LangChain",
      "url": "https://python.langchain.com/docs/tutorials/rag/"
    },
    {
      "id": "tv_e5f6g7h8",
      "title": "Understanding RAG",
      "url": "https://example.com/rag"
    }
  ],
  "sources": [...],
  "connectors_used": ["searxng", "tavily"],
  "model": "tongyi-deepresearch-30b",
  "usage": {
    "prompt_tokens": 4521,
    "completion_tokens": 1823
  }
}
```

---

### POST /api/v1/ask

Quick answer endpoint. Same as `/research` but with `reasoning_effort: "low"`.

**Request Body**

Same as `/api/v1/research` (reasoning_effort is ignored and set to "low").

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

---

## Data Types

### Source

```typescript
interface Source {
  id: string;        // Unique identifier (e.g., "sx_a1b2c3d4")
  title: string;     // Document title
  url: string;       // Source URL
  content: string;   // Snippet/content
  score: float;      // RRF fusion score
  connector: string; // Source connector name
}
```

### Citation

```typescript
interface Citation {
  id: string;    // Source ID
  title: string; // Document title
  url: string;   // Source URL
}
```

## Source ID Format

Source IDs follow the pattern `{connector_prefix}_{hash}`:

| Prefix | Connector |
|--------|-----------|
| `sx_` | SearXNG |
| `tv_` | Tavily |
| `lu_` | LinkUp |

The hash is derived from the URL for deduplication.

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid request body"
}
```

### 404 Not Found

```json
{
  "detail": "No search results found"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Synthesis error: Connection refused"
}
```

### 503 Service Unavailable

```json
{
  "detail": "No search connectors configured"
}
```

## Rate Limits

No rate limits are enforced by default. Rate limiting should be handled by a reverse proxy (nginx, Traefik) in production.

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
