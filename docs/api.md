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
| `preset` | string | No | null | Synthesis preset (enables P1 features) |
| `focus_mode` | string | No | null | Discovery focus mode |

**Presets** (P1 Features)

| Preset | Latency | Features |
|--------|---------|----------|
| `fast` | ~4-5s | Basic synthesis, no verification |
| `comprehensive` | ~45-55s | Quality gate + RCS + contradiction detection |
| `contracrow` | ~45-55s | Focus on contradiction detection |
| `academic` | ~40-50s | Academic style with outline |
| `tutorial` | ~40-50s | Tutorial style with outline |

**Focus Modes**

`general`, `academic`, `documentation`, `comparison`, `debugging`, `tutorial`, `news`

**Example Request (Basic)**

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG and how does it work?",
    "reasoning_effort": "high"
  }'
```

**Example Request (With Preset)**

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "preset": "comprehensive",
    "focus_mode": "documentation"
  }'
```

**Response**

```json
{
  "query": "What is RAG and how does it work?",
  "content": "Retrieval Augmented Generation (RAG) is a technique...",
  "citations": [...],
  "sources": [...],
  "connectors_used": ["searxng", "tavily"],
  "model": "tongyi-deepresearch-30b",
  "usage": {"prompt_tokens": 4521, "completion_tokens": 1823},
  "preset_used": "Comprehensive",
  "focus_mode_used": "documentation",
  "quality_gate": {
    "decision": "proceed",
    "passed_sources": 5,
    "filtered_sources": 0
  },
  "contradictions": [],
  "rcs_summaries": [
    {"source_title": "RAG Tutorial", "summary": "...", "relevance_score": 0.85}
  ]
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

### POST /api/v1/discover

Exploratory discovery with knowledge gap analysis. Optimized for the EXPLORATORY workflow.

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Topic to explore |
| `top_k` | integer | No | 10 | Results per source (1-50) |
| `identify_gaps` | boolean | No | true | Analyze knowledge gaps |
| `focus_mode` | string | No | "general" | Domain-specific discovery mode |

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/discover \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer architecture in NLP",
    "identify_gaps": true,
    "focus_mode": "academic"
  }'
```

**Response**

```json
{
  "query": "transformer architecture in NLP",
  "landscape": {
    "explicit_topics": ["self-attention", "encoder-decoder"],
    "implicit_topics": ["positional encoding"],
    "related_concepts": ["BERT", "GPT"],
    "contrasting_views": []
  },
  "knowledge_gaps": [
    {"gap": "efficiency", "description": "How to reduce quadratic attention complexity", "importance": "high"}
  ],
  "sources": [...],
  "synthesis_preview": "Transformers revolutionized NLP by...",
  "recommended_deep_dives": ["https://arxiv.org/abs/1706.03762"]
}
```

---

### POST /api/v1/synthesize

Synthesize pre-gathered content into coherent analysis. Use when you already have sources from other tools.

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Synthesis focus/question |
| `sources` | PreGatheredSource[] | Yes | - | Pre-gathered sources |
| `style` | string | No | "comprehensive" | Output format |
| `max_tokens` | integer | No | 3000 | Maximum output tokens |

**PreGatheredSource Schema**

```json
{
  "origin": "ref",
  "url": "https://example.com/article",
  "title": "Article Title",
  "content": "Full content already fetched...",
  "source_type": "article"
}
```

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare RAG vs fine-tuning",
    "sources": [
      {"origin": "jina", "url": "https://...", "title": "RAG Guide", "content": "..."},
      {"origin": "exa", "url": "https://...", "title": "Fine-tuning Tutorial", "content": "..."}
    ],
    "style": "comparative"
  }'
```

---

### POST /api/v1/reason

Advanced reasoning with chain-of-thought analysis.

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Research query |
| `sources` | PreGatheredSource[] | Yes | - | Pre-gathered sources |
| `style` | string | No | "comprehensive" | Output format |

**Example Request**

```bash
curl -X POST http://localhost:8000/api/v1/reason \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is attention better than RNNs for long sequences?",
    "sources": [...],
    "style": "academic"
  }'
```

---

### POST /api/v1/synthesize/enhanced

Enhanced synthesis with P0 reliability features (quality gate, contradiction detection, citation verification).

**Request Body**

Same as `/api/v1/synthesize` plus:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_quality_gate` | boolean | No | true | Evaluate source quality first |
| `detect_contradictions` | boolean | No | true | Surface source contradictions |
| `verify_citations` | boolean | No | false | NLI verify citations (slower) |

---

### POST /api/v1/synthesize/p1

P1 enhanced synthesis with presets, outline-guided synthesis, and RCS.

**Request Body**

Same as `/api/v1/synthesize/enhanced` plus:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `preset` | string | No | null | Use preset configuration |
| `use_outline` | boolean | No | false | Use SciRAG outline-guided synthesis |
| `use_rcs` | boolean | No | false | Use PaperQA2-style contextual summarization |
| `rcs_top_k` | integer | No | 5 | Top sources to keep after RCS ranking |

---

### GET /api/v1/presets

List available synthesis presets.

**Response**

```json
{
  "presets": [
    {
      "name": "fast",
      "description": "Quick synthesis without verification",
      "features": ["basic_synthesis"]
    },
    {
      "name": "comprehensive",
      "description": "Full pipeline with quality gate, RCS, and contradiction detection",
      "features": ["quality_gate", "rcs", "contradictions", "outline"]
    }
  ]
}
```

---

### GET /api/v1/focus-modes

List available discovery focus modes.

**Response**

```json
{
  "focus_modes": [
    {
      "name": "general",
      "description": "Balanced search across all sources"
    },
    {
      "name": "academic",
      "description": "Prioritize scholarly and research sources"
    },
    {
      "name": "documentation",
      "description": "Focus on official docs and technical references"
    }
  ]
}
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
