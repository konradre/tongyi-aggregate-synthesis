# Enhancement Roadmap: Outperforming Perplexity

## Current State Assessment

**Overall Rating: 3/10** relative to Perplexity

| Dimension | Current | Target | Gap |
|-----------|---------|--------|-----|
| Query Intelligence | 1/10 | 8/10 | Critical |
| Source Diversity | 3/10 | 8/10 | High |
| Ranking Quality | 2/10 | 9/10 | Critical |
| Passage Extraction | 1/10 | 8/10 | Critical |
| Synthesis Depth | 4/10 | 9/10 | High |
| Citation Precision | 2/10 | 8/10 | High |
| Latency/Streaming | 3/10 | 7/10 | Medium |

---

## Gap Analysis

### Critical Gaps (P0)

#### 1. No Query Intelligence
**Current:** Raw query passed directly to connectors
**Target:** Intent classification, query expansion, temporal awareness, entity extraction

**Implementation:** `src/query/processor.py` ✅ Created
```python
processor = QueryProcessor()
processed = processor.process("latest AI safety research")
# → intent: ACADEMIC
# → temporal: RECENT
# → entities: ["AI", "safety"]
# → expansions: ["AI alignment", "ML safety"]
# → sub_queries: ["AI safety papers 2024", "alignment research"]
```

#### 2. Position-Only Ranking
**Current:** Pure RRF (1/(k+rank))
**Target:** Multi-signal: semantic + authority + freshness + RRF

**Implementation:** `src/ranking/hybrid.py` ✅ Created
```python
ranker = HybridRanker(embedding_model="BAAI/bge-large-en-v1.5")
ranked = ranker.rank(query, sources, weights=RankingWeights(
    semantic=0.35,
    authority=0.25,
    freshness=0.15,
    rrf=0.25
))
```

#### 3. Full Document Retrieval
**Current:** Entire documents sent to LLM
**Target:** Relevant passages only (500-token chunks)

**Implementation:** `src/ranking/passage.py` ✅ Created
```python
extractor = PassageExtractor(chunk_size=500, embedding_model="BAAI/bge-large-en-v1.5")
passages = extractor.extract_passages(query, content, source_id, url, title, top_k=3)
```

### High Priority Gaps (P1)

#### 4. Single-Pass Synthesis
**Current:** One LLM call with all context
**Target:** Multi-stage: outline → draft → cite → refine → evaluate

**Implementation:** `src/synthesis/enhanced.py` ✅ Created
```python
synthesizer = EnhancedSynthesizer(llm_client)
result = await synthesizer.synthesize(
    query, sources, passages,
    depth=SynthesisDepth.COMPREHENSIVE
)
# Returns: content, citations, confidence, coverage, trace
```

#### 5. Weak Citation Binding
**Current:** Regex extraction [source_id]
**Target:** Claim-to-evidence binding with verification

**Status:** Partially implemented in `enhanced.py`

#### 6. Limited Source Diversity
**Current:** 3 connectors (SearXNG, Tavily, LinkUp)
**Target:** 10+ specialized connectors

**TODO:** Add connectors:
- Academic: ArXiv, Semantic Scholar, PubMed
- Code: GitHub Search, StackOverflow
- News: NewsAPI, Google News
- Knowledge: Wikipedia, Wolfram Alpha

### Medium Priority (P2)

#### 7. No Streaming
**Current:** Wait for full response
**Target:** Token-by-token streaming

#### 8. No Caching
**Current:** Every request hits all sources
**Target:** Multi-level caching (query, source, embedding)

#### 9. Single-Turn Only
**Current:** No conversation memory
**Target:** Context-aware follow-ups

---

## Implementation Priority Matrix

| Enhancement | Impact | Effort | Dependencies | Status |
|-------------|--------|--------|--------------|--------|
| Query Processing | High | Low | None | ✅ Done |
| Hybrid Ranking | High | Medium | Embeddings | ✅ Done |
| Passage Extraction | High | Medium | Embeddings | ✅ Done |
| Multi-Stage Synthesis | High | High | None | ✅ Done |
| Citation Binding | High | Medium | Embeddings | 🔄 Partial |
| ArXiv Connector | Medium | Low | None | ❌ TODO |
| GitHub Connector | Medium | Low | API Key | ❌ TODO |
| Streaming | Medium | Medium | Async refactor | ❌ TODO |
| Caching Layer | Medium | Medium | Redis/Disk | ❌ TODO |
| Wikipedia Connector | Low | Low | None | ❌ TODO |

---

## Differentiation Strategy

We **cannot** compete with Perplexity on:
- User feedback data (they have millions of queries)
- Engineering team size
- Custom model training
- Infrastructure scale

We **can** exceed Perplexity on:

### 1. Radical Transparency
```python
# Every decision is traceable
result = synthesizer.synthesize(query, sources)
print(result.trace)  # Shows: query_processing → source_selection →
                     #        passage_extraction → claim_binding → refinement
```

### 2. User Control
```python
# Users control synthesis behavior
result = await synthesizer.synthesize(
    query, sources,
    perspective="critical",      # or "neutral", "supportive"
    citation_density="high",     # or "low", "medium"
    freshness_weight=0.4,        # Custom ranking weights
    authority_weight=0.3,
)
```

### 3. Domain Specialization
Instead of being generalist, be THE BEST for specific domains:
- ML Research (arXiv + PapersWithCode + GitHub)
- Legal Research (case law + statutes)
- Medical Research (PubMed + Cochrane + ClinicalTrials)

### 4. Local-First Privacy
- All processing happens locally
- No data leaves user infrastructure
- Audit log of all operations
- GDPR/HIPAA compatible

---

## Phase 1 Integration (Next Steps)

### 1. Wire Up Query Processor

```python
# In src/search/aggregator.py
from ..query import QueryProcessor

class SearchAggregator:
    def __init__(self, ...):
        self.query_processor = QueryProcessor()

    async def search(self, query: str, ...):
        # Process query first
        processed = self.query_processor.process(query)

        # Use processed.sub_queries for parallel search
        # Use processed.suggested_connectors for connector selection
        # Pass processed.freshness_weight to ranker
```

### 2. Wire Up Hybrid Ranking

```python
# In src/search/aggregator.py
from ..ranking import HybridRanker, PassageExtractor

class SearchAggregator:
    def __init__(self, ...):
        self.ranker = HybridRanker(embedding_model="BAAI/bge-large-en-v1.5")
        self.passage_extractor = PassageExtractor(...)

    async def search(self, query: str, ...):
        # After RRF fusion
        sources = rrf_fusion(results_lists)

        # Apply hybrid ranking
        sources = self.ranker.rank(query, sources)

        # Extract passages
        passages = []
        for source in sources[:10]:
            source_passages = self.passage_extractor.extract_passages(
                query, source.content, source.id, source.url, source.title
            )
            passages.extend(source_passages)

        return sources, passages
```

### 3. Wire Up Enhanced Synthesis

```python
# In src/api/main.py
from ..synthesis.enhanced import EnhancedSynthesizer

@app.post("/research")
async def research(request: ResearchRequest):
    # Get sources and passages
    sources, passages = await aggregator.search(request.query)

    # Use enhanced synthesis
    synthesizer = EnhancedSynthesizer(llm_client)
    result = await synthesizer.synthesize(
        request.query,
        sources,
        passages,
        depth=SynthesisDepth.MEDIUM
    )

    return {
        "synthesis": result.content,
        "sources": [s.dict() for s in sources],
        "citations": result.citations,
        "confidence": result.confidence,
        "methodology": result.methodology,
    }
```

---

## Dependencies to Add

```toml
# pyproject.toml additions
[project.dependencies]
sentence-transformers = "^2.2.0"  # For embeddings
numpy = "^1.24.0"                  # For vector operations

[project.optional-dependencies]
full = [
    "sentence-transformers>=2.2.0",
]
```

---

## Benchmarking Plan

### Test Queries by Type

```python
BENCHMARK_QUERIES = {
    "factual": [
        "What is the population of Tokyo?",
        "When was Python first released?",
    ],
    "academic": [
        "What are the latest advances in transformer efficiency?",
        "Explain the Self-RAG paper's key contributions",
    ],
    "comparative": [
        "Compare PostgreSQL vs MySQL for analytics workloads",
        "FastAPI vs Django for production APIs",
    ],
    "tutorial": [
        "How do I implement rate limiting in FastAPI?",
        "How to set up Kubernetes autoscaling?",
    ],
    "current_events": [
        "Latest developments in AI regulation 2025",
        "Recent breakthroughs in quantum computing",
    ],
}
```

### Metrics to Track

1. **Latency** (ms): Time to first byte, time to complete
2. **Source Quality**: Authority scores, freshness
3. **Citation Accuracy**: % of claims with valid citations
4. **Coverage**: % of query aspects addressed
5. **User Preference**: A/B test vs Perplexity (when possible)

---

## Timeline-Free Milestones

### Milestone 1: Core Intelligence
- [x] Query processor
- [x] Hybrid ranker
- [x] Passage extractor
- [x] Enhanced synthesizer
- [ ] Integration testing
- [ ] Benchmark baseline

### Milestone 2: Source Expansion
- [ ] ArXiv connector
- [ ] GitHub Search connector
- [ ] Wikipedia connector
- [ ] Connector selection intelligence

### Milestone 3: Production Hardening
- [ ] Streaming responses
- [ ] Caching layer
- [ ] Error recovery
- [ ] Monitoring/observability

### Milestone 4: Differentiation
- [ ] Transparency dashboard
- [ ] User control API
- [ ] Domain specialization (pick 1-2)
- [ ] Privacy compliance

---

## Research Papers to Study

1. **RAG Fundamentals**
   - "Retrieval-Augmented Generation for Knowledge-Intensive NLP" (Lewis et al.)
   - "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al.)

2. **Retrieval Optimization**
   - "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia)
   - "Learning to Retrieve Passages without Supervision" (Ram et al.)

3. **Self-Improvement**
   - "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al.)
   - "Chain-of-Thought Prompting" (Wei et al.)

4. **Context Management**
   - "Lost in the Middle" (Liu et al.) - LLMs struggle with long contexts
   - "Extending Context Window" (various)

---

## Success Criteria

**Target: 8/10** overall sophistication

| Dimension | Current | Target | How to Measure |
|-----------|---------|--------|----------------|
| Query Intelligence | 1 | 8 | Query expansion accuracy |
| Source Diversity | 3 | 8 | # of specialized sources |
| Ranking Quality | 2 | 9 | NDCG@10 on relevance |
| Passage Extraction | 1 | 8 | Passage relevance score |
| Synthesis Depth | 4 | 9 | Human evaluation |
| Citation Precision | 2 | 8 | Claim verification rate |
| Latency | 3 | 7 | P95 latency (ms) |

**Ultimate Goal:** On domain-specific queries, produce higher quality research outputs than Perplexity with full transparency and user control.
