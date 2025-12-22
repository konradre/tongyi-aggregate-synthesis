# P1 Enhancements Implementation

Implementation of P1 priority enhancements from ENHANCEMENT_ROADMAP.md, focused on synthesis quality and domain-specific discovery.

## New Endpoints

### GET /presets
List available synthesis presets (PaperQA2-inspired).

```json
{
  "presets": [
    {"name": "Comprehensive", "value": "comprehensive", "description": "Full analysis with all verification steps", "style": "comprehensive", "max_tokens": 4000},
    {"name": "Fast", "value": "fast", "description": "Quick synthesis, skip verification for speed", "style": "concise", "max_tokens": 1000},
    {"name": "Contracrow", "value": "contracrow", "description": "Optimized for finding contradictions", "style": "comparative", "max_tokens": 3000},
    {"name": "Academic", "value": "academic", "description": "Scholarly synthesis with rigorous citations", "style": "academic", "max_tokens": 5000},
    {"name": "Tutorial", "value": "tutorial", "description": "Step-by-step guide format", "style": "tutorial", "max_tokens": 3000}
  ]
}
```

### GET /focus-modes
List available focus modes for discovery (Perplexica-inspired).

```json
{
  "modes": [
    {"name": "General", "value": "general", "description": "Broad technical questions", "search_expansion": true, "gap_categories": ["documentation", "examples", "alternatives", "gotchas"]},
    {"name": "Academic", "value": "academic", "description": "Research papers, scientific studies", "search_expansion": true, "gap_categories": ["methodology", "limitations", "replications", "critiques", "citations"]},
    {"name": "Documentation", "value": "documentation", "description": "Library/framework docs, API references", "search_expansion": false, "gap_categories": ["api_reference", "examples", "migration", "changelog", "configuration"]},
    {"name": "Comparison", "value": "comparison", "description": "X vs Y evaluations", "search_expansion": true, "gap_categories": ["criteria", "tradeoffs", "edge_cases", "benchmarks", "community_preference"]},
    {"name": "Debugging", "value": "debugging", "description": "Error messages, bug investigation", "search_expansion": true, "gap_categories": ["error_context", "similar_issues", "root_cause", "workarounds", "fixes"]},
    {"name": "Tutorial", "value": "tutorial", "description": "How-to guides, step-by-step learning", "search_expansion": false, "gap_categories": ["prerequisites", "step_by_step", "common_mistakes", "next_steps"]},
    {"name": "News", "value": "news", "description": "Recent events, announcements", "search_expansion": true, "gap_categories": ["announcement", "reaction", "impact", "timeline"]}
  ]
}
```

### POST /synthesize/p1
Enhanced synthesis with presets, outline-guided synthesis (SciRAG), and RCS contextual summarization.

**Request:**
```json
{
  "query": "Compare FastAPI vs Flask for production APIs",
  "sources": [...],
  "preset": "comprehensive",  // Or null for manual config
  "use_outline": true,        // SciRAG outline-guided synthesis
  "use_rcs": true,            // RCS contextual summarization
  "rcs_top_k": 5,             // Top sources to keep
  "run_quality_gate": true,
  "detect_contradictions": true,
  "verify_citations": false
}
```

**Response:**
```json
{
  "query": "...",
  "content": "...",
  "citations": [...],
  "confidence": 0.85,
  "preset_used": "Comprehensive",
  "outline": ["Overview", "Key Differences", "Use Cases", "Recommendations"],
  "sections": {"Overview": "...", "Key Differences": "..."},
  "critique": {"issues": [...], "has_critical": false},
  "rcs_summaries": [{"source_title": "...", "summary": "...", "relevance_score": 0.9, "key_points": [...]}],
  "sources_filtered": 3,
  "quality_gate": {...},
  "contradictions": [...]
}
```

## New Modules

### 1. Outline-Guided Synthesis (SciRAG)
**File:** `src/synthesis/outline.py`
**Research:** SciRAG (arXiv:2511.14362)

Plan-critique-refine cycle:
1. Generate outline from query + sources
2. Fill each section with cited content
3. Critique the draft for gaps/errors
4. Refine based on critique

```python
from src.synthesis import OutlineGuidedSynthesizer, generate_outline_heuristic

synthesizer = OutlineGuidedSynthesizer(llm_client, model="qwen-plus")
result = await synthesizer.synthesize(
    query="Compare FastAPI vs Flask",
    sources=sources,
    style=SynthesisStyle.COMPARATIVE,
)
# result.outline.sections = ["Overview", "Key Differences", ...]
# result.sections = {"Overview": "...", ...}
# result.critique = CritiqueResult(issues=[...])

# Heuristic fallback (no LLM)
outline = generate_outline_heuristic("How to implement OAuth2", SynthesisStyle.TUTORIAL)
# outline.sections = ["Prerequisites", "Step-by-Step Guide", "Common Issues", "Next Steps"]
```

### 2. Contextual Summarization (RCS)
**File:** `src/synthesis/rcs.py`
**Research:** PaperQA2

Ranking & Contextual Summarization - summarize sources in query context before synthesis:
1. Summarize each source specifically for the query
2. Score relevance (0.0-1.0)
3. Re-rank using LLM
4. Keep top-k most relevant

```python
from src.synthesis import RCSPreprocessor

rcs = RCSPreprocessor(llm_client, model="qwen-plus", min_relevance=0.3)
result = await rcs.prepare(
    query="How does React useState work?",
    sources=sources,
    top_k=5,
)
# result.summaries = [ContextualSummary(...), ...]
# result.total_sources = 10
# result.kept_sources = 5

# Heuristic mode (no LLM)
result = rcs.prepare_sync(query, sources, top_k=5)
```

### 3. Focus Modes (Perplexica)
**File:** `src/discovery/focus_modes.py`
**Research:** Perplexica (11k+ stars)

Domain-specific search configurations:

| Mode | Priority Engines | Gap Categories |
|------|------------------|----------------|
| General | google, bing | documentation, examples, alternatives, gotchas |
| Academic | arxiv, google scholar | methodology, limitations, critiques, citations |
| Documentation | google, github | api_reference, examples, migration, changelog |
| Comparison | google, stackoverflow, reddit | criteria, tradeoffs, benchmarks |
| Debugging | stackoverflow, github | error_context, root_cause, workarounds |
| Tutorial | google, youtube | prerequisites, step_by_step, common_mistakes |
| News | google news, bing news | announcement, reaction, impact, timeline |

```python
from src.discovery import FocusModeSelector, FocusModeType, get_focus_mode

# Auto-detect mode
selector = FocusModeSelector(llm_client)
mode_type = await selector.select("Compare React vs Vue")  # FocusModeType.COMPARISON

# Heuristic detection (no LLM)
mode_type = selector.select_sync("How to implement OAuth2")  # FocusModeType.TUTORIAL

# Get mode config
mode = get_focus_mode("academic")
# mode.priority_engines = ["arxiv", "google scholar", "semantic scholar"]
# mode.gap_categories = ["methodology", "limitations", ...]
```

### 4. Synthesis Presets (PaperQA2)
**File:** `src/synthesis/presets.py`
**Research:** PaperQA2 bundled settings

Pre-configured setting bundles:

| Preset | Style | Max Tokens | Quality Gate | Contradictions | Outline | RCS |
|--------|-------|------------|--------------|----------------|---------|-----|
| comprehensive | comprehensive | 4000 | ✓ | ✓ | ✓ | ✓ |
| fast | concise | 1000 | ✗ | ✗ | ✗ | ✗ |
| contracrow | comparative | 3000 | ✓ | ✓ | ✗ | ✓ |
| academic | academic | 5000 | ✓ | ✓ | ✓ | ✓ |
| tutorial | tutorial | 3000 | ✗ | ✗ | ✓ | ✗ |

```python
from src.synthesis import get_preset, list_presets, apply_overrides, PresetOverrides

# List all presets
presets = list_presets()  # [{"name": "Comprehensive", "value": "comprehensive", ...}, ...]

# Get preset by name
preset = get_preset("academic")
# preset.style = SynthesisStyle.ACADEMIC
# preset.verify_citations = True
# preset.use_outline = True

# Apply overrides
custom = apply_overrides(preset, PresetOverrides(max_tokens=8000))
```

## Files Changed

### New Files
- `src/synthesis/outline.py` - Outline-guided synthesis (SciRAG)
- `src/synthesis/rcs.py` - RCS contextual summarization
- `src/discovery/focus_modes.py` - Focus modes (Perplexica)
- `src/synthesis/presets.py` - Synthesis presets

### Modified Files
- `src/synthesis/__init__.py` - Export P1 modules
- `src/discovery/__init__.py` - Export focus_modes
- `src/api/schemas.py` - P1 request/response schemas
- `src/api/routes.py` - P1 endpoints

## Usage Examples

### Preset-Driven Synthesis
```python
# Fast mode for quick answers
response = await client.post("/synthesize/p1", json={
    "query": "What is React?",
    "sources": [...],
    "preset": "fast",
})

# Academic mode for rigorous analysis
response = await client.post("/synthesize/p1", json={
    "query": "Effects of transformer attention on language modeling",
    "sources": [...],
    "preset": "academic",
})

# Contracrow mode for contradiction hunting
response = await client.post("/synthesize/p1", json={
    "query": "Is Redux still relevant in 2025?",
    "sources": [...],
    "preset": "contracrow",
})
```

### Manual Configuration
```python
response = await client.post("/synthesize/p1", json={
    "query": "How to implement WebSockets in FastAPI",
    "sources": [...],
    "preset": null,
    "style": "tutorial",
    "use_outline": true,
    "use_rcs": true,
    "rcs_top_k": 3,
    "run_quality_gate": true,
    "detect_contradictions": false,
})
```

## Research References

1. **SciRAG (arXiv:2511.14362)** - Outline-guided synthesis with critique
2. **PaperQA2** - RCS contextual summarization, contracrow preset
3. **Perplexica (11k+ stars)** - Focus modes for domain-specific search
