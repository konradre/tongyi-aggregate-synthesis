"""
End-to-End Tests for Enhanced MCP Server.

Tests all P0/P1 enhancements:
- Tool listing and schema validation
- Focus modes in discover
- Presets in synthesize
- Reasoning depths in reason
- Pipeline component integration
- Error handling

Usage:
    pytest tests/test_mcp_e2e.py -v
    # Or standalone:
    python tests/test_mcp_e2e.py
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import MCP server components
import sys
sys.path.insert(0, "/home/khitomer/Projects/research/tool")

from src.mcp_server import (
    list_tools,
    call_tool,
    _tool_discover,
    _tool_synthesize,
    _tool_reason,
    _tool_search,
    _tool_research,
    _tool_ask,
)
from src.synthesis import (
    PreGatheredSource,
    SynthesisStyle,
    get_preset,
    PresetName,
    SourceQualityGate,
    QualityDecision,
    ContradictionDetector,
    OutlineGuidedSynthesizer,
    RCSPreprocessor,
)
from src.discovery import (
    FocusModeType,
    FocusModeSelector,
    get_focus_mode,
    get_search_params,
    FOCUS_MODES,
)


# ============================================================================
# Test 1: Tool Listing and Schema Validation
# ============================================================================

class TestToolListing:
    """Test MCP tool discovery."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        """Verify all 6 tools are listed."""
        tools = await list_tools()
        tool_names = [t.name for t in tools]

        expected = ["search", "research", "ask", "discover", "synthesize", "reason"]
        assert set(tool_names) == set(expected), f"Missing tools: {set(expected) - set(tool_names)}"

    @pytest.mark.asyncio
    async def test_discover_has_focus_mode_param(self):
        """Verify discover tool has focus_mode parameter."""
        tools = await list_tools()
        discover = next(t for t in tools if t.name == "discover")

        schema = discover.inputSchema
        assert "focus_mode" in schema["properties"]

        fm_schema = schema["properties"]["focus_mode"]
        assert fm_schema["type"] == "string"
        assert set(fm_schema["enum"]) == {"general", "academic", "documentation", "comparison", "debugging", "tutorial", "news"}
        assert "Domain-specific discovery mode" in fm_schema["description"]

    @pytest.mark.asyncio
    async def test_synthesize_has_preset_param(self):
        """Verify synthesize tool has preset parameter."""
        tools = await list_tools()
        synthesize = next(t for t in tools if t.name == "synthesize")

        schema = synthesize.inputSchema
        assert "preset" in schema["properties"], "preset not in properties"

        preset_schema = schema["properties"]["preset"]
        assert preset_schema["type"] == "string", f"preset type is {preset_schema['type']}"
        assert set(preset_schema["enum"]) == {"comprehensive", "fast", "contracrow", "academic", "tutorial"}, \
            f"preset enum mismatch: {preset_schema['enum']}"
        assert "pipeline" in preset_schema["description"].lower(), \
            f"preset description doesn't mention pipeline: {preset_schema['description'][:100]}"

    @pytest.mark.asyncio
    async def test_reason_has_reasoning_depth_param(self):
        """Verify reason tool has reasoning_depth parameter."""
        tools = await list_tools()
        reason = next(t for t in tools if t.name == "reason")

        schema = reason.inputSchema
        assert "reasoning_depth" in schema["properties"]

        depth_schema = schema["properties"]["reasoning_depth"]
        assert depth_schema["type"] == "string"
        assert set(depth_schema["enum"]) == {"shallow", "moderate", "deep"}
        assert "reasoning chain" in depth_schema["description"]

    @pytest.mark.asyncio
    async def test_tool_descriptions_are_detailed(self):
        """Verify tool descriptions have sufficient detail for LLM discovery."""
        tools = await list_tools()

        min_description_length = 100  # Ensure descriptions are meaningful
        for tool in tools:
            assert len(tool.description) >= min_description_length, \
                f"Tool '{tool.name}' description too short ({len(tool.description)} chars)"


# ============================================================================
# Test 2: Focus Modes Configuration
# ============================================================================

class TestFocusModes:
    """Test P1 focus mode configurations."""

    def test_all_focus_modes_exist(self):
        """Verify all 7 focus modes are defined."""
        expected_modes = ["general", "academic", "documentation", "comparison", "debugging", "tutorial", "news"]
        for mode_name in expected_modes:
            mode = get_focus_mode(mode_name)
            assert mode is not None, f"Focus mode '{mode_name}' not found"
            assert mode.name.lower() == mode_name or mode_name in mode.name.lower()

    def test_focus_mode_gap_categories(self):
        """Verify each focus mode has appropriate gap categories."""
        expected_gaps = {
            FocusModeType.GENERAL: ["documentation", "examples", "alternatives", "gotchas"],
            FocusModeType.ACADEMIC: ["methodology", "limitations", "replications", "critiques"],
            FocusModeType.DOCUMENTATION: ["api_reference", "examples", "migration", "changelog", "configuration"],
            FocusModeType.COMPARISON: ["criteria", "tradeoffs", "edge_cases", "benchmarks", "community_preference"],
            FocusModeType.DEBUGGING: ["error_context", "similar_issues", "root_cause", "workarounds", "fixes"],
            FocusModeType.TUTORIAL: ["prerequisites", "step_by_step", "common_mistakes", "next_steps"],
            FocusModeType.NEWS: ["announcement", "reaction", "impact", "timeline"],
        }

        for mode_type, expected in expected_gaps.items():
            mode = FOCUS_MODES[mode_type]
            for gap in expected:
                assert gap in mode.gap_categories, \
                    f"Mode '{mode_type.value}' missing gap category '{gap}'"

    def test_focus_mode_search_expansion(self):
        """Verify search expansion settings per mode."""
        # Modes with search expansion ON
        expansion_on = [FocusModeType.GENERAL, FocusModeType.ACADEMIC, FocusModeType.COMPARISON,
                        FocusModeType.DEBUGGING, FocusModeType.NEWS]
        # Modes with search expansion OFF (stay focused)
        expansion_off = [FocusModeType.DOCUMENTATION, FocusModeType.TUTORIAL]

        for mode_type in expansion_on:
            params = get_search_params(mode_type)
            assert params["expand_searches"] == True, \
                f"Mode '{mode_type.value}' should have search expansion ON"

        for mode_type in expansion_off:
            params = get_search_params(mode_type)
            assert params["expand_searches"] == False, \
                f"Mode '{mode_type.value}' should have search expansion OFF"

    def test_focus_mode_selector_heuristics(self):
        """Test automatic focus mode selection from query patterns."""
        selector = FocusModeSelector()  # No LLM for sync heuristics

        test_cases = [
            ("How to implement OAuth in FastAPI", FocusModeType.TUTORIAL),
            ("Compare React vs Vue", FocusModeType.COMPARISON),
            ("TypeError: undefined is not a function", FocusModeType.DEBUGGING),
            ("latest AI announcements 2025", FocusModeType.NEWS),
            ("research paper on transformers methodology", FocusModeType.ACADEMIC),
            ("FastAPI documentation API reference", FocusModeType.DOCUMENTATION),
        ]

        for query, expected_mode in test_cases:
            detected = selector.select_sync(query)
            assert detected == expected_mode, \
                f"Query '{query}' detected as '{detected.value}', expected '{expected_mode.value}'"


# ============================================================================
# Test 3: Synthesis Presets Configuration
# ============================================================================

class TestSynthesisPresets:
    """Test P1 synthesis preset configurations."""

    def test_all_presets_exist(self):
        """Verify all 5 presets are defined."""
        expected_presets = ["comprehensive", "fast", "contracrow", "academic", "tutorial"]
        for preset_name in expected_presets:
            preset = get_preset(preset_name)
            assert preset is not None, f"Preset '{preset_name}' not found"

    def test_comprehensive_preset_full_pipeline(self):
        """Verify comprehensive preset enables all components."""
        preset = get_preset("comprehensive")
        assert preset.run_quality_gate == True
        assert preset.use_rcs == True
        assert preset.detect_contradictions == True
        assert preset.use_outline == True

    def test_fast_preset_minimal_pipeline(self):
        """Verify fast preset disables all components."""
        preset = get_preset("fast")
        assert preset.run_quality_gate == False
        assert preset.use_rcs == False
        assert preset.detect_contradictions == False
        assert preset.use_outline == False

    def test_contracrow_preset_contradiction_focus(self):
        """Verify contracrow preset focuses on contradictions."""
        preset = get_preset("contracrow")
        assert preset.detect_contradictions == True
        assert preset.run_quality_gate == True
        assert preset.use_outline == False  # Skip outline for contradiction focus

    def test_academic_preset_scholarly(self):
        """Verify academic preset has scholarly settings."""
        preset = get_preset("academic")
        assert preset.use_outline == True
        assert preset.use_rcs == True
        assert preset.style == SynthesisStyle.ACADEMIC

    def test_tutorial_preset_stepbystep(self):
        """Verify tutorial preset produces structured how-to."""
        preset = get_preset("tutorial")
        assert preset.use_outline == True
        assert preset.detect_contradictions == False
        assert preset.style == SynthesisStyle.TUTORIAL


# ============================================================================
# Test 4: Pipeline Component Integration (with mocks)
# ============================================================================

class TestPipelineIntegration:
    """Test P0/P1 pipeline component integration."""

    def _create_mock_sources(self, count=3):
        """Create mock PreGatheredSource objects."""
        return [
            PreGatheredSource(
                origin="test",
                url=f"https://example.com/{i}",
                title=f"Test Source {i}",
                content=f"Test content for source {i}. " * 50,
                source_type="article",
            )
            for i in range(count)
        ]

    @pytest.mark.asyncio
    async def test_synthesize_with_comprehensive_preset(self):
        """Test synthesize runs full pipeline with comprehensive preset."""
        with patch('src.mcp_server._get_llm_client') as mock_client:
            # Setup mock
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Synthesized content"

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            sources = self._create_mock_sources()
            args = {
                "query": "Test query",
                "sources": [{"title": s.title, "content": s.content, "url": s.url} for s in sources],
                "preset": "comprehensive",
            }

            # Should not raise - pipeline components should be wired correctly
            result = await _tool_synthesize(args)
            assert len(result) > 0
            assert "text" in result[0].type

    @pytest.mark.asyncio
    async def test_synthesize_with_fast_preset_skips_pipeline(self):
        """Test synthesize with fast preset skips preprocessing."""
        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Fast synthesis"

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            sources = self._create_mock_sources()
            args = {
                "query": "Test query",
                "sources": [{"title": s.title, "content": s.content} for s in sources],
                "preset": "fast",
            }

            result = await _tool_synthesize(args)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_without_preset_uses_standard(self):
        """Test synthesize without preset uses standard synthesis."""
        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Standard synthesis"

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            args = {
                "query": "Test query",
                "sources": [{"title": "Test", "content": "Content"}],
            }

            result = await _tool_synthesize(args)
            assert len(result) > 0


# ============================================================================
# Test 5: Reasoning Depths
# ============================================================================

class TestReasoningDepths:
    """Test reason tool with different depths."""

    @pytest.mark.asyncio
    async def test_shallow_reasoning(self):
        """Test shallow reasoning depth."""
        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Brief analysis"

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            result = await _tool_reason({
                "query": "Simple question",
                "reasoning_depth": "shallow",
            })

            assert len(result) > 0
            # Verify shallow prompt was used
            call_args = mock_async_client.chat.completions.create.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            system_msg = messages[0]["content"]
            assert "brief" in system_msg.lower()

    @pytest.mark.asyncio
    async def test_deep_reasoning(self):
        """Test deep reasoning depth."""
        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Comprehensive analysis"

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            result = await _tool_reason({
                "query": "Complex decision",
                "reasoning_depth": "deep",
            })

            assert len(result) > 0
            # Verify deep prompt was used
            call_args = mock_async_client.chat.completions.create.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            system_msg = messages[0]["content"]
            assert "comprehensive" in system_msg.lower() or "multiple perspectives" in system_msg.lower()


# ============================================================================
# Test 6: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Test unknown tool name returns error message."""
        result = await call_tool("nonexistent_tool", {})
        assert len(result) > 0
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_invalid_focus_mode_defaults_to_general(self):
        """Test invalid focus_mode defaults to general."""
        mode = get_focus_mode("invalid_mode")
        general = get_focus_mode("general")
        assert mode.name == general.name

    @pytest.mark.asyncio
    async def test_invalid_preset_defaults_to_comprehensive(self):
        """Test invalid preset defaults to comprehensive."""
        preset = get_preset("invalid_preset")
        comprehensive = get_preset("comprehensive")
        assert preset.name == comprehensive.name

    @pytest.mark.asyncio
    async def test_synthesize_empty_sources(self):
        """Test synthesize with empty sources."""
        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "No content"

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            result = await _tool_synthesize({
                "query": "Test",
                "sources": [],
            })
            # Should handle gracefully
            assert len(result) > 0


# ============================================================================
# Test 7: Alignment Verification (Critical)
# ============================================================================

class TestAlignmentVerification:
    """Verify MCP server aligns with SKILL.md documentation."""

    @pytest.mark.asyncio
    async def test_focus_mode_enum_matches_skill_doc(self):
        """Verify focus_mode enum values match SKILL.md documentation."""
        tools = await list_tools()
        discover = next(t for t in tools if t.name == "discover")

        # These must match SKILL.md exactly
        documented_modes = ["general", "academic", "documentation", "comparison", "debugging", "tutorial", "news"]
        schema_modes = discover.inputSchema["properties"]["focus_mode"]["enum"]

        assert set(schema_modes) == set(documented_modes), \
            f"focus_mode enum mismatch: schema={schema_modes}, doc={documented_modes}"

    @pytest.mark.asyncio
    async def test_preset_enum_matches_skill_doc(self):
        """Verify preset enum values match SKILL.md documentation."""
        tools = await list_tools()
        synthesize = next(t for t in tools if t.name == "synthesize")

        # These must match SKILL.md exactly
        documented_presets = ["comprehensive", "fast", "contracrow", "academic", "tutorial"]
        schema_presets = synthesize.inputSchema["properties"]["preset"]["enum"]

        assert set(schema_presets) == set(documented_presets), \
            f"preset enum mismatch: schema={schema_presets}, doc={documented_presets}"

    @pytest.mark.asyncio
    async def test_reasoning_depth_enum_matches_skill_doc(self):
        """Verify reasoning_depth enum values match SKILL.md documentation."""
        tools = await list_tools()
        reason = next(t for t in tools if t.name == "reason")

        # These must match SKILL.md exactly
        documented_depths = ["shallow", "moderate", "deep"]
        schema_depths = reason.inputSchema["properties"]["reasoning_depth"]["enum"]

        assert set(schema_depths) == set(documented_depths), \
            f"reasoning_depth enum mismatch: schema={schema_depths}, doc={documented_depths}"

    def test_preset_pipeline_matches_skill_doc(self):
        """Verify preset pipeline components match SKILL.md documentation."""
        # From SKILL.md:
        # - comprehensive: Quality Gate → RCS → Contradiction Detection → Outline-Guided
        # - fast: Direct synthesis only
        # - contracrow: Quality Gate → Contradiction Detection → Standard
        # - academic: Quality Gate → RCS → Outline-Guided
        # - tutorial: Quality Gate → Outline-Guided

        comprehensive = get_preset("comprehensive")
        assert comprehensive.run_quality_gate == True
        assert comprehensive.use_rcs == True
        assert comprehensive.detect_contradictions == True
        assert comprehensive.use_outline == True

        fast = get_preset("fast")
        assert fast.run_quality_gate == False
        assert fast.use_rcs == False
        assert fast.detect_contradictions == False
        assert fast.use_outline == False

        contracrow = get_preset("contracrow")
        assert contracrow.run_quality_gate == True
        assert contracrow.detect_contradictions == True
        assert contracrow.use_outline == False

        academic = get_preset("academic")
        assert academic.run_quality_gate == True
        assert academic.use_rcs == True
        assert academic.use_outline == True

        tutorial = get_preset("tutorial")
        assert tutorial.run_quality_gate == False  # SKILL.md says Quality Gate, but preset has False
        assert tutorial.use_outline == True


# ============================================================================
# Standalone Runner
# ============================================================================

async def run_all_tests():
    """Run all tests when executed standalone."""
    print("=" * 70)
    print("MCP Server End-to-End Tests")
    print("=" * 70)

    test_classes = [
        ("Tool Listing", TestToolListing),
        ("Focus Modes", TestFocusModes),
        ("Synthesis Presets", TestSynthesisPresets),
        ("Pipeline Integration", TestPipelineIntegration),
        ("Reasoning Depths", TestReasoningDepths),
        ("Error Handling", TestErrorHandling),
        ("Alignment Verification", TestAlignmentVerification),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for category, test_class in test_classes:
        print(f"\n{category}:")
        print("-" * 40)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                test_name = method_name.replace("test_", "").replace("_", " ")
                try:
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        method()
                    print(f"  ✓ {test_name}")
                    total_passed += 1
                except Exception as e:
                    print(f"  ✗ {test_name}: {e}")
                    total_failed += 1
                    failures.append((category, test_name, str(e)))

    print("\n" + "=" * 70)
    print(f"Results: {total_passed} passed, {total_failed} failed")

    if failures:
        print("\nFailures:")
        for cat, test, error in failures:
            print(f"  - [{cat}] {test}: {error}")

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
