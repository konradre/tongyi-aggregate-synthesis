"""
Synthesis Presets (OpenRouter-Optimized).

This branch uses OpenRouter which has per-request costs and rate limits.
Heavy presets (comprehensive, contracrow, academic) are disabled to avoid:
- Multiple sequential LLM calls (5-8 per request)
- 30-60 second response times
- Higher API costs

Available presets optimized for OpenRouter:
- fast: Single LLM call, quick responses (recommended)
- tutorial: Outline-guided but no verification overhead
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .aggregator import SynthesisStyle


class PresetName(str, Enum):
    """Available synthesis presets (OpenRouter-optimized)."""
    FAST = "fast"
    TUTORIAL = "tutorial"


@dataclass
class SynthesisPreset:
    """Configuration for a synthesis preset."""
    name: str
    description: str
    style: SynthesisStyle
    max_tokens: int
    verify_citations: bool
    detect_contradictions: bool
    use_outline: bool
    use_rcs: bool  # Contextual summarization
    run_quality_gate: bool
    temperature: float = 0.7
    min_sources: int = 1


# Pre-defined presets (OpenRouter-optimized - single LLM call each)
SYNTHESIS_PRESETS: dict[PresetName, SynthesisPreset] = {
    PresetName.FAST: SynthesisPreset(
        name="Fast",
        description="Quick synthesis, optimized for OpenRouter (recommended)",
        style=SynthesisStyle.CONCISE,
        max_tokens=2000,
        verify_citations=False,
        detect_contradictions=False,
        use_outline=False,
        use_rcs=False,
        run_quality_gate=False,
        temperature=0.5,
        min_sources=1,
    ),

    PresetName.TUTORIAL: SynthesisPreset(
        name="Tutorial",
        description="Step-by-step guide format with outline",
        style=SynthesisStyle.TUTORIAL,
        max_tokens=3000,
        verify_citations=False,
        detect_contradictions=False,
        use_outline=True,  # Single extra LLM call for structure
        use_rcs=False,
        run_quality_gate=False,
        min_sources=1,
    ),
}

# Default preset for OpenRouter
DEFAULT_PRESET = PresetName.FAST


def get_preset(name: str) -> SynthesisPreset:
    """
    Get preset by name string.

    Args:
        name: Preset name (fast, tutorial)

    Returns:
        SynthesisPreset configuration

    Note:
        OpenRouter branch only supports fast and tutorial presets.
        Legacy preset names (comprehensive, contracrow, academic) fall back to fast.
    """
    try:
        preset_name = PresetName(name.lower())
        return SYNTHESIS_PRESETS[preset_name]
    except (ValueError, KeyError):
        # Fall back to fast for any unknown/legacy preset
        return SYNTHESIS_PRESETS[DEFAULT_PRESET]


def get_preset_by_enum(preset: PresetName) -> SynthesisPreset:
    """Get preset by enum value."""
    return SYNTHESIS_PRESETS.get(preset, SYNTHESIS_PRESETS[DEFAULT_PRESET])


def list_presets() -> list[dict]:
    """List all available presets with descriptions."""
    return [
        {
            "name": preset.name,
            "value": name.value,
            "description": preset.description,
            "style": preset.style.value,
            "max_tokens": preset.max_tokens,
        }
        for name, preset in SYNTHESIS_PRESETS.items()
    ]


@dataclass
class PresetOverrides:
    """Allow partial overrides of preset settings."""
    max_tokens: Optional[int] = None
    verify_citations: Optional[bool] = None
    detect_contradictions: Optional[bool] = None
    use_outline: Optional[bool] = None
    use_rcs: Optional[bool] = None
    run_quality_gate: Optional[bool] = None
    temperature: Optional[float] = None


def apply_overrides(preset: SynthesisPreset, overrides: PresetOverrides) -> SynthesisPreset:
    """
    Apply partial overrides to a preset.

    Args:
        preset: Base preset
        overrides: Values to override

    Returns:
        New preset with overrides applied
    """
    return SynthesisPreset(
        name=preset.name,
        description=preset.description,
        style=preset.style,
        max_tokens=overrides.max_tokens if overrides.max_tokens is not None else preset.max_tokens,
        verify_citations=overrides.verify_citations if overrides.verify_citations is not None else preset.verify_citations,
        detect_contradictions=overrides.detect_contradictions if overrides.detect_contradictions is not None else preset.detect_contradictions,
        use_outline=overrides.use_outline if overrides.use_outline is not None else preset.use_outline,
        use_rcs=overrides.use_rcs if overrides.use_rcs is not None else preset.use_rcs,
        run_quality_gate=overrides.run_quality_gate if overrides.run_quality_gate is not None else preset.run_quality_gate,
        temperature=overrides.temperature if overrides.temperature is not None else preset.temperature,
        min_sources=preset.min_sources,
    )
