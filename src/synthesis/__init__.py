"""Research synthesis with LLM integration."""

from .engine import SynthesisEngine
from .prompts import RESEARCH_SYSTEM_PROMPT, build_research_prompt
from .aggregator import (
    SynthesisAggregator,
    SynthesisStyle,
    PreGatheredSource,
    AggregatedSynthesis,
)

__all__ = [
    "SynthesisEngine",
    "RESEARCH_SYSTEM_PROMPT",
    "build_research_prompt",
    "SynthesisAggregator",
    "SynthesisStyle",
    "PreGatheredSource",
    "AggregatedSynthesis",
]
