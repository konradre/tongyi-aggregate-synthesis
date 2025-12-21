"""Research synthesis with LLM integration."""

from .engine import SynthesisEngine
from .prompts import RESEARCH_SYSTEM_PROMPT, build_research_prompt

__all__ = ["SynthesisEngine", "RESEARCH_SYSTEM_PROMPT", "build_research_prompt"]
