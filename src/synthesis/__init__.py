"""Research synthesis with LLM integration."""

from .engine import SynthesisEngine
from .prompts import RESEARCH_SYSTEM_PROMPT, build_research_prompt
from .aggregator import (
    SynthesisAggregator,
    SynthesisStyle,
    PreGatheredSource,
    AggregatedSynthesis,
)
from .verification import (
    CitationVerifier,
    VerifiedClaim,
    VerificationResult,
    extract_claims_with_citations,
)
from .binding import (
    BidirectionalBinder,
    BidirectionalBinding,
    EvidenceExcerpt,
)
from .quality_gate import (
    SourceQualityGate,
    QualityGateResult,
    QualityDecision,
)
from .contradictions import (
    ContradictionDetector,
    Contradiction,
    ContradictionReport,
    ContradictionSeverity,
)

__all__ = [
    # Engine
    "SynthesisEngine",
    "RESEARCH_SYSTEM_PROMPT",
    "build_research_prompt",
    # Aggregator
    "SynthesisAggregator",
    "SynthesisStyle",
    "PreGatheredSource",
    "AggregatedSynthesis",
    # Verification
    "CitationVerifier",
    "VerifiedClaim",
    "VerificationResult",
    "extract_claims_with_citations",
    # Binding
    "BidirectionalBinder",
    "BidirectionalBinding",
    "EvidenceExcerpt",
    # Quality Gate
    "SourceQualityGate",
    "QualityGateResult",
    "QualityDecision",
    # Contradiction Detection
    "ContradictionDetector",
    "Contradiction",
    "ContradictionReport",
    "ContradictionSeverity",
]
