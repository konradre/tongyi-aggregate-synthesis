"""Discovery module for exploratory research."""

from .explorer import Explorer, DiscoveryResult, KnowledgeGap, KnowledgeLandscape, ScoredSource
from .routing import ConnectorRouter, RoutingDecision, QueryType
from .expansion import QueryExpander, ExpandedQuery
from .decomposer import QueryDecomposer, QueryAspect
from .gap_filler import GapFiller, GapFillingResult

__all__ = [
    # Explorer
    "Explorer",
    "DiscoveryResult",
    "KnowledgeGap",
    "KnowledgeLandscape",
    "ScoredSource",
    # Routing
    "ConnectorRouter",
    "RoutingDecision",
    "QueryType",
    # Expansion
    "QueryExpander",
    "ExpandedQuery",
    # Decomposition
    "QueryDecomposer",
    "QueryAspect",
    # Gap Filling
    "GapFiller",
    "GapFillingResult",
]
