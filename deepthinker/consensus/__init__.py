"""
Consensus Engine for DeepThinker 2.0 Council System.

Provides multiple algorithms for reaching consensus from multiple LLM outputs:
- MajorityVoteConsensus: Semantic voting based on embedding similarity
- WeightedBlendConsensus: Weighted merging of responses
- CritiqueConsensus: Models critique each other, then re-evaluate
- SemanticDistanceConsensus: Picks response furthest from hallucination cluster

DeepThinker 2.0 Additions:
- ConsensusPolicyEngine: Determines when to skip consensus (high agreement/few models)
"""

from .voting import MajorityVoteConsensus
from .weighted_blend import WeightedBlendConsensus
from .critique_exchange import CritiqueConsensus
from .semantic_distance import SemanticDistanceConsensus
from .policy_engine import (
    ConsensusPolicyEngine,
    ConsensusPolicyResult,
    get_consensus_policy_engine,
)

__all__ = [
    "MajorityVoteConsensus",
    "WeightedBlendConsensus",
    "CritiqueConsensus",
    "SemanticDistanceConsensus",
    # Policy engine (new in 2.0)
    "ConsensusPolicyEngine",
    "ConsensusPolicyResult",
    "get_consensus_policy_engine",
]


def get_consensus_engine(algorithm: str):
    """
    Factory function to get consensus engine by name.
    
    Args:
        algorithm: Name of consensus algorithm
        
    Returns:
        Consensus engine instance
    """
    engines = {
        "voting": MajorityVoteConsensus,
        "weighted_blend": WeightedBlendConsensus,
        "critique_exchange": CritiqueConsensus,
        "semantic_distance": SemanticDistanceConsensus,
    }
    
    if algorithm not in engines:
        raise ValueError(f"Unknown consensus algorithm: {algorithm}. "
                        f"Available: {list(engines.keys())}")
    
    return engines[algorithm]()

