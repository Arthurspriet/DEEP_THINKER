"""
Claim Graph for DeepThinker Epistemics.

Provides a graph-based representation of claims and their relationships:
- Nodes: Claims from ClaimRegistry
- Edges: supports / contradicts relationships

Integrates with existing ClaimRegistry for claim management.
Runs contradiction detection only on top-K load-bearing claims.

Gated with config flag: CLAIM_GRAPH_ENABLED
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .claim_validator import Claim, ClaimStatus, ClaimType

logger = logging.getLogger(__name__)


class EdgeType(str, Enum):
    """Types of edges between claims."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    SUPERSEDES = "supersedes"


@dataclass
class ClaimEdge:
    """
    Edge between two claims in the graph.
    
    Attributes:
        source_claim_id: ID of the source claim
        target_claim_id: ID of the target claim
        edge_type: Type of relationship
        confidence: Confidence in the relationship (0-1)
        detected_by: How the relationship was detected
        timestamp: When the edge was created
    """
    source_claim_id: str
    target_claim_id: str
    edge_type: EdgeType
    confidence: float = 0.5
    detected_by: str = "heuristic"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_claim_id": self.source_claim_id,
            "target_claim_id": self.target_claim_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "detected_by": self.detected_by,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEdge":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            source_claim_id=data.get("source_claim_id", ""),
            target_claim_id=data.get("target_claim_id", ""),
            edge_type=EdgeType(data.get("edge_type", "supports")),
            confidence=data.get("confidence", 0.5),
            detected_by=data.get("detected_by", "heuristic"),
            timestamp=timestamp,
        )


@dataclass
class ClaimNode:
    """
    Node in the claim graph.
    
    Attributes:
        claim: The underlying claim
        incoming_edges: Edges pointing to this claim
        outgoing_edges: Edges from this claim
        load_bearing_score: How central/important this claim is
    """
    claim: Claim
    incoming_edges: List[ClaimEdge] = field(default_factory=list)
    outgoing_edges: List[ClaimEdge] = field(default_factory=list)
    load_bearing_score: float = 0.0
    
    @property
    def claim_id(self) -> str:
        return self.claim.id
    
    @property
    def contradiction_count(self) -> int:
        """Count incoming contradiction edges."""
        return sum(
            1 for e in self.incoming_edges
            if e.edge_type == EdgeType.CONTRADICTS
        )
    
    @property
    def support_count(self) -> int:
        """Count incoming support edges."""
        return sum(
            1 for e in self.incoming_edges
            if e.edge_type == EdgeType.SUPPORTS
        )
    
    def compute_load_bearing_score(self) -> float:
        """
        Compute how central/important this claim is.
        
        Based on:
        - Number of outgoing edges (supports other claims)
        - Claim type (facts are more load-bearing)
        - Status (grounded claims are more important)
        """
        score = 0.0
        
        # Outgoing edges increase importance
        score += len(self.outgoing_edges) * 0.2
        
        # Claim type weight
        if self.claim.claim_type == ClaimType.FACT:
            score += 0.3
        elif self.claim.claim_type == ClaimType.INFERENCE:
            score += 0.2
        
        # Grounded claims are more important
        if self.claim.status == ClaimStatus.GROUNDED:
            score += 0.3
        elif self.claim.status == ClaimStatus.CONTESTED:
            score += 0.1
        
        # Confidence contributes
        score += self.claim.confidence * 0.2
        
        self.load_bearing_score = min(1.0, score)
        return self.load_bearing_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "incoming_edges": [e.to_dict() for e in self.incoming_edges],
            "outgoing_edges": [e.to_dict() for e in self.outgoing_edges],
            "load_bearing_score": self.load_bearing_score,
        }


class ClaimGraph:
    """
    Graph of claims and their relationships.
    
    Provides:
    - Building graph from ClaimRegistry claims
    - Support/contradiction edge detection
    - Top-K load-bearing claim selection
    - Consistency score computation
    
    Usage:
        from deepthinker.epistemics import ClaimRegistry
        
        registry = ClaimRegistry()
        # ... add claims ...
        
        graph = ClaimGraph()
        graph.build_from_registry(registry)
        
        top_claims = graph.get_top_k_load_bearing(k=20)
        consistency = graph.compute_consistency_score()
    """
    
    def __init__(self):
        """Initialize the claim graph."""
        self._nodes: Dict[str, ClaimNode] = {}
        self._edges: List[ClaimEdge] = []
        self._contradiction_pairs: Set[Tuple[str, str]] = set()
    
    def add_claim(self, claim: Claim) -> ClaimNode:
        """
        Add a claim to the graph.
        
        Args:
            claim: Claim to add
            
        Returns:
            Created ClaimNode
        """
        if claim.id in self._nodes:
            return self._nodes[claim.id]
        
        node = ClaimNode(claim=claim)
        self._nodes[claim.id] = node
        return node
    
    def add_edge(
        self,
        source_claim_id: str,
        target_claim_id: str,
        edge_type: EdgeType,
        confidence: float = 0.5,
        detected_by: str = "heuristic",
    ) -> Optional[ClaimEdge]:
        """
        Add an edge between two claims.
        
        Args:
            source_claim_id: Source claim ID
            target_claim_id: Target claim ID
            edge_type: Type of relationship
            confidence: Confidence in relationship
            detected_by: Detection method
            
        Returns:
            Created ClaimEdge, or None if claims not found
        """
        if source_claim_id not in self._nodes or target_claim_id not in self._nodes:
            logger.warning(
                f"[CLAIM_GRAPH] Cannot add edge: claim(s) not in graph"
            )
            return None
        
        edge = ClaimEdge(
            source_claim_id=source_claim_id,
            target_claim_id=target_claim_id,
            edge_type=edge_type,
            confidence=confidence,
            detected_by=detected_by,
        )
        
        self._edges.append(edge)
        self._nodes[source_claim_id].outgoing_edges.append(edge)
        self._nodes[target_claim_id].incoming_edges.append(edge)
        
        # Track contradiction pairs
        if edge_type == EdgeType.CONTRADICTS:
            pair = tuple(sorted([source_claim_id, target_claim_id]))
            self._contradiction_pairs.add(pair)
        
        return edge
    
    def build_from_claims(self, claims: List[Claim]) -> None:
        """
        Build graph from a list of claims.
        
        Adds all claims and detects support relationships
        based on upstream_claim_ids.
        
        Args:
            claims: List of claims to add
        """
        # Add all claims first
        for claim in claims:
            self.add_claim(claim)
        
        # Add support edges based on upstream references
        for claim in claims:
            for upstream_id in claim.upstream_claim_ids:
                if upstream_id in self._nodes:
                    self.add_edge(
                        source_claim_id=upstream_id,
                        target_claim_id=claim.id,
                        edge_type=EdgeType.SUPPORTS,
                        confidence=0.7,
                        detected_by="upstream_reference",
                    )
        
        # Compute load-bearing scores
        self._compute_all_load_bearing_scores()
        
        logger.debug(
            f"[CLAIM_GRAPH] Built graph: {len(self._nodes)} claims, "
            f"{len(self._edges)} edges"
        )
    
    def build_from_registry(self, registry: Any) -> None:
        """
        Build graph from a ClaimRegistry.
        
        Args:
            registry: ClaimRegistry instance
        """
        # Get all claims from registry
        claims = list(registry._claims.values())
        self.build_from_claims(claims)
    
    def _compute_all_load_bearing_scores(self) -> None:
        """Compute load-bearing scores for all nodes."""
        for node in self._nodes.values():
            node.compute_load_bearing_score()
    
    def get_top_k_load_bearing(self, k: int = 20) -> List[ClaimNode]:
        """
        Get the top-K most load-bearing claims.
        
        Args:
            k: Number of claims to return
            
        Returns:
            List of top-K ClaimNodes sorted by load-bearing score
        """
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: n.load_bearing_score,
            reverse=True,
        )
        return sorted_nodes[:k]
    
    def get_grounded_claims(self) -> List[ClaimNode]:
        """Get all grounded claim nodes."""
        return [
            node for node in self._nodes.values()
            if node.claim.status == ClaimStatus.GROUNDED
        ]
    
    def get_contradicting_pairs(self) -> List[Tuple[Claim, Claim]]:
        """
        Get all pairs of contradicting claims.
        
        Returns:
            List of (claim1, claim2) tuples
        """
        pairs = []
        for id1, id2 in self._contradiction_pairs:
            if id1 in self._nodes and id2 in self._nodes:
                pairs.append((
                    self._nodes[id1].claim,
                    self._nodes[id2].claim,
                ))
        return pairs
    
    def compute_consistency_score(self) -> float:
        """
        Compute overall consistency score.
        
        Based on contradiction rate among claims.
        
        Returns:
            Consistency score (0-1, where 1 is fully consistent)
        """
        if not self._nodes:
            return 1.0
        
        num_claims = len(self._nodes)
        num_contradictions = len(self._contradiction_pairs)
        
        # Maximum possible contradictions for n claims
        max_contradictions = num_claims * (num_claims - 1) / 2
        
        if max_contradictions == 0:
            return 1.0
        
        # Consistency is inverse of contradiction rate
        # Use a softer penalty for contradictions
        contradiction_rate = num_contradictions / min(max_contradictions, num_claims)
        consistency = max(0.0, 1.0 - contradiction_rate)
        
        return consistency
    
    def get_claim(self, claim_id: str) -> Optional[ClaimNode]:
        """Get a claim node by ID."""
        return self._nodes.get(claim_id)
    
    def get_claims_by_type(self, claim_type: ClaimType) -> List[ClaimNode]:
        """Get all claims of a specific type."""
        return [
            node for node in self._nodes.values()
            if node.claim.claim_type == claim_type
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        edge_type_counts = {}
        for edge in self._edges:
            edge_type_counts[edge.edge_type.value] = (
                edge_type_counts.get(edge.edge_type.value, 0) + 1
            )
        
        return {
            "num_claims": len(self._nodes),
            "num_edges": len(self._edges),
            "num_contradictions": len(self._contradiction_pairs),
            "consistency_score": self.compute_consistency_score(),
            "edge_types": edge_type_counts,
            "grounded_claims": len(self.get_grounded_claims()),
            "avg_load_bearing_score": (
                sum(n.load_bearing_score for n in self._nodes.values()) /
                len(self._nodes) if self._nodes else 0
            ),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {
                claim_id: node.to_dict()
                for claim_id, node in self._nodes.items()
            },
            "edges": [e.to_dict() for e in self._edges],
            "contradiction_pairs": [
                list(pair) for pair in self._contradiction_pairs
            ],
            "statistics": self.get_statistics(),
        }
    
    def clear(self) -> None:
        """Clear the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._contradiction_pairs.clear()


def rank_claims_by_load_bearing(
    claims: List[Claim],
    k: int = 20,
) -> List[Claim]:
    """
    Convenience function to rank claims by load-bearing score.
    
    Args:
        claims: List of claims to rank
        k: Maximum number to return
        
    Returns:
        Top-K claims sorted by load-bearing score
    """
    graph = ClaimGraph()
    graph.build_from_claims(claims)
    top_nodes = graph.get_top_k_load_bearing(k)
    return [node.claim for node in top_nodes]

