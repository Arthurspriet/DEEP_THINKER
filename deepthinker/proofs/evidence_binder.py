"""
Evidence Binder for Proof Packets.

Binds claims to evidence objects by querying the mission RAG store
and computing coverage scores via semantic similarity.

Enforces the key invariant:
- No evidence -> claim confidence capped at 0.3
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .proof_packet import (
    ClaimEntry,
    EvidenceBinding,
    EvidenceTypeProof,
)

if TYPE_CHECKING:
    from ..memory.rag_store import MissionRAGStore
    from ..metrics.evidence_object import EvidenceObject, EvidenceType

logger = logging.getLogger(__name__)


# Confidence cap for claims without evidence
NO_EVIDENCE_CONFIDENCE_CAP = 0.3

# Minimum similarity score to consider evidence relevant
MIN_SIMILARITY_THRESHOLD = 0.3

# Maximum evidence items to bind per claim
MAX_EVIDENCE_PER_CLAIM = 5


class EvidenceBinder:
    """
    Binds claims to evidence from the mission RAG store.
    
    Queries the RAG store for each claim to find semantically similar
    evidence, computes coverage scores, and returns EvidenceBinding objects.
    
    Usage:
        binder = EvidenceBinder(rag_store=mission_rag)
        bindings = binder.bind_evidence(claims)
    """
    
    def __init__(
        self,
        rag_store: Optional["MissionRAGStore"] = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        max_evidence_per_claim: int = MAX_EVIDENCE_PER_CLAIM,
    ):
        """
        Initialize the evidence binder.
        
        Args:
            rag_store: Mission RAG store for evidence retrieval
            min_similarity: Minimum similarity for evidence to be considered
            max_evidence_per_claim: Maximum evidence items per claim
        """
        self._rag_store = rag_store
        self._min_similarity = min_similarity
        self._max_evidence = max_evidence_per_claim
        
        # Cache for evidence lookups
        self._evidence_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def set_rag_store(self, rag_store: "MissionRAGStore") -> None:
        """Set the RAG store for evidence retrieval."""
        self._rag_store = rag_store
        self._evidence_cache.clear()
    
    def bind_evidence(
        self,
        claims: List[ClaimEntry],
    ) -> List[EvidenceBinding]:
        """
        Bind evidence to a list of claims.
        
        Args:
            claims: List of claims to bind evidence to
            
        Returns:
            List of EvidenceBinding objects
        """
        bindings = []
        
        for claim in claims:
            binding = self._bind_single_claim(claim)
            bindings.append(binding)
        
        logger.debug(
            f"[EVIDENCE_BINDER] Bound evidence for {len(claims)} claims, "
            f"{sum(1 for b in bindings if b.evidence_ids)} with evidence"
        )
        
        return bindings
    
    def _bind_single_claim(self, claim: ClaimEntry) -> EvidenceBinding:
        """
        Bind evidence to a single claim.
        
        Args:
            claim: Claim to bind evidence to
            
        Returns:
            EvidenceBinding for the claim
        """
        # Default binding with no evidence
        default_binding = EvidenceBinding(
            claim_id=claim.claim_id,
            evidence_ids=[],
            evidence_type=EvidenceTypeProof.ASSUMPTION,
            coverage_score=0.0,
        )
        
        if self._rag_store is None:
            return default_binding
        
        try:
            # Query RAG store for similar evidence
            results = self._query_evidence(claim.normalized_text)
            
            if not results:
                return default_binding
            
            # Filter by similarity threshold
            relevant_results = [
                r for r in results
                if r.get("similarity", 0) >= self._min_similarity
            ]
            
            if not relevant_results:
                return default_binding
            
            # Limit to max evidence per claim
            relevant_results = relevant_results[:self._max_evidence]
            
            # Extract evidence IDs
            evidence_ids = [
                r.get("id", r.get("doc_id", ""))
                for r in relevant_results
                if r.get("id") or r.get("doc_id")
            ]
            
            # Determine evidence type from first result
            evidence_type = self._infer_evidence_type(relevant_results[0])
            
            # Compute coverage score as average similarity
            coverage_score = sum(
                r.get("similarity", 0) for r in relevant_results
            ) / len(relevant_results)
            
            return EvidenceBinding(
                claim_id=claim.claim_id,
                evidence_ids=evidence_ids,
                evidence_type=evidence_type,
                coverage_score=min(1.0, coverage_score),
            )
            
        except Exception as e:
            logger.warning(f"[EVIDENCE_BINDER] Failed to bind evidence: {e}")
            return default_binding
    
    def _query_evidence(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Query the RAG store for evidence.
        
        Args:
            query_text: Text to query for
            
        Returns:
            List of evidence results with similarity scores
        """
        # Check cache
        cache_key = query_text[:100]  # Use truncated text as key
        if cache_key in self._evidence_cache:
            return self._evidence_cache[cache_key]
        
        if self._rag_store is None:
            return []
        
        try:
            # Query RAG store
            results = self._rag_store.query(
                query=query_text,
                top_k=self._max_evidence * 2,  # Get more than needed for filtering
            )
            
            # Convert to standard format
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(result)
                elif hasattr(result, "to_dict"):
                    formatted_results.append(result.to_dict())
                else:
                    # Try to extract common attributes
                    formatted_results.append({
                        "id": getattr(result, "id", ""),
                        "text": getattr(result, "text", str(result)),
                        "similarity": getattr(result, "similarity", 0.5),
                        "artifact_type": getattr(result, "artifact_type", "general"),
                    })
            
            # Cache results
            self._evidence_cache[cache_key] = formatted_results
            
            return formatted_results
            
        except Exception as e:
            logger.debug(f"[EVIDENCE_BINDER] RAG query failed: {e}")
            return []
    
    def _infer_evidence_type(self, result: Dict[str, Any]) -> EvidenceTypeProof:
        """
        Infer evidence type from a RAG result.
        
        Args:
            result: RAG query result
            
        Returns:
            EvidenceTypeProof for the result
        """
        artifact_type = result.get("artifact_type", "").lower()
        source = result.get("source", "").lower()
        
        # Check artifact type
        if artifact_type in ("web", "web_search", "search"):
            return EvidenceTypeProof.WEB
        elif artifact_type in ("code", "code_output", "execution"):
            return EvidenceTypeProof.CODE
        elif artifact_type in ("simulation", "sim"):
            return EvidenceTypeProof.SIMULATION
        elif artifact_type in ("memory", "rag", "retrieval"):
            return EvidenceTypeProof.MEMORY
        
        # Check source
        if "http" in source or "www" in source:
            return EvidenceTypeProof.WEB
        elif "code" in source or "exec" in source:
            return EvidenceTypeProof.CODE
        
        # Default to memory (from RAG)
        return EvidenceTypeProof.MEMORY
    
    def adjust_claim_confidence(
        self,
        claims: List[ClaimEntry],
        bindings: List[EvidenceBinding],
    ) -> List[ClaimEntry]:
        """
        Adjust claim confidence based on evidence coverage.
        
        Claims without evidence have confidence capped.
        
        Args:
            claims: List of claims
            bindings: Evidence bindings for claims
            
        Returns:
            Claims with adjusted confidence
        """
        binding_map = {b.claim_id: b for b in bindings}
        adjusted_claims = []
        
        for claim in claims:
            binding = binding_map.get(claim.claim_id)
            
            # Check if claim has evidence
            has_evidence = (
                binding is not None and
                binding.coverage_score > 0 and
                len(binding.evidence_ids) > 0
            )
            
            if not has_evidence:
                # Cap confidence for claims without evidence
                new_confidence = min(claim.confidence_estimate, NO_EVIDENCE_CONFIDENCE_CAP)
                adjusted_claims.append(ClaimEntry(
                    claim_id=claim.claim_id,
                    normalized_text=claim.normalized_text,
                    claim_type=claim.claim_type,
                    confidence_estimate=new_confidence,
                ))
            else:
                adjusted_claims.append(claim)
        
        return adjusted_claims
    
    def compute_evidence_stats(
        self,
        bindings: List[EvidenceBinding],
    ) -> Dict[str, Any]:
        """
        Compute statistics about evidence bindings.
        
        Args:
            bindings: List of evidence bindings
            
        Returns:
            Dictionary with evidence statistics
        """
        if not bindings:
            return {
                "total_claims": 0,
                "claims_with_evidence": 0,
                "evidence_coverage_ratio": 0.0,
                "average_coverage_score": 0.0,
                "by_type": {},
            }
        
        with_evidence = [b for b in bindings if b.evidence_ids]
        
        by_type = {}
        for evidence_type in EvidenceTypeProof:
            count = len([b for b in bindings if b.evidence_type == evidence_type])
            if count > 0:
                by_type[evidence_type.value] = count
        
        avg_coverage = (
            sum(b.coverage_score for b in bindings) / len(bindings)
            if bindings else 0.0
        )
        
        return {
            "total_claims": len(bindings),
            "claims_with_evidence": len(with_evidence),
            "evidence_coverage_ratio": len(with_evidence) / len(bindings),
            "average_coverage_score": avg_coverage,
            "by_type": by_type,
        }
    
    def clear_cache(self) -> None:
        """Clear the evidence cache."""
        self._evidence_cache.clear()


# Global binder instance
_binder: Optional[EvidenceBinder] = None


def get_evidence_binder(
    rag_store: Optional["MissionRAGStore"] = None,
) -> EvidenceBinder:
    """Get the global evidence binder instance."""
    global _binder
    if _binder is None:
        _binder = EvidenceBinder(rag_store=rag_store)
    elif rag_store is not None:
        _binder.set_rag_store(rag_store)
    return _binder

