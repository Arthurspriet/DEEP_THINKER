"""
Integration helper functions for easy integration into existing DeepThinker components.
"""

import logging
from typing import List, Dict, Any, Optional

from .epistemic import ClaimExtractorTool, ClaimConfidenceTool, CitationGate
from .search import SearchJustificationGenerator, SearchBudgetAllocator, EvidenceCompressorTool
from .memory import MemoryProvenanceTracker, MemoryRetrievalAuditor
from .reasoning import DepthController, PhaseROIEvaluator
from .resources import CPUGPUCapabilityRouter, ExecutionTierEscalator
from .output import AnswerConfidenceHeader, CounterfactualChecker
from .debug import MissionTraceVisualizer, CouncilDisagreementDetector
from .schemas import Claim, ConfidenceScore

logger = logging.getLogger(__name__)


class ToolingIntegrationHelper:
    """
    Helper class that provides easy integration points for the tooling layer.
    
    This class can be instantiated once and used throughout a mission.
    """
    
    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        gpu_available: bool = False
    ):
        """
        Initialize integration helper.
        
        Args:
            memory_manager: Optional MemoryManager instance
            gpu_available: Whether GPU is available
        """
        # Initialize tools
        self.claim_extractor = ClaimExtractorTool()
        self.confidence_estimator = ClaimConfidenceTool(memory_manager=memory_manager)
        self.citation_gate = CitationGate(require_citations=True)
        
        self.search_justifier = SearchJustificationGenerator()
        self.budget_allocator = SearchBudgetAllocator()
        self.evidence_compressor = EvidenceCompressorTool()
        
        self.provenance_tracker = MemoryProvenanceTracker()
        self.retrieval_auditor = MemoryRetrievalAuditor()
        
        self.depth_controller = DepthController()
        self.roi_evaluator = PhaseROIEvaluator()
        
        self.capability_router = CPUGPUCapabilityRouter(gpu_available=gpu_available)
        self.tier_escalator = ExecutionTierEscalator()
        
        self.confidence_header = AnswerConfidenceHeader()
        self.counterfactual_checker = CounterfactualChecker()
        
        self.trace_visualizer = MissionTraceVisualizer()
        self.disagreement_detector = CouncilDisagreementDetector()
    
    def process_council_output(
        self,
        output: str,
        council_name: str,
        phase_name: str,
        council_agreement: Optional[float] = None
    ) -> tuple[List[Claim], List[ConfidenceScore]]:
        """
        Process council output: extract claims and estimate confidence.
        
        Args:
            output: Council output text
            council_name: Name of council
            phase_name: Phase name
            council_agreement: Optional agreement level from consensus
            
        Returns:
            Tuple of (claims, confidence_scores)
        """
        # Extract claims
        claims = self.claim_extractor.extract_from_council_output(
            output=output,
            council_name=council_name,
            phase_name=phase_name
        )
        
        # Estimate confidence
        if council_agreement is not None:
            confidence_scores = [
                self.confidence_estimator.estimate_confidence(
                    claim=claim,
                    council_agreement=council_agreement
                )
                for claim in claims
            ]
        else:
            confidence_scores = self.confidence_estimator.estimate_batch(claims)
        
        return claims, confidence_scores
    
    def check_citations(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        memory_references: Optional[Dict[str, List[str]]] = None,
        web_evidence: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, bool]:
        """
        Check if claims have required citations.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores
            memory_references: Optional memory references
            web_evidence: Optional web evidence
            
        Returns:
            Dict mapping claim_id -> bool (True if verified)
        """
        return self.citation_gate.check_claims(
            claims=claims,
            confidence_scores=confidence_scores,
            memory_references=memory_references,
            web_evidence=web_evidence
        )
    
    def generate_search_plan(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        time_remaining_minutes: Optional[float] = None
    ) -> tuple[List[Any], Any]:
        """
        Generate search justifications and allocate budget.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores
            time_remaining_minutes: Optional time remaining
            
        Returns:
            Tuple of (justifications, budget)
        """
        # Generate justifications
        justifications = self.search_justifier.generate_justifications(
            claims=claims,
            confidence_scores=confidence_scores
        )
        
        # Allocate budget
        budget = self.budget_allocator.allocate_budget(
            time_remaining_minutes=time_remaining_minutes,
            claims=claims,
            confidence_scores=confidence_scores,
            justifications=justifications
        )
        
        return justifications, budget
    
    def compress_search_results(
        self,
        claim_id: str,
        raw_results: List[Dict[str, Any]],
        claim_text: Optional[str] = None
    ) -> Any:
        """
        Compress raw search results into evidence format.
        
        Args:
            claim_id: Claim ID
            raw_results: Raw search results
            claim_text: Optional claim text
            
        Returns:
            CompressedEvidence
        """
        return self.evidence_compressor.compress_evidence(
            claim_id=claim_id,
            raw_results=raw_results,
            claim_text=claim_text
        )
    
    def check_reasoning_depth(
        self,
        iteration_count: int,
        new_facts_detected: bool = True,
        decisions_changed: bool = False
    ) -> Any:
        """
        Check if reasoning should continue.
        
        Args:
            iteration_count: Current iteration
            new_facts_detected: Whether new facts detected
            decisions_changed: Whether decisions changed
            
        Returns:
            DepthDecision
        """
        return self.depth_controller.check_depth(
            iteration_count=iteration_count,
            new_facts_detected=new_facts_detected,
            decisions_changed=decisions_changed
        )
    
    def evaluate_phase_roi(self, phase_name: str) -> Any:
        """
        Evaluate ROI for a phase.
        
        Args:
            phase_name: Phase name
            
        Returns:
            ROIMetrics
        """
        return self.roi_evaluator.evaluate_phase(phase_name)
    
    def generate_confidence_header(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore],
        verified_claims: Dict[str, bool]
    ) -> Any:
        """
        Generate confidence header for final output.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores
            verified_claims: Dict of verified claims
            
        Returns:
            ConfidenceHeader
        """
        return self.confidence_header.generate_header(
            claims=claims,
            confidence_scores=confidence_scores,
            verified_claims=verified_claims
        )
    
    def check_counterfactuals(
        self,
        claims: List[Claim],
        confidence_scores: List[ConfidenceScore]
    ) -> Any:
        """
        Check counterfactual fragility.
        
        Args:
            claims: List of claims
            confidence_scores: Confidence scores
            
        Returns:
            CounterfactualResult
        """
        return self.counterfactual_checker.check_fragility(
            claims=claims,
            confidence_scores=confidence_scores
        )


# Global helper instance (can be set by mission orchestrator)
_global_helper: Optional[ToolingIntegrationHelper] = None


def get_tooling_helper() -> Optional[ToolingIntegrationHelper]:
    """Get the global tooling helper instance."""
    return _global_helper


def set_tooling_helper(helper: ToolingIntegrationHelper) -> None:
    """Set the global tooling helper instance."""
    global _global_helper
    _global_helper = helper

