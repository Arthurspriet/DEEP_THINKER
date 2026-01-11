"""
Consensus Policy Engine for DeepThinker 2.0.

Controls when consensus algorithms should run:
- Skip if only 1-2 models
- Skip if semantic agreement is high
- Skip if MajorityVote would add no signal

Prevents wasteful consensus computation when outputs already agree.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)

# Verbose logging integration
try:
    from deepthinker.cli import verbose_logger
    VERBOSE_LOGGER_AVAILABLE = True
except ImportError:
    VERBOSE_LOGGER_AVAILABLE = False
    verbose_logger = None


@dataclass
class ConsensusPolicyResult:
    """
    Result of consensus policy evaluation.
    
    Attributes:
        should_run: Whether to run full consensus
        reason: Reason for the decision
        agreement_score: Quick agreement estimate (if computed)
        recommended_action: What to do instead if not running consensus
        selected_output: Pre-selected output if consensus skipped
    """
    should_run: bool
    reason: str
    agreement_score: float = 0.0
    recommended_action: str = "run_consensus"
    selected_output: Optional[Any] = None


class ConsensusPolicyEngine:
    """
    Determines when consensus algorithms should run.
    
    Consensus is expensive (embedding computation, clustering).
    Skip it when:
    - Only 1-2 models produced output (no majority possible)
    - Quick agreement check shows high semantic similarity
    - All outputs are nearly identical
    
    This prevents MajorityVote from running when it adds no value.
    """
    
    # Thresholds
    MIN_MODELS_FOR_CONSENSUS = 3
    HIGH_AGREEMENT_THRESHOLD = 0.85
    IDENTICAL_HASH_THRESHOLD = 0.8  # If 80%+ have same hash, skip
    
    def __init__(
        self,
        min_models: int = 3,
        agreement_threshold: float = 0.85,
        enable_quick_check: bool = True
    ):
        """
        Initialize the policy engine.
        
        Args:
            min_models: Minimum models needed for consensus
            agreement_threshold: Threshold above which consensus is skipped
            enable_quick_check: Enable quick agreement estimation
        """
        self.min_models = min_models
        self.agreement_threshold = agreement_threshold
        self.enable_quick_check = enable_quick_check
        self._decision_log: List[Dict] = []
    
    def should_run_consensus(
        self,
        outputs: Dict[str, Any],
        council_name: str = ""
    ) -> ConsensusPolicyResult:
        """
        Determine if consensus should run.
        
        Args:
            outputs: Dict mapping model_name -> output
            council_name: Name of the council (for logging)
            
        Returns:
            ConsensusPolicyResult with decision
        """
        # Extract successful outputs
        successful = self._extract_successful_outputs(outputs)
        
        if not successful:
            return ConsensusPolicyResult(
                should_run=False,
                reason="no_successful_outputs",
                recommended_action="fail",
            )
        
        # Check model count
        if len(successful) < self.min_models:
            # Just select the best single output
            best = self._select_best_single(successful)
            
            self._log_decision(
                council_name,
                "skipped",
                f"insufficient_models ({len(successful)} < {self.min_models})",
                0.0
            )
            
            # Log consensus skip
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_consensus_panel(
                    council_name=council_name,
                    skipped=True,
                    skip_reason=f"insufficient models ({len(successful)} < {self.min_models})"
                )
            
            return ConsensusPolicyResult(
                should_run=False,
                reason=f"insufficient_models ({len(successful)} < {self.min_models})",
                recommended_action="select_best",
                selected_output=best,
            )
        
        # Check for identical outputs (cheap)
        hash_agreement = self._check_hash_agreement(successful)
        if hash_agreement >= self.IDENTICAL_HASH_THRESHOLD:
            # Outputs are nearly identical
            best = self._select_best_single(successful)
            
            self._log_decision(
                council_name,
                "skipped",
                f"outputs_identical (hash_agreement={hash_agreement:.2f})",
                hash_agreement
            )
            
            # Log consensus skip
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_consensus_panel(
                    council_name=council_name,
                    skipped=True,
                    skip_reason=f"outputs identical (similarity: {hash_agreement:.2f})"
                )
            
            return ConsensusPolicyResult(
                should_run=False,
                reason=f"outputs_identical (agreement={hash_agreement:.1%})",
                agreement_score=hash_agreement,
                recommended_action="select_any",
                selected_output=best,
            )
        
        # Quick semantic agreement check
        if self.enable_quick_check:
            quick_agreement = self._quick_agreement_estimate(successful)
            
            if quick_agreement >= self.agreement_threshold:
                best = self._select_best_single(successful)
                
                self._log_decision(
                    council_name,
                    "skipped",
                    f"high_agreement ({quick_agreement:.2f})",
                    quick_agreement
                )
                
                # Log consensus skip
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_consensus_panel(
                        council_name=council_name,
                        skipped=True,
                        skip_reason=f"high agreement (similarity: {quick_agreement:.2f})"
                    )
                
                return ConsensusPolicyResult(
                    should_run=False,
                    reason=f"high_agreement ({quick_agreement:.1%})",
                    agreement_score=quick_agreement,
                    recommended_action="select_best",
                    selected_output=best,
                )
        
        # Consensus should run
        self._log_decision(
            council_name,
            "run",
            "disagreement_detected",
            hash_agreement
        )
        
        return ConsensusPolicyResult(
            should_run=True,
            reason="disagreement_detected",
            agreement_score=hash_agreement,
            recommended_action="run_consensus",
        )
    
    def _extract_successful_outputs(
        self,
        outputs: Dict[str, Any]
    ) -> Dict[str, str]:
        """Extract successful text outputs."""
        successful = {}
        
        for name, output in outputs.items():
            if hasattr(output, 'success') and hasattr(output, 'output'):
                if output.success and output.output:
                    successful[name] = str(output.output)
            elif isinstance(output, str) and output:
                successful[name] = output
        
        return successful
    
    def _check_hash_agreement(
        self,
        outputs: Dict[str, str]
    ) -> float:
        """
        Check agreement based on content hashes.
        
        Fast check - if most outputs hash to the same value,
        they're essentially identical.
        """
        if not outputs:
            return 0.0
        
        # Normalize and hash each output
        hashes = {}
        for name, text in outputs.items():
            # Normalize: lowercase, strip whitespace, remove punctuation variations
            normalized = text.lower().strip()
            normalized = ' '.join(normalized.split())  # Normalize whitespace
            
            content_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
            hashes[name] = content_hash
        
        # Find most common hash
        hash_counts: Dict[str, int] = {}
        for h in hashes.values():
            hash_counts[h] = hash_counts.get(h, 0) + 1
        
        if not hash_counts:
            return 0.0
        
        max_count = max(hash_counts.values())
        agreement = max_count / len(outputs)
        
        return agreement
    
    def _quick_agreement_estimate(
        self,
        outputs: Dict[str, str]
    ) -> float:
        """
        Quick semantic agreement estimate without embeddings.
        
        Uses word overlap (Jaccard similarity) as a fast proxy.
        """
        if len(outputs) < 2:
            return 1.0
        
        # Get word sets for each output
        word_sets = {}
        for name, text in outputs.items():
            words = set(text.lower().split())
            # Remove very common words
            words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'to', 'of', 'in', 'for', 'on', 'with',
                     'at', 'by', 'from', 'as', 'or', 'and', 'but', 'if', 'that',
                     'this', 'it', 'its', 'we', 'you', 'they', 'their', 'our'}
            word_sets[name] = words
        
        # Compute pairwise Jaccard similarities
        similarities = []
        names = list(word_sets.keys())
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                set_a = word_sets[names[i]]
                set_b = word_sets[names[j]]
                
                if not set_a or not set_b:
                    continue
                
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                
                if union > 0:
                    similarities.append(intersection / union)
        
        if not similarities:
            return 0.5
        
        return sum(similarities) / len(similarities)
    
    def _select_best_single(
        self,
        outputs: Dict[str, str]
    ) -> Any:
        """
        Select the best single output when consensus is skipped.
        
        Prefers:
        1. Longest output (more detail)
        2. First in order (stable selection)
        """
        if not outputs:
            return None
        
        # Score by length (proxy for detail/quality)
        scored = [(name, text, len(text)) for name, text in outputs.items()]
        scored.sort(key=lambda x: x[2], reverse=True)
        
        return scored[0][1]  # Return the text
    
    def _log_decision(
        self,
        council_name: str,
        decision: str,
        reason: str,
        agreement: float
    ) -> None:
        """Log a policy decision."""
        entry = {
            "council": council_name,
            "decision": decision,
            "reason": reason,
            "agreement": agreement,
        }
        self._decision_log.append(entry)
        
        if decision == "skipped":
            logger.debug(
                f"Consensus skipped for {council_name}: {reason} "
                f"(agreement={agreement:.1%})"
            )
    
    def get_decision_log(self) -> List[Dict]:
        """Get the decision log."""
        return self._decision_log.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        total = len(self._decision_log)
        skipped = sum(1 for d in self._decision_log if d["decision"] == "skipped")
        
        return {
            "total_decisions": total,
            "consensus_skipped": skipped,
            "consensus_run": total - skipped,
            "skip_rate": skipped / total if total > 0 else 0,
        }
    
    def reset(self) -> None:
        """Reset decision log."""
        self._decision_log.clear()


# Global instance
_policy_engine: Optional[ConsensusPolicyEngine] = None


def get_consensus_policy_engine(
    min_models: int = 3,
    agreement_threshold: float = 0.85
) -> ConsensusPolicyEngine:
    """Get the global consensus policy engine."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = ConsensusPolicyEngine(
            min_models=min_models,
            agreement_threshold=agreement_threshold
        )
    return _policy_engine

