"""
Phase ROI Evaluator - Tracks return on investment for each phase.
"""

import logging
from typing import List, Dict, Any, Optional

from ..schemas import ROIMetrics

logger = logging.getLogger(__name__)


class PhaseROIEvaluator:
    """
    Tracks per-phase metrics:
    - Tokens spent
    - Facts added (new claims verified)
    - Decisions changed (plan revisions)
    
    Calculates ROI and aborts phases with zero ROI.
    """
    
    def __init__(self, min_roi_threshold: float = 0.001):
        """
        Initialize ROI evaluator.
        
        Args:
            min_roi_threshold: Minimum ROI to consider phase valuable (default: 0.001)
        """
        self.min_roi_threshold = min_roi_threshold
        self._phase_metrics: Dict[str, ROIMetrics] = {}
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start tracking a phase.
        
        Args:
            phase_name: Name of the phase
        """
        self._phase_metrics[phase_name] = ROIMetrics(
            phase_name=phase_name,
            tokens_spent=0,
            facts_added=0,
            decisions_changed=0,
            roi_score=0.0,
            should_abort=False
        )
        logger.debug(f"Started ROI tracking for phase: {phase_name}")
    
    def record_tokens(self, phase_name: str, tokens: int) -> None:
        """
        Record tokens spent in a phase.
        
        Args:
            phase_name: Phase name
            tokens: Number of tokens spent
        """
        if phase_name in self._phase_metrics:
            self._phase_metrics[phase_name].tokens_spent += tokens
    
    def record_facts(self, phase_name: str, facts_count: int) -> None:
        """
        Record facts added in a phase.
        
        Args:
            phase_name: Phase name
            facts_count: Number of new facts/claims verified
        """
        if phase_name in self._phase_metrics:
            self._phase_metrics[phase_name].facts_added += facts_count
    
    def record_decisions(self, phase_name: str, decisions_count: int) -> None:
        """
        Record decisions changed in a phase.
        
        Args:
            phase_name: Phase name
            decisions_count: Number of plan revisions or goal changes
        """
        if phase_name in self._phase_metrics:
            self._phase_metrics[phase_name].decisions_changed += decisions_count
    
    def evaluate_phase(self, phase_name: str) -> ROIMetrics:
        """
        Evaluate ROI for a phase.
        
        Args:
            phase_name: Phase name to evaluate
            
        Returns:
            ROIMetrics with calculated ROI and abort recommendation
        """
        if phase_name not in self._phase_metrics:
            logger.warning(f"Phase {phase_name} not found in metrics")
            return ROIMetrics(
                phase_name=phase_name,
                tokens_spent=0,
                facts_added=0,
                decisions_changed=0,
                roi_score=0.0,
                should_abort=True
            )
        
        metrics = self._phase_metrics[phase_name]
        
        # Calculate ROI: (facts_added + decisions_changed) / tokens_spent
        if metrics.tokens_spent > 0:
            metrics.roi_score = (
                (metrics.facts_added + metrics.decisions_changed) / metrics.tokens_spent
            )
        else:
            # No tokens spent - consider it neutral
            metrics.roi_score = 0.0
        
        # Determine if should abort
        metrics.should_abort = metrics.roi_score < self.min_roi_threshold
        
        logger.info(
            f"Phase ROI evaluation for {phase_name}: "
            f"tokens={metrics.tokens_spent}, facts={metrics.facts_added}, "
            f"decisions={metrics.decisions_changed}, ROI={metrics.roi_score:.4f}, "
            f"abort={metrics.should_abort}"
        )
        
        return metrics
    
    def get_phase_metrics(self, phase_name: str) -> Optional[ROIMetrics]:
        """
        Get metrics for a phase.
        
        Args:
            phase_name: Phase name
            
        Returns:
            ROIMetrics or None if phase not found
        """
        return self._phase_metrics.get(phase_name)
    
    def get_all_metrics(self) -> Dict[str, ROIMetrics]:
        """
        Get all phase metrics.
        
        Returns:
            Dict mapping phase_name -> ROIMetrics
        """
        return self._phase_metrics.copy()
    
    def should_abort_phase(self, phase_name: str) -> bool:
        """
        Check if a phase should be aborted based on ROI.
        
        Args:
            phase_name: Phase name to check
            
        Returns:
            True if phase should be aborted
        """
        metrics = self.evaluate_phase(phase_name)
        return metrics.should_abort

