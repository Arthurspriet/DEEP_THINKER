"""
Council Value Estimator for DeepThinker.

Non-ML estimator that evaluates ROI of invoking councils.
Only suppresses optional councils, never mandated ones.
"""

import logging
from typing import Tuple

from .policy_memory import PolicyMemory

logger = logging.getLogger(__name__)


class CouncilValueEstimator:
    """
    Estimates ROI of invoking a council in a phase.
    
    Based purely on historical statistics, never blocks
    councils mandated by PhaseSpec.
    """
    
    def __init__(
        self,
        policy_memory: PolicyMemory,
        cold_start_threshold: int = 10,
        negative_roi_threshold: float = 0.1
    ):
        """
        Initialize the council value estimator.
        
        Args:
            policy_memory: PolicyMemory for statistics
            cold_start_threshold: Minimum invocations before suppressing
            negative_roi_threshold: ROI below which to suppress
        """
        self._policy_memory = policy_memory
        self._cold_start_threshold = cold_start_threshold
        self._negative_roi_threshold = negative_roi_threshold
    
    def estimate_roi(
        self,
        council_name: str,
        phase_type: str,
        is_mandated: bool,
    ) -> Tuple[float, bool]:
        """
        Estimate ROI and decide if council should be invoked.
        
        Args:
            council_name: Name of the council
            phase_type: Type of phase
            is_mandated: Whether council is mandated by PhaseSpec
            
        Returns:
            Tuple of (roi_estimate, should_invoke)
        """
        # Never suppress mandated councils
        if is_mandated:
            return (1.0, True)
        
        stats = self._policy_memory.get_council_stats(council_name, phase_type)
        
        # Cold start: permissive (gather data)
        if stats.invocation_count < self._cold_start_threshold:
            logger.debug(
                f"Council {council_name} in {phase_type}: cold start "
                f"(count={stats.invocation_count}), allowing"
            )
            return (0.5, True)  # Default to invoke, gather data
        
        # Calculate ROI (quality per GPU second)
        roi = stats.quality_per_gpu_second
        
        # Suppress only if strong negative signal
        if roi < self._negative_roi_threshold and stats.invocation_count > 20:
            logger.info(
                f"Council {council_name} in {phase_type}: negative ROI "
                f"(roi={roi:.3f}, count={stats.invocation_count}), suppressing"
            )
            return (roi, False)
        
        # Also check for negative quality gain
        if stats.avg_quality_gain < -0.1 and stats.invocation_count > 15:
            logger.info(
                f"Council {council_name} in {phase_type}: negative quality gain "
                f"(gain={stats.avg_quality_gain:.3f}), suppressing"
            )
            return (roi, False)
        
        logger.debug(
            f"Council {council_name} in {phase_type}: ROI={roi:.3f}, "
            f"allowing (count={stats.invocation_count})"
        )
        return (roi, True)

