"""
Depth Controller - Enforces max reasoning loops and detects diminishing returns.
"""

import logging
from typing import List, Dict, Any, Optional

from ..schemas import DepthDecision

logger = logging.getLogger(__name__)


class DepthController:
    """
    Enforces max reasoning loops per phase and detects diminishing returns.
    
    Rule: If no new signal → force synthesis.
    """
    
    def __init__(
        self,
        max_iterations_per_phase: int = 5,
        diminishing_returns_threshold: int = 2
    ):
        """
        Initialize depth controller.
        
        Args:
            max_iterations_per_phase: Maximum iterations allowed per phase
            diminishing_returns_threshold: Consecutive iterations with no new signal before forcing stop
        """
        self.max_iterations = max_iterations_per_phase
        self.diminishing_returns_threshold = diminishing_returns_threshold
    
    def check_depth(
        self,
        iteration_count: int,
        previous_outputs: Optional[List[str]] = None,
        current_output: Optional[str] = None,
        new_facts_detected: bool = True,
        decisions_changed: bool = False
    ) -> DepthDecision:
        """
        Check if reasoning should continue or stop.
        
        Args:
            iteration_count: Current iteration number (1-indexed)
            previous_outputs: Optional list of previous iteration outputs
            current_output: Optional current iteration output
            new_facts_detected: Whether new facts were detected in this iteration
            decisions_changed: Whether any decisions/plans changed in this iteration
            
        Returns:
            DepthDecision with continue flag and reason
        """
        # Check max iterations
        if iteration_count >= self.max_iterations:
            return DepthDecision(
                continue_reasoning=False,
                max_remaining=0,
                reason=f"Maximum iterations ({self.max_iterations}) reached",
                diminishing_returns_detected=False,
                new_signal_detected=False
            )
        
        # Check for new signal
        new_signal = new_facts_detected or decisions_changed
        
        # Detect diminishing returns
        diminishing_returns = False
        if previous_outputs and current_output:
            # Check if current output is similar to previous outputs
            # (simple heuristic: check if significant new content)
            recent_outputs = previous_outputs[-self.diminishing_returns_threshold:]
            
            # Count how many recent iterations had no new signal
            no_signal_count = 0
            for prev_output in recent_outputs:
                # Simple similarity check (could be improved)
                if self._outputs_similar(prev_output, current_output):
                    no_signal_count += 1
            
            if no_signal_count >= self.diminishing_returns_threshold:
                diminishing_returns = True
                return DepthDecision(
                    continue_reasoning=False,
                    max_remaining=0,
                    reason=f"Diminishing returns detected: {no_signal_count} consecutive iterations with no new signal",
                    diminishing_returns_detected=True,
                    new_signal_detected=False
                )
        
        # If no new signal in current iteration, check if we should continue
        if not new_signal:
            # Allow a few iterations without new signal, but not too many
            if iteration_count >= self.diminishing_returns_threshold:
                return DepthDecision(
                    continue_reasoning=False,
                    max_remaining=0,
                    reason="No new signal detected and iteration threshold reached",
                    diminishing_returns_detected=True,
                    new_signal_detected=False
                )
        
        # Continue reasoning
        max_remaining = self.max_iterations - iteration_count
        return DepthDecision(
            continue_reasoning=True,
            max_remaining=max_remaining,
            reason=f"Continuing reasoning ({max_remaining} iterations remaining)",
            diminishing_returns_detected=False,
            new_signal_detected=new_signal
        )
    
    def _outputs_similar(self, output1: str, output2: str, threshold: float = 0.8) -> bool:
        """
        Check if two outputs are similar (simple heuristic).
        
        Args:
            output1: First output
            output2: Second output
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if outputs are similar
        """
        if not output1 or not output2:
            return False
        
        # Simple word overlap check
        words1 = set(output1.lower().split())
        words2 = set(output2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= threshold
    
    def should_force_synthesis(
        self,
        iteration_count: int,
        new_signal_detected: bool,
        diminishing_returns_detected: bool
    ) -> bool:
        """
        Determine if synthesis should be forced.
        
        Args:
            iteration_count: Current iteration count
            new_signal_detected: Whether new signal was detected
            diminishing_returns_detected: Whether diminishing returns detected
            
        Returns:
            True if synthesis should be forced
        """
        if diminishing_returns_detected:
            return True
        
        if iteration_count >= self.max_iterations:
            return True
        
        if not new_signal_detected and iteration_count >= self.diminishing_returns_threshold:
            return True
        
        return False

