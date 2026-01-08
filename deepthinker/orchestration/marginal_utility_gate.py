"""
Marginal Utility Gate for DeepThinker.

Gates expensive operations (DEEP_ANALYSIS, SIMULATION, extra consensus)
based on expected marginal utility vs cost.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .policy_memory import PolicyMemory

logger = logging.getLogger(__name__)


@dataclass
class MarginalUtilityDecision:
    """
    Decision from marginal utility gate.
    """
    should_proceed: bool
    expected_quality_gain: float
    expected_cost_increase: Tuple[int, float]  # tokens, seconds
    marginal_utility: float
    skip_reason: Optional[str] = None
    alternative_action: Optional[str] = None


class MarginalUtilityGate:
    """
    Gates expensive operations based on marginal utility analysis.
    
    Compares expected quality gain vs expected cost increase.
    Only proceeds if marginal utility is positive.
    """
    
    def __init__(
        self,
        policy_memory: PolicyMemory,
        min_marginal_utility: float = 0.05,  # Conservative default
        cost_weight_tokens: float = 0.0001,  # 10k tokens = 1.0 cost
        cost_weight_time: float = 0.01,  # 100s = 1.0 cost
    ):
        """
        Initialize the marginal utility gate.
        
        Args:
            policy_memory: PolicyMemory for historical statistics
            min_marginal_utility: Minimum MU threshold to proceed
            cost_weight_tokens: Weight for token cost
            cost_weight_time: Weight for time cost
        """
        self._policy_memory = policy_memory
        self._min_marginal_utility = min_marginal_utility
        self._cost_weight_tokens = cost_weight_tokens
        self._cost_weight_time = cost_weight_time
    
    def check_deep_analysis(
        self,
        current_quality: float,
        time_remaining: float,
        mission_preferences,
    ) -> MarginalUtilityDecision:
        """
        Check if DEEP_ANALYSIS phase should proceed.
        
        Args:
            current_quality: Current quality score (0-1)
            time_remaining: Time remaining in mission (seconds)
            mission_preferences: MissionPreferences instance
            
        Returns:
            MarginalUtilityDecision
        """
        # Get expected gain and cost for deep analysis
        expected_gain = self._policy_memory.expected_quality_gain(
            "deep_analysis", "synthesis"
        )
        
        # If no data, default to allowing (cold start)
        if expected_gain == 0.0:
            expected_gain = 0.15  # Conservative estimate
            expected_tokens = 5000
            expected_time = 120.0
        else:
            expected_tokens, expected_time = self._policy_memory.expected_cost(
                "deep_analysis", "synthesis"
            )
        
        # Cost-weighted utility
        cost = (
            expected_tokens * self._cost_weight_tokens +
            expected_time * self._cost_weight_time
        ) * mission_preferences.cost_sensitivity
        
        marginal_utility = expected_gain - cost
        
        # Apply preference biases
        if mission_preferences.quality_priority > 0.8:
            marginal_utility *= 1.2  # Boost for quality-focused missions
        if mission_preferences.latency_sensitivity > 0.8:
            marginal_utility *= 0.7  # Penalize for latency-sensitive missions
        
        # Check if we have enough time
        if time_remaining < expected_time * 1.5:
            return MarginalUtilityDecision(
                should_proceed=False,
                expected_quality_gain=expected_gain,
                expected_cost_increase=(expected_tokens, expected_time),
                marginal_utility=marginal_utility,
                skip_reason=f"Insufficient time: {time_remaining:.1f}s < {expected_time * 1.5:.1f}s",
                alternative_action="proceed_to_synthesis"
            )
        
        if marginal_utility <= self._min_marginal_utility:
            return MarginalUtilityDecision(
                should_proceed=False,
                expected_quality_gain=expected_gain,
                expected_cost_increase=(expected_tokens, expected_time),
                marginal_utility=marginal_utility,
                skip_reason=f"MU={marginal_utility:.3f} <= threshold={self._min_marginal_utility}",
                alternative_action="proceed_to_synthesis"
            )
        
        return MarginalUtilityDecision(
            should_proceed=True,
            expected_quality_gain=expected_gain,
            expected_cost_increase=(expected_tokens, expected_time),
            marginal_utility=marginal_utility,
        )
    
    def check_simulation(
        self,
        current_quality: float,
        complexity_score: float,
        mission_preferences,
    ) -> MarginalUtilityDecision:
        """
        Check if SIMULATION phase should proceed.
        
        Args:
            current_quality: Current quality score (0-1)
            complexity_score: Complexity of the problem (0-1)
            mission_preferences: MissionPreferences instance
            
        Returns:
            MarginalUtilityDecision
        """
        # Get expected gain and cost for simulation
        expected_gain = self._policy_memory.expected_quality_gain(
            "simulation", "simulation"
        )
        
        if expected_gain == 0.0:
            expected_gain = 0.10  # Conservative estimate
            expected_tokens = 3000
            expected_time = 90.0
        else:
            expected_tokens, expected_time = self._policy_memory.expected_cost(
                "simulation", "simulation"
            )
        
        # Adjust gain based on complexity (higher complexity = more value from simulation)
        expected_gain *= (0.5 + complexity_score * 0.5)
        
        # Cost-weighted utility
        cost = (
            expected_tokens * self._cost_weight_tokens +
            expected_time * self._cost_weight_time
        ) * mission_preferences.cost_sensitivity
        
        marginal_utility = expected_gain - cost
        
        # Apply preference biases
        if mission_preferences.quality_priority > 0.8:
            marginal_utility *= 1.15
        
        if marginal_utility <= self._min_marginal_utility:
            return MarginalUtilityDecision(
                should_proceed=False,
                expected_quality_gain=expected_gain,
                expected_cost_increase=(expected_tokens, expected_time),
                marginal_utility=marginal_utility,
                skip_reason=f"MU={marginal_utility:.3f} <= threshold",
                alternative_action="skip_simulation"
            )
        
        return MarginalUtilityDecision(
            should_proceed=True,
            expected_quality_gain=expected_gain,
            expected_cost_increase=(expected_tokens, expected_time),
            marginal_utility=marginal_utility,
        )
    
    def check_consensus_round(
        self,
        round_number: int,
        disagreement_score: float,
        mission_preferences,
    ) -> MarginalUtilityDecision:
        """
        Check if an extra consensus round should proceed.
        
        Args:
            round_number: Current consensus round number (1-indexed)
            disagreement_score: Disagreement between models (0-1)
            mission_preferences: MissionPreferences instance
            
        Returns:
            MarginalUtilityDecision
        """
        # Extra consensus rounds have diminishing returns
        if round_number > 2:
            # Very conservative for rounds > 2
            return MarginalUtilityDecision(
                should_proceed=False,
                expected_quality_gain=0.02,
                expected_cost_increase=(500, 10.0),
                marginal_utility=-0.1,
                skip_reason=f"Round {round_number} exceeds limit",
                alternative_action="use_current_consensus"
            )
        
        # Expected gain decreases with each round
        base_gain = 0.05 * (1.0 - (round_number - 1) * 0.3)
        # Higher disagreement = more value from consensus
        expected_gain = base_gain * disagreement_score
        
        # Consensus is relatively cheap
        expected_tokens = 500
        expected_time = 10.0
        
        cost = (
            expected_tokens * self._cost_weight_tokens +
            expected_time * self._cost_weight_time
        ) * mission_preferences.cost_sensitivity
        
        marginal_utility = expected_gain - cost
        
        if marginal_utility <= self._min_marginal_utility * 0.5:  # Stricter for consensus
            return MarginalUtilityDecision(
                should_proceed=False,
                expected_quality_gain=expected_gain,
                expected_cost_increase=(expected_tokens, expected_time),
                marginal_utility=marginal_utility,
                skip_reason=f"MU={marginal_utility:.3f} too low for round {round_number}",
                alternative_action="use_current_consensus"
            )
        
        return MarginalUtilityDecision(
            should_proceed=True,
            expected_quality_gain=expected_gain,
            expected_cost_increase=(expected_tokens, expected_time),
            marginal_utility=marginal_utility,
        )

