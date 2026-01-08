"""
Bandit Registry for DeepThinker.

Central registry for all bandit decision classes.
"""

import logging
from typing import Any, Dict, List, Optional

from .config import BanditConfig, get_bandit_config
from .generalized_bandit import GeneralizedBandit

logger = logging.getLogger(__name__)


# Decision class definitions
DECISION_CLASSES = {
    "model_tier": {
        "arms": ["SMALL", "MEDIUM", "LARGE"],
        "description": "Model tier selection for council/scout",
    },
    "council_set": {
        "arms": ["standard", "deep", "fast", "research"],
        "description": "Council set selection",
    },
    "tool_selection": {
        "arms": ["web_search", "code_exec", "rag_query", "file_read", "none"],
        "description": "Tool selection for steps",
    },
    "escalation_depth": {
        "arms": ["1_round", "2_rounds", "3_rounds"],
        "description": "Number of rounds per phase",
    },
}


class BanditRegistry:
    """
    Central registry for all bandit decision classes.
    
    Provides:
    - Lazy initialization of bandits
    - Unified access to all decision classes
    - Statistics aggregation
    
    Usage:
        registry = BanditRegistry()
        
        # Get bandit for a decision class
        bandit = registry.get("model_tier")
        arm = bandit.select()
        bandit.update(arm, reward)
        
        # Get all statistics
        stats = registry.get_all_stats()
    """
    
    def __init__(self, config: Optional[BanditConfig] = None):
        """
        Initialize the registry.
        
        Args:
            config: Optional BanditConfig. Uses global if None.
        """
        self.config = config or get_bandit_config()
        self._bandits: Dict[str, GeneralizedBandit] = {}
    
    def get(self, decision_class: str) -> GeneralizedBandit:
        """
        Get bandit for a decision class.
        
        Lazy-initializes if not already created.
        
        Args:
            decision_class: Name of the decision class
            
        Returns:
            GeneralizedBandit instance
            
        Raises:
            ValueError: If decision class is not registered
        """
        if decision_class not in DECISION_CLASSES:
            raise ValueError(
                f"Unknown decision class: {decision_class}. "
                f"Available: {list(DECISION_CLASSES.keys())}"
            )
        
        if decision_class not in self._bandits:
            arms = DECISION_CLASSES[decision_class]["arms"]
            self._bandits[decision_class] = GeneralizedBandit(
                decision_class=decision_class,
                arms=arms,
                config=self.config,
            )
            logger.info(f"[BANDIT_REGISTRY] Initialized: {decision_class}")
        
        return self._bandits[decision_class]
    
    def get_or_create(
        self,
        decision_class: str,
        arms: List[str],
    ) -> GeneralizedBandit:
        """
        Get or create a bandit for a custom decision class.
        
        Args:
            decision_class: Name of the decision class
            arms: List of arm names
            
        Returns:
            GeneralizedBandit instance
        """
        if decision_class not in self._bandits:
            self._bandits[decision_class] = GeneralizedBandit(
                decision_class=decision_class,
                arms=arms,
                config=self.config,
            )
            logger.info(
                f"[BANDIT_REGISTRY] Created custom bandit: {decision_class} "
                f"with arms: {arms}"
            )
        
        return self._bandits[decision_class]
    
    def list_classes(self) -> List[str]:
        """List all registered decision classes."""
        return list(DECISION_CLASSES.keys())
    
    def list_initialized(self) -> List[str]:
        """List all initialized bandits."""
        return list(self._bandits.keys())
    
    def get_stats(self, decision_class: str) -> Dict[str, Any]:
        """
        Get statistics for a specific bandit.
        
        Args:
            decision_class: Name of the decision class
            
        Returns:
            Statistics dictionary
        """
        if decision_class not in self._bandits:
            return {"initialized": False, "decision_class": decision_class}
        
        return self._bandits[decision_class].get_stats()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all initialized bandits.
        
        Returns:
            Dictionary of decision_class -> stats
        """
        return {
            decision_class: bandit.get_stats()
            for decision_class, bandit in self._bandits.items()
        }
    
    def reset(self, decision_class: str) -> bool:
        """
        Reset a specific bandit.
        
        Args:
            decision_class: Name of the decision class
            
        Returns:
            True if reset succeeded
        """
        if decision_class not in self._bandits:
            return False
        
        self._bandits[decision_class].reset()
        return True
    
    def reset_all(self) -> None:
        """Reset all initialized bandits."""
        for bandit in self._bandits.values():
            bandit.reset()
        logger.info("[BANDIT_REGISTRY] Reset all bandits")
    
    def close(self) -> None:
        """Close registry and clear bandits."""
        self._bandits.clear()


# Global registry instance
_registry: Optional[BanditRegistry] = None


def get_bandit_registry(config: Optional[BanditConfig] = None) -> BanditRegistry:
    """Get global bandit registry instance."""
    global _registry
    if _registry is None:
        _registry = BanditRegistry(config=config)
    return _registry


def reset_bandit_registry() -> None:
    """Reset global registry (mainly for testing)."""
    global _registry
    if _registry is not None:
        _registry.close()
    _registry = None


# Convenience functions for common decision classes
def get_model_tier_bandit() -> GeneralizedBandit:
    """Get the model tier bandit."""
    return get_bandit_registry().get("model_tier")


def get_council_set_bandit() -> GeneralizedBandit:
    """Get the council set bandit."""
    return get_bandit_registry().get("council_set")


def get_tool_selection_bandit() -> GeneralizedBandit:
    """Get the tool selection bandit."""
    return get_bandit_registry().get("tool_selection")


def get_escalation_depth_bandit() -> GeneralizedBandit:
    """Get the escalation depth bandit."""
    return get_bandit_registry().get("escalation_depth")

