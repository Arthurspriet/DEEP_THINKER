"""
Reward Store for DeepThinker.

JSONL-based append-only persistence for reward history.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import RewardConfig, get_reward_config
from .reward_signal import RewardSignal

logger = logging.getLogger(__name__)


class RewardStore:
    """
    Persistent storage for reward signals.
    
    Storage structure:
        kb/rewards/reward_history.jsonl - All reward records (append-only)
    
    Key properties:
    - Append-only: Never modifies existing records
    - Human-readable: Plain JSONL format
    - Versioned: Each record includes reward_version
    
    Usage:
        store = RewardStore()
        store.write(reward_signal)
        
        # Query rewards
        rewards = store.read_by_mission("mission-123")
        rewards = store.read_by_decision_type("model_selection")
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize the reward store.
        
        Args:
            config: RewardConfig (uses global if None)
        """
        self.config = config or get_reward_config()
        self.store_path = Path(self.config.store_path)
    
    def _ensure_dir(self) -> bool:
        """Ensure store directory exists."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to create store dir: {e}")
            return False
    
    def write(self, signal: RewardSignal) -> bool:
        """
        Write a reward signal to the store.
        
        Appends to the JSONL file.
        
        Args:
            signal: RewardSignal to persist
            
        Returns:
            True if write succeeded
        """
        if not self.config.enabled:
            return False
        
        try:
            if not self._ensure_dir():
                return False
            
            record = signal.to_dict()
            json_line = json.dumps(record, ensure_ascii=False)
            
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            
            logger.debug(
                f"[REWARDS] Wrote reward={signal.reward:.3f} "
                f"for {signal.decision_type}/{signal.phase_id}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to write reward: {e}")
            return False
    
    def read_all(self) -> List[RewardSignal]:
        """
        Read all reward records.
        
        Returns:
            List of RewardSignal objects, sorted by timestamp
        """
        if not self.store_path.exists():
            return []
        
        records = []
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        signal = RewardSignal.from_dict(data)
                        records.append(signal)
                    except Exception as e:
                        logger.warning(
                            f"[REWARDS] Skipping malformed line {line_num}: {e}"
                        )
            
            # Sort by timestamp
            records.sort(key=lambda r: r.timestamp)
            return records
            
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to read rewards: {e}")
            return []
    
    def iter_records(self) -> Iterator[RewardSignal]:
        """
        Iterate over reward records without loading all into memory.
        
        Yields:
            RewardSignal objects in file order
        """
        if not self.store_path.exists():
            return
        
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield RewardSignal.from_dict(data)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to iterate rewards: {e}")
    
    def read_by_mission(self, mission_id: str) -> List[RewardSignal]:
        """
        Read rewards for a specific mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            List of matching RewardSignal objects
        """
        return [r for r in self.read_all() if r.mission_id == mission_id]
    
    def read_by_decision_type(self, decision_type: str) -> List[RewardSignal]:
        """
        Read rewards for a specific decision type.
        
        Args:
            decision_type: Decision type to filter by
            
        Returns:
            List of matching RewardSignal objects
        """
        return [r for r in self.read_all() if r.decision_type == decision_type]
    
    def read_by_phase(self, mission_id: str, phase_id: str) -> List[RewardSignal]:
        """
        Read rewards for a specific phase.
        
        Args:
            mission_id: Mission identifier
            phase_id: Phase identifier
            
        Returns:
            List of matching RewardSignal objects
        """
        return [
            r for r in self.read_all()
            if r.mission_id == mission_id and r.phase_id == phase_id
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over all rewards.
        
        Returns:
            Statistics dictionary
        """
        rewards = self.read_all()
        
        if not rewards:
            return {
                "total_records": 0,
                "avg_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }
        
        reward_values = [r.reward for r in rewards]
        
        # Count by decision type
        type_counts: Dict[str, int] = {}
        type_rewards: Dict[str, List[float]] = {}
        for r in rewards:
            dt = r.decision_type or "unknown"
            type_counts[dt] = type_counts.get(dt, 0) + 1
            if dt not in type_rewards:
                type_rewards[dt] = []
            type_rewards[dt].append(r.reward)
        
        return {
            "total_records": len(rewards),
            "avg_reward": sum(reward_values) / len(reward_values),
            "min_reward": min(reward_values),
            "max_reward": max(reward_values),
            "by_decision_type": {
                dt: {
                    "count": type_counts[dt],
                    "avg_reward": sum(type_rewards[dt]) / len(type_rewards[dt]),
                }
                for dt in type_counts
            },
            "first_record_at": rewards[0].timestamp.isoformat(),
            "last_record_at": rewards[-1].timestamp.isoformat(),
        }
    
    def get_recent(self, limit: int = 100) -> List[RewardSignal]:
        """
        Get most recent rewards.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of RewardSignal objects (most recent first)
        """
        rewards = self.read_all()
        rewards.reverse()  # Most recent first
        return rewards[:limit]
    
    def clear(self) -> bool:
        """
        Clear all rewards (for testing only).
        
        Returns:
            True if cleared successfully
        """
        try:
            if self.store_path.exists():
                self.store_path.unlink()
            return True
        except Exception as e:
            logger.warning(f"[REWARDS] Failed to clear store: {e}")
            return False


# Global store instance
_store: Optional[RewardStore] = None


def get_reward_store(config: Optional[RewardConfig] = None) -> RewardStore:
    """Get global reward store instance."""
    global _store
    if _store is None:
        _store = RewardStore(config=config)
    return _store


def reset_reward_store() -> None:
    """Reset global store (mainly for testing)."""
    global _store
    _store = None


