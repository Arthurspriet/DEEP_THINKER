"""
Tests for Bandit module.

Tests the UCB bandit for model tier selection,
state persistence, and reward computation.
"""

import pytest
import json
import tempfile
import os


class TestBanditArm:
    """Test BanditArm class."""
    
    def test_import(self):
        """Test that bandit module can be imported."""
        from deepthinker.routing import BanditArm, BanditState, ModelTierBandit
        assert BanditArm is not None
        assert BanditState is not None
        assert ModelTierBandit is not None
    
    def test_creation(self):
        """Test arm creation."""
        from deepthinker.routing import BanditArm
        
        arm = BanditArm(name="MEDIUM")
        
        assert arm.name == "MEDIUM"
        assert arm.pulls == 0
        assert arm.total_reward == 0.0
    
    def test_mean_reward(self):
        """Test mean reward computation."""
        from deepthinker.routing import BanditArm
        
        arm = BanditArm(name="MEDIUM", pulls=4, total_reward=2.0)
        
        assert arm.mean_reward == 0.5
    
    def test_mean_reward_zero_pulls(self):
        """Test mean reward with zero pulls."""
        from deepthinker.routing import BanditArm
        
        arm = BanditArm(name="MEDIUM", pulls=0, total_reward=0.0)
        
        assert arm.mean_reward == 0.0
    
    def test_ucb_score_unexplored(self):
        """Test UCB score for unexplored arm is infinity."""
        from deepthinker.routing import BanditArm
        import math
        
        arm = BanditArm(name="MEDIUM", pulls=0)
        
        score = arm.ucb_score(total_pulls=10, exploration_bonus=1.0)
        
        assert score == float('inf')
    
    def test_ucb_score_explored(self):
        """Test UCB score for explored arm."""
        from deepthinker.routing import BanditArm
        
        arm = BanditArm(name="MEDIUM", pulls=5, total_reward=2.5)
        
        score = arm.ucb_score(total_pulls=10, exploration_bonus=1.0)
        
        # Should be mean + exploration bonus
        assert score > arm.mean_reward
    
    def test_update(self):
        """Test updating arm with reward."""
        from deepthinker.routing import BanditArm
        
        arm = BanditArm(name="MEDIUM")
        
        arm.update(reward=0.5)
        
        assert arm.pulls == 1
        assert arm.total_reward == 0.5
        assert arm.mean_reward == 0.5
        
        arm.update(reward=1.0)
        
        assert arm.pulls == 2
        assert arm.total_reward == 1.5
        assert arm.mean_reward == 0.75
    
    def test_to_dict_from_dict(self):
        """Test arm serialization round-trip."""
        from deepthinker.routing import BanditArm
        
        arm = BanditArm(name="LARGE", pulls=10, total_reward=5.0)
        
        data = arm.to_dict()
        restored = BanditArm.from_dict(data)
        
        assert restored.name == arm.name
        assert restored.pulls == arm.pulls
        assert restored.total_reward == arm.total_reward


class TestBanditState:
    """Test BanditState class."""
    
    def test_creation(self):
        """Test state creation."""
        from deepthinker.routing import BanditState, BanditArm
        
        state = BanditState(
            arms={
                "SMALL": BanditArm(name="SMALL"),
                "MEDIUM": BanditArm(name="MEDIUM"),
                "LARGE": BanditArm(name="LARGE"),
            }
        )
        
        assert len(state.arms) == 3
        assert state.total_pulls == 0
    
    def test_to_dict_from_dict(self):
        """Test state serialization round-trip."""
        from deepthinker.routing import BanditState, BanditArm
        
        state = BanditState(
            arms={
                "SMALL": BanditArm(name="SMALL", pulls=5, total_reward=2.0),
                "MEDIUM": BanditArm(name="MEDIUM", pulls=10, total_reward=7.0),
            },
            total_pulls=15,
        )
        
        data = state.to_dict()
        restored = BanditState.from_dict(data)
        
        assert len(restored.arms) == 2
        assert restored.total_pulls == 15
        assert restored.arms["MEDIUM"].pulls == 10


class TestModelTierBandit:
    """Test ModelTierBandit class."""
    
    def test_creation(self):
        """Test bandit creation."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            assert bandit is not None
            assert bandit._state is not None
            assert len(bandit._state.arms) == 3
    
    def test_select_disabled(self):
        """Test selection when disabled returns default."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            tier = bandit.select()
            
            assert tier == "MEDIUM"
    
    def test_select_exploration(self):
        """Test exploration phase selects all arms."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(
            bandit_enabled=True,
            bandit_min_observations=2,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            # Select multiple times - should explore all arms
            selected = set()
            for _ in range(10):
                tier = bandit.select()
                selected.add(tier)
                # Simulate update
                bandit._state.arms[tier].pulls += 1
                bandit._state.total_pulls += 1
            
            # Should have explored all arms
            assert len(selected) == 3
    
    def test_update(self):
        """Test updating bandit with outcome."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(
            bandit_enabled=True,
            bandit_lambda=0.1,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            # Update MEDIUM arm
            reward = bandit.update(
                tier="MEDIUM",
                score_delta=0.2,
                cost_delta=0.1,
            )
            
            # Reward = score_delta - lambda * cost_delta = 0.2 - 0.1 * 0.1 = 0.19
            assert reward == pytest.approx(0.19)
            assert bandit._state.arms["MEDIUM"].pulls == 1
            assert bandit._state.total_pulls == 1
    
    def test_state_persistence(self):
        """Test that state is persisted to file."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandits", "bandit.json")
            
            # Create and update bandit
            bandit1 = ModelTierBandit(config=config, state_path=state_path)
            bandit1.update("MEDIUM", 0.5, 0.1)
            
            # Create new bandit from same path
            bandit2 = ModelTierBandit(config=config, state_path=state_path)
            
            # Should have loaded state
            assert bandit2._state.arms["MEDIUM"].pulls == 1
            assert bandit2._state.total_pulls == 1
    
    def test_get_stats(self):
        """Test getting bandit statistics."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            bandit.update("MEDIUM", 0.5, 0.1)
            bandit.update("LARGE", 0.7, 0.2)
            
            stats = bandit.get_stats()
            
            assert stats["initialized"] == True
            assert stats["total_pulls"] == 2
            assert "MEDIUM" in stats["arms"]
            assert "LARGE" in stats["arms"]
            assert stats["best_arm"] is not None
    
    def test_reset(self):
        """Test resetting bandit state."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            bandit.update("MEDIUM", 0.5, 0.1)
            bandit.reset()
            
            assert bandit._state.total_pulls == 0
            assert all(arm.pulls == 0 for arm in bandit._state.arms.values())

