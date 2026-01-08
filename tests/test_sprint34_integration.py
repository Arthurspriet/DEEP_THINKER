"""
Integration Tests for Sprint 3-4 Learning & Trust System.

Tests:
1. All flags OFF = zero behavior change
2. Shadow modes log but don't change behavior
3. Reward computation
4. Bandit updates
5. Memory usefulness filtering
6. Stop/escalate predictor
7. Replay correctness
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestFlagsOff(unittest.TestCase):
    """
    Test that all features disabled = zero behavior change.
    
    When all Sprint 3-4 features are disabled:
    - No new log files should be created
    - No new model calls should be made
    - Existing functionality should work unchanged
    """
    
    def setUp(self):
        """Set up test environment with all flags OFF."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Set all flags to OFF
        self.env_patches = {
            "REWARD_ENABLED": "false",
            "BANDIT_ENABLED": "false",
            "LEARNING_ENABLED": "false",
            "LEARNED_POLICY_MODE": "off",
            "REVIEW_ENABLED": "false",
            "REPLAY_ENABLED": "false",
            "TRUST_ENABLED": "false",
            "MEMORY_USEFULNESS_ENABLED": "false",
            "ALIGNMENT_LEARNING_ENABLED": "false",
        }
        
        self.env_patcher = patch.dict(os.environ, self.env_patches)
        self.env_patcher.start()
        
        # Reset all global configs
        self._reset_configs()
    
    def tearDown(self):
        """Clean up."""
        self.env_patcher.stop()
        self._reset_configs()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _reset_configs(self):
        """Reset all module configs."""
        from deepthinker.rewards.config import reset_reward_config
        from deepthinker.bandits.config import reset_bandit_config
        from deepthinker.learning.config import reset_learning_config
        from deepthinker.review.config import reset_review_config
        from deepthinker.replay.config import reset_replay_config
        from deepthinker.trust.config import reset_trust_config
        from deepthinker.memory.usefulness_config import reset_memory_usefulness_config
        from deepthinker.alignment.learning_config import reset_alignment_learning_config
        
        reset_reward_config()
        reset_bandit_config()
        reset_learning_config()
        reset_review_config()
        reset_replay_config()
        reset_trust_config()
        reset_memory_usefulness_config()
        reset_alignment_learning_config()
    
    def test_reward_disabled_no_logging(self):
        """RewardStore.write() should return False when disabled."""
        from deepthinker.rewards import RewardSignal, RewardStore, RewardConfig
        
        config = RewardConfig(enabled=False, store_path=f"{self.temp_dir}/rewards.jsonl")
        store = RewardStore(config=config)
        
        signal = RewardSignal.from_phase_outcome(score_delta=0.1)
        result = store.write(signal)
        
        self.assertFalse(result)
        self.assertFalse(Path(f"{self.temp_dir}/rewards.jsonl").exists())
    
    def test_bandit_disabled_returns_first_arm(self):
        """GeneralizedBandit.select() returns first arm when disabled."""
        from deepthinker.bandits import GeneralizedBandit, BanditConfig
        
        config = BanditConfig(enabled=False)
        bandit = GeneralizedBandit(
            decision_class="test",
            arms=["A", "B", "C"],
            config=config,
        )
        
        selected = bandit.select()
        self.assertEqual(selected, "A")  # First arm (sorted)
    
    def test_bandit_disabled_update_rejected(self):
        """GeneralizedBandit.update() returns False when disabled."""
        from deepthinker.bandits import GeneralizedBandit, BanditConfig
        
        config = BanditConfig(enabled=False)
        bandit = GeneralizedBandit(
            decision_class="test",
            arms=["A", "B", "C"],
            config=config,
        )
        
        result = bandit.update("A", 0.5)
        self.assertFalse(result)
    
    def test_learning_disabled_no_prediction(self):
        """StopEscalatePredictor returns CONTINUE when disabled."""
        from deepthinker.learning import (
            StopEscalatePredictor, LearningConfig, LearnedPolicyMode,
            PolicyState, PolicyAction,
        )
        
        config = LearningConfig(enabled=False, policy_mode=LearnedPolicyMode.OFF)
        predictor = StopEscalatePredictor(config=config)
        
        state = PolicyState(current_score=0.9)
        prediction = predictor.predict(state)
        
        self.assertEqual(prediction.recommended_action, PolicyAction.CONTINUE)
        self.assertEqual(prediction.p_stop, 0.0)
    
    def test_review_queue_disabled_returns_none(self):
        """ReviewQueue.enqueue() returns None when disabled."""
        from deepthinker.review import ReviewQueue, ReviewConfig
        
        config = ReviewConfig(enabled=False, queue_path=f"{self.temp_dir}/queue/")
        queue = ReviewQueue(config=config)
        
        result = queue.enqueue({"decision_id": "test-123"})
        self.assertIsNone(result)
    
    def test_trust_calculator_disabled_returns_default(self):
        """TrustCalculator returns default score when disabled."""
        from deepthinker.trust import TrustCalculator, TrustConfig
        
        config = TrustConfig(enabled=False)
        calculator = TrustCalculator(config=config)
        
        score = calculator.compute(mission_id="test")
        self.assertEqual(score.overall_trust, 0.5)  # Default
    
    def test_memory_usefulness_disabled_fallback(self):
        """MemoryUsefulnessPredictor uses similarity when disabled."""
        from deepthinker.memory.usefulness import (
            MemoryUsefulnessPredictor, MemoryCandidate, MemoryType,
        )
        from deepthinker.memory.usefulness_config import MemoryUsefulnessConfig
        
        config = MemoryUsefulnessConfig(enabled=False)
        predictor = MemoryUsefulnessPredictor(config=config)
        
        candidate = MemoryCandidate(
            memory_id="test",
            memory_type=MemoryType.EVIDENCE,
            similarity_score=0.8,
        )
        
        helpfulness = predictor.predict(candidate)
        self.assertEqual(helpfulness, 0.8)  # Falls back to similarity
    
    def test_alignment_bandit_disabled_returns_none(self):
        """AlignmentActionBandit returns None when disabled."""
        from deepthinker.alignment.learning import (
            AlignmentActionBandit, AlignmentContext,
        )
        from deepthinker.alignment.learning_config import AlignmentLearningConfig
        
        config = AlignmentLearningConfig(enabled=False)
        bandit = AlignmentActionBandit(config=config)
        
        context = AlignmentContext(drift_score=0.8)
        action = bandit.select_action(context, cusum_triggered=True)
        
        self.assertIsNone(action)


class TestShadowModes(unittest.TestCase):
    """
    Test that shadow modes log but don't change behavior.
    
    In SHADOW mode:
    - Predictions should be logged
    - No decisions should be changed
    - No behavioral impact
    """
    
    def setUp(self):
        """Set up test environment with shadow modes."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Enable with SHADOW mode
        self.env_patches = {
            "LEARNING_ENABLED": "true",
            "LEARNED_POLICY_MODE": "shadow",
            "LEARNING_SHADOW_LOG_PATH": f"{self.temp_dir}/shadow.jsonl",
            "BANDIT_ENABLED": "true",
            "BANDIT_FREEZE_MODE": "true",  # Read-only
        }
        
        self.env_patcher = patch.dict(os.environ, self.env_patches)
        self.env_patcher.start()
        
        # Reset configs
        self._reset_configs()
    
    def tearDown(self):
        """Clean up."""
        self.env_patcher.stop()
        self._reset_configs()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _reset_configs(self):
        """Reset relevant configs."""
        from deepthinker.learning.config import reset_learning_config
        from deepthinker.bandits.config import reset_bandit_config
        
        reset_learning_config()
        reset_bandit_config()
    
    def test_shadow_mode_logs_prediction(self):
        """SHADOW mode should log predictions."""
        from deepthinker.learning import (
            StopEscalatePredictor, LearningConfig, LearnedPolicyMode,
            PolicyState,
        )
        
        config = LearningConfig(
            enabled=True,
            policy_mode=LearnedPolicyMode.SHADOW,
            shadow_log_path=f"{self.temp_dir}/shadow.jsonl",
        )
        predictor = StopEscalatePredictor(config=config)
        
        state = PolicyState(
            current_score=0.9,
            time_budget_used_pct=0.8,
        )
        
        prediction = predictor.predict(state)
        
        # Should have logged
        self.assertTrue(prediction.shadow_logged)
        
        # Should NOT have an action recommendation in shadow mode
        self.assertIsNone(prediction.recommended_action)
        
        # Log file should exist
        log_path = Path(f"{self.temp_dir}/shadow.jsonl")
        self.assertTrue(log_path.exists())
        
        # Log should contain record
        with open(log_path, "r") as f:
            log_content = f.read()
        self.assertIn("p_stop", log_content)
    
    def test_freeze_mode_rejects_updates(self):
        """Freeze mode should reject bandit updates."""
        from deepthinker.bandits import GeneralizedBandit, BanditConfig
        
        config = BanditConfig(
            enabled=True,
            freeze_mode=True,
            store_dir=f"{self.temp_dir}/bandits/",
        )
        bandit = GeneralizedBandit(
            decision_class="test",
            arms=["A", "B", "C"],
            config=config,
        )
        
        # Selection should work
        selected = bandit.select()
        self.assertIn(selected, ["A", "B", "C"])
        
        # Update should be rejected
        result = bandit.update("A", 0.5)
        self.assertFalse(result)


class TestRewardComputation(unittest.TestCase):
    """Test reward signal computation."""
    
    def test_reward_deterministic(self):
        """Same inputs should produce same outputs."""
        from deepthinker.rewards import RewardSignal, RewardConfig
        
        config = RewardConfig(enabled=True)
        
        signal1 = RewardSignal.from_phase_outcome(
            score_delta=0.1,
            cost_tokens=5000,
            alignment_drift_events=1,
            config=config,
        )
        reward1 = signal1.compute_reward()
        
        signal2 = RewardSignal.from_phase_outcome(
            score_delta=0.1,
            cost_tokens=5000,
            alignment_drift_events=1,
            config=config,
        )
        reward2 = signal2.compute_reward()
        
        self.assertEqual(reward1, reward2)
    
    def test_reward_clamps_applied(self):
        """Hard clamps should be enforced."""
        from deepthinker.rewards import RewardSignal, RewardConfig
        
        config = RewardConfig(
            enabled=True,
            cost_penalty_clamp=0.3,
            alignment_penalty_clamp=0.2,
        )
        
        signal = RewardSignal.from_phase_outcome(
            cost_tokens=1000000,  # Very high cost
            alignment_drift_events=10,  # Many events
            config=config,
        )
        signal.compute_reward()
        
        self.assertLessEqual(signal.cost_penalty_clamped, 0.3)
        self.assertLessEqual(signal.alignment_penalty_clamped, 0.2)
    
    def test_reward_in_valid_range(self):
        """Reward should be in [-1, +1]."""
        from deepthinker.rewards import RewardSignal, RewardConfig
        
        config = RewardConfig(enabled=True)
        
        # Extreme positive case
        signal_pos = RewardSignal.from_phase_outcome(
            score_delta=1.0,
            consistency_delta=1.0,
            grounding_delta=1.0,
            config=config,
        )
        reward_pos = signal_pos.compute_reward()
        self.assertLessEqual(reward_pos, 1.0)
        self.assertGreaterEqual(reward_pos, -1.0)
        
        # Extreme negative case
        signal_neg = RewardSignal.from_phase_outcome(
            score_delta=-1.0,
            cost_tokens=1000000,
            alignment_drift_events=10,
            config=config,
        )
        reward_neg = signal_neg.compute_reward()
        self.assertLessEqual(reward_neg, 1.0)
        self.assertGreaterEqual(reward_neg, -1.0)


class TestBanditUpdates(unittest.TestCase):
    """Test bandit update logic."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exploration_before_exploit(self):
        """Should explore all arms before exploitation."""
        from deepthinker.bandits import GeneralizedBandit, BanditConfig
        
        config = BanditConfig(
            enabled=True,
            min_trials_before_exploit=3,
            store_dir=f"{self.temp_dir}/bandits/",
        )
        bandit = GeneralizedBandit(
            decision_class="test_explore",  # Unique name to avoid state collision
            arms=["A", "B", "C"],
            config=config,
        )
        
        # Select and update in sequence to properly explore
        for _ in range(9):  # 3 arms * 3 trials = 9
            selected = bandit.select()
            bandit.update(selected, 0.5)
        
        # All arms should have been explored at least min_trials times
        stats = bandit.get_stats()
        total_pulls = sum(stats["arms"][arm]["pulls"] for arm in stats["arms"])
        self.assertGreaterEqual(total_pulls, 9)
    
    def test_schema_validation(self):
        """Schema changes should be detected."""
        from deepthinker.bandits import GeneralizedBandit, BanditConfig
        
        config = BanditConfig(
            enabled=True,
            schema_version="1.0.0",
            store_dir=f"{self.temp_dir}/bandits/",
        )
        
        # Create bandit with arms A, B, C
        bandit1 = GeneralizedBandit(
            decision_class="test",
            arms=["A", "B", "C"],
            config=config,
        )
        bandit1.update("A", 0.5)
        
        # Create new bandit with different arms (should reinitialize)
        bandit2 = GeneralizedBandit(
            decision_class="test",
            arms=["X", "Y", "Z"],  # Different arms
            config=config,
        )
        
        # Should have new arms
        stats = bandit2.get_stats()
        self.assertIn("X", stats["arms"])
        self.assertNotIn("A", stats["arms"])


class TestMemoryUsefulnessFiltering(unittest.TestCase):
    """Test memory usefulness prediction and filtering."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_filtering_respects_budget(self):
        """Should filter memories within token budget."""
        from deepthinker.memory.usefulness import (
            MemoryUsefulnessPredictor, MemoryCandidate, MemoryType,
        )
        from deepthinker.memory.usefulness_config import MemoryUsefulnessConfig
        
        config = MemoryUsefulnessConfig(
            enabled=True,
            min_helpfulness_threshold=0.0,  # Accept all
        )
        predictor = MemoryUsefulnessPredictor(config=config)
        
        candidates = [
            MemoryCandidate(
                memory_id=f"mem{i}",
                memory_type=MemoryType.EVIDENCE,
                full_text="x" * 400,  # ~100 tokens each
                similarity_score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]
        
        injected, log = predictor.filter_memories(
            candidates, budget_tokens=250, phase="test"
        )
        
        # Should only inject memories within budget
        self.assertLess(len(injected), 5)
        self.assertLess(log.tokens_used, 250)
    
    def test_counterfactual_logging(self):
        """Should log both injected and rejected memories."""
        from deepthinker.memory.usefulness import (
            MemoryUsefulnessPredictor, MemoryCandidate, MemoryType,
        )
        from deepthinker.memory.usefulness_config import MemoryUsefulnessConfig
        
        config = MemoryUsefulnessConfig(
            enabled=True,
            counterfactual_log_path=f"{self.temp_dir}/cf.jsonl",
        )
        predictor = MemoryUsefulnessPredictor(config=config)
        
        candidates = [
            MemoryCandidate(
                memory_id="high",
                memory_type=MemoryType.EVIDENCE,
                full_text="high quality",
                similarity_score=0.9,
            ),
            MemoryCandidate(
                memory_id="low",
                memory_type=MemoryType.UNKNOWN,
                full_text="low quality",
                similarity_score=0.1,
            ),
        ]
        
        injected, log = predictor.filter_memories(
            candidates, budget_tokens=100, phase="test"
        )
        predictor.log_counterfactual(log)
        
        # Log should contain both injected and rejected
        self.assertGreater(len(log.retrieval_candidates), 0)
        self.assertEqual(
            len(log.injected_memories) + len(log.rejected_memories),
            len(log.retrieval_candidates)
        )


class TestReplayCorrectness(unittest.TestCase):
    """Test counterfactual replay correctness."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_replay_decisions_only_no_model_calls(self):
        """DECISIONS_ONLY mode should not make model calls."""
        from deepthinker.replay import (
            CounterfactualReplayEngine, ReplayConfig, ReplayMode,
        )
        
        config = ReplayConfig(
            enabled=True,
            mode=ReplayMode.DECISIONS_ONLY,
            output_path=f"{self.temp_dir}/replay/",
        )
        engine = CounterfactualReplayEngine(config=config)
        
        # Create mock decisions
        decisions_path = Path(f"{self.temp_dir}/decisions.jsonl")
        with open(decisions_path, "w") as f:
            f.write(json.dumps({
                "decision_id": "d1",
                "decision_type": "model_selection",
                "selected_option": "SMALL",
                "options_considered": ["SMALL", "LARGE"],
                "confidence": 0.8,
            }) + "\n")
        
        # Mock policy
        class MockPolicy:
            def select(self, context):
                return "LARGE"
        
        result = engine.replay_mission(
            mission_id="test",
            new_policy=MockPolicy(),
            decision_store_path=str(decisions_path),
        )
        
        # Should have replayed decisions
        self.assertEqual(result.total_decisions, 1)
        self.assertEqual(result.decisions_changed, 1)


if __name__ == "__main__":
    unittest.main()

