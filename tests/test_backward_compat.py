"""
Backward Compatibility Tests for Sprint 1-2 Features.

CRITICAL: Ensures that:
- With all feature flags OFF, no new artifacts or decision types are produced
- With feature flags ON, scorecards and routing decisions are produced

This test validates that the new metrics/routing/policy system doesn't
break existing functionality when disabled.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock


class TestBackwardCompatibilityFlagsOff:
    """
    Test that with all feature flags OFF, the system behaves exactly
    as it did before Sprint 1-2.
    """
    
    def test_metrics_config_defaults_off(self):
        """Test that all feature flags default to OFF."""
        from deepthinker.metrics import MetricsConfig, reset_metrics_config
        
        reset_metrics_config()
        
        config = MetricsConfig()
        
        assert config.scorecard_enabled == False
        assert config.scorecard_policy_enabled == False
        assert config.learning_router_enabled == False
        assert config.bandit_enabled == False
        assert config.claim_graph_enabled == False
        
        assert config.is_any_enabled() == False
    
    def test_metrics_hook_noop_when_disabled(self):
        """Test that metrics hook is a no-op when disabled."""
        from deepthinker.metrics import (
            MetricsOrchestrationHook,
            MetricsConfig,
            PhaseMetricsContext,
        )
        
        config = MetricsConfig(
            scorecard_enabled=False,
            scorecard_policy_enabled=False,
            learning_router_enabled=False,
            bandit_enabled=False,
        )
        
        hook = MetricsOrchestrationHook(config=config)
        
        # Mock state and phase
        mock_state = Mock()
        mock_state.mission_id = "test_mission"
        mock_state.objective = "Test objective"
        mock_state.remaining_minutes.return_value = 30.0
        mock_state.phases = []
        
        mock_phase = Mock()
        mock_phase.name = "test_phase"
        mock_phase.artifacts = {}
        
        # on_phase_start should return a context but not score
        ctx = hook.on_phase_start(mock_state, mock_phase)
        
        # Should have created context
        assert ctx is not None
        assert ctx.mission_id == "test_mission"
        
        # score_before should be None (scoring disabled)
        assert ctx.score_before is None
    
    def test_policy_returns_continue_when_disabled(self):
        """Test that policy returns CONTINUE when disabled."""
        from deepthinker.policy import ScorecardPolicy, PolicyAction
        from deepthinker.metrics import MetricsConfig, Scorecard
        
        config = MetricsConfig(scorecard_policy_enabled=False)
        policy = ScorecardPolicy(config=config)
        
        # Even with low scores, should return CONTINUE when disabled
        scorecard = Scorecard.from_scores(0.1, 0.1, 0.1, 0.1)
        
        decision = policy.decide(scorecard)
        
        assert decision.action == PolicyAction.CONTINUE
        assert decision.rationale == "Policy disabled"
    
    def test_router_returns_low_confidence_when_disabled(self):
        """Test that router returns low confidence when disabled."""
        from deepthinker.routing import MLRouterAdvisor, RoutingContext
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=False)
        router = MLRouterAdvisor(config=config)
        
        ctx = RoutingContext(objective="Test")
        decision = router.advise(ctx)
        
        assert decision.confidence == 0.0
        assert decision.rationale == "Router disabled"
    
    def test_bandit_returns_default_when_disabled(self):
        """Test that bandit returns default tier when disabled."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            tier = bandit.select()
            
            assert tier == "MEDIUM"  # Default
    
    def test_bandit_update_noop_when_disabled(self):
        """Test that bandit update is noop when disabled."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(bandit_enabled=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            result = bandit.update("LARGE", 0.5, 0.1)
            
            assert result == 0.0
            # State should not have been updated
            assert not os.path.exists(state_path)
    
    def test_no_new_decision_types_emitted_when_disabled(self):
        """Test that no new decision types are emitted when flags are OFF."""
        from deepthinker.metrics import (
            MetricsOrchestrationHook,
            MetricsConfig,
        )
        
        config = MetricsConfig(
            scorecard_enabled=False,
            scorecard_policy_enabled=False,
        )
        
        hook = MetricsOrchestrationHook(config=config)
        
        # Mock emitter
        mock_emitter = Mock()
        
        # Mock state and phase
        mock_state = Mock()
        mock_state.mission_id = "test_mission"
        mock_state.objective = "Test objective"
        mock_state.remaining_minutes.return_value = 30.0
        mock_state.phases = []
        
        mock_phase = Mock()
        mock_phase.name = "test_phase"
        mock_phase.artifacts = {}
        
        # Run phase start/end
        ctx = hook.on_phase_start(mock_state, mock_phase)
        scorecard, decision = hook.on_phase_end(ctx, mock_phase, mock_emitter)
        
        # Should NOT have called emitter (features disabled)
        mock_emitter.emit.assert_not_called()
        
        # Should return None for both
        assert scorecard is None
        assert decision is None


class TestBackwardCompatibilityFlagsOn:
    """
    Test that with feature flags ON, the new functionality works correctly.
    """
    
    def test_scorecard_produced_when_enabled(self):
        """Test that scorecards are produced when enabled."""
        from deepthinker.metrics import (
            MetricsConfig,
            Scorecard,
            JudgeEnsemble,
            JudgeScores,
        )
        
        config = MetricsConfig(
            scorecard_enabled=True,
            judge_sample_rate=1.0,
        )
        
        # Test that we can create scorecards
        scorecard = Scorecard.from_scores(0.8, 0.7, 0.6, 0.9)
        
        assert scorecard is not None
        assert scorecard.overall > 0
        assert scorecard.goal_coverage == 0.8
    
    def test_policy_makes_decisions_when_enabled(self):
        """Test that policy makes stop/escalate decisions when enabled."""
        from deepthinker.policy import ScorecardPolicy, PolicyAction
        from deepthinker.metrics import MetricsConfig, Scorecard
        
        config = MetricsConfig(
            scorecard_policy_enabled=True,
            stop_overall_threshold=0.8,
            stop_consistency_threshold=0.7,
        )
        
        policy = ScorecardPolicy(config=config)
        
        # High score should trigger STOP
        high_scorecard = Scorecard.from_scores(0.9, 0.9, 0.9, 0.9)
        decision = policy.decide(high_scorecard)
        
        assert decision.action == PolicyAction.STOP
        assert "stop_threshold_met" in decision.triggered_rules
    
    def test_policy_escalates_on_low_scores(self):
        """Test that policy escalates on low scores when enabled."""
        from deepthinker.policy import ScorecardPolicy, PolicyAction
        from deepthinker.metrics import MetricsConfig, Scorecard
        
        config = MetricsConfig(
            scorecard_policy_enabled=True,
            escalate_goal_coverage_threshold=0.4,
        )
        
        policy = ScorecardPolicy(config=config)
        
        # Low goal coverage should trigger ESCALATE
        low_scorecard = Scorecard.from_scores(0.2, 0.8, 0.8, 0.8)
        decision = policy.decide(low_scorecard)
        
        assert decision.action == PolicyAction.ESCALATE
        assert "low_quality_escalation" in decision.triggered_rules
    
    def test_router_provides_advice_when_enabled(self):
        """Test that router provides advice when enabled."""
        from deepthinker.routing import MLRouterAdvisor, RoutingContext
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "weights.json")
            router = MLRouterAdvisor(config=config, weights_path=weights_path)
            
            ctx = RoutingContext(
                objective="Research important topic",
                phase_name="research",
                time_remaining_minutes=30.0,
                time_budget_minutes=60.0,
            )
            
            decision = router.advise(ctx)
            
            assert decision.confidence > 0  # Should have some confidence
            assert decision.model_tier in ["SMALL", "MEDIUM", "LARGE"]
            assert decision.num_rounds in [1, 2, 3]
    
    def test_bandit_learns_when_enabled(self):
        """Test that bandit learns from outcomes when enabled."""
        from deepthinker.routing import ModelTierBandit
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(
            bandit_enabled=True,
            bandit_lambda=0.1,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bandit.json")
            bandit = ModelTierBandit(config=config, state_path=state_path)
            
            # Update with positive outcome for LARGE
            bandit.update("LARGE", score_delta=0.5, cost_delta=0.1)
            bandit.update("LARGE", score_delta=0.6, cost_delta=0.1)
            
            # Update with negative outcome for SMALL
            bandit.update("SMALL", score_delta=0.1, cost_delta=0.1)
            
            stats = bandit.get_stats()
            
            # LARGE should have higher mean reward
            assert stats["arms"]["LARGE"]["mean_reward"] > stats["arms"]["SMALL"]["mean_reward"]
    
    def test_new_decision_types_exist(self):
        """Test that new decision types are defined."""
        from deepthinker.decisions.decision_record import DecisionType
        
        # Check new decision types exist
        assert hasattr(DecisionType, "SCORECARD_STOP")
        assert hasattr(DecisionType, "SCORECARD_ESCALATE")
        assert hasattr(DecisionType, "ROUTING_DECISION")
        assert hasattr(DecisionType, "TOOL_USAGE")
        
        # Check values
        assert DecisionType.SCORECARD_STOP.value == "scorecard_stop"
        assert DecisionType.SCORECARD_ESCALATE.value == "scorecard_escalate"
        assert DecisionType.ROUTING_DECISION.value == "routing_decision"
        assert DecisionType.TOOL_USAGE.value == "tool_usage"


class TestEnvVariableOverrides:
    """Test that environment variables correctly override config."""
    
    def test_env_enables_scorecard(self):
        """Test that SCORECARD_ENABLED=true enables scorecard."""
        from deepthinker.metrics import MetricsConfig, reset_metrics_config
        
        reset_metrics_config()
        
        with patch.dict(os.environ, {"SCORECARD_ENABLED": "true"}):
            config = MetricsConfig.from_env()
            assert config.scorecard_enabled == True
    
    def test_env_enables_policy(self):
        """Test that SCORECARD_POLICY_ENABLED=true enables policy."""
        from deepthinker.metrics import MetricsConfig
        
        with patch.dict(os.environ, {"SCORECARD_POLICY_ENABLED": "true"}):
            config = MetricsConfig.from_env()
            assert config.scorecard_policy_enabled == True
    
    def test_env_enables_router(self):
        """Test that LEARNING_ROUTER_ENABLED=true enables router."""
        from deepthinker.metrics import MetricsConfig
        
        with patch.dict(os.environ, {"LEARNING_ROUTER_ENABLED": "true"}):
            config = MetricsConfig.from_env()
            assert config.learning_router_enabled == True
    
    def test_env_enables_bandit(self):
        """Test that BANDIT_ENABLED=true enables bandit."""
        from deepthinker.metrics import MetricsConfig
        
        with patch.dict(os.environ, {"BANDIT_ENABLED": "true"}):
            config = MetricsConfig.from_env()
            assert config.bandit_enabled == True
    
    def test_env_sets_thresholds(self):
        """Test that threshold env variables work."""
        from deepthinker.metrics import MetricsConfig
        
        with patch.dict(os.environ, {
            "STOP_OVERALL_THRESHOLD": "0.9",
            "STOP_CONSISTENCY_THRESHOLD": "0.85",
            "BANDIT_LAMBDA": "0.2",
        }):
            config = MetricsConfig.from_env()
            assert config.stop_overall_threshold == 0.9
            assert config.stop_consistency_threshold == 0.85
            assert config.bandit_lambda == 0.2
    
    def test_env_sets_sampling_rates(self):
        """Test that sampling rate env variables work."""
        from deepthinker.metrics import MetricsConfig
        
        with patch.dict(os.environ, {
            "JUDGE_SAMPLE_RATE": "0.5",
            "TOOL_TRACK_SAMPLE_RATE": "0.3",
        }):
            config = MetricsConfig.from_env()
            assert config.judge_sample_rate == 0.5
            assert config.tool_track_sample_rate == 0.3


class TestIntegrationNoBreakage:
    """Test that existing code paths still work."""
    
    def test_orchestrator_import_works(self):
        """Test that mission orchestrator can still be imported."""
        from deepthinker.missions.mission_orchestrator import MissionOrchestrator
        assert MissionOrchestrator is not None
    
    def test_arbiter_import_works(self):
        """Test that arbiter can still be imported."""
        from deepthinker.arbiter.arbiter import Arbiter, ArbiterDecision
        assert Arbiter is not None
        assert ArbiterDecision is not None
    
    def test_step_executor_import_works(self):
        """Test that step executor can still be imported."""
        from deepthinker.steps.step_executor import StepExecutor
        assert StepExecutor is not None
    
    def test_decision_types_backward_compat(self):
        """Test that existing decision types still work."""
        from deepthinker.decisions.decision_record import DecisionType
        
        # Existing types should still be present
        assert hasattr(DecisionType, "MODEL_SELECTION")
        assert hasattr(DecisionType, "RETRY_ESCALATION")
        assert hasattr(DecisionType, "GOVERNANCE_INTERVENTION")
        assert hasattr(DecisionType, "PHASE_TERMINATION")
        assert hasattr(DecisionType, "EMPTY_OUTPUT_ESCALATION")
    
    def test_epistemics_backward_compat(self):
        """Test that existing epistemics exports still work."""
        from deepthinker.epistemics import (
            Claim,
            ClaimType,
            ClaimStatus,
            ClaimValidator,
            ClaimRegistry,
            Source,
        )
        
        assert Claim is not None
        assert ClaimType is not None
        assert ClaimStatus is not None
        assert ClaimValidator is not None
        assert ClaimRegistry is not None
        assert Source is not None


class TestLegacySimulationPath:
    """Test that the legacy simulation path works correctly."""
    
    def test_run_simulation_phase_import(self):
        """Test that _run_simulation_phase can be called without NameError.
        
        Regression test for missing Task import in run_workflow.py.
        """
        from deepthinker.execution.run_workflow import DeepThinkerWorkflow
        
        # Verify the class can be imported (Task is used in _run_simulation_phase)
        assert DeepThinkerWorkflow is not None
        
        # Verify Task is properly imported in the module
        from deepthinker.execution import run_workflow
        assert hasattr(run_workflow, 'Task') or 'Task' in dir(run_workflow)
    
    def test_simulation_phase_method_exists(self):
        """Test that _run_simulation_phase method exists on DeepThinkerWorkflow."""
        from deepthinker.execution.run_workflow import DeepThinkerWorkflow
        
        assert hasattr(DeepThinkerWorkflow, '_run_simulation_phase')
    
    def test_legacy_scenarios_parameter_accepted(self):
        """Test that run_deepthinker_workflow accepts scenarios parameter."""
        from deepthinker.execution.run_workflow import run_deepthinker_workflow
        import inspect
        
        sig = inspect.signature(run_deepthinker_workflow)
        assert 'scenarios' in sig.parameters
        
        # Verify it accepts Optional[List[str]]
        param = sig.parameters['scenarios']
        assert param.default is None
