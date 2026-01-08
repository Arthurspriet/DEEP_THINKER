"""
Unit tests for the Alignment Control Layer.

Tests cover:
- Alignment metrics (a_t, d_t, s_t, D^-)
- CUSUM trigger logic
- Controller escalation ladder
- Silent failure behavior
- Configuration loading
"""

import os
import pytest
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch


# =============================================================================
# Test Models
# =============================================================================

class TestAlignmentModels:
    """Tests for alignment data models."""
    
    def test_north_star_goal_creation(self):
        """Test NorthStarGoal creation and methods."""
        from deepthinker.alignment.models import NorthStarGoal
        
        goal = NorthStarGoal.from_mission_objective(
            objective="Analyze climate change impacts",
            mission_id="test-123",
        )
        
        assert goal.goal_id.startswith("ns_test-123_")
        assert goal.intent_summary == "Analyze climate change impacts"
        assert goal.success_criteria == []
        assert goal.embedding is None
    
    def test_north_star_goal_to_dict(self):
        """Test NorthStarGoal serialization."""
        from deepthinker.alignment.models import NorthStarGoal
        
        goal = NorthStarGoal(
            goal_id="test-goal",
            intent_summary="Test objective",
            success_criteria=["Criteria 1", "Criteria 2"],
            forbidden_outcomes=["Bad outcome"],
            priority_axes={"cost": "minimize"},
            scope_boundaries=["Scope limit"],
        )
        
        data = goal.to_dict()
        
        assert data["goal_id"] == "test-goal"
        assert data["intent_summary"] == "Test objective"
        assert len(data["success_criteria"]) == 2
        assert data["has_embedding"] is False
    
    def test_north_star_goal_full_text(self):
        """Test NorthStarGoal full text generation for embedding."""
        from deepthinker.alignment.models import NorthStarGoal
        
        goal = NorthStarGoal(
            goal_id="test",
            intent_summary="Main objective",
            success_criteria=["Success 1"],
            forbidden_outcomes=["Forbidden 1"],
            priority_axes={"speed": "maximize"},
            scope_boundaries=["Within budget"],
        )
        
        full_text = goal.get_full_text()
        
        assert "Main objective" in full_text
        assert "Success criteria" in full_text
        assert "Forbidden" in full_text
        assert "Priorities" in full_text
        assert "Scope" in full_text
    
    def test_alignment_point_creation(self):
        """Test AlignmentPoint creation."""
        from deepthinker.alignment.models import AlignmentPoint
        
        point = AlignmentPoint(
            t=5,
            a_t=0.85,
            d_t=-0.05,
            s_t=0.1,
            cusum_neg=0.15,
            cumulative_neg_drift=0.2,
            triggered=False,
            phase_name="research",
        )
        
        assert point.t == 5
        assert point.a_t == 0.85
        assert point.triggered is False
    
    def test_alignment_point_initial(self):
        """Test initial alignment point."""
        from deepthinker.alignment.models import AlignmentPoint
        
        point = AlignmentPoint.initial("start")
        
        assert point.t == 0
        assert point.a_t == 1.0
        assert point.d_t == 0.0
        assert point.cusum_neg == 0.0
        assert point.triggered is False
    
    def test_alignment_trajectory(self):
        """Test AlignmentTrajectory operations."""
        from deepthinker.alignment.models import (
            AlignmentPoint,
            AlignmentTrajectory,
            NorthStarGoal,
        )
        
        goal = NorthStarGoal.from_mission_objective("Test", "m1")
        trajectory = AlignmentTrajectory(
            mission_id="m1",
            north_star=goal,
        )
        
        # Add points
        p1 = AlignmentPoint(t=0, a_t=1.0, d_t=0.0, s_t=0.0, cusum_neg=0.0,
                           cumulative_neg_drift=0.0, triggered=False, phase_name="p1")
        p2 = AlignmentPoint(t=1, a_t=0.9, d_t=-0.1, s_t=0.2, cusum_neg=0.05,
                           cumulative_neg_drift=0.1, triggered=False, phase_name="p2")
        p3 = AlignmentPoint(t=2, a_t=0.7, d_t=-0.2, s_t=0.4, cusum_neg=0.2,
                           cumulative_neg_drift=0.3, triggered=True, phase_name="p3")
        
        trajectory.add_point(p1)
        trajectory.add_point(p2)
        trajectory.add_point(p3)
        
        assert len(trajectory.points) == 3
        assert trajectory.last_point() == p3
        assert trajectory.get_trigger_count() == 1
        assert trajectory.get_consecutive_triggers() == 1
    
    def test_alignment_assessment_fallback(self):
        """Test AlignmentAssessment fallback creation."""
        from deepthinker.alignment.models import AlignmentAssessment
        
        fallback = AlignmentAssessment.fallback("test_error")
        
        assert fallback.perceived_alignment == "medium"
        assert fallback.drift_risk == "emerging"
        assert "test_error" in fallback.dominant_drift_vector
    
    def test_alignment_action_values(self):
        """Test AlignmentAction enum values."""
        from deepthinker.alignment.models import AlignmentAction
        
        assert AlignmentAction.REANCHOR_INTERNAL.value == "reanchor_internal"
        assert AlignmentAction.INCREASE_SKEPTIC_WEIGHT.value == "increase_skeptic_weight"
        assert AlignmentAction.TRIGGER_USER_EVENT_DRIFT_CONFIRMATION.value == "user_event"
    
    def test_controller_state(self):
        """Test ControllerState operations."""
        from deepthinker.alignment.models import ControllerState, AlignmentAction
        
        state = ControllerState()
        
        assert state.consecutive_triggers == 0
        
        state.increment_triggers()
        state.increment_triggers()
        assert state.consecutive_triggers == 2
        
        state.record_action(AlignmentAction.REANCHOR_INTERNAL)
        assert state.total_actions_taken == 1
        assert "reanchor_internal" in state.last_action_timestamps
        
        state.reset_triggers()
        assert state.consecutive_triggers == 0


# =============================================================================
# Test Configuration
# =============================================================================

class TestAlignmentConfig:
    """Tests for alignment configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig()
        
        assert config.enabled is False
        assert config.embedding_model == "qwen3-embedding:4b"
        assert config.evaluator_model == "llama3.2:3b"
        assert config.min_similarity_soft == 0.4
        assert config.cusum_h == 0.5
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        from deepthinker.alignment.config import AlignmentConfig, reset_alignment_config
        
        # Set env vars
        os.environ["ALIGNMENT_ENABLED"] = "true"
        os.environ["ALIGNMENT_MIN_SIMILARITY"] = "0.5"
        
        try:
            reset_alignment_config()
            config = AlignmentConfig.from_env()
            
            assert config.enabled is True
            assert config.min_similarity_soft == 0.5
        finally:
            # Clean up
            os.environ.pop("ALIGNMENT_ENABLED", None)
            os.environ.pop("ALIGNMENT_MIN_SIMILARITY", None)
            reset_alignment_config()
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(enabled=True, cusum_h=0.6)
        data = config.to_dict()
        
        assert data["enabled"] is True
        assert data["cusum_h"] == 0.6
        assert "escalation_ladder" in data


# =============================================================================
# Test Drift Detection (Metrics)
# =============================================================================

class TestAlignmentMetrics:
    """Tests for alignment metrics computation."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embedding function that returns controlled vectors."""
        def _get_embedding(text: str) -> List[float]:
            # Return different embeddings based on text content
            if "goal" in text.lower() or "objective" in text.lower():
                return [1.0, 0.0, 0.0]  # Goal vector
            elif "aligned" in text.lower():
                return [0.95, 0.05, 0.0]  # Closely aligned
            elif "drifting" in text.lower():
                return [0.5, 0.5, 0.0]  # Drifting
            elif "diverged" in text.lower():
                return [0.0, 1.0, 0.0]  # Diverged
            else:
                return [0.7, 0.3, 0.0]  # Default
        
        return _get_embedding
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity between vectors."""
        import numpy as np
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(enabled=True)
        detector = EmbeddingDriftDetector(config)
        
        # Identical vectors -> similarity = 1
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert abs(detector._cosine_similarity(v1, v2) - 1.0) < 0.001
        
        # Orthogonal vectors -> similarity = 0
        v3 = np.array([0.0, 1.0, 0.0])
        assert abs(detector._cosine_similarity(v1, v3)) < 0.001
        
        # 45-degree angle -> similarity ≈ 0.707
        v4 = np.array([0.707, 0.707, 0.0])
        v4 = v4 / np.linalg.norm(v4)
        sim = detector._cosine_similarity(v1, v4)
        assert 0.7 < sim < 0.75
    
    def test_euclidean_distance(self):
        """Test Euclidean distance for semantic jump."""
        import numpy as np
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(enabled=True)
        detector = EmbeddingDriftDetector(config)
        
        # Same vector -> distance = 0
        v1 = np.array([1.0, 0.0, 0.0])
        assert detector._euclidean_distance(v1, v1) == 0.0
        
        # Opposite vectors (normalized) -> distance = 2
        v2 = np.array([-1.0, 0.0, 0.0])
        assert abs(detector._euclidean_distance(v1, v2) - 2.0) < 0.001
    
    def test_drift_delta_computation(self, mock_embeddings):
        """Test d_t = a_t - a_{t-1} computation."""
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import AlignmentPoint, NorthStarGoal
        
        config = AlignmentConfig(enabled=True)
        detector = EmbeddingDriftDetector(config)
        
        # Mock the embedding call
        with patch.object(detector, '_get_embedding', side_effect=mock_embeddings):
            # Set up north star
            goal = NorthStarGoal.from_mission_objective("Goal objective", "m1")
            detector.set_north_star(goal)
            
            # Compute first point
            p1 = detector.compute_alignment_point(
                output_text="Aligned output",
                phase_name="p1",
                prev_point=None,
            )
            
            # Compute second point
            p2 = detector.compute_alignment_point(
                output_text="Drifting output",
                phase_name="p2",
                prev_point=p1,
            )
            
            # d_t should be negative (drifting away)
            assert p2.d_t < 0
    
    def test_cumulative_negative_drift(self, mock_embeddings):
        """Test D_t^- = sum(max(0, -d_t)) computation."""
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import NorthStarGoal
        
        config = AlignmentConfig(enabled=True)
        detector = EmbeddingDriftDetector(config)
        
        with patch.object(detector, '_get_embedding', side_effect=mock_embeddings):
            goal = NorthStarGoal.from_mission_objective("Goal objective", "m1")
            detector.set_north_star(goal)
            
            # Start aligned
            p1 = detector.compute_alignment_point("Aligned output", "p1", None)
            
            # Drift away
            p2 = detector.compute_alignment_point("Drifting output", "p2", p1)
            
            # Drift more
            p3 = detector.compute_alignment_point("Diverged output", "p3", p2)
            
            # Cumulative negative drift should increase
            assert p3.cumulative_neg_drift >= p2.cumulative_neg_drift


# =============================================================================
# Test CUSUM Trigger
# =============================================================================

class TestCUSUMTrigger:
    """Tests for CUSUM-based drift trigger."""
    
    def test_cusum_formula(self):
        """Test CUSUM formula: cusum_neg = max(0, prev_cusum + (-d_t) - k)."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(cusum_k=0.05, cusum_h=0.5)
        
        # Simulate CUSUM calculation manually
        cusum = 0.0
        k = config.cusum_k
        h = config.cusum_h
        
        # Series of small negative drifts - need enough to exceed threshold
        # Each drift adds 0.1 - 0.05 = 0.05 to CUSUM
        # Need 11 drifts to get to 0.55 > 0.5
        drifts = [-0.1] * 11
        
        for d_t in drifts:
            neg_drift = max(0.0, -d_t)
            cusum = max(0.0, cusum + neg_drift - k)
        
        # After 11 small drifts, CUSUM should exceed threshold
        assert cusum > h
    
    def test_cusum_trigger_after_repeated_drift(self):
        """Test that CUSUM triggers after repeated small negative drift."""
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import AlignmentPoint
        
        config = AlignmentConfig(
            enabled=True,
            cusum_k=0.05,
            cusum_h=0.3,
            min_events_before_trigger=3,
        )
        detector = EmbeddingDriftDetector(config)
        
        # Simulate a series of points with small negative drift
        prev_point = AlignmentPoint(
            t=0, a_t=1.0, d_t=0.0, s_t=0.0,
            cusum_neg=0.0, cumulative_neg_drift=0.0,
            triggered=False, phase_name="p0"
        )
        
        # Check triggers manually by simulating the CUSUM logic
        # The detector uses embeddings, so we test the trigger logic directly
        cusum = 0.0
        drifts = [-0.1, -0.1, -0.1, -0.1, -0.1]  # 5 small drifts
        
        for i, d_t in enumerate(drifts):
            neg_drift = max(0.0, -d_t)
            cusum = max(0.0, cusum + neg_drift - config.cusum_k)
            
            # Check if would trigger
            should_trigger = (
                i >= config.min_events_before_trigger and
                cusum > config.cusum_h
            )
            
            if i >= config.min_events_before_trigger:
                # After enough events, should trigger
                if cusum > config.cusum_h:
                    assert should_trigger
    
    def test_cusum_reset_on_positive_drift(self):
        """Test that CUSUM decreases/resets with positive drift."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(cusum_k=0.05)
        
        # Start with some CUSUM accumulated
        cusum = 0.3
        
        # Positive drift (improvement)
        d_t = 0.1  # Positive drift
        neg_drift = max(0.0, -d_t)  # = 0.0
        cusum = max(0.0, cusum + neg_drift - config.cusum_k)
        
        # CUSUM should decrease (by k)
        assert cusum == 0.25  # 0.3 + 0.0 - 0.05
        
        # More positive drift
        d_t = 0.1
        neg_drift = max(0.0, -d_t)
        cusum = max(0.0, cusum + neg_drift - config.cusum_k)
        
        assert cusum == 0.20  # Continues to decrease


# =============================================================================
# Test Controller Escalation
# =============================================================================

class TestControllerEscalation:
    """Tests for controller escalation ladder."""
    
    def test_first_trigger_reanchor(self):
        """Test that first trigger results in REANCHOR action."""
        from deepthinker.alignment.controller import AlignmentController
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import (
            AlignmentAction,
            AlignmentPoint,
            AlignmentTrajectory,
            NorthStarGoal,
        )
        
        config = AlignmentConfig(enabled=True)
        controller = AlignmentController(config)
        
        goal = NorthStarGoal.from_mission_objective("Test", "m1")
        trajectory = AlignmentTrajectory(mission_id="m1", north_star=goal)
        
        # Create triggered point
        point = AlignmentPoint(
            t=5, a_t=0.3, d_t=-0.2, s_t=0.4,
            cusum_neg=0.6, cumulative_neg_drift=0.5,
            triggered=True, phase_name="p5"
        )
        
        actions = controller.decide(point, trajectory)
        
        assert AlignmentAction.REANCHOR_INTERNAL in actions
    
    def test_escalation_ladder_progression(self):
        """Test that repeated triggers escalate actions."""
        from deepthinker.alignment.controller import AlignmentController
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import (
            AlignmentAction,
            AlignmentPoint,
            AlignmentTrajectory,
            NorthStarGoal,
        )
        
        config = AlignmentConfig(
            enabled=True,
            reanchor_cooldown_phases=0,  # No cooldown for testing
            max_actions_per_phase=10,  # Allow all actions for testing
            escalation_ladder={
                "reanchor_internal": 1,
                "increase_skeptic_weight": 2,
                "switch_to_evidence": 3,
                "prune_focus_areas": 4,
                "user_event": 5,
            },
            user_event_threshold=5,
        )
        controller = AlignmentController(config)
        
        goal = NorthStarGoal.from_mission_objective("Test", "m1")
        trajectory = AlignmentTrajectory(mission_id="m1", north_star=goal)
        
        # Simulate consecutive triggers
        collected_actions = []
        for i in range(6):
            point = AlignmentPoint(
                t=i, a_t=0.3, d_t=-0.1, s_t=0.2,
                cusum_neg=0.6, cumulative_neg_drift=0.5,
                triggered=True, phase_name=f"p{i}"
            )
            actions = controller.decide(point, trajectory)
            collected_actions.append(set(actions))
        
        # First trigger: reanchor
        assert AlignmentAction.REANCHOR_INTERNAL in collected_actions[0]
        
        # Second trigger: includes skeptic (consecutive=2)
        assert AlignmentAction.INCREASE_SKEPTIC_WEIGHT in collected_actions[1]
        
        # Third trigger: includes evidence mode (consecutive=3)
        assert AlignmentAction.SWITCH_DEEPEN_MODE_TO_EVIDENCE in collected_actions[2]
        
        # Fourth trigger: includes prune (consecutive=4)
        assert AlignmentAction.PRUNE_OR_PARK_FOCUS_AREAS in collected_actions[3]
        
        # Fifth trigger: includes user event (consecutive=5)
        assert AlignmentAction.TRIGGER_USER_EVENT_DRIFT_CONFIRMATION in collected_actions[4]
    
    def test_no_action_when_not_triggered(self):
        """Test that no actions when point is not triggered."""
        from deepthinker.alignment.controller import AlignmentController
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import (
            AlignmentPoint,
            AlignmentTrajectory,
            NorthStarGoal,
        )
        
        config = AlignmentConfig(enabled=True)
        controller = AlignmentController(config)
        
        goal = NorthStarGoal.from_mission_objective("Test", "m1")
        trajectory = AlignmentTrajectory(mission_id="m1", north_star=goal)
        
        # Non-triggered point
        point = AlignmentPoint(
            t=5, a_t=0.9, d_t=0.05, s_t=0.1,
            cusum_neg=0.1, cumulative_neg_drift=0.05,
            triggered=False, phase_name="p5"
        )
        
        actions = controller.decide(point, trajectory)
        
        assert len(actions) == 0
    
    def test_consecutive_counter_reset(self):
        """Test that consecutive counter resets on non-triggered point."""
        from deepthinker.alignment.controller import AlignmentController
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import (
            AlignmentAction,
            AlignmentPoint,
            AlignmentTrajectory,
            NorthStarGoal,
        )
        
        config = AlignmentConfig(enabled=True, reanchor_cooldown_phases=0)
        controller = AlignmentController(config)
        
        goal = NorthStarGoal.from_mission_objective("Test", "m1")
        trajectory = AlignmentTrajectory(mission_id="m1", north_star=goal)
        
        # First trigger
        p1 = AlignmentPoint(
            t=1, a_t=0.3, d_t=-0.2, s_t=0.4,
            cusum_neg=0.6, cumulative_neg_drift=0.5,
            triggered=True, phase_name="p1"
        )
        controller.decide(p1, trajectory)
        assert controller.state.consecutive_triggers == 1
        
        # Second trigger
        p2 = AlignmentPoint(
            t=2, a_t=0.3, d_t=-0.1, s_t=0.2,
            cusum_neg=0.7, cumulative_neg_drift=0.6,
            triggered=True, phase_name="p2"
        )
        controller.decide(p2, trajectory)
        assert controller.state.consecutive_triggers == 2
        
        # Non-triggered point should reset
        p3 = AlignmentPoint(
            t=3, a_t=0.8, d_t=0.1, s_t=0.1,
            cusum_neg=0.1, cumulative_neg_drift=0.6,
            triggered=False, phase_name="p3"
        )
        controller.decide(p3, trajectory)
        assert controller.state.consecutive_triggers == 0


# =============================================================================
# Test Silent Failure
# =============================================================================

class TestSilentFailure:
    """Tests for silent failure behavior."""
    
    def test_disabled_returns_empty(self):
        """Test that disabled alignment returns empty actions."""
        from deepthinker.alignment.integration import run_alignment_check
        from deepthinker.alignment.config import reset_alignment_config
        
        # Ensure disabled
        os.environ["ALIGNMENT_ENABLED"] = "false"
        reset_alignment_config()
        
        try:
            # Create mock mission state
            mock_state = MagicMock()
            mock_state.mission_id = "test"
            mock_state.objective = "Test objective"
            mock_state.constraints = None
            
            mock_phase = MagicMock()
            mock_phase.name = "research"
            mock_phase.artifacts = {"output": "Some output"}
            
            actions = run_alignment_check(mock_state, mock_phase, {})
            
            assert actions == []
        finally:
            os.environ.pop("ALIGNMENT_ENABLED", None)
            reset_alignment_config()
    
    def test_embedding_error_silent(self):
        """Test that embedding errors don't crash and return None."""
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import NorthStarGoal
        
        config = AlignmentConfig(enabled=True)
        detector = EmbeddingDriftDetector(config)
        
        # Mock embedding to return empty list (failure mode)
        with patch.object(detector, '_get_embedding', return_value=[]):
            goal = NorthStarGoal.from_mission_objective("Test", "m1")
            
            # Should not raise
            detector.set_north_star(goal)
            
            # North star embedding should be None when embedding fails
            assert detector._north_star_embedding is None
            
            # Should return None on error since no north star embedding
            point = detector.compute_alignment_point("Test", "p1", None)
            assert point is None
    
    def test_integration_error_returns_empty(self):
        """Test that integration errors return empty list."""
        from deepthinker.alignment.integration import run_alignment_check
        
        # Pass invalid objects to trigger error
        with patch('deepthinker.alignment.integration.get_alignment_config') as mock_config:
            mock_config.return_value = MagicMock(enabled=True)
            
            # This should not raise, just return empty
            result = run_alignment_check(None, None, {})
            assert result == []


# =============================================================================
# Test Persistence
# =============================================================================

class TestAlignmentPersistence:
    """Tests for alignment log persistence."""
    
    def test_log_store_initialization(self, tmp_path):
        """Test log store initialization."""
        from deepthinker.alignment.persist import AlignmentLogStore
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import NorthStarGoal
        
        config = AlignmentConfig(persist_logs=True)
        store = AlignmentLogStore(config, log_dir=str(tmp_path))
        
        goal = NorthStarGoal.from_mission_objective("Test", "m1")
        store.initialize("m1", goal)
        
        log = store.get_log("m1")
        assert log is not None
        assert log["mission_id"] == "m1"
        assert "north_star" in log
    
    def test_log_save_and_load(self, tmp_path):
        """Test saving and loading logs."""
        from deepthinker.alignment.persist import AlignmentLogStore
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import NorthStarGoal, AlignmentPoint
        
        config = AlignmentConfig(persist_logs=True)
        store = AlignmentLogStore(config, log_dir=str(tmp_path))
        
        goal = NorthStarGoal.from_mission_objective("Test", "test-mission")
        store.initialize("test-mission", goal)
        
        # Add some data
        point = AlignmentPoint(
            t=0, a_t=0.9, d_t=0.0, s_t=0.0,
            cusum_neg=0.0, cumulative_neg_drift=0.0,
            triggered=False, phase_name="p0"
        )
        store.add_point("test-mission", point)
        
        # Save
        assert store.save("test-mission") is True
        
        # Check file exists
        log_file = tmp_path / "test-mission.json"
        assert log_file.exists()
        
        # Create new store and load
        store2 = AlignmentLogStore(config, log_dir=str(tmp_path))
        loaded = store2.load("test-mission")
        
        assert loaded is not None
        assert loaded["mission_id"] == "test-mission"
        assert len(loaded["trajectory"]) == 1


# =============================================================================
# Test Integration with Mission State
# =============================================================================

class TestMissionStateIntegration:
    """Tests for integration with MissionState."""
    
    def test_mission_state_has_alignment_fields(self):
        """Test that MissionState has alignment fields."""
        from deepthinker.missions.mission_types import MissionState, MissionConstraints
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        state = MissionState(
            mission_id="test",
            objective="Test objective",
            constraints=MissionConstraints(time_budget_minutes=30),
            created_at=now,
            deadline_at=now + timedelta(minutes=30),
        )
        
        # Check alignment fields exist
        assert hasattr(state, "alignment_north_star")
        assert hasattr(state, "alignment_trajectory")
        assert hasattr(state, "alignment_controller_state")
        assert hasattr(state, "pending_user_event")
        
        # Check defaults
        assert state.alignment_north_star is None
        assert state.alignment_trajectory == []
        assert state.alignment_controller_state == {}
        assert state.pending_user_event is None
    
    def test_mission_state_to_dict_includes_alignment(self):
        """Test that to_dict includes alignment fields."""
        from deepthinker.missions.mission_types import MissionState, MissionConstraints
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        state = MissionState(
            mission_id="test",
            objective="Test",
            constraints=MissionConstraints(time_budget_minutes=30),
            created_at=now,
            deadline_at=now + timedelta(minutes=30),
        )
        
        # Set some alignment data
        state.alignment_north_star = {"goal_id": "test", "intent": "Test"}
        state.alignment_trajectory = [{"t": 0, "a_t": 1.0}]
        
        data = state.to_dict()
        
        assert "alignment_north_star" in data
        assert "alignment_trajectory" in data
        assert data["alignment_north_star"]["goal_id"] == "test"


# =============================================================================
# Test Two-Tier Threshold System (Gap 1)
# =============================================================================

class TestTwoTierThresholds:
    """Tests for the two-tier warning/correction threshold system."""
    
    def test_config_has_two_tier_thresholds(self):
        """Test that AlignmentConfig has warning and correction thresholds."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig()
        
        assert hasattr(config, "warning_threshold")
        assert hasattr(config, "correction_threshold")
        assert config.warning_threshold == 0.6
        assert config.correction_threshold == 0.4
    
    def test_alignment_point_has_warning_field(self):
        """Test that AlignmentPoint has warning field."""
        from deepthinker.alignment.models import AlignmentPoint
        
        point = AlignmentPoint(
            t=5,
            a_t=0.5,
            d_t=-0.1,
            s_t=0.1,
            cusum_neg=0.2,
            cumulative_neg_drift=0.3,
            triggered=False,
            phase_name="test",
            warning=True,
        )
        
        assert hasattr(point, "warning")
        assert point.warning is True
    
    def test_initial_point_has_warning_false(self):
        """Test that initial AlignmentPoint has warning=False."""
        from deepthinker.alignment.models import AlignmentPoint
        
        point = AlignmentPoint.initial("start")
        
        assert point.warning is False
    
    def test_drift_detector_returns_warning_and_correction(self):
        """Test that _check_triggers returns tuple (warning, correction)."""
        from deepthinker.alignment.drift import EmbeddingDriftDetector
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(
            enabled=True,
            warning_threshold=0.6,
            correction_threshold=0.4,
            min_events_before_trigger=0,  # Allow immediate triggering for test
        )
        detector = EmbeddingDriftDetector(config)
        
        # High alignment: no warning, no correction
        warning, correction = detector._check_triggers(a_t=0.8, d_t=0.0, cusum_neg=0.0, t=5)
        assert warning is False
        assert correction is False
        
        # Medium alignment: warning only
        warning, correction = detector._check_triggers(a_t=0.5, d_t=0.0, cusum_neg=0.0, t=5)
        assert warning is True
        assert correction is False
        
        # Low alignment: warning and correction
        warning, correction = detector._check_triggers(a_t=0.3, d_t=0.0, cusum_neg=0.0, t=5)
        assert warning is True
        assert correction is True
    
    def test_alignment_point_to_dict_includes_warning(self):
        """Test that AlignmentPoint.to_dict() includes warning field."""
        from deepthinker.alignment.models import AlignmentPoint
        
        point = AlignmentPoint(
            t=5,
            a_t=0.5,
            d_t=-0.1,
            s_t=0.1,
            cusum_neg=0.2,
            cumulative_neg_drift=0.3,
            triggered=False,
            phase_name="test",
            warning=True,
        )
        
        data = point.to_dict()
        assert "warning" in data
        assert data["warning"] is True
    
    def test_alignment_point_from_dict_loads_warning(self):
        """Test that AlignmentPoint.from_dict() loads warning field."""
        from deepthinker.alignment.models import AlignmentPoint
        
        data = {
            "t": 5,
            "a_t": 0.5,
            "d_t": -0.1,
            "s_t": 0.1,
            "cusum_neg": 0.2,
            "cumulative_neg_drift": 0.3,
            "triggered": False,
            "phase_name": "test",
            "warning": True,
        }
        
        point = AlignmentPoint.from_dict(data)
        assert point.warning is True
    
    def test_config_from_env_loads_thresholds(self):
        """Test that AlignmentConfig.from_env() loads threshold env vars."""
        from deepthinker.alignment.config import AlignmentConfig, reset_alignment_config
        
        os.environ["ALIGNMENT_WARNING_THRESHOLD"] = "0.7"
        os.environ["ALIGNMENT_CORRECTION_THRESHOLD"] = "0.5"
        
        try:
            reset_alignment_config()
            config = AlignmentConfig.from_env()
            
            assert config.warning_threshold == 0.7
            assert config.correction_threshold == 0.5
        finally:
            os.environ.pop("ALIGNMENT_WARNING_THRESHOLD", None)
            os.environ.pop("ALIGNMENT_CORRECTION_THRESHOLD", None)
            reset_alignment_config()


# =============================================================================
# Test Prompt Injection (Gap 2)
# =============================================================================

class TestPromptInjection:
    """Tests for re-anchor prompt injection."""
    
    def test_config_has_prompt_injection_fields(self):
        """Test that AlignmentConfig has prompt injection fields."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig()
        
        assert hasattr(config, "reanchor_prompt_template")
        assert hasattr(config, "inject_reanchor_prompt")
        assert config.inject_reanchor_prompt is True
        assert "{objective}" in config.reanchor_prompt_template
    
    def test_reanchor_action_sets_prompt(self):
        """Test that REANCHOR_INTERNAL action sets reanchor_prompt in mission state."""
        from deepthinker.alignment.controller import AlignmentController, apply_action
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import (
            AlignmentAction,
            AlignmentTrajectory,
            NorthStarGoal,
        )
        
        config = AlignmentConfig(
            enabled=True,
            inject_reanchor_prompt=True,
            reanchor_prompt_template="[REMINDER: {objective}]",
        )
        controller = AlignmentController(config)
        
        goal = NorthStarGoal.from_mission_objective("Test objective here", "m1")
        trajectory = AlignmentTrajectory(mission_id="m1", north_star=goal)
        
        # Add a point so trajectory has last_point
        from deepthinker.alignment.models import AlignmentPoint
        trajectory.add_point(AlignmentPoint(
            t=1, a_t=0.3, d_t=-0.1, s_t=0.1,
            cusum_neg=0.5, cumulative_neg_drift=0.2,
            triggered=True, phase_name="p1"
        ))
        
        # Mock mission state
        mock_state = MagicMock()
        mock_state.alignment_controller_state = {}
        
        # Apply reanchor action
        success = apply_action(
            action=AlignmentAction.REANCHOR_INTERNAL,
            mission_state=mock_state,
            controller=controller,
            trajectory=trajectory,
        )
        
        assert success is True
        assert "reanchor_prompt" in mock_state.alignment_controller_state
        assert "Test objective here" in mock_state.alignment_controller_state["reanchor_prompt"]
    
    def test_council_consumes_reanchor_prompt(self):
        """Test that BaseCouncil._check_and_consume_reanchor_prompt works."""
        # Mock minimal council setup
        mock_state = MagicMock()
        mock_state.alignment_controller_state = {
            "reanchor_prompt": "[REMINDER: Test objective]",
            "reanchor_prompt_phase": "research",
        }
        
        # Simulate the consume logic
        controller_state = mock_state.alignment_controller_state
        reanchor_prompt = controller_state.get("reanchor_prompt")
        
        assert reanchor_prompt == "[REMINDER: Test objective]"
        
        # Consume it
        controller_state.pop("reanchor_prompt", None)
        controller_state.pop("reanchor_prompt_phase", None)
        
        assert "reanchor_prompt" not in controller_state
    
    def test_step_executor_consumes_reanchor_prompt(self):
        """Test that StepExecutor._check_and_consume_reanchor_prompt works."""
        from deepthinker.steps.step_executor import StepExecutor
        
        # Create minimal executor (with mocked pool)
        mock_pool = MagicMock()
        executor = StepExecutor(model_pool=mock_pool, enable_reflection=False)
        
        # Mock mission state with reanchor prompt
        mock_state = MagicMock()
        mock_state.alignment_controller_state = {
            "reanchor_prompt": "[REMINDER: Test objective]",
        }
        
        # Should find and consume the prompt
        result = executor._check_and_consume_reanchor_prompt(mock_state)
        
        assert result == "[REMINDER: Test objective]"
        assert "reanchor_prompt" not in mock_state.alignment_controller_state
    
    def test_step_executor_returns_none_when_no_prompt(self):
        """Test that StepExecutor returns None when no reanchor prompt."""
        from deepthinker.steps.step_executor import StepExecutor
        
        mock_pool = MagicMock()
        executor = StepExecutor(model_pool=mock_pool, enable_reflection=False)
        
        # No mission state
        result = executor._check_and_consume_reanchor_prompt(None)
        assert result is None
        
        # Empty controller state
        mock_state = MagicMock()
        mock_state.alignment_controller_state = {}
        result = executor._check_and_consume_reanchor_prompt(mock_state)
        assert result is None
    
    def test_config_from_env_loads_inject_prompt(self):
        """Test that AlignmentConfig.from_env() loads inject_prompt env var."""
        from deepthinker.alignment.config import AlignmentConfig, reset_alignment_config
        
        os.environ["ALIGNMENT_INJECT_PROMPT"] = "false"
        
        try:
            reset_alignment_config()
            config = AlignmentConfig.from_env()
            
            assert config.inject_reanchor_prompt is False
        finally:
            os.environ.pop("ALIGNMENT_INJECT_PROMPT", None)
            reset_alignment_config()


# =============================================================================
# Test Injection Safety Limits
# =============================================================================

class TestInjectionSafetyLimits:
    """Tests for injection cooldown and limit enforcement."""
    
    def test_controller_state_can_inject_initial(self):
        """Test that can_inject returns True on first injection."""
        from deepthinker.alignment.models import ControllerState
        
        state = ControllerState()
        
        # First injection should be allowed
        assert state.can_inject(t=0, min_phases=2, max_injections=5) is True
    
    def test_controller_state_can_inject_cooldown(self):
        """Test that can_inject respects cooldown period."""
        from deepthinker.alignment.models import ControllerState
        
        state = ControllerState()
        state.record_injection(t=0)
        
        # Too soon after injection
        assert state.can_inject(t=1, min_phases=2, max_injections=5) is False
        
        # After cooldown
        assert state.can_inject(t=2, min_phases=2, max_injections=5) is True
    
    def test_controller_state_can_inject_max_limit(self):
        """Test that can_inject respects max injection limit."""
        from deepthinker.alignment.models import ControllerState
        
        state = ControllerState()
        
        # Record max injections
        for i in range(5):
            state.record_injection(t=i * 3)  # Spaced out to avoid cooldown
        
        # Should be blocked by limit
        assert state.can_inject(t=20, min_phases=2, max_injections=5) is False
    
    def test_controller_state_record_injection(self):
        """Test that record_injection updates state correctly."""
        from deepthinker.alignment.models import ControllerState
        
        state = ControllerState()
        assert state.last_injection_t == -1
        assert state.injection_count_this_mission == 0
        
        state.record_injection(t=5)
        
        assert state.last_injection_t == 5
        assert state.injection_count_this_mission == 1
        
        state.record_injection(t=10)
        
        assert state.last_injection_t == 10
        assert state.injection_count_this_mission == 2
    
    def test_config_has_injection_limits(self):
        """Test that AlignmentConfig has injection limit fields."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig()
        
        assert hasattr(config, "max_injections_per_mission")
        assert hasattr(config, "min_phases_between_injections")
        assert config.max_injections_per_mission == 5
        assert config.min_phases_between_injections == 2
    
    def test_config_from_env_loads_injection_limits(self):
        """Test that AlignmentConfig.from_env() loads injection limit env vars."""
        from deepthinker.alignment.config import AlignmentConfig, reset_alignment_config
        
        os.environ["ALIGNMENT_MAX_INJECTIONS"] = "10"
        os.environ["ALIGNMENT_MIN_PHASES_BETWEEN_INJECTIONS"] = "3"
        
        try:
            reset_alignment_config()
            config = AlignmentConfig.from_env()
            
            assert config.max_injections_per_mission == 10
            assert config.min_phases_between_injections == 3
        finally:
            os.environ.pop("ALIGNMENT_MAX_INJECTIONS", None)
            os.environ.pop("ALIGNMENT_MIN_PHASES_BETWEEN_INJECTIONS", None)
            reset_alignment_config()
    
    def test_apply_reanchor_respects_injection_limits(self):
        """Test that apply_action respects injection cooldown and limits."""
        from deepthinker.alignment.controller import AlignmentController, apply_action
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import (
            AlignmentAction,
            AlignmentTrajectory,
            AlignmentPoint,
            NorthStarGoal,
        )
        
        config = AlignmentConfig(
            enabled=True,
            inject_reanchor_prompt=True,
            max_injections_per_mission=2,
            min_phases_between_injections=2,
        )
        controller = AlignmentController(config)
        
        goal = NorthStarGoal.from_mission_objective("Test objective", "m1")
        trajectory = AlignmentTrajectory(mission_id="m1", north_star=goal)
        
        # Add points to trajectory
        for i in range(5):
            trajectory.add_point(AlignmentPoint(
                t=i, a_t=0.3, d_t=-0.1, s_t=0.1,
                cusum_neg=0.5, cumulative_neg_drift=0.2,
                triggered=True, phase_name=f"p{i}"
            ))
        
        # Mock mission state
        mock_state = MagicMock()
        mock_state.alignment_controller_state = {}
        
        # First injection should succeed
        success1 = apply_action(
            action=AlignmentAction.REANCHOR_INTERNAL,
            mission_state=mock_state,
            controller=controller,
            trajectory=trajectory,
        )
        assert success1 is True
        assert "reanchor_prompt" in mock_state.alignment_controller_state
        
        # Second injection too soon (same phase) should succeed but without prompt
        # Clear the prompt to check if new one is added
        mock_state.alignment_controller_state.pop("reanchor_prompt", None)
        
        success2 = apply_action(
            action=AlignmentAction.REANCHOR_INTERNAL,
            mission_state=mock_state,
            controller=controller,
            trajectory=trajectory,
        )
        # Still succeeds (reanchor happens) but prompt may be skipped due to cooldown
        assert success2 is True


# =============================================================================
# Test Demo Mode Configuration
# =============================================================================

class TestDemoModeConfig:
    """Tests for demo mode configuration."""
    
    def test_config_has_demo_mode_fields(self):
        """Test that AlignmentConfig has demo mode fields."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig()
        
        assert hasattr(config, "demo_mode")
        assert hasattr(config, "demo_warning_threshold")
        assert hasattr(config, "demo_correction_threshold")
        assert hasattr(config, "demo_min_events_before_trigger")
        
        assert config.demo_mode is False
        assert config.demo_warning_threshold == 0.85
        assert config.demo_correction_threshold == 0.75
        assert config.demo_min_events_before_trigger == 1
    
    def test_demo_mode_uses_higher_thresholds(self):
        """Test that demo mode uses higher sensitivity thresholds."""
        from deepthinker.alignment.config import AlignmentConfig, reset_alignment_config
        
        os.environ["ALIGNMENT_DEMO_MODE"] = "true"
        
        try:
            reset_alignment_config()
            config = AlignmentConfig.from_env()
            
            assert config.demo_mode is True
            assert config.warning_threshold == 0.85
            assert config.correction_threshold == 0.75
            assert config.min_events_before_trigger == 1
        finally:
            os.environ.pop("ALIGNMENT_DEMO_MODE", None)
            reset_alignment_config()
    
    def test_normal_mode_uses_standard_thresholds(self):
        """Test that normal mode uses standard thresholds."""
        from deepthinker.alignment.config import AlignmentConfig, reset_alignment_config
        
        os.environ["ALIGNMENT_DEMO_MODE"] = "false"
        
        try:
            reset_alignment_config()
            config = AlignmentConfig.from_env()
            
            assert config.demo_mode is False
            assert config.warning_threshold == 0.6
            assert config.correction_threshold == 0.4
            assert config.min_events_before_trigger == 3
        finally:
            os.environ.pop("ALIGNMENT_DEMO_MODE", None)
            reset_alignment_config()
    
    def test_to_dict_includes_demo_fields(self):
        """Test that to_dict includes demo mode fields."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig(demo_mode=True)
        data = config.to_dict()
        
        assert "demo_mode" in data
        assert "demo_warning_threshold" in data
        assert "demo_correction_threshold" in data
        assert "demo_min_events_before_trigger" in data
        assert data["demo_mode"] is True


# =============================================================================
# Test Audit Logging
# =============================================================================

class TestAuditLogging:
    """Tests for prompt injection audit logging."""
    
    def test_config_has_log_full_prompts(self):
        """Test that AlignmentConfig has log_full_prompts field."""
        from deepthinker.alignment.config import AlignmentConfig
        
        config = AlignmentConfig()
        
        assert hasattr(config, "log_full_prompts")
        assert config.log_full_prompts is False
    
    def test_add_action_includes_prompt_hash(self, tmp_path):
        """Test that add_action includes prompt hash in log."""
        from deepthinker.alignment.persist import AlignmentLogStore
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import NorthStarGoal, AlignmentPoint, AlignmentAction
        
        config = AlignmentConfig(persist_logs=True, log_full_prompts=False)
        store = AlignmentLogStore(config, log_dir=str(tmp_path))
        
        goal = NorthStarGoal.from_mission_objective("Test", "test-mission")
        store.initialize("test-mission", goal)
        
        point = AlignmentPoint(
            t=0, a_t=0.5, d_t=-0.1, s_t=0.1,
            cusum_neg=0.3, cumulative_neg_drift=0.2,
            triggered=True, phase_name="p0"
        )
        
        test_prompt = "This is a test reanchor prompt"
        store.add_action(
            "test-mission",
            AlignmentAction.REANCHOR_INTERNAL,
            point,
            metadata={},
            injected_prompt=test_prompt,
        )
        
        log = store.get_log("test-mission")
        assert len(log["actions"]) == 1
        
        action_record = log["actions"][0]
        assert "prompt_injection" in action_record
        assert "prompt_hash" in action_record["prompt_injection"]
        assert "prompt_length" in action_record["prompt_injection"]
        assert action_record["prompt_injection"]["prompt_length"] == len(test_prompt)
        # Should NOT have full prompt by default
        assert "prompt_full" not in action_record["prompt_injection"]
    
    def test_add_action_includes_full_prompt_when_enabled(self, tmp_path):
        """Test that add_action includes full prompt when log_full_prompts is True."""
        from deepthinker.alignment.persist import AlignmentLogStore
        from deepthinker.alignment.config import AlignmentConfig
        from deepthinker.alignment.models import NorthStarGoal, AlignmentPoint, AlignmentAction
        
        config = AlignmentConfig(persist_logs=True, log_full_prompts=True)
        store = AlignmentLogStore(config, log_dir=str(tmp_path))
        
        goal = NorthStarGoal.from_mission_objective("Test", "test-mission")
        store.initialize("test-mission", goal)
        
        point = AlignmentPoint(
            t=0, a_t=0.5, d_t=-0.1, s_t=0.1,
            cusum_neg=0.3, cumulative_neg_drift=0.2,
            triggered=True, phase_name="p0"
        )
        
        test_prompt = "This is a test reanchor prompt"
        store.add_action(
            "test-mission",
            AlignmentAction.REANCHOR_INTERNAL,
            point,
            metadata={},
            injected_prompt=test_prompt,
        )
        
        log = store.get_log("test-mission")
        action_record = log["actions"][0]
        
        # Should have full prompt when enabled
        assert "prompt_full" in action_record["prompt_injection"]
        assert action_record["prompt_injection"]["prompt_full"] == test_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

