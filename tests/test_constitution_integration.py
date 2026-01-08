"""
Integration Tests for Constitution.

Tests:
- Shadow mode emits events but doesn't block
- Enforce mode blocks learning and stops deepening
- Full flow from phase start to phase end
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepthinker.constitution.config import (
    ConstitutionConfig,
    ConstitutionMode,
    get_constitution_config,
    reset_constitution_config,
)
from deepthinker.constitution.engine import (
    ConstitutionEngine,
    PhaseEvaluationContext,
    get_engine,
    clear_engine_cache,
)
from deepthinker.constitution.ledger import (
    ConstitutionLedger,
    get_ledger,
    clear_ledger_cache,
)
from deepthinker.constitution.enforcement import ConstitutionFlags
from deepthinker.constitution.types import BaselineSnapshot, ConstitutionEventType


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory."""
    return tmp_path


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    clear_engine_cache()
    clear_ledger_cache()
    reset_constitution_config()


class TestShadowMode:
    """Tests for shadow mode behavior."""
    
    def test_shadow_mode_emits_events(self, temp_dir):
        """Test that shadow mode emits events to ledger."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Create mock phase and state
        mock_phase = MagicMock()
        mock_phase.name = "research"
        mock_phase.deepening_rounds = 0
        
        mock_state = MagicMock()
        mock_state.mission_id = "test-mission"
        
        # Run through phase lifecycle
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        flags = engine.evaluate_phase(ctx, evidence_added=5, rounds_used=1)
        
        # Check events were written
        ledger = get_ledger("test-mission", config)
        events = ledger.read_all()
        
        # Should have at least baseline and score events
        assert len(events) >= 1
    
    def test_shadow_mode_logs_violations(self, temp_dir):
        """Test that violations are logged in shadow mode."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
            evidence_threshold=0.01,
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Create context that should trigger violation
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.7,  # Large increase
            evidence_added=0,  # No evidence
        )
        
        flags = engine.evaluate_phase(ctx, evidence_added=0, rounds_used=1)
        
        # Should flag but not block (shadow mode)
        assert not flags.ok
        assert len(flags.violations) > 0
        
        # Check violation was logged
        ledger = get_ledger("test-mission", config)
        violations = ledger.get_violations()
        assert len(violations) >= 1
    
    def test_shadow_mode_does_not_block(self, temp_dir):
        """Test that shadow mode records but doesn't enforce blocks."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # The engine should still identify violations
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.7,
            evidence_added=0,
        )
        
        flags = engine.evaluate_phase(ctx, evidence_added=0, rounds_used=1)
        
        # Engine flags the issue, but orchestrator decides whether to enforce
        # based on config.is_enforcing
        assert not config.is_enforcing


class TestEnforceMode:
    """Tests for enforce mode behavior."""
    
    def test_enforce_mode_blocks_learning(self, temp_dir):
        """Test that enforce mode blocks learning on violations."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.ENFORCE,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Create context with Goodhart violation
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.7,  # Large target improvement
            shadow_metrics_before={"contradiction_rate": 0.3},
            shadow_metrics_after={"contradiction_rate": 0.3},  # No shadow improvement
        )
        
        flags = engine.evaluate_phase(ctx, evidence_added=0, rounds_used=1)
        
        # Should block learning
        assert flags.block_learning
        assert config.is_enforcing
    
    def test_enforce_mode_stops_deepening(self, temp_dir):
        """Test that enforce mode stops deepening on no-free-lunch violation."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.ENFORCE,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Create context with unproductive depth
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.5,  # No improvement
            evidence_added=0,
            rounds_used=3,  # Multiple rounds
        )
        
        flags = engine.evaluate_phase(ctx, evidence_added=0, rounds_used=3)
        
        # Should stop deepening
        assert flags.stop_deepening


class TestFullFlow:
    """Tests for full phase lifecycle."""
    
    def test_full_phase_flow(self, temp_dir):
        """Test complete flow from phase start to end."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Create mock objects
        mock_phase = MagicMock()
        mock_phase.name = "research"
        mock_phase.deepening_rounds = 0
        
        mock_state = MagicMock()
        mock_state.mission_id = "test-mission"
        
        mock_scorecard = MagicMock()
        mock_scorecard.overall = 0.7
        mock_scorecard.goal_coverage = 0.8
        mock_scorecard.evidence_grounding = 0.6
        mock_scorecard.consistency = 0.7
        mock_scorecard.judge_disagreement = 0.1
        
        # Phase start
        ctx = engine.snapshot_baseline(mock_state, mock_phase, mock_scorecard)
        
        assert ctx.baseline is not None
        assert ctx.phase_id == "research"
        
        # Phase end
        flags = engine.evaluate_phase(
            ctx=ctx,
            scorecard=mock_scorecard,
            evidence_added=5,
            rounds_used=1,
            tools_used=["web_search", "code_exec"],
        )
        
        # Should complete without error
        assert isinstance(flags, ConstitutionFlags)
        
        # Check ledger has events
        ledger = get_ledger("test-mission", config)
        events = ledger.read_all()
        
        # Should have baseline, score, and possibly other events
        event_types = [e["event_type"] for e in events]
        assert ConstitutionEventType.BASELINE_SNAPSHOT.value in event_types
    
    def test_learning_update_recording(self, temp_dir):
        """Test recording of learning updates."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Record a learning update
        engine.record_learning_update(
            component="bandit",
            allowed=True,
            reward=0.5,
            arm="MEDIUM",
        )
        
        # Record a blocked update
        engine.record_learning_update(
            component="bandit",
            allowed=False,
            reason="constitution:goodhart_violation",
            reward=0.5,
            arm="LARGE",
        )
        
        # Check ledger
        ledger = get_ledger("test-mission", config)
        events = list(ledger.read_events(
            event_type=ConstitutionEventType.LEARNING_UPDATE
        ))
        
        assert len(events) == 2
        assert events[0]["allowed"]
        assert not events[1]["allowed"]


class TestOrchestratorIntegration:
    """Tests for integration with MissionOrchestrator patterns."""
    
    def test_engine_handles_missing_scorecard(self, temp_dir):
        """Test that engine handles missing scorecard gracefully."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "research"
        mock_phase.deepening_rounds = 0
        
        mock_state = MagicMock()
        mock_state.mission_id = "test-mission"
        
        # Snapshot without scorecard
        ctx = engine.snapshot_baseline(mock_state, mock_phase, scorecard=None)
        
        # Evaluate without scorecard
        flags = engine.evaluate_phase(ctx, scorecard=None)
        
        # Should not error
        assert isinstance(flags, ConstitutionFlags)
    
    def test_engine_handles_missing_claim_graph(self, temp_dir):
        """Test that engine handles missing claim graph gracefully."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=True,
            ledger_base_dir=str(temp_dir / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "research"
        mock_phase.deepening_rounds = 0
        
        mock_state = MagicMock()
        mock_state.mission_id = "test-mission"
        
        # Snapshot without claim graph
        ctx = engine.snapshot_baseline(mock_state, mock_phase, claim_graph=None)
        
        # Evaluate without claim graph
        flags = engine.evaluate_phase(ctx, claim_graph=None)
        
        # Should not error
        assert isinstance(flags, ConstitutionFlags)


class TestConfigurationModes:
    """Tests for configuration modes."""
    
    def test_off_mode_does_nothing(self):
        """Test that off mode does nothing."""
        config = ConstitutionConfig(mode=ConstitutionMode.OFF)
        engine = ConstitutionEngine("test-mission", config)
        
        assert not engine.is_enabled
        assert not engine.is_enforcing
        
        mock_phase = MagicMock()
        mock_phase.name = "test"
        mock_state = MagicMock()
        
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        flags = engine.evaluate_phase(ctx)
        
        # Should return all_ok without any checks
        assert flags.ok
    
    def test_mode_properties(self):
        """Test mode detection properties."""
        # Off mode
        config_off = ConstitutionConfig(mode=ConstitutionMode.OFF)
        assert not config_off.is_enabled
        assert not config_off.is_enforcing
        
        # Shadow mode
        config_shadow = ConstitutionConfig(mode=ConstitutionMode.SHADOW)
        assert config_shadow.is_enabled
        assert not config_shadow.is_enforcing
        
        # Enforce mode
        config_enforce = ConstitutionConfig(mode=ConstitutionMode.ENFORCE)
        assert config_enforce.is_enabled
        assert config_enforce.is_enforcing

