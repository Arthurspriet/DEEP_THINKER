"""
Tests for Constitution Invariant Checking.

Tests all four constitutional invariants:
1. Conservation of Evidence
2. Monotonic Uncertainty Under Compression
3. No-Free-Lunch Depth
4. Anti-Gaming Divergence (Goodhart Shield)
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from deepthinker.constitution.config import ConstitutionConfig, ConstitutionMode
from deepthinker.constitution.engine import (
    ConstitutionEngine,
    PhaseEvaluationContext,
)
from deepthinker.constitution.enforcement import ConstitutionFlags, EnforcementAction
from deepthinker.constitution.types import BaselineSnapshot


@pytest.fixture
def constitution_config():
    """Create a test constitution config in shadow mode."""
    return ConstitutionConfig(
        mode=ConstitutionMode.SHADOW,
        evidence_threshold=0.01,
        min_evidence_for_score_increase=1,
        min_gain_per_round=0.02,
        max_unproductive_rounds=2,
        target_improvement_threshold=0.03,
        shadow_improvement_threshold=0.01,
        ledger_enabled=False,  # Disable ledger for unit tests
    )


@pytest.fixture
def engine(constitution_config):
    """Create a test constitution engine."""
    return ConstitutionEngine(
        mission_id="test-mission",
        config=constitution_config,
    )


class TestEvidenceConservation:
    """Tests for Conservation of Evidence invariant."""
    
    def test_score_increase_without_evidence_flags_violation(self, engine):
        """Test that score increase without evidence is flagged."""
        # Create context with baseline
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(
                overall_score=0.5,
                evidence_count=5,
            ),
            score_after=0.6,  # 0.1 increase
            evidence_added=0,  # No new evidence
        )
        
        flags = engine._check_evidence_conservation(ctx)
        
        assert not flags.ok
        assert flags.block_learning
        assert len(flags.violations) > 0
        assert "evidence" in flags.violations[0].lower()
    
    def test_score_increase_with_evidence_passes(self, engine):
        """Test that score increase with evidence passes."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(
                overall_score=0.5,
                evidence_count=5,
            ),
            score_after=0.6,
            evidence_added=3,  # New evidence added
        )
        
        flags = engine._check_evidence_conservation(ctx)
        
        assert flags.ok
        assert len(flags.violations) == 0
    
    def test_score_increase_with_contradiction_reduction_passes(self, engine):
        """Test that score increase with contradiction reduction passes."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(
                overall_score=0.5,
                contradiction_rate=0.3,
            ),
            score_after=0.6,
            evidence_added=0,
            contradiction_rate_after=0.1,  # Reduced from 0.3 to 0.1
        )
        
        flags = engine._check_evidence_conservation(ctx)
        
        assert flags.ok
    
    def test_small_score_increase_allowed(self, engine):
        """Test that small score increases below threshold are allowed."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.505,  # Very small increase
            evidence_added=0,
        )
        
        flags = engine._check_evidence_conservation(ctx)
        
        assert flags.ok  # Below threshold


class TestNoFreeLunch:
    """Tests for No-Free-Lunch Depth invariant."""
    
    def test_unproductive_rounds_flag_violation(self, engine):
        """Test that unproductive rounds are flagged."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.5,  # No improvement
            evidence_added=0,
            rounds_used=3,  # Multiple rounds
        )
        
        flags = engine._check_no_free_lunch(ctx)
        
        assert not flags.ok
        assert flags.stop_deepening
        assert len(flags.violations) > 0
    
    def test_productive_rounds_pass(self, engine):
        """Test that productive rounds pass."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.6,  # Improvement
            evidence_added=5,
            rounds_used=3,
        )
        
        flags = engine._check_no_free_lunch(ctx)
        
        assert flags.ok
    
    def test_single_round_always_passes(self, engine):
        """Test that single rounds always pass (no depth to check)."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.5,  # No improvement
            evidence_added=0,
            rounds_used=1,  # Single round
        )
        
        flags = engine._check_no_free_lunch(ctx)
        
        assert flags.ok
    
    def test_evidence_without_score_improvement_passes(self, engine):
        """Test that adding evidence counts as gain even without score improvement."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.5,  # No score change
            evidence_added=10,  # But evidence added
            rounds_used=3,
        )
        
        flags = engine._check_no_free_lunch(ctx)
        
        assert flags.ok


class TestGoodhartShield:
    """Tests for Anti-Gaming Divergence (Goodhart Shield) invariant."""
    
    def test_target_up_shadow_flat_flags_divergence(self, engine):
        """Test that target improvement without shadow improvement is flagged."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.6,  # Target improved
            shadow_metrics_before={
                "contradiction_rate": 0.3,
                "judge_disagreement": 0.2,
            },
            shadow_metrics_after={
                "contradiction_rate": 0.3,  # No improvement
                "judge_disagreement": 0.2,  # No improvement
            },
        )
        
        flags = engine._check_goodhart_divergence(ctx)
        
        assert not flags.ok
        assert flags.block_learning
        assert flags.force_evidence_mode
    
    def test_target_and_shadow_both_improve_passes(self, engine):
        """Test that both target and shadow improving passes."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.6,  # Target improved
            shadow_metrics_before={
                "contradiction_rate": 0.3,
                "judge_disagreement": 0.2,
            },
            shadow_metrics_after={
                "contradiction_rate": 0.2,  # Improved (lower is better)
                "judge_disagreement": 0.15,
            },
        )
        
        flags = engine._check_goodhart_divergence(ctx)
        
        assert flags.ok
    
    def test_no_target_improvement_skips_check(self, engine):
        """Test that no target improvement skips the check."""
        ctx = PhaseEvaluationContext(
            mission_id="test-mission",
            phase_id="research",
            baseline=BaselineSnapshot(overall_score=0.5),
            score_after=0.5,  # No target improvement
            shadow_metrics_before={"contradiction_rate": 0.3},
            shadow_metrics_after={"contradiction_rate": 0.4},  # Shadow got worse
        )
        
        flags = engine._check_goodhart_divergence(ctx)
        
        assert flags.ok  # No check because target didn't improve


class TestMonotonicUncertainty:
    """Tests for Monotonic Uncertainty Under Compression invariant."""
    
    def test_uncertainty_reduction_without_validation_flags(self, engine):
        """Test that uncertainty reduction without validation is flagged."""
        flags = engine.check_compression_uncertainty(
            uncertainty_before=0.5,
            uncertainty_after=0.2,  # Large reduction
            validated=False,
        )
        
        assert not flags.ok
        assert flags.warn
    
    def test_validated_uncertainty_reduction_passes(self, engine):
        """Test that validated uncertainty reduction passes."""
        flags = engine.check_compression_uncertainty(
            uncertainty_before=0.5,
            uncertainty_after=0.2,
            validated=True,
        )
        
        assert flags.ok
    
    def test_small_uncertainty_reduction_allowed(self, engine):
        """Test that small uncertainty reduction is allowed."""
        flags = engine.check_compression_uncertainty(
            uncertainty_before=0.5,
            uncertainty_after=0.48,  # Very small reduction
            validated=False,
        )
        
        assert flags.ok


class TestConstitutionFlagsLogic:
    """Tests for ConstitutionFlags behavior."""
    
    def test_all_ok_creates_clean_flags(self):
        """Test that all_ok creates clean flags."""
        flags = ConstitutionFlags.all_ok()
        
        assert flags.ok
        assert not flags.warn
        assert not flags.block_learning
        assert not flags.stop_deepening
        assert len(flags.violations) == 0
    
    def test_add_violation_updates_flags(self):
        """Test that add_violation updates flags correctly."""
        flags = ConstitutionFlags.all_ok()
        flags.add_violation("Test violation", EnforcementAction.BLOCK_LEARNING)
        
        assert not flags.ok
        assert flags.block_learning
        assert len(flags.violations) == 1
    
    def test_merge_combines_flags(self):
        """Test that merge combines flags with OR logic."""
        flags1 = ConstitutionFlags(ok=True, block_learning=True)
        flags2 = ConstitutionFlags(ok=True, stop_deepening=True)
        
        merged = flags1.merge(flags2)
        
        assert merged.block_learning
        assert merged.stop_deepening
    
    def test_merge_or_on_booleans(self):
        """Test OR logic on boolean fields."""
        flags1 = ConstitutionFlags(ok=True, warn=False)
        flags2 = ConstitutionFlags(ok=False, warn=True)
        
        merged = flags1.merge(flags2)
        
        assert not merged.ok  # AND for ok
        assert merged.warn  # OR for warn


class TestEngineIntegration:
    """Integration tests for the full engine."""
    
    def test_evaluate_phase_combines_all_invariants(self, engine):
        """Test that evaluate_phase checks all invariants."""
        # Create a mock phase and state
        mock_phase = MagicMock()
        mock_phase.name = "research"
        mock_phase.deepening_rounds = 0
        
        mock_state = MagicMock()
        mock_state.mission_id = "test-mission"
        
        # Snapshot baseline
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        
        # Evaluate - should run without errors
        flags = engine.evaluate_phase(
            ctx=ctx,
            evidence_added=0,
            rounds_used=1,
        )
        
        assert isinstance(flags, ConstitutionFlags)
    
    def test_engine_disabled_returns_all_ok(self):
        """Test that disabled engine returns all_ok."""
        config = ConstitutionConfig(mode=ConstitutionMode.OFF)
        engine = ConstitutionEngine("test", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "test"
        mock_state = MagicMock()
        
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        flags = engine.evaluate_phase(ctx)
        
        assert flags.ok




