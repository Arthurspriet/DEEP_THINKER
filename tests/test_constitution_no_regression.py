"""
No-Regression Tests for Constitution.

Tests that:
- Constitution OFF mode produces no side effects
- No new artifacts when disabled
- Existing behavior unchanged
"""

import pytest
import os
import tempfile
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
    clear_engine_cache,
)
from deepthinker.constitution.ledger import (
    ConstitutionLedger,
    clear_ledger_cache,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    clear_engine_cache()
    clear_ledger_cache()
    reset_constitution_config()


class TestOffModeNoSideEffects:
    """Tests that OFF mode produces no side effects."""
    
    def test_off_mode_creates_no_files(self, tmp_path):
        """Test that OFF mode creates no ledger files."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.OFF,
            ledger_base_dir=str(tmp_path / "constitution"),
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        # Run through a phase
        mock_phase = MagicMock()
        mock_phase.name = "research"
        mock_phase.deepening_rounds = 0
        
        mock_state = MagicMock()
        mock_state.mission_id = "test-mission"
        
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        engine.evaluate_phase(ctx, evidence_added=5, rounds_used=1)
        
        # No constitution directory should be created
        constitution_dir = tmp_path / "constitution"
        if constitution_dir.exists():
            # If it exists, it should be empty
            assert len(list(constitution_dir.glob("*"))) == 0
    
    def test_off_mode_engine_disabled(self):
        """Test that OFF mode engine reports disabled."""
        config = ConstitutionConfig(mode=ConstitutionMode.OFF)
        engine = ConstitutionEngine("test-mission", config)
        
        assert not engine.is_enabled
        assert not engine.is_enforcing
    
    def test_off_mode_always_returns_ok(self):
        """Test that OFF mode always returns all_ok."""
        config = ConstitutionConfig(mode=ConstitutionMode.OFF)
        engine = ConstitutionEngine("test-mission", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "test"
        mock_phase.deepening_rounds = 0
        mock_state = MagicMock()
        
        # Even with conditions that would normally trigger violations
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        ctx.score_after = 0.9  # Huge improvement
        ctx.evidence_added = 0  # No evidence
        
        flags = engine.evaluate_phase(ctx)
        
        assert flags.ok
        assert not flags.block_learning
        assert not flags.stop_deepening
    
    def test_off_mode_no_logging(self, tmp_path, caplog):
        """Test that OFF mode produces minimal logging."""
        import logging
        
        config = ConstitutionConfig(
            mode=ConstitutionMode.OFF,
            ledger_base_dir=str(tmp_path / "constitution"),
        )
        
        with caplog.at_level(logging.DEBUG, logger="deepthinker.constitution"):
            engine = ConstitutionEngine("test-mission", config)
            
            mock_phase = MagicMock()
            mock_phase.name = "test"
            mock_phase.deepening_rounds = 0
            mock_state = MagicMock()
            
            ctx = engine.snapshot_baseline(mock_state, mock_phase)
            engine.evaluate_phase(ctx)
        
        # Should have very few or no constitution-specific logs
        constitution_logs = [r for r in caplog.records 
                           if "CONSTITUTION" in r.message]
        # Allow for debug-level init messages, but no violation/event logs
        assert all("violation" not in r.message.lower() 
                  for r in constitution_logs)


class TestLedgerDisabled:
    """Tests for disabled ledger behavior."""
    
    def test_ledger_disabled_writes_nothing(self, tmp_path):
        """Test that disabled ledger writes nothing."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=False,
            ledger_base_dir=str(tmp_path / "constitution"),
        )
        
        ledger = ConstitutionLedger("test-mission", config, tmp_path / "constitution")
        
        from deepthinker.constitution.types import EvidenceEvent
        
        # Try to write
        result = ledger.write_event(EvidenceEvent(
            mission_id="test",
            phase_id="test",
            count_added=5,
        ))
        
        assert not result  # Write should return False
        
        # No file should exist
        assert not ledger.ledger_path.exists()
    
    def test_ledger_disabled_reads_empty(self, tmp_path):
        """Test that disabled ledger reads empty."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_enabled=False,
            ledger_base_dir=str(tmp_path / "constitution"),
        )
        
        ledger = ConstitutionLedger("test-mission", config, tmp_path / "constitution")
        
        events = ledger.read_all()
        assert events == []


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing components."""
    
    def test_config_from_env_defaults_off(self):
        """Test that environment config defaults to OFF."""
        # Clear any environment variables
        for key in list(os.environ.keys()):
            if key.startswith("CONSTITUTION_"):
                del os.environ[key]
        
        reset_constitution_config()
        config = get_constitution_config()
        
        assert config.mode == ConstitutionMode.OFF
    
    def test_judge_ensemble_without_constitution(self):
        """Test that JudgeEnsemble works without constitution."""
        # This tests the import guard in judge_ensemble
        try:
            from deepthinker.metrics.judge_ensemble import JudgeEnsemble
            # If import works, constitution integration is optional
            assert True
        except ImportError:
            pytest.fail("JudgeEnsemble should import without constitution")
    
    def test_bandit_update_without_blocked_reason(self):
        """Test that bandit.update() works without blocked_reason parameter."""
        from deepthinker.bandits.generalized_bandit import GeneralizedBandit, BanditConfig
        
        config = BanditConfig(
            enabled=True,
            freeze_mode=False,
            store_dir=str(Path(tempfile.gettempdir()) / "test_bandits"),
        )
        
        bandit = GeneralizedBandit(
            decision_class="test_class",
            arms=["A", "B", "C"],
            config=config,
        )
        
        # Should work without blocked_reason (backward compatible)
        result = bandit.update("A", 0.5)
        
        # Result depends on config, but should not error
        assert isinstance(result, bool)


class TestInvariantToggling:
    """Tests for individual invariant toggling."""
    
    def test_disable_evidence_conservation(self, tmp_path):
        """Test that evidence conservation can be disabled."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            evidence_conservation_enabled=False,
            ledger_enabled=False,
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "test"
        mock_phase.deepening_rounds = 0
        mock_state = MagicMock()
        
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        ctx.baseline.overall_score = 0.5
        ctx.score_after = 0.9  # Would normally trigger
        ctx.evidence_added = 0
        
        flags = engine.evaluate_phase(ctx)
        
        # Evidence conservation is disabled, so no violation from it
        evidence_violations = [v for v in flags.violations 
                             if "evidence" in v.lower()]
        assert len(evidence_violations) == 0
    
    def test_disable_no_free_lunch(self, tmp_path):
        """Test that no-free-lunch can be disabled."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            no_free_lunch_enabled=False,
            ledger_enabled=False,
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "test"
        mock_phase.deepening_rounds = 0
        mock_state = MagicMock()
        
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        ctx.rounds_used = 5  # Would normally trigger
        ctx.evidence_added = 0
        ctx.score_after = ctx.baseline.overall_score  # No improvement
        
        flags = engine.evaluate_phase(ctx, rounds_used=5)
        
        # Should not trigger stop_deepening
        assert not flags.stop_deepening
    
    def test_disable_goodhart_shield(self, tmp_path):
        """Test that Goodhart shield can be disabled."""
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            goodhart_shield_enabled=False,
            ledger_enabled=False,
        )
        
        engine = ConstitutionEngine("test-mission", config)
        
        mock_phase = MagicMock()
        mock_phase.name = "test"
        mock_phase.deepening_rounds = 0
        mock_state = MagicMock()
        
        ctx = engine.snapshot_baseline(mock_state, mock_phase)
        ctx.baseline.overall_score = 0.5
        ctx.score_after = 0.8  # Large improvement
        ctx.shadow_metrics_before = {"contradiction_rate": 0.3}
        ctx.shadow_metrics_after = {"contradiction_rate": 0.3}  # No improvement
        
        flags = engine.evaluate_phase(ctx)
        
        # Should not trigger block_learning from Goodhart
        goodhart_violations = [v for v in flags.violations 
                              if "goodhart" in v.lower() or "divergence" in v.lower()]
        assert len(goodhart_violations) == 0




