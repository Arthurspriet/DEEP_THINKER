"""
Tests for Bio Priors Package.

Tests cover:
- Schema validation
- Pattern loading (all 10 YAML cards)
- PressureSignals clamp/merge/scale behavior
- Engine determinism (golden test)
- No-op when disabled
- Advisory mode log structure
- Shadow mode "would_apply" diff
- Soft mode application
- Config from_env() parsing
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Import bio priors components
from deepthinker.bio_priors.config import (
    BioPriorConfig,
    get_bio_prior_config,
    reset_bio_prior_config,
    VALID_MODES,
)
from deepthinker.bio_priors.signals import (
    PressureSignals,
    BOUNDS,
)
from deepthinker.bio_priors.schema import (
    BioPattern,
    BioPatternValidationError,
    VALID_SYSTEM_MAPPING_KEYS,
    VALID_MATURITY,
)
from deepthinker.bio_priors.loader import (
    load_patterns,
    validate_pattern,
    validate_all_patterns,
    get_patterns_summary,
    PATTERNS_DIR,
)
from deepthinker.bio_priors.metrics import (
    BioPriorContext,
    RECENT_WINDOW_STEPS,
)
from deepthinker.bio_priors.engine import (
    BioPriorEngine,
    BioPriorOutput,
)
from deepthinker.bio_priors.integration import (
    apply_bio_pressures_to_deepening_plan,
    compute_would_apply_diff,
    V1_APPLIED_FIELDS,
    V1_LOG_ONLY_FIELDS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default disabled config."""
    return BioPriorConfig(enabled=False, mode="off")


@pytest.fixture
def enabled_advisory_config():
    """Config with advisory mode enabled."""
    return BioPriorConfig(enabled=True, mode="advisory", topk=3)


@pytest.fixture
def enabled_shadow_config():
    """Config with shadow mode enabled."""
    return BioPriorConfig(enabled=True, mode="shadow", topk=3)


@pytest.fixture
def enabled_soft_config():
    """Config with soft mode enabled."""
    return BioPriorConfig(enabled=True, mode="soft", topk=3)


@pytest.fixture
def sample_context():
    """Sample BioPriorContext for testing."""
    return BioPriorContext(
        phase="research",
        step_index=5,
        time_remaining_s=300.0,
        evidence_new_count_recent=0,  # Stagnation
        contradiction_rate=0.3,        # High contradiction
        uncertainty_trend=0.1,
        drift_score=0.4,               # High drift
        plan_branching_factor=2.0,
        last_step_evidence_delta=0,
        recent_window_steps=RECENT_WINDOW_STEPS,
    )


@pytest.fixture
def mock_deepening_plan():
    """Mock DeepeningPlan for testing integration."""
    @dataclass
    class MockDeepeningPlan:
        run_web_research: bool = False
        run_code_analysis: bool = False
        run_additional_council: bool = False
        focus_areas: List[str] = field(default_factory=list)
        max_deepening_rounds: int = 3
        reason: str = "test"
        
        @property
        def has_work(self) -> bool:
            return self.run_web_research or self.run_code_analysis or self.run_additional_council
    
    return MockDeepeningPlan(max_deepening_rounds=3)


# =============================================================================
# Config Tests
# =============================================================================

class TestBioPriorConfig:
    """Test BioPriorConfig behavior."""
    
    def test_default_config_is_disabled(self, default_config):
        """Default config should be disabled."""
        assert default_config.enabled is False
        assert default_config.mode == "off"
        assert default_config.is_active is False
        assert default_config.should_apply is False
    
    def test_enabled_but_off_mode_is_inactive(self):
        """Enabled with mode=off should be inactive."""
        config = BioPriorConfig(enabled=True, mode="off")
        assert config.is_active is False
        assert config.should_apply is False
    
    def test_advisory_mode(self, enabled_advisory_config):
        """Advisory mode should be active but not apply."""
        assert enabled_advisory_config.is_active is True
        assert enabled_advisory_config.should_apply is False
        assert enabled_advisory_config.should_compute_diff is False
    
    def test_shadow_mode(self, enabled_shadow_config):
        """Shadow mode should compute diff but not apply."""
        assert enabled_shadow_config.is_active is True
        assert enabled_shadow_config.should_apply is False
        assert enabled_shadow_config.should_compute_diff is True
    
    def test_soft_mode(self, enabled_soft_config):
        """Soft mode should be active and apply."""
        assert enabled_soft_config.is_active is True
        assert enabled_soft_config.should_apply is True
        assert enabled_soft_config.should_compute_diff is True
    
    def test_invalid_mode_defaults_to_off(self):
        """Invalid mode should default to off."""
        config = BioPriorConfig(enabled=True, mode="invalid")
        assert config.mode == "off"
        assert config.is_active is False
    
    def test_topk_validation(self):
        """topk must be >= 1."""
        config = BioPriorConfig(enabled=True, mode="soft", topk=0)
        assert config.topk == 3  # Default
    
    def test_max_pressure_clamping(self):
        """max_pressure should be clamped to [0, 2]."""
        config = BioPriorConfig(max_pressure=5.0)
        assert config.max_pressure == 2.0
        
        config2 = BioPriorConfig(max_pressure=-1.0)
        assert config2.max_pressure == 0.0
    
    def test_from_env(self):
        """Test loading config from environment variables."""
        reset_bio_prior_config()
        
        with patch.dict(os.environ, {
            "DEEPTHINKER_BIO_PRIORS_ENABLED": "true",
            "DEEPTHINKER_BIO_PRIORS_MODE": "soft",
            "DEEPTHINKER_BIO_PRIORS_TOPK": "5",
            "DEEPTHINKER_BIO_PRIORS_MAX_PRESSURE": "0.8",
        }):
            config = BioPriorConfig.from_env()
            assert config.enabled is True
            assert config.mode == "soft"
            assert config.topk == 5
            assert config.max_pressure == 0.8
        
        reset_bio_prior_config()
    
    def test_to_dict(self, enabled_soft_config):
        """Test config serialization."""
        d = enabled_soft_config.to_dict()
        assert d["enabled"] is True
        assert d["mode"] == "soft"
        assert d["topk"] == 3
        assert "is_active" in d
        assert "should_apply" in d


# =============================================================================
# PressureSignals Tests
# =============================================================================

class TestPressureSignals:
    """Test PressureSignals dataclass and methods."""
    
    def test_default_signals_are_neutral(self):
        """Default signals should be neutral/zero."""
        signals = PressureSignals()
        assert signals.exploration_bias_delta == 0.0
        assert signals.depth_budget_delta == 0
        assert signals.redundancy_check is False
        assert signals.force_falsification_step is False
        assert signals.branch_pruning_suggested is False
        assert signals.confidence_penalty_delta == 0.0
        assert signals.retrieval_diversify is False
        assert signals.council_diversity_min == 1
        assert signals.bounds_version == "v1"
        assert signals.intent == "modulation_only"
    
    def test_clamp_exploration_bias_delta(self):
        """exploration_bias_delta should clamp to [-0.2, +0.2]."""
        signals = PressureSignals(exploration_bias_delta=0.5)
        clamped = signals.clamp()
        assert clamped.exploration_bias_delta == 0.2
        
        signals2 = PressureSignals(exploration_bias_delta=-0.5)
        clamped2 = signals2.clamp()
        assert clamped2.exploration_bias_delta == -0.2
    
    def test_clamp_depth_budget_delta(self):
        """depth_budget_delta should clamp to [-2, +2]."""
        signals = PressureSignals(depth_budget_delta=5)
        clamped = signals.clamp()
        assert clamped.depth_budget_delta == 2
        
        signals2 = PressureSignals(depth_budget_delta=-5)
        clamped2 = signals2.clamp()
        assert clamped2.depth_budget_delta == -2
    
    def test_clamp_confidence_penalty_delta(self):
        """confidence_penalty_delta should clamp to [0.0, 0.2]."""
        signals = PressureSignals(confidence_penalty_delta=0.5)
        clamped = signals.clamp()
        assert clamped.confidence_penalty_delta == 0.2
        
        signals2 = PressureSignals(confidence_penalty_delta=-0.1)
        clamped2 = signals2.clamp()
        assert clamped2.confidence_penalty_delta == 0.0
    
    def test_clamp_council_diversity_min(self):
        """council_diversity_min should clamp to [1, 4]."""
        signals = PressureSignals(council_diversity_min=10)
        clamped = signals.clamp()
        assert clamped.council_diversity_min == 4
        
        signals2 = PressureSignals(council_diversity_min=0)
        clamped2 = signals2.clamp()
        assert clamped2.council_diversity_min == 1
    
    def test_merge_floats_weighted(self):
        """Merge should compute weighted average for floats."""
        s1 = PressureSignals(exploration_bias_delta=0.1)
        s2 = PressureSignals(exploration_bias_delta=0.2)
        
        merged = s1.merge(s2, weight_self=1.0, weight_other=1.0)
        assert abs(merged.exploration_bias_delta - 0.15) < 0.01
    
    def test_merge_ints_weighted(self):
        """Merge should compute weighted round for ints."""
        s1 = PressureSignals(depth_budget_delta=1)
        s2 = PressureSignals(depth_budget_delta=2)
        
        merged = s1.merge(s2, weight_self=1.0, weight_other=1.0)
        assert merged.depth_budget_delta == 2  # Round of 1.5
    
    def test_merge_bools_or(self):
        """Merge should OR booleans."""
        s1 = PressureSignals(redundancy_check=True, force_falsification_step=False)
        s2 = PressureSignals(redundancy_check=False, force_falsification_step=True)
        
        merged = s1.merge(s2)
        assert merged.redundancy_check is True
        assert merged.force_falsification_step is True
    
    def test_merge_and_clamp(self):
        """Merge should clamp result."""
        s1 = PressureSignals(exploration_bias_delta=0.2)
        s2 = PressureSignals(exploration_bias_delta=0.2)
        
        # Weighted sum would be 0.2, not 0.4 (due to normalization)
        merged = s1.merge(s2, weight_self=1.0, weight_other=1.0)
        assert merged.exploration_bias_delta <= 0.2
    
    def test_scale_floats_and_ints(self):
        """Scale should multiply float/int fields."""
        signals = PressureSignals(
            exploration_bias_delta=0.1,
            depth_budget_delta=2,
            confidence_penalty_delta=0.1,
        )
        
        scaled = signals.scale(0.5)
        assert abs(scaled.exploration_bias_delta - 0.05) < 0.01
        assert scaled.depth_budget_delta == 1
        assert abs(scaled.confidence_penalty_delta - 0.05) < 0.01
    
    def test_scale_bools_unchanged(self):
        """Scale should NOT change booleans."""
        signals = PressureSignals(
            redundancy_check=True,
            force_falsification_step=True,
        )
        
        scaled = signals.scale(0.0)
        assert scaled.redundancy_check is True
        assert scaled.force_falsification_step is True
    
    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict should roundtrip."""
        signals = PressureSignals(
            exploration_bias_delta=0.1,
            depth_budget_delta=1,
            redundancy_check=True,
        )
        
        d = signals.to_dict()
        restored = PressureSignals.from_dict(d)
        
        assert restored.exploration_bias_delta == signals.exploration_bias_delta
        assert restored.depth_budget_delta == signals.depth_budget_delta
        assert restored.redundancy_check == signals.redundancy_check
    
    def test_zero_factory(self):
        """zero() should return neutral signals."""
        signals = PressureSignals.zero()
        assert signals.exploration_bias_delta == 0.0
        assert signals.depth_budget_delta == 0
        assert not signals.has_any_signal()
    
    def test_has_any_signal(self):
        """has_any_signal should detect non-default values."""
        assert not PressureSignals().has_any_signal()
        assert PressureSignals(redundancy_check=True).has_any_signal()
        assert PressureSignals(depth_budget_delta=1).has_any_signal()


# =============================================================================
# Schema Tests
# =============================================================================

class TestBioPattern:
    """Test BioPattern schema validation."""
    
    def test_valid_pattern(self):
        """Valid pattern should pass validation."""
        pattern = BioPattern(
            id="BIO_TEST_001",
            name="Test Pattern",
            problem_class=["testing"],
            conditions=["test condition"],
            mechanism="test mechanism",
            system_mapping={"redundancy_check": True},
            maturity="draft",
            weight=0.5,
        )
        assert pattern.id == "BIO_TEST_001"
        assert pattern.maturity == "draft"
    
    def test_id_must_start_with_bio(self):
        """ID must start with BIO_."""
        with pytest.raises(BioPatternValidationError, match="id must start with 'BIO_'"):
            BioPattern(
                id="INVALID_001",
                name="Test",
                problem_class=["test"],
                conditions=["test"],
                mechanism="test",
                system_mapping={"redundancy_check": True},
            )
    
    def test_weight_bounds(self):
        """Weight must be in [0, 1]."""
        with pytest.raises(BioPatternValidationError, match="weight must be in"):
            BioPattern(
                id="BIO_TEST_001",
                name="Test",
                problem_class=["test"],
                conditions=["test"],
                mechanism="test",
                system_mapping={"redundancy_check": True},
                weight=1.5,
            )
    
    def test_maturity_must_be_valid(self):
        """Maturity must be draft or stable."""
        with pytest.raises(BioPatternValidationError, match="maturity must be"):
            BioPattern(
                id="BIO_TEST_001",
                name="Test",
                problem_class=["test"],
                conditions=["test"],
                mechanism="test",
                system_mapping={"redundancy_check": True},
                maturity="invalid",
            )
    
    def test_must_have_problem_class(self):
        """Must have at least one problem_class."""
        with pytest.raises(BioPatternValidationError, match="at least 1 problem_class"):
            BioPattern(
                id="BIO_TEST_001",
                name="Test",
                problem_class=[],
                conditions=["test"],
                mechanism="test",
                system_mapping={"redundancy_check": True},
            )
    
    def test_must_have_system_mapping(self):
        """Must have at least one system_mapping key."""
        with pytest.raises(BioPatternValidationError, match="at least 1 system_mapping"):
            BioPattern(
                id="BIO_TEST_001",
                name="Test",
                problem_class=["test"],
                conditions=["test"],
                mechanism="test",
                system_mapping={},
            )
    
    def test_system_mapping_keys_must_be_valid(self):
        """system_mapping keys must be PressureSignals fields."""
        with pytest.raises(BioPatternValidationError, match="invalid keys"):
            BioPattern(
                id="BIO_TEST_001",
                name="Test",
                problem_class=["test"],
                conditions=["test"],
                mechanism="test",
                system_mapping={"invalid_field": True},
            )
    
    def test_to_pressure_signals(self):
        """to_pressure_signals should convert system_mapping."""
        pattern = BioPattern(
            id="BIO_TEST_001",
            name="Test",
            problem_class=["test"],
            conditions=["test"],
            mechanism="test",
            system_mapping={
                "redundancy_check": True,
                "depth_budget_delta": 1,
            },
        )
        
        signals = pattern.to_pressure_signals()
        assert signals.redundancy_check is True
        assert signals.depth_budget_delta == 1


# =============================================================================
# Loader Tests
# =============================================================================

class TestLoader:
    """Test YAML loader functionality."""
    
    def test_load_all_patterns(self):
        """Should load all 10 YAML pattern cards."""
        patterns = load_patterns()
        assert len(patterns) == 10
    
    def test_all_patterns_have_bio_prefix(self):
        """All pattern IDs should start with BIO_."""
        patterns = load_patterns()
        for pattern in patterns:
            assert pattern.id.startswith("BIO_"), f"Pattern {pattern.id} missing BIO_ prefix"
    
    def test_all_patterns_have_valid_maturity(self):
        """All patterns should have valid maturity."""
        patterns = load_patterns()
        for pattern in patterns:
            assert pattern.maturity in VALID_MATURITY, f"Pattern {pattern.id} has invalid maturity"
    
    def test_all_patterns_have_valid_weights(self):
        """All patterns should have weights in [0.2, 0.6]."""
        patterns = load_patterns()
        for pattern in patterns:
            assert 0.2 <= pattern.weight <= 0.6, f"Pattern {pattern.id} has weight outside recommended range"
    
    def test_validate_all_patterns_passes(self):
        """validate_all_patterns should pass for all cards."""
        all_valid, errors = validate_all_patterns()
        assert all_valid, f"Validation errors: {errors}"
    
    def test_patterns_summary(self):
        """get_patterns_summary should return all patterns."""
        summaries = get_patterns_summary()
        assert len(summaries) == 10
        for summary in summaries:
            assert "id" in summary
            assert "name" in summary
            assert "maturity" in summary
            assert "weight" in summary
    
    def test_expected_patterns_exist(self):
        """Check that expected pattern IDs exist."""
        patterns = load_patterns()
        ids = {p.id for p in patterns}
        
        expected = {
            "BIO_IMMUNE_001",
            "BIO_HOMEOSTASIS_001",
            "BIO_FORAGING_001",
            "BIO_ANT_TRAILS_001",
            "BIO_FLOCKING_001",
            "BIO_REDUNDANCY_001",
            "BIO_METABOLIC_BUDGET_001",
            "BIO_ERROR_CORRECTION_001",
            "BIO_DEVELOPMENTAL_STAGES_001",
            "BIO_PREDATOR_PREY_001",
        }
        
        assert ids == expected, f"Missing patterns: {expected - ids}"


# =============================================================================
# Engine Tests
# =============================================================================

class TestBioPriorEngine:
    """Test BioPriorEngine functionality."""
    
    def test_engine_disabled_returns_empty(self, default_config, sample_context):
        """Disabled engine should return empty output."""
        engine = BioPriorEngine(config=default_config)
        output = engine.evaluate(sample_context)
        
        assert output.mode == "off"
        assert output.applied is False
        assert len(output.selected_patterns) == 0
        assert not output.signals.has_any_signal()
    
    def test_engine_advisory_mode(self, enabled_advisory_config, sample_context):
        """Advisory mode should evaluate but not apply."""
        engine = BioPriorEngine(config=enabled_advisory_config)
        output = engine.evaluate(sample_context)
        
        assert output.mode == "advisory"
        assert output.applied is False
        assert len(output.applied_fields) == 0
        assert len(output.selected_patterns) > 0
    
    def test_engine_shadow_mode(self, enabled_shadow_config, sample_context):
        """Shadow mode should compute would_apply diff."""
        engine = BioPriorEngine(config=enabled_shadow_config)
        output = engine.evaluate(sample_context)
        
        assert output.mode == "shadow"
        assert output.applied is False
        assert "would_apply_diff" in output.trace
    
    def test_engine_soft_mode(self, enabled_soft_config, sample_context):
        """Soft mode should mark as applied."""
        engine = BioPriorEngine(config=enabled_soft_config)
        output = engine.evaluate(sample_context)
        
        assert output.mode == "soft"
        assert output.applied is True
        # applied_fields may be empty if no signals affect v1 fields
    
    def test_engine_topk_selection(self, enabled_soft_config, sample_context):
        """Engine should select topk patterns."""
        config = BioPriorConfig(enabled=True, mode="soft", topk=2)
        engine = BioPriorEngine(config=config)
        output = engine.evaluate(sample_context)
        
        assert len(output.selected_patterns) <= 2
    
    def test_engine_determinism(self, enabled_soft_config, sample_context):
        """Engine should be deterministic - same input, same output."""
        engine = BioPriorEngine(config=enabled_soft_config)
        
        output1 = engine.evaluate(sample_context)
        output2 = engine.evaluate(sample_context)
        
        # Same patterns selected
        assert output1.selected_patterns == output2.selected_patterns
        
        # Same signals
        assert output1.signals == output2.signals
        
        # Same trace structure
        assert output1.trace.get("patterns_selected") == output2.trace.get("patterns_selected")
    
    def test_engine_golden_determinism(self):
        """
        Golden test: fixed context MUST produce identical, ordered output.
        
        This test ensures the engine is fully deterministic and reproducible.
        """
        config = BioPriorConfig(enabled=True, mode="soft", topk=3)
        engine = BioPriorEngine(config=config)
        
        # Fixed context fixture (simulates stagnation + drift)
        ctx = BioPriorContext(
            phase="research",
            step_index=5,
            time_remaining_s=300.0,
            evidence_new_count_recent=0,
            contradiction_rate=0.3,
            uncertainty_trend=0.1,
            drift_score=0.4,
            plan_branching_factor=2.0,
            last_step_evidence_delta=0,
            recent_window_steps=RECENT_WINDOW_STEPS,
        )
        
        output = engine.evaluate(ctx)
        
        # Assert expected structure
        assert len(output.selected_patterns) == 3
        assert output.signals.bounds_version == "v1"
        assert output.signals.intent == "modulation_only"
        
        # Assert trace includes required fields
        assert "missing_metrics" in output.trace
        assert "recent_window_steps" in output.trace
        assert output.trace["recent_window_steps"] == RECENT_WINDOW_STEPS
        
        # Run twice to verify determinism
        output2 = engine.evaluate(ctx)
        assert output.selected_patterns == output2.selected_patterns
        assert output.signals == output2.signals
        
        # Verify patterns are ordered by score * weight (descending)
        for i in range(len(output.selected_patterns) - 1):
            p1 = output.selected_patterns[i]
            p2 = output.selected_patterns[i + 1]
            assert p1["score"] * p1["weight"] >= p2["score"] * p2["weight"]
    
    def test_engine_trace_includes_context(self, enabled_soft_config, sample_context):
        """Trace should include context snapshot."""
        engine = BioPriorEngine(config=enabled_soft_config)
        output = engine.evaluate(sample_context)
        
        assert "context_snapshot" in output.trace
        assert output.trace["context_snapshot"]["phase"] == "research"
    
    def test_engine_trace_includes_missing_metrics(self, enabled_soft_config):
        """Trace should include list of missing metrics."""
        ctx = BioPriorContext(
            phase="test",
            step_index=0,
            # All optional metrics are None
        )
        
        engine = BioPriorEngine(config=enabled_soft_config)
        output = engine.evaluate(ctx)
        
        assert "missing_metrics" in output.trace
        assert len(output.trace["missing_metrics"]) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test bio prior integration functions."""
    
    def test_apply_to_deepening_plan_depth_delta(self, mock_deepening_plan):
        """Should apply depth_budget_delta to max_deepening_rounds."""
        signals = PressureSignals(depth_budget_delta=1)
        
        original_rounds = mock_deepening_plan.max_deepening_rounds
        applied = apply_bio_pressures_to_deepening_plan(mock_deepening_plan, signals)
        
        assert "depth_budget_delta" in applied
        assert mock_deepening_plan.max_deepening_rounds == original_rounds + 1
    
    def test_apply_to_deepening_plan_respects_bounds(self, mock_deepening_plan):
        """Should respect max_deepening_rounds_limit."""
        signals = PressureSignals(depth_budget_delta=10)
        
        applied = apply_bio_pressures_to_deepening_plan(
            mock_deepening_plan,
            signals,
            max_deepening_rounds_limit=5,
        )
        
        assert mock_deepening_plan.max_deepening_rounds <= 5
    
    def test_apply_with_zero_delta_no_change(self, mock_deepening_plan):
        """Zero delta should not change plan."""
        signals = PressureSignals(depth_budget_delta=0)
        
        original_rounds = mock_deepening_plan.max_deepening_rounds
        applied = apply_bio_pressures_to_deepening_plan(mock_deepening_plan, signals)
        
        assert applied == []
        assert mock_deepening_plan.max_deepening_rounds == original_rounds
    
    def test_compute_would_apply_diff(self, mock_deepening_plan):
        """Should compute would_apply diff correctly."""
        signals = PressureSignals(
            depth_budget_delta=1,
            redundancy_check=True,
            force_falsification_step=True,
        )
        
        diff = compute_would_apply_diff(mock_deepening_plan, signals)
        
        assert "v1_applied" in diff
        assert "v1_log_only" in diff
        assert "depth_budget_delta" in diff["v1_applied"]
        assert "redundancy_check" in diff["v1_log_only"]
        assert "force_falsification_step" in diff["v1_log_only"]
    
    def test_v1_scope_fields(self):
        """V1 should only apply depth_budget_delta."""
        assert "depth_budget_delta" in V1_APPLIED_FIELDS
        assert len(V1_APPLIED_FIELDS) == 1
        
        # All other fields should be log-only
        assert "redundancy_check" in V1_LOG_ONLY_FIELDS
        assert "force_falsification_step" in V1_LOG_ONLY_FIELDS
        assert "branch_pruning_suggested" in V1_LOG_ONLY_FIELDS


# =============================================================================
# Context Tests
# =============================================================================

class TestBioPriorContext:
    """Test BioPriorContext helper properties."""
    
    def test_is_early_phase(self):
        """is_early_phase should be True for step_index < 3."""
        ctx = BioPriorContext(phase="test", step_index=2)
        assert ctx.is_early_phase is True
        
        ctx2 = BioPriorContext(phase="test", step_index=5)
        assert ctx2.is_early_phase is False
    
    def test_is_late_phase(self):
        """is_late_phase should be True for high step_index or low time."""
        ctx = BioPriorContext(phase="test", step_index=11)
        assert ctx.is_late_phase is True
        
        ctx2 = BioPriorContext(phase="test", step_index=5, time_remaining_s=30)
        assert ctx2.is_late_phase is True
    
    def test_has_stagnation_signal(self):
        """has_stagnation_signal should detect zero evidence."""
        ctx = BioPriorContext(phase="test", step_index=5, evidence_new_count_recent=0)
        assert ctx.has_stagnation_signal is True
        
        ctx2 = BioPriorContext(phase="test", step_index=5, evidence_new_count_recent=5)
        assert ctx2.has_stagnation_signal is False
    
    def test_has_high_contradiction(self):
        """has_high_contradiction should detect rate > 0.2."""
        ctx = BioPriorContext(phase="test", step_index=5, contradiction_rate=0.3)
        assert ctx.has_high_contradiction is True
        
        ctx2 = BioPriorContext(phase="test", step_index=5, contradiction_rate=0.1)
        assert ctx2.has_high_contradiction is False
    
    def test_get_missing_metrics(self):
        """get_missing_metrics should list None fields."""
        ctx = BioPriorContext(
            phase="test",
            step_index=5,
            time_remaining_s=None,
            contradiction_rate=0.2,
        )
        
        missing = ctx.get_missing_metrics()
        assert "time_remaining_s" in missing
        assert "contradiction_rate" not in missing


# =============================================================================
# Constitution Integration Tests
# =============================================================================

class TestConstitutionIntegration:
    """Test constitution types integration."""
    
    def test_prior_influence_event_exists(self):
        """PriorInfluenceEvent should be importable."""
        from deepthinker.constitution.types import (
            PriorInfluenceEvent,
            ConstitutionEventType,
            is_evidence_event,
        )
        
        assert ConstitutionEventType.PRIOR_INFLUENCE.value == "prior_influence"
    
    def test_prior_influence_event_is_non_evidence(self):
        """PriorInfluenceEvent should have is_evidence=False."""
        from deepthinker.constitution.types import PriorInfluenceEvent, is_evidence_event
        
        event = PriorInfluenceEvent(
            mission_id="test",
            phase_id="test",
            mode="soft",
            selected_patterns=["BIO_TEST_001"],
        )
        
        assert event.is_evidence is False
        assert event.affects_confidence is False
        assert is_evidence_event(event) is False
    
    def test_prior_influence_event_cannot_be_evidence(self):
        """is_evidence should always be False even if set to True."""
        from deepthinker.constitution.types import PriorInfluenceEvent
        
        # Attempt to create with is_evidence=True (should be forced to False)
        event = PriorInfluenceEvent(
            mission_id="test",
            phase_id="test",
            is_evidence=True,  # Should be ignored
        )
        
        assert event.is_evidence is False
    
    def test_prior_influence_event_to_dict(self):
        """to_dict should always have is_evidence=False."""
        from deepthinker.constitution.types import PriorInfluenceEvent
        
        event = PriorInfluenceEvent(
            mission_id="test",
            phase_id="test",
            mode="soft",
        )
        
        d = event.to_dict()
        assert d["is_evidence"] is False
        assert d["affects_confidence"] is False
        assert d["source"] == "bio_priors"


# =============================================================================
# CLI Tests
# =============================================================================

class TestCLI:
    """Test CLI commands."""
    
    def test_cli_validate_succeeds(self):
        """CLI validate should pass for valid patterns."""
        from deepthinker.bio_priors.cli import cmd_validate
        exit_code = cmd_validate()
        assert exit_code == 0
    
    def test_cli_list_succeeds(self):
        """CLI list should succeed."""
        from deepthinker.bio_priors.cli import cmd_list
        exit_code = cmd_list()
        assert exit_code == 0
    
    def test_cli_run_demo_succeeds(self):
        """CLI run-demo should succeed."""
        from deepthinker.bio_priors.cli import cmd_run_demo
        exit_code = cmd_run_demo()
        assert exit_code == 0

