"""
Tests for the Normative Control Layer (Governance).

Tests governance evaluation, rule engine, and verdict generation.
"""

import pytest
from deepthinker.governance import (
    NormativeController,
    NormativeVerdict,
    VerdictStatus,
    RecommendedAction,
    Violation,
    ViolationType,
    RuleEngine,
    GovernanceConfig,
    load_governance_config,
    GOVERNANCE_PHASE_CONTRACTS,
    GovernancePhaseContract,
    get_governance_contract,
    create_violation,
)


class TestViolation:
    """Tests for Violation dataclass."""
    
    def test_violation_creation(self):
        """Test basic violation creation."""
        violation = Violation(
            type=ViolationType.STRUCTURAL_MISSING_SCENARIOS,
            severity=0.9,
            description="Expected 3 scenarios, found 0",
            phase_name="synthesis",
            is_hard=True,
        )
        
        assert violation.type == ViolationType.STRUCTURAL_MISSING_SCENARIOS
        assert violation.severity == 0.9
        assert violation.is_hard is True
        assert "synthesis" in violation.phase_name
    
    def test_violation_severity_clamping(self):
        """Test that severity is clamped to valid range."""
        violation = Violation(
            type=ViolationType.PHASE_CONTAMINATION,
            severity=1.5,  # Invalid - should be clamped
            description="Test",
            phase_name="test",
        )
        assert violation.severity == 1.0
        
        violation2 = Violation(
            type=ViolationType.PHASE_CONTAMINATION,
            severity=-0.5,  # Invalid - should be clamped
            description="Test",
            phase_name="test",
        )
        assert violation2.severity == 0.0
    
    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = Violation(
            type=ViolationType.EPISTEMIC_LOW_SOURCE_COUNT,
            severity=0.5,
            description="Found 1 source, required 3",
            phase_name="reconnaissance",
        )
        
        data = violation.to_dict()
        
        assert data["type"] == "epistemic_low_source_count"
        assert data["severity"] == 0.5
        assert "reconnaissance" in data["phase_name"]
    
    def test_create_violation_factory(self):
        """Test violation factory function."""
        violation = create_violation(
            ViolationType.STRUCTURAL_MISSING_SCENARIOS,
            "Missing scenarios",
            "synthesis",
        )
        
        # Should use default severity from mapping
        assert violation.severity == 0.9
        assert violation.is_hard is True


class TestGovernancePhaseContracts:
    """Tests for governance phase contracts."""
    
    def test_synthesis_contract(self):
        """Test synthesis phase contract."""
        contract = GOVERNANCE_PHASE_CONTRACTS["synthesis"]
        
        assert "scenarios" in contract.allowed
        assert "recommendations" in contract.allowed
        assert "raw_research" in contract.forbidden
        assert contract.required_scenario_count == 3
        assert contract.strictness_weight == 1.0
    
    def test_reconnaissance_contract(self):
        """Test reconnaissance phase contract."""
        contract = GOVERNANCE_PHASE_CONTRACTS["reconnaissance"]
        
        assert "sources" in contract.allowed
        assert "recommendations" in contract.forbidden
        assert contract.min_sources == 2
        assert contract.requires_web_search is True
    
    def test_get_governance_contract_direct(self):
        """Test direct contract retrieval."""
        contract = get_governance_contract("synthesis")
        assert contract.phase_name == "synthesis"
        assert contract.required_scenario_count == 3
    
    def test_get_governance_contract_fuzzy(self):
        """Test fuzzy contract matching."""
        contract = get_governance_contract("recon_phase")
        assert "reconnaissance" in contract.phase_name
    
    def test_get_governance_contract_unknown(self):
        """Test fallback for unknown phases."""
        contract = get_governance_contract("unknown_phase_xyz")
        assert contract.phase_name == "unknown_phase_xyz"
        assert len(contract.forbidden) == 0


class TestGovernanceConfig:
    """Tests for governance configuration."""
    
    def test_config_loading(self):
        """Test config loads from YAML."""
        config = load_governance_config()
        
        assert config.warn_threshold == 0.3
        assert config.block_threshold == 0.7
        assert "structural_missing_scenarios" in config.hard_rules
    
    def test_config_severity_lookup(self):
        """Test severity lookup."""
        config = load_governance_config()
        
        severity = config.get_severity("structural_missing_scenarios")
        assert severity == 0.9
        
        # Unknown type returns default
        severity_unknown = config.get_severity("unknown_type")
        assert severity_unknown == 0.5
    
    def test_config_phase_strictness(self):
        """Test phase strictness lookup."""
        config = load_governance_config()
        
        assert config.get_phase_strictness("synthesis") == 1.0
        assert config.get_phase_strictness("analysis") == 0.7
        assert config.get_phase_strictness("unknown") == 0.6  # Default
    
    def test_config_resource_modifier(self):
        """Test resource modifier lookup."""
        config = load_governance_config()
        
        modifier = config.get_resource_modifier("high")
        assert modifier["strictness_multiplier"] == 1.2
        
        modifier_critical = config.get_resource_modifier("critical")
        assert modifier_critical["max_retries"] == 0


class TestRuleEngine:
    """Tests for the rule engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a rule engine instance."""
        return RuleEngine()
    
    def test_evaluate_empty_output(self, engine):
        """Test evaluation of empty output."""
        violations = engine.evaluate_rules(
            phase_name="synthesis",
            phase_output={},
            mission_state=None,
            gpu_pressure="low",
        )
        
        # Should detect structural issues
        assert len(violations) > 0
    
    def test_evaluate_missing_scenarios(self, engine):
        """Test detection of missing scenarios."""
        violations = engine.evaluate_rules(
            phase_name="synthesis",
            phase_output={
                "scenarios": [{"name": "Only One", "description": "Single scenario"}],  # Only 1, need 3
                "content": "Some output with insufficient scenarios",
            },
            mission_state=None,
            gpu_pressure="low",
        )
        
        # Should have structural violation for wrong scenario count
        structural_violations = [
            v for v in violations
            if v.type == ViolationType.STRUCTURAL_MISSING_SCENARIOS
        ]
        assert len(structural_violations) > 0
    
    def test_evaluate_insufficient_content(self, engine):
        """Test detection of insufficient content."""
        violations = engine.evaluate_rules(
            phase_name="synthesis",
            phase_output={"content": "Too short"},
            mission_state=None,
            gpu_pressure="low",
        )
        
        # Should have structural violation for insufficient content
        insufficient_violations = [
            v for v in violations
            if v.type == ViolationType.STRUCTURAL_INSUFFICIENT_CONTENT
        ]
        assert len(insufficient_violations) > 0
    
    def test_evaluate_valid_synthesis(self, engine):
        """Test evaluation of valid synthesis output."""
        good_output = {
            'scenarios': [
                {'name': 'Scenario A', 'description': 'Detailed description of scenario A with comprehensive analysis of implications and outcomes for stakeholders.'},
                {'name': 'Scenario B', 'description': 'Detailed description of scenario B covering different aspects and alternative perspectives on the situation.'},
                {'name': 'Scenario C', 'description': 'Detailed description of scenario C with unique viewpoint and distinct drivers from other scenarios.'}
            ],
            'synthesis_report': '''
# Strategic Analysis

## Recommendations
Based on our analysis, we provide the following recommendations...
''' * 3  # Make it long enough
        }
        
        violations = engine.evaluate_rules(
            phase_name="synthesis",
            phase_output=good_output,
            mission_state=None,
            gpu_pressure="low",
        )
        
        # Should have minimal violations (possibly none)
        hard_violations = [v for v in violations if v.is_hard]
        assert len(hard_violations) == 0
    
    def test_resource_modifier_application(self, engine):
        """Test that resource pressure affects severity."""
        output = {"content": "Test output without scenarios"}
        
        violations_low = engine.evaluate_rules(
            phase_name="synthesis",
            phase_output=output,
            mission_state=None,
            gpu_pressure="low",
        )
        
        # Reset history
        engine.clear_history()
        
        violations_critical = engine.evaluate_rules(
            phase_name="synthesis",
            phase_output=output,
            mission_state=None,
            gpu_pressure="critical",
        )
        
        # Critical pressure should increase severity
        if violations_low and violations_critical:
            avg_severity_low = sum(v.severity for v in violations_low) / len(violations_low)
            avg_severity_critical = sum(v.severity for v in violations_critical) / len(violations_critical)
            assert avg_severity_critical >= avg_severity_low


class TestNormativeController:
    """Tests for the NormativeController."""
    
    @pytest.fixture
    def controller(self):
        """Create a controller instance."""
        return NormativeController()
    
    def test_evaluate_block_verdict(self, controller):
        """Test that invalid output produces BLOCK verdict."""
        verdict = controller.evaluate(
            phase_name="synthesis",
            phase_output={"content": "Short invalid output"},
            mission_state=None,
        )
        
        assert verdict.status == VerdictStatus.BLOCK
        assert len(verdict.violations) > 0
        assert verdict.recommended_action in [
            RecommendedAction.RETRY_PHASE,
            RecommendedAction.SCOPE_REDUCTION,
        ]
    
    def test_evaluate_allow_verdict(self, controller):
        """Test that valid output produces ALLOW verdict."""
        good_output = {
            'scenarios': [
                {'name': 'Scenario A: Tech Growth', 'description': 'This scenario explores a future where technology investment drives economic growth across all sectors with significant implications for employment.'},
                {'name': 'Scenario B: Regulatory Focus', 'description': 'This scenario examines increased regulatory oversight creating new compliance requirements and changing competitive dynamics in major markets.'},
                {'name': 'Scenario C: Market Evolution', 'description': 'This scenario considers natural market evolution driven by consumer preferences and emerging business models transforming industries.'}
            ],
            'synthesis_report': '''
# Comprehensive Strategic Analysis

## Executive Summary
This analysis provides three distinct scenarios for strategic planning.

## Scenario Details
Each scenario represents a different trajectory based on key drivers.

## Recommendations
Based on this analysis, organizations should prepare for multiple futures.
''' * 2
        }
        
        verdict = controller.evaluate(
            phase_name="synthesis",
            phase_output=good_output,
            mission_state=None,
        )
        
        # Should ALLOW or WARN (not BLOCK)
        assert verdict.status in [VerdictStatus.ALLOW, VerdictStatus.WARN]
    
    def test_verdict_to_dict(self, controller):
        """Test verdict serialization."""
        verdict = controller.evaluate(
            phase_name="analysis",
            phase_output={"content": "Analysis content " * 50},
            mission_state=None,
        )
        
        data = verdict.to_dict()
        
        assert "status" in data
        assert "violations" in data
        assert "recommended_action" in data
        assert "epistemic_risk" in data
    
    def test_retry_tracking(self, controller):
        """Test retry count tracking."""
        controller.record_retry("test_phase")
        controller.record_retry("test_phase")
        
        assert controller.get_retry_count("test_phase") == 2
        assert controller.get_retry_count("other_phase") == 0
        
        controller.reset_phase_tracking("test_phase")
        assert controller.get_retry_count("test_phase") == 0
    
    def test_governance_report(self, controller):
        """Test governance summary report."""
        # Evaluate something to generate data
        controller.evaluate(
            phase_name="synthesis",
            phase_output={"content": "Test"},
            mission_state=None,
        )
        
        report = controller.get_governance_report()
        
        assert "epistemic_risk_score" in report
        assert "total_violations" in report
        assert "phases_blocked" in report
    
    def test_controller_reset(self, controller):
        """Test controller state reset."""
        controller.record_retry("phase_a")
        controller.evaluate(
            phase_name="synthesis",
            phase_output={"content": "Test"},
            mission_state=None,
        )
        
        controller.reset()
        
        assert controller.get_retry_count("phase_a") == 0
        report = controller.get_governance_report()
        assert report["total_violations"] == 0


class TestVerdictStatus:
    """Tests for verdict status enum."""
    
    def test_verdict_status_values(self):
        """Test verdict status values."""
        assert VerdictStatus.ALLOW.value == "ALLOW"
        assert VerdictStatus.WARN.value == "WARN"
        assert VerdictStatus.BLOCK.value == "BLOCK"


class TestRecommendedAction:
    """Tests for recommended action enum."""
    
    def test_action_values(self):
        """Test action values."""
        assert RecommendedAction.NONE.value == "NONE"
        assert RecommendedAction.RETRY_PHASE.value == "RETRY_PHASE"
        assert RecommendedAction.FORCE_WEB_SEARCH.value == "FORCE_WEB_SEARCH"
        assert RecommendedAction.SCOPE_REDUCTION.value == "SCOPE_REDUCTION"


class TestNormativeVerdictMethods:
    """Tests for NormativeVerdict methods."""
    
    def test_has_hard_violations(self):
        """Test hard violation detection."""
        verdict = NormativeVerdict(
            status=VerdictStatus.BLOCK,
            violations=[
                Violation(
                    type=ViolationType.STRUCTURAL_MISSING_SCENARIOS,
                    severity=0.9,
                    description="Test",
                    phase_name="test",
                    is_hard=True,
                ),
                Violation(
                    type=ViolationType.PHASE_CONTAMINATION,
                    severity=0.4,
                    description="Test",
                    phase_name="test",
                    is_hard=False,
                ),
            ],
        )
        
        assert verdict.has_hard_violations() is True
        assert len(verdict.get_hard_violations()) == 1
    
    def test_get_violations_by_type(self):
        """Test filtering violations by type."""
        verdict = NormativeVerdict(
            status=VerdictStatus.WARN,
            violations=[
                Violation(
                    type=ViolationType.EPISTEMIC_LOW_SOURCE_COUNT,
                    severity=0.5,
                    description="Test",
                    phase_name="test",
                ),
                Violation(
                    type=ViolationType.EPISTEMIC_LOW_SOURCE_COUNT,
                    severity=0.5,
                    description="Test 2",
                    phase_name="test",
                ),
                Violation(
                    type=ViolationType.PHASE_CONTAMINATION,
                    severity=0.4,
                    description="Test",
                    phase_name="test",
                ),
            ],
        )
        
        epistemic = verdict.get_violations_by_type(ViolationType.EPISTEMIC_LOW_SOURCE_COUNT)
        assert len(epistemic) == 2

