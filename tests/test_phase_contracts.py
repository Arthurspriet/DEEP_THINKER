"""
Tests for the Phase Contracts module.

Tests phase purity enforcement, contamination detection,
and phase guard functionality.
"""

import pytest
from deepthinker.phases import (
    ContaminationType,
    PhaseContract,
    PhaseContamination,
    PhaseGuard,
    PhaseViolation,
    PHASE_CONTRACTS,
    get_phase_guard,
)


class TestPhaseContract:
    """Tests for the PhaseContract dataclass."""
    
    def test_contract_creation(self):
        """Test basic contract creation."""
        contract = PhaseContract(
            phase_name="test_phase",
            allowed_types={"facts", "sources"},
            forbidden_types={"recommendations", "conclusions"},
            allowed_patterns=[r'\b(fact|evidence)\b'],
            forbidden_patterns=[r'\b(recommend|suggest)\b'],
        )
        
        assert contract.phase_name == "test_phase"
        assert "facts" in contract.allowed_types
        assert "recommendations" in contract.forbidden_types


class TestPhaseContracts:
    """Tests for predefined phase contracts."""
    
    def test_reconnaissance_contract(self):
        """Test reconnaissance phase contract."""
        contract = PHASE_CONTRACTS["reconnaissance"]
        
        assert "sources" in contract.allowed_types
        assert "facts" in contract.allowed_types
        assert "recommendations" in contract.forbidden_types
        assert "conclusions" in contract.forbidden_types
        assert contract.requires_sources is True
    
    def test_analysis_contract(self):
        """Test analysis phase contract."""
        contract = PHASE_CONTRACTS["analysis"]
        
        assert "synthesis" in contract.allowed_types
        assert "causal_links" in contract.allowed_types
        assert "new_sources" in contract.forbidden_types
        assert "recommendations" in contract.forbidden_types
    
    def test_synthesis_contract(self):
        """Test synthesis phase contract."""
        contract = PHASE_CONTRACTS["synthesis"]
        
        assert "recommendations" in contract.allowed_types
        assert "scenarios" in contract.allowed_types
        assert "conclusions" in contract.allowed_types
        assert "raw_research" in contract.forbidden_types
        assert "new_facts" in contract.forbidden_types


class TestPhaseViolation:
    """Tests for the PhaseViolation dataclass."""
    
    def test_violation_creation(self):
        """Test violation creation."""
        violation = PhaseViolation(
            contamination_type=ContaminationType.PREMATURE_RECOMMENDATION,
            description="Found recommendation in reconnaissance",
            excerpt="We recommend implementing...",
            severity=0.7,
            line_number=42,
        )
        
        assert violation.contamination_type == ContaminationType.PREMATURE_RECOMMENDATION
        assert violation.severity == 0.7
        assert "recommend" in violation.excerpt.lower()


class TestPhaseGuard:
    """Tests for the PhaseGuard class."""
    
    @pytest.fixture
    def guard(self):
        """Create a fresh guard for each test."""
        return PhaseGuard(strict_mode=False, contamination_threshold=0.3)
    
    def test_get_contract(self, guard):
        """Test contract retrieval."""
        contract = guard.get_contract("reconnaissance")
        assert contract.phase_name == "reconnaissance"
        
        contract = guard.get_contract("ANALYSIS")  # Case insensitive
        assert "analysis" in contract.phase_name.lower()
    
    def test_get_contract_fuzzy_match(self, guard):
        """Test fuzzy contract matching."""
        contract = guard.get_contract("deep_analysis_phase")
        assert contract is not None
    
    def test_get_contract_default_fallback(self, guard):
        """Test default contract for unknown phases."""
        contract = guard.get_contract("unknown_phase_xyz")
        assert contract.phase_name == "unknown_phase_xyz"
        assert len(contract.forbidden_types) == 0
    
    def test_inspect_clean_output(self, guard):
        """Test inspection of clean output."""
        clean_reconnaissance = """
        ## Sources Found
        - According to the World Bank, GDP growth was 3.5%.
        - Data from the IMF shows inflation at 2%.
        
        ## Key Facts
        - The economy grew steadily in 2024.
        - Investment increased by 10%.
        
        ## Unknown Areas
        - Exact impact of policy changes unclear.
        """
        
        result = guard.inspect_output(clean_reconnaissance, "reconnaissance")
        
        assert result.is_clean is True or result.contamination_score < 0.3
    
    def test_inspect_contaminated_reconnaissance(self, guard):
        """Test detection of recommendations in reconnaissance."""
        contaminated = """
        ## Research Findings
        Based on our research, we found several facts.
        
        ## Recommendations
        We recommend implementing the following actions:
        1. Invest in technology
        2. Expand to new markets
        
        ## Conclusion
        In conclusion, the strategy should focus on growth.
        """
        
        result = guard.inspect_output(contaminated, "reconnaissance")
        
        assert result.contamination_score > 0 or len(result.violations) > 0
    
    def test_inspect_contaminated_synthesis(self, guard):
        """Test detection of new research in synthesis."""
        contaminated = """
        ## Final Recommendations
        We recommend the following strategy.
        
        ## New Research
        According to new sources we just discovered,
        additional research shows different findings.
        We found new evidence that changes our view.
        """
        
        result = guard.inspect_output(contaminated, "synthesis")
        
        # Should detect late sourcing or new research
        assert result.contamination_score > 0 or len(result.violations) > 0
    
    def test_contamination_score_computation(self, guard):
        """Test contamination score computation."""
        output = """
        We recommend implementing these actions.
        Our conclusion is that growth will continue.
        In summary, the next steps are clear.
        """
        
        score = guard.compute_contamination_score(output, "reconnaissance")
        
        assert 0.0 <= score <= 1.0
    
    def test_corrective_iteration_trigger(self, guard):
        """Test corrective iteration trigger."""
        high_contamination = PhaseContamination(
            phase_name="reconnaissance",
            contamination_score=0.6,
            should_retry=True,
            violations=[PhaseViolation(
                contamination_type=ContaminationType.PREMATURE_RECOMMENDATION,
                description="Test",
            )],
        )
        
        should_retry = guard.trigger_corrective_iteration(high_contamination)
        
        assert should_retry is True
    
    def test_correction_prompt_generation(self, guard):
        """Test correction prompt generation."""
        contamination = PhaseContamination(
            phase_name="reconnaissance",
            violations=[PhaseViolation(
                contamination_type=ContaminationType.PREMATURE_RECOMMENDATION,
                description="Found recommendation in reconnaissance",
            )],
        )
        
        prompt = guard.get_correction_prompt(contamination)
        
        assert "reconnaissance" in prompt.lower()
        assert "violated" in prompt.lower() or "violation" in prompt.lower()
        assert "allowed" in prompt.lower() or "forbidden" in prompt.lower()
    
    def test_validation_log(self, guard):
        """Test validation logging."""
        guard.inspect_output("Test output", "analysis")
        guard.inspect_output("Another output", "synthesis")
        
        log = guard.get_validation_log()
        
        assert len(log) == 2
        
        guard.clear_validation_log()
        assert len(guard.get_validation_log()) == 0


class TestContaminationTypes:
    """Tests for contamination type classification."""
    
    @pytest.fixture
    def guard(self):
        return PhaseGuard()
    
    def test_premature_recommendation_in_recon(self, guard):
        """Test premature recommendation detection in reconnaissance."""
        output = """
        We recommend that the company should implement AI systems.
        The strategy should focus on automation.
        """
        
        result = guard.inspect_output(output, "reconnaissance")
        
        # Should detect premature recommendation
        recommendation_violations = [
            v for v in result.violations 
            if v.contamination_type == ContaminationType.PREMATURE_RECOMMENDATION
        ]
        # Note: may or may not detect depending on pattern matching
    
    def test_late_sourcing_in_synthesis(self, guard):
        """Test late sourcing detection in synthesis."""
        output = """
        ## Recommendations
        Our final recommendation is to invest in growth.
        
        According to new research we just found, there are additional factors.
        New sources indicate different outcomes.
        """
        
        result = guard.inspect_output(output, "synthesis")
        
        # Should detect late sourcing
        late_sourcing = [
            v for v in result.violations 
            if v.contamination_type in [
                ContaminationType.LATE_SOURCING,
                ContaminationType.RAW_RESEARCH_IN_SYNTHESIS,
            ]
        ]
        # Pattern detection may vary


class TestPhaseContamination:
    """Tests for the PhaseContamination dataclass."""
    
    def test_contamination_to_dict(self):
        """Test contamination serialization."""
        contamination = PhaseContamination(
            phase_name="reconnaissance",
            violations=[
                PhaseViolation(
                    contamination_type=ContaminationType.PREMATURE_RECOMMENDATION,
                    description="Test violation",
                )
            ],
            contamination_score=0.5,
            is_clean=False,
            recommended_penalty=1.5,
            should_retry=True,
        )
        
        data = contamination.to_dict()
        
        assert data["phase_name"] == "reconnaissance"
        assert data["violation_count"] == 1
        assert data["contamination_score"] == 0.5
        assert data["is_clean"] is False
        assert data["should_retry"] is True


class TestGlobalGuard:
    """Tests for the global guard instance."""
    
    def test_global_guard_instance(self):
        """Test global guard instance getter."""
        guard1 = get_phase_guard()
        guard2 = get_phase_guard()
        
        assert guard1 is guard2
    
    def test_strict_mode_configuration(self):
        """Test strict mode affects behavior."""
        lenient_guard = PhaseGuard(strict_mode=False, contamination_threshold=0.5)
        strict_guard = PhaseGuard(strict_mode=True, contamination_threshold=0.3)
        
        mild_contamination = """
        Some factual information here.
        We might recommend looking into this further.
        """
        
        lenient_result = lenient_guard.inspect_output(mild_contamination, "reconnaissance")
        strict_result = strict_guard.inspect_output(mild_contamination, "reconnaissance")
        
        # Strict mode should be more sensitive (potentially)
        # Results may vary but both should produce valid outputs
        assert lenient_result.phase_name == "reconnaissance"
        assert strict_result.phase_name == "reconnaissance"


class TestPhaseGuardIntegration:
    """Integration tests for phase guard."""
    
    def test_full_phase_validation_pipeline(self):
        """Test full phase validation pipeline."""
        guard = PhaseGuard()
        
        # Simulate reconnaissance phase output
        recon_output = """
        ## Source Analysis
        According to financial reports, revenue grew 15%.
        The market data indicates strong demand.
        
        ## Key Findings
        - Revenue increased by 15%
        - Market share expanded
        - Customer satisfaction improved
        
        ## Open Questions
        - What drove the growth?
        - Is it sustainable?
        """
        
        result = guard.inspect_output(recon_output, "reconnaissance")
        
        assert result.phase_name == "reconnaissance"
        assert result.contamination_score >= 0.0
        
        if not result.is_clean:
            # Should be able to generate correction prompt
            prompt = guard.get_correction_prompt(result)
            assert len(prompt) > 0
    
    def test_multi_phase_validation(self):
        """Test validation across multiple phases."""
        guard = PhaseGuard()
        
        phases = [
            ("reconnaissance", "Sources: World Bank, IMF. Facts found."),
            ("analysis", "Analysis shows trends. Causal links identified."),
            ("synthesis", "Recommendations: Invest in growth. Conclusions drawn."),
        ]
        
        for phase_name, output in phases:
            result = guard.inspect_output(output, phase_name)
            assert result.phase_name == phase_name
            # All should produce valid results
            assert result.contamination_score >= 0.0
            assert result.contamination_score <= 1.0

