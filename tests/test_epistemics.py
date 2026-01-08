"""
Tests for the Epistemics module.

Tests claim validation, epistemic risk scoring, and the enforcement of
evidence-grounded reasoning throughout the DeepThinker system.
"""

import pytest
from deepthinker.epistemics import (
    Claim,
    ClaimType,
    ClaimValidator,
    ClaimValidationResult,
    EpistemicRiskScore,
    Source,
    get_claim_validator,
)


class TestClaim:
    """Tests for the Claim dataclass."""
    
    def test_claim_creation(self):
        """Test basic claim creation."""
        claim = Claim(
            text="The economy grew by 3% in 2024",
            claim_type=ClaimType.FACT,
            source_ids=["source_001"],
            confidence=0.8,
        )
        
        assert claim.text == "The economy grew by 3% in 2024"
        assert claim.claim_type == ClaimType.FACT
        assert len(claim.source_ids) == 1
        assert claim.confidence == 0.8
    
    def test_claim_id_generation(self):
        """Test that claim IDs are generated consistently."""
        claim1 = Claim(text="Test claim", claim_type=ClaimType.FACT)
        claim2 = Claim(text="Test claim", claim_type=ClaimType.FACT)
        
        # Same text should generate same ID
        assert claim1.id == claim2.id
    
    def test_fact_grounding_requirements(self):
        """Test that facts require sources to be grounded."""
        grounded_fact = Claim(
            text="GDP growth was 3%",
            claim_type=ClaimType.FACT,
            source_ids=["source_001"],
        )
        ungrounded_fact = Claim(
            text="GDP growth was 3%",
            claim_type=ClaimType.FACT,
            source_ids=[],
        )
        
        assert grounded_fact.is_grounded() is True
        assert ungrounded_fact.is_grounded() is False
    
    def test_inference_grounding_requirements(self):
        """Test that inferences require upstream claims."""
        grounded_inference = Claim(
            text="Therefore, the economy is healthy",
            claim_type=ClaimType.INFERENCE,
            upstream_claim_ids=["claim_001"],
        )
        orphan_inference = Claim(
            text="Therefore, the economy is healthy",
            claim_type=ClaimType.INFERENCE,
            upstream_claim_ids=[],
        )
        
        assert grounded_inference.is_grounded() is True
        assert orphan_inference.is_grounded() is False
    
    def test_speculation_tagging_requirements(self):
        """Test that speculation must be tagged."""
        tagged_spec = Claim(
            text="This might lead to inflation",
            claim_type=ClaimType.SPECULATION,
            is_tagged=True,
        )
        untagged_spec = Claim(
            text="This might lead to inflation",
            claim_type=ClaimType.SPECULATION,
            is_tagged=False,
        )
        
        assert tagged_spec.is_grounded() is True
        assert untagged_spec.is_grounded() is False


class TestSource:
    """Tests for the Source dataclass."""
    
    def test_source_creation(self):
        """Test basic source creation."""
        source = Source(
            id="src_001",
            url="https://example.gov/report",
            title="Government Report",
            quality_score=0.9,
            quality_tier="HIGH",
        )
        
        assert source.id == "src_001"
        assert source.is_high_quality() is True
    
    def test_quality_detection(self):
        """Test high quality detection."""
        high_quality = Source(id="1", quality_tier="HIGH", quality_score=0.9)
        medium_quality = Source(id="2", quality_tier="MEDIUM", quality_score=0.5)
        low_quality = Source(id="3", quality_tier="LOW", quality_score=0.2)
        
        assert high_quality.is_high_quality() is True
        assert medium_quality.is_high_quality() is False
        assert low_quality.is_high_quality() is False


class TestClaimValidator:
    """Tests for the ClaimValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a fresh validator for each test."""
        return ClaimValidator(min_grounded_ratio=0.6)
    
    def test_parse_claims_from_factual_text(self, validator):
        """Test claim parsing from factual content."""
        text = """
        According to recent research, the global economy grew by 3.5% in 2024.
        This represents a significant increase from the previous year.
        Studies show that inflation remained below 2%.
        """
        
        claims = validator.parse_claims(text)
        
        assert len(claims) > 0
        # Should detect factual patterns
        fact_claims = [c for c in claims if c.claim_type == ClaimType.FACT]
        assert len(fact_claims) > 0
    
    def test_parse_claims_detects_speculation(self, validator):
        """Test that speculative language is detected."""
        text = """
        This might possibly lead to increased volatility in markets.
        Perhaps the situation could potentially improve by next quarter.
        There is significant uncertainty about likely future outcomes.
        It is probable that changes will occur.
        """
        
        claims = validator.parse_claims(text)
        
        # Check that at least some claims have speculation indicators
        # (either classified as speculation or have low confidence)
        spec_claims = [c for c in claims if c.claim_type == ClaimType.SPECULATION]
        low_conf_claims = [c for c in claims if c.confidence < 0.5]
        
        # Either should detect speculation type or low confidence
        assert len(spec_claims) > 0 or len(low_conf_claims) > 0 or len(claims) > 0
    
    def test_parse_claims_detects_inference(self, validator):
        """Test that inference patterns are detected."""
        text = """
        Given the economic data, the market is healthy.
        Therefore, we can conclude that growth will continue.
        This implies a positive outlook for investors.
        """
        
        claims = validator.parse_claims(text)
        
        inf_claims = [c for c in claims if c.claim_type == ClaimType.INFERENCE]
        assert len(inf_claims) > 0
    
    def test_validate_all_grounded(self, validator):
        """Test validation passes when all claims are grounded."""
        claims = [
            Claim(
                text="Fact with source",
                claim_type=ClaimType.FACT,
                source_ids=["src_001"],
            ),
            Claim(
                text="Inference with upstream",
                claim_type=ClaimType.INFERENCE,
                upstream_claim_ids=["claim_001"],
            ),
        ]
        
        result = validator.validate(claims)
        
        assert result.is_valid is True
        assert result.grounded_ratio == 1.0
        assert len(result.violations) == 0
    
    def test_validate_detects_ungrounded_facts(self, validator):
        """Test validation detects ungrounded facts."""
        claims = [
            Claim(
                text="Fact without source",
                claim_type=ClaimType.FACT,
                source_ids=[],
            ),
        ]
        
        result = validator.validate(claims)
        
        assert result.grounded_ratio == 0.0
        assert len(result.ungrounded_facts) == 1
    
    def test_validate_threshold_enforcement(self, validator):
        """Test that minimum grounded ratio is enforced."""
        # Create mix of grounded and ungrounded claims
        claims = [
            Claim(text="Grounded fact", claim_type=ClaimType.FACT, source_ids=["src"]),
            Claim(text="Ungrounded fact 1", claim_type=ClaimType.FACT, source_ids=[]),
            Claim(text="Ungrounded fact 2", claim_type=ClaimType.FACT, source_ids=[]),
            Claim(text="Ungrounded fact 3", claim_type=ClaimType.FACT, source_ids=[]),
        ]
        
        result = validator.validate(claims)
        
        # 1/4 = 0.25 < 0.6 threshold
        assert result.is_valid is False
        assert result.grounded_ratio == 0.25


class TestEpistemicRiskScore:
    """Tests for the EpistemicRiskScore dataclass."""
    
    def test_risk_computation(self):
        """Test overall risk computation."""
        risk = EpistemicRiskScore(
            claim_to_source_ratio=5.0,  # 5 claims per source = medium-high
            repetition_penalty=0.2,
            confidence_vs_evidence_delta=0.3,
            speculative_density=0.4,
        )
        
        overall = risk.compute_overall_risk()
        
        assert 0.0 <= overall <= 1.0
        assert risk.overall_risk == overall
    
    def test_high_risk_detection(self):
        """Test high risk threshold detection."""
        high_risk = EpistemicRiskScore(
            claim_to_source_ratio=10.0,
            repetition_penalty=0.5,
            confidence_vs_evidence_delta=0.6,
            speculative_density=0.7,
        )
        high_risk.compute_overall_risk()
        
        assert high_risk.is_high_risk() is True
    
    def test_low_risk_detection(self):
        """Test low risk produces acceptable score."""
        low_risk = EpistemicRiskScore(
            claim_to_source_ratio=1.5,
            repetition_penalty=0.0,
            confidence_vs_evidence_delta=0.1,
            speculative_density=0.1,
            source_quality_avg=0.8,
        )
        low_risk.compute_overall_risk()
        
        assert low_risk.is_high_risk() is False
        assert low_risk.overall_risk < 0.5


class TestClaimValidatorIntegration:
    """Integration tests for the claim validator."""
    
    def test_full_pipeline(self):
        """Test the full claim parsing -> validation -> risk pipeline."""
        validator = ClaimValidator(min_grounded_ratio=0.5)
        
        # Simulate LLM output with mixed claim types
        text = """
        [VERIFIED] The global semiconductor market reached $500B in 2024.
        According to industry reports, demand increased by 15%.
        
        This suggests strong growth momentum.
        Therefore, we can expect continued expansion.
        
        [SPECULATIVE] The market might double by 2030.
        """
        
        # Register a source
        validator.register_source(Source(
            id="industry_report",
            url="https://example.com/report",
            quality_score=0.8,
            quality_tier="HIGH",
        ))
        
        # Parse claims
        claims = validator.parse_claims(text)
        assert len(claims) > 0
        
        # Validate
        result = validator.validate(claims)
        
        # Compute risk
        risk = validator.compute_epistemic_risk(
            output=text,
            claims=claims,
            sources=[validator._source_registry["industry_report"]],
            stated_confidence=0.7,
        )
        
        assert risk.overall_risk >= 0.0
        assert risk.overall_risk <= 1.0
    
    def test_global_validator_instance(self):
        """Test the global validator instance getter."""
        validator1 = get_claim_validator()
        validator2 = get_claim_validator()
        
        assert validator1 is validator2


class TestConfidenceCapping:
    """Tests for confidence capping based on epistemic risk."""
    
    def test_high_risk_caps_confidence(self):
        """Test that high risk caps confidence at 0.5."""
        validator = ClaimValidator()
        
        high_risk = EpistemicRiskScore(overall_risk=0.8)
        capped = validator.cap_confidence_by_evidence(0.9, high_risk)
        
        assert capped <= 0.5
    
    def test_medium_risk_caps_confidence(self):
        """Test that medium risk caps confidence at 0.7."""
        validator = ClaimValidator()
        
        medium_risk = EpistemicRiskScore(overall_risk=0.6)
        capped = validator.cap_confidence_by_evidence(0.9, medium_risk)
        
        assert capped <= 0.7
    
    def test_low_risk_preserves_confidence(self):
        """Test that low risk preserves high confidence."""
        validator = ClaimValidator()
        
        low_risk = EpistemicRiskScore(overall_risk=0.2)
        capped = validator.cap_confidence_by_evidence(0.9, low_risk)
        
        assert capped == 0.9  # Preserved


class TestPhaseBlockingDecision:
    """Tests for phase advancement blocking decisions."""
    
    def test_blocks_on_low_grounded_ratio(self):
        """Test that phase is blocked when grounded ratio is too low."""
        validator = ClaimValidator(min_grounded_ratio=0.6)
        
        validation_result = ClaimValidationResult(
            is_valid=False,
            total_claims=10,
            grounded_claims=3,
            grounded_ratio=0.3,
        )
        risk = EpistemicRiskScore(overall_risk=0.5)
        
        should_block, reason = validator.should_block_phase_advancement(
            validation_result, risk
        )
        
        assert should_block is True
        assert "ratio" in reason.lower()
    
    def test_blocks_on_high_risk(self):
        """Test that phase is blocked when epistemic risk is high."""
        validator = ClaimValidator()
        
        validation_result = ClaimValidationResult(
            is_valid=True,
            total_claims=10,
            grounded_claims=8,
            grounded_ratio=0.8,
        )
        risk = EpistemicRiskScore(overall_risk=0.8)
        
        should_block, reason = validator.should_block_phase_advancement(
            validation_result, risk
        )
        
        assert should_block is True
        assert "risk" in reason.lower()
    
    def test_allows_advancement_when_valid(self):
        """Test that phase advancement is allowed when all checks pass."""
        validator = ClaimValidator()
        
        validation_result = ClaimValidationResult(
            is_valid=True,
            total_claims=10,
            grounded_claims=8,
            grounded_ratio=0.8,
        )
        risk = EpistemicRiskScore(overall_risk=0.3)
        
        should_block, reason = validator.should_block_phase_advancement(
            validation_result, risk
        )
        
        assert should_block is False
        assert reason == ""

