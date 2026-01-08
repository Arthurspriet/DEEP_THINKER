"""
Tests for Scorecard module.

Tests the Scorecard dataclass, weighted overall computation,
and serialization.
"""

import pytest
from datetime import datetime


class TestScorecard:
    """Test Scorecard dataclass."""
    
    def test_import(self):
        """Test that scorecard module can be imported."""
        from deepthinker.metrics import Scorecard, ScorecardCost, ScorecardRuntime, ScorecardMetadata
        assert Scorecard is not None
        assert ScorecardCost is not None
        assert ScorecardRuntime is not None
        assert ScorecardMetadata is not None
    
    def test_scorecard_creation(self):
        """Test basic scorecard creation."""
        from deepthinker.metrics import Scorecard
        
        scorecard = Scorecard(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
        )
        
        assert scorecard.goal_coverage == 0.8
        assert scorecard.evidence_grounding == 0.7
        assert scorecard.actionability == 0.6
        assert scorecard.consistency == 0.9
    
    def test_compute_overall(self):
        """Test weighted overall score computation."""
        from deepthinker.metrics import Scorecard
        
        scorecard = Scorecard(
            goal_coverage=0.8,
            evidence_grounding=0.8,
            actionability=0.8,
            consistency=0.8,
        )
        
        overall = scorecard.compute_overall()
        
        # Should be close to 0.8 with default weights
        assert 0.75 <= overall <= 0.85
        assert scorecard.overall == overall
    
    def test_compute_overall_custom_weights(self):
        """Test overall with custom weights."""
        from deepthinker.metrics import Scorecard
        
        scorecard = Scorecard(
            goal_coverage=1.0,
            evidence_grounding=0.0,
            actionability=0.0,
            consistency=0.0,
        )
        
        # All weight on goal_coverage
        overall = scorecard.compute_overall(weights={
            "goal_coverage": 1.0,
            "evidence_grounding": 0.0,
            "actionability": 0.0,
            "consistency": 0.0,
        })
        
        assert overall == 1.0
    
    def test_from_scores(self):
        """Test creating scorecard from individual scores."""
        from deepthinker.metrics import Scorecard, ScorecardMetadata
        
        metadata = ScorecardMetadata(
            mission_id="test_mission",
            phase_id="test_phase",
        )
        
        scorecard = Scorecard.from_scores(
            goal_coverage=0.7,
            evidence_grounding=0.6,
            actionability=0.5,
            consistency=0.8,
            metadata=metadata,
            previous_overall=0.5,
        )
        
        assert scorecard.goal_coverage == 0.7
        assert scorecard.metadata.mission_id == "test_mission"
        assert scorecard.score_delta is not None
        assert scorecard.score_delta == scorecard.overall - 0.5
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        from deepthinker.metrics import Scorecard, ScorecardMetadata
        
        original = Scorecard.from_scores(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
            metadata=ScorecardMetadata(
                mission_id="test",
                phase_id="phase1",
            ),
        )
        
        data = original.to_dict()
        restored = Scorecard.from_dict(data)
        
        assert restored.goal_coverage == original.goal_coverage
        assert restored.evidence_grounding == original.evidence_grounding
        assert restored.actionability == original.actionability
        assert restored.consistency == original.consistency
        assert restored.overall == original.overall
        assert restored.metadata.mission_id == original.metadata.mission_id
    
    def test_is_high_quality(self):
        """Test quality threshold check."""
        from deepthinker.metrics import Scorecard
        
        high_quality = Scorecard.from_scores(0.9, 0.9, 0.9, 0.9)
        low_quality = Scorecard.from_scores(0.3, 0.3, 0.3, 0.3)
        
        assert high_quality.is_high_quality(threshold=0.7)
        assert not low_quality.is_high_quality(threshold=0.7)
    
    def test_can_stop(self):
        """Test stop condition check."""
        from deepthinker.metrics import Scorecard, get_metrics_config, reset_metrics_config
        
        # Reset to ensure clean state
        reset_metrics_config()
        
        # High scores should allow stop
        high_scorecard = Scorecard.from_scores(0.9, 0.9, 0.9, 0.9)
        
        # Low scores should not allow stop
        low_scorecard = Scorecard.from_scores(0.3, 0.3, 0.3, 0.3)
        
        config = get_metrics_config()
        
        assert high_scorecard.can_stop(config)
        assert not low_scorecard.can_stop(config)
    
    def test_needs_escalation(self):
        """Test escalation condition check."""
        from deepthinker.metrics import Scorecard, get_metrics_config, reset_metrics_config
        
        reset_metrics_config()
        
        # Low goal coverage should need escalation
        low_goal = Scorecard.from_scores(0.2, 0.8, 0.8, 0.8)
        
        # High scores should not need escalation
        high_scores = Scorecard.from_scores(0.8, 0.8, 0.8, 0.8)
        
        config = get_metrics_config()
        
        assert low_goal.needs_escalation(config)
        assert not high_scores.needs_escalation(config)


class TestScorecardCost:
    """Test ScorecardCost dataclass."""
    
    def test_creation(self):
        """Test cost creation."""
        from deepthinker.metrics import ScorecardCost
        
        cost = ScorecardCost(
            tokens=1000,
            usd=0.05,
            latency_ms=500.0,
        )
        
        assert cost.tokens == 1000
        assert cost.usd == 0.05
        assert cost.latency_ms == 500.0
    
    def test_to_dict_from_dict(self):
        """Test cost serialization."""
        from deepthinker.metrics import ScorecardCost
        
        original = ScorecardCost(tokens=1000, usd=0.05, latency_ms=500.0)
        data = original.to_dict()
        restored = ScorecardCost.from_dict(data)
        
        assert restored.tokens == original.tokens
        assert restored.usd == original.usd
        assert restored.latency_ms == original.latency_ms


class TestScorecardMetadata:
    """Test ScorecardMetadata dataclass."""
    
    def test_creation(self):
        """Test metadata creation."""
        from deepthinker.metrics import ScorecardMetadata
        
        meta = ScorecardMetadata(
            mission_id="mission_123",
            phase_id="research",
            models_used=["model1", "model2"],
            councils_used=["ResearcherCouncil"],
            is_final=True,
        )
        
        assert meta.mission_id == "mission_123"
        assert meta.phase_id == "research"
        assert len(meta.models_used) == 2
        assert meta.is_final
    
    def test_timestamp(self):
        """Test default timestamp."""
        from deepthinker.metrics import ScorecardMetadata
        
        before = datetime.utcnow()
        meta = ScorecardMetadata()
        after = datetime.utcnow()
        
        assert before <= meta.timestamp <= after

