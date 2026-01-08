"""
Tests for Judge Ensemble module.

Tests single judge, ensemble scoring, disagreement,
and sampling behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestJudgeEnsemble:
    """Test JudgeEnsemble class."""
    
    def test_import(self):
        """Test that judge ensemble can be imported."""
        from deepthinker.metrics import JudgeEnsemble, JudgeResult, JudgeScores
        assert JudgeEnsemble is not None
        assert JudgeResult is not None
        assert JudgeScores is not None
    
    def test_creation(self):
        """Test ensemble creation."""
        from deepthinker.metrics import JudgeEnsemble, MetricsConfig
        
        config = MetricsConfig(
            scorecard_enabled=True,
            judge_sample_rate=1.0,
        )
        
        ensemble = JudgeEnsemble(config=config)
        assert ensemble.config == config
    
    def test_parse_scores(self):
        """Test parsing scores from judge output."""
        from deepthinker.metrics import JudgeEnsemble, MetricsConfig
        
        config = MetricsConfig(scorecard_enabled=True)
        ensemble = JudgeEnsemble(config=config)
        
        raw_output = """
        GOAL_COVERAGE: 8
        EVIDENCE_GROUNDING: 7
        ACTIONABILITY: 6
        CONSISTENCY: 9
        """
        
        scores = ensemble._parse_scores(raw_output, "test_model")
        
        assert scores.goal_coverage == 0.8
        assert scores.evidence_grounding == 0.7
        assert scores.actionability == 0.6
        assert scores.consistency == 0.9
        assert scores.model_name == "test_model"
    
    def test_parse_scores_decimal(self):
        """Test parsing decimal scores."""
        from deepthinker.metrics import JudgeEnsemble, MetricsConfig
        
        config = MetricsConfig(scorecard_enabled=True)
        ensemble = JudgeEnsemble(config=config)
        
        raw_output = """
        GOAL_COVERAGE: 7.5
        EVIDENCE_GROUNDING: 8.5
        ACTIONABILITY: 6.0
        CONSISTENCY: 9.5
        """
        
        scores = ensemble._parse_scores(raw_output, "test_model")
        
        assert scores.goal_coverage == 0.75
        assert scores.evidence_grounding == 0.85
        assert scores.consistency == 0.95
    
    def test_compute_disagreement(self):
        """Test disagreement computation between judges."""
        from deepthinker.metrics import JudgeEnsemble, JudgeScores, MetricsConfig
        
        config = MetricsConfig(scorecard_enabled=True)
        ensemble = JudgeEnsemble(config=config)
        
        cheap_scores = JudgeScores(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
        )
        
        strong_scores = JudgeScores(
            goal_coverage=0.9,
            evidence_grounding=0.6,
            actionability=0.7,
            consistency=0.8,
        )
        
        disagreement = ensemble._compute_disagreement(cheap_scores, strong_scores)
        
        # Mean absolute difference: (0.1 + 0.1 + 0.1 + 0.1) / 4 = 0.1
        assert abs(disagreement - 0.1) < 0.001  # Handle floating point precision
    
    def test_aggregate_scores_single_judge(self):
        """Test score aggregation with single judge."""
        from deepthinker.metrics import JudgeEnsemble, JudgeScores, MetricsConfig
        
        config = MetricsConfig(scorecard_enabled=True)
        ensemble = JudgeEnsemble(config=config)
        
        cheap_scores = JudgeScores(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
        )
        
        goal, evidence, action, consistency = ensemble._aggregate_scores(
            cheap_scores, None
        )
        
        assert goal == 0.8
        assert evidence == 0.7
        assert action == 0.6
        assert consistency == 0.9
    
    def test_aggregate_scores_ensemble(self):
        """Test score aggregation with two judges."""
        from deepthinker.metrics import JudgeEnsemble, JudgeScores, MetricsConfig
        
        config = MetricsConfig(scorecard_enabled=True)
        ensemble = JudgeEnsemble(config=config)
        
        cheap_scores = JudgeScores(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
        )
        
        strong_scores = JudgeScores(
            goal_coverage=1.0,
            evidence_grounding=0.9,
            actionability=0.8,
            consistency=1.0,
        )
        
        goal, evidence, action, consistency = ensemble._aggregate_scores(
            cheap_scores, strong_scores
        )
        
        # Weighted average: 0.4 * cheap + 0.6 * strong
        assert goal == pytest.approx(0.4 * 0.8 + 0.6 * 1.0)
        assert evidence == pytest.approx(0.4 * 0.7 + 0.6 * 0.9)
        assert action == pytest.approx(0.4 * 0.6 + 0.6 * 0.8)
        assert consistency == pytest.approx(0.4 * 0.9 + 0.6 * 1.0)
    
    def test_sampling_skips_when_rate_zero(self):
        """Test that sampling=0 skips scoring."""
        from deepthinker.metrics import JudgeEnsemble, MetricsConfig
        
        config = MetricsConfig(
            scorecard_enabled=True,
            judge_sample_rate=0.0,  # Never sample
        )
        
        ensemble = JudgeEnsemble(config=config)
        
        result = ensemble.score_artifact(
            output="Test output",
            objective="Test objective",
            force_sample=False,
        )
        
        assert result.sampled == False
        # Should return placeholder scores
        assert result.scorecard.overall == 0.5


class TestJudgeScores:
    """Test JudgeScores dataclass."""
    
    def test_creation(self):
        """Test scores creation."""
        from deepthinker.metrics import JudgeScores
        
        scores = JudgeScores(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
            raw_output="test output",
            model_name="test_model",
        )
        
        assert scores.goal_coverage == 0.8
        assert scores.model_name == "test_model"
    
    def test_to_dict(self):
        """Test scores serialization."""
        from deepthinker.metrics import JudgeScores
        
        scores = JudgeScores(
            goal_coverage=0.8,
            evidence_grounding=0.7,
            actionability=0.6,
            consistency=0.9,
            model_name="test_model",
        )
        
        data = scores.to_dict()
        
        assert data["goal_coverage"] == 0.8
        assert data["model_name"] == "test_model"


class TestJudgeResult:
    """Test JudgeResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        from deepthinker.metrics import JudgeResult, Scorecard
        
        scorecard = Scorecard.from_scores(0.8, 0.7, 0.6, 0.9)
        
        result = JudgeResult(
            scorecard=scorecard,
            disagreement=0.1,
            sampled=True,
        )
        
        assert result.scorecard == scorecard
        assert result.disagreement == 0.1
        assert result.sampled == True
    
    def test_to_dict(self):
        """Test result serialization."""
        from deepthinker.metrics import JudgeResult, Scorecard
        
        scorecard = Scorecard.from_scores(0.8, 0.7, 0.6, 0.9)
        
        result = JudgeResult(
            scorecard=scorecard,
            disagreement=0.1,
            sampled=True,
        )
        
        data = result.to_dict()
        
        assert "scorecard" in data
        assert data["disagreement"] == 0.1
        assert data["sampled"] == True

