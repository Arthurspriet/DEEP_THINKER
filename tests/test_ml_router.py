"""
Tests for ML Router module.

Tests feature extraction, advisory output, and fallback behavior.
"""

import pytest
import json
import tempfile
import os


class TestRoutingFeatures:
    """Test feature extraction for routing."""
    
    def test_import(self):
        """Test that routing module can be imported."""
        from deepthinker.routing import (
            extract_routing_features,
            RoutingContext,
            RoutingFeatures,
        )
        assert extract_routing_features is not None
        assert RoutingContext is not None
        assert RoutingFeatures is not None
    
    def test_context_creation(self):
        """Test routing context creation."""
        from deepthinker.routing import RoutingContext
        
        ctx = RoutingContext(
            objective="Analyze stock market trends",
            phase_name="research",
            input_text="Some input text",
            time_remaining_minutes=30.0,
            time_budget_minutes=60.0,
        )
        
        assert ctx.objective == "Analyze stock market trends"
        assert ctx.phase_name == "research"
        assert ctx.time_remaining_minutes == 30.0
    
    def test_extract_features(self):
        """Test feature extraction from context."""
        from deepthinker.routing import extract_routing_features, RoutingContext
        
        ctx = RoutingContext(
            objective="Research the latest AI developments",
            phase_name="research",
            input_text="This is some input text with URLs https://example.com",
            time_remaining_minutes=30.0,
            time_budget_minutes=60.0,
            recent_scores=[0.6, 0.7, 0.8],
        )
        
        features = extract_routing_features(ctx)
        
        # Check task type features
        assert features["task_type_research"] == 1.0
        
        # Check phase features
        assert features["phase_research"] == 1.0
        
        # Check time features
        assert features["time_remaining_ratio"] == 0.5
        
        # Check URL detection
        assert features["has_urls"] == 1.0
        
        # Check recent scores
        assert features["recent_score_mean"] == pytest.approx(0.7)
        assert features["recent_score_trend"] > 0  # Positive trend
    
    def test_task_type_classification(self):
        """Test task type classification from objective."""
        from deepthinker.routing import extract_routing_features, RoutingContext
        
        # Research task
        research_ctx = RoutingContext(objective="Research market trends")
        research_features = extract_routing_features(research_ctx)
        assert research_features["task_type_research"] == 1.0
        
        # Code task
        code_ctx = RoutingContext(objective="Implement a new feature")
        code_features = extract_routing_features(code_ctx)
        assert code_features["task_type_code"] == 1.0
        
        # Analysis task
        analysis_ctx = RoutingContext(objective="Analyze the data")
        analysis_features = extract_routing_features(analysis_ctx)
        assert analysis_features["task_type_analysis"] == 1.0
    
    def test_difficulty_estimate(self):
        """Test difficulty estimation from heuristics."""
        from deepthinker.routing import extract_routing_features, RoutingContext
        
        # Simple task
        simple_ctx = RoutingContext(
            objective="Do something simple",
            input_text="Short input",
            retry_count=0,
        )
        simple_features = extract_routing_features(simple_ctx)
        
        # Complex task
        complex_ctx = RoutingContext(
            objective="Do something very complex " * 50,
            input_text="Long input " * 1000,
            retry_count=3,
            recent_scores=[0.3, 0.2],
        )
        complex_features = extract_routing_features(complex_ctx)
        
        assert complex_features["difficulty_estimate"] > simple_features["difficulty_estimate"]


class TestRoutingFeaturesList:
    """Test RoutingFeatures dataclass."""
    
    def test_to_vector(self):
        """Test converting features to vector."""
        from deepthinker.routing import RoutingFeatures
        from deepthinker.routing.features import get_feature_names
        
        features = RoutingFeatures(
            features={
                "task_type_research": 1.0,
                "phase_research": 1.0,
                "time_remaining_ratio": 0.5,
            }
        )
        
        vector = features.to_vector()
        
        assert len(vector) == len(get_feature_names())
        assert vector[0] == 1.0  # task_type_research is first


class TestMLRouterAdvisor:
    """Test MLRouterAdvisor class."""
    
    def test_creation(self):
        """Test router creation."""
        from deepthinker.routing import MLRouterAdvisor
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "weights.json")
            router = MLRouterAdvisor(config=config, weights_path=weights_path)
            
            assert router is not None
    
    def test_advise_disabled(self):
        """Test advise when disabled returns low confidence."""
        from deepthinker.routing import MLRouterAdvisor, RoutingContext
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=False)
        
        router = MLRouterAdvisor(config=config)
        ctx = RoutingContext(objective="Test")
        
        decision = router.advise(ctx)
        
        assert decision.confidence == 0.0
        assert decision.rationale == "Router disabled"
    
    def test_advise_heuristic(self):
        """Test heuristic-based advice."""
        from deepthinker.routing import MLRouterAdvisor, RoutingContext
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "weights.json")
            router = MLRouterAdvisor(config=config, weights_path=weights_path)
            
            ctx = RoutingContext(
                objective="Research complex topic",
                phase_name="research",
                time_remaining_minutes=30.0,
                time_budget_minutes=60.0,
            )
            
            decision = router.advise(ctx)
            
            assert decision.model_tier in ["SMALL", "MEDIUM", "LARGE"]
            assert decision.num_rounds in [1, 2, 3]
            assert 0.0 <= decision.confidence <= 1.0
            assert decision.rationale != ""
    
    def test_advise_research_task(self):
        """Test advice for research task."""
        from deepthinker.routing import MLRouterAdvisor, RoutingContext
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "weights.json")
            router = MLRouterAdvisor(config=config, weights_path=weights_path)
            
            ctx = RoutingContext(
                objective="Research the topic",
                phase_name="research",
            )
            
            decision = router.advise(ctx)
            
            assert decision.council_set == "research"
    
    def test_save_weights(self):
        """Test saving weights."""
        from deepthinker.routing import MLRouterAdvisor
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "models", "weights.json")
            router = MLRouterAdvisor(config=config, weights_path=weights_path)
            
            custom_weights = {"tier": {"difficulty_estimate": 0.7}}
            router.save_weights(custom_weights)
            
            assert os.path.exists(weights_path)
            with open(weights_path) as f:
                loaded = json.load(f)
            assert loaded["tier"]["difficulty_estimate"] == 0.7
    
    def test_record_outcome(self):
        """Test recording outcome for training."""
        from deepthinker.routing import MLRouterAdvisor, RoutingContext, RoutingDecision
        from deepthinker.metrics import MetricsConfig
        
        config = MetricsConfig(learning_router_enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "models", "weights.json")
            router = MLRouterAdvisor(config=config, weights_path=weights_path)
            
            ctx = RoutingContext(objective="Test")
            decision = router.advise(ctx)
            
            router.record_outcome(
                decision=decision,
                score_delta=0.2,
                cost_delta=0.1,
            )
            
            # Check outcomes file was created
            outcomes_path = os.path.join(tmpdir, "models", "outcomes.jsonl")
            assert os.path.exists(outcomes_path)


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""
    
    def test_creation(self):
        """Test decision creation."""
        from deepthinker.routing import RoutingDecision
        
        decision = RoutingDecision(
            council_set="research",
            model_tier="MEDIUM",
            num_rounds=2,
            confidence=0.8,
            rationale="Test rationale",
        )
        
        assert decision.council_set == "research"
        assert decision.model_tier == "MEDIUM"
        assert decision.num_rounds == 2
    
    def test_to_dict_from_dict(self):
        """Test decision serialization."""
        from deepthinker.routing import RoutingDecision
        
        original = RoutingDecision(
            council_set="coder",
            model_tier="LARGE",
            num_rounds=3,
            confidence=0.9,
            rationale="Complex coding task",
        )
        
        data = original.to_dict()
        restored = RoutingDecision.from_dict(data)
        
        assert restored.council_set == original.council_set
        assert restored.model_tier == original.model_tier
        assert restored.num_rounds == original.num_rounds
        assert restored.confidence == original.confidence

