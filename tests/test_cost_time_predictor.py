"""
Unit tests for Cost & Time Predictor.

Tests feature encoder, model registry, and predictor components.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSchemas:
    """Tests for schema dataclasses."""
    
    def test_phase_context_creation(self):
        """Test PhaseContext dataclass creation and serialization."""
        from deepthinker.resources.cost_time_predictor.schemas import PhaseContext
        
        ctx = PhaseContext(
            phase_name="Reconnaissance",
            phase_type="research",
            effort_level="standard",
            mission_time_budget_seconds=3600.0,
            time_remaining_seconds=3000.0,
            iteration_index=0,
        )
        
        assert ctx.phase_name == "Reconnaissance"
        assert ctx.phase_type == "research"
        
        # Test serialization
        d = ctx.to_dict()
        assert d["phase_name"] == "Reconnaissance"
        assert d["time_remaining_seconds"] == 3000.0
    
    def test_execution_plan_creation(self):
        """Test ExecutionPlan dataclass."""
        from deepthinker.resources.cost_time_predictor.schemas import ExecutionPlan
        
        plan = ExecutionPlan(
            model_tier="medium",
            model_names=["cogito:14b", "gemma3:12b"],
            councils_invoked=["research_council"],
            consensus_enabled=True,
            max_iterations=3,
            per_call_timeout_seconds=90.0,
            search_enabled=False,
        )
        
        assert plan.model_tier == "medium"
        assert len(plan.model_names) == 2
        assert plan.consensus_enabled is True
        
        d = plan.to_dict()
        assert d["model_names"] == ["cogito:14b", "gemma3:12b"]
    
    def test_system_state_creation(self):
        """Test SystemState dataclass."""
        from deepthinker.resources.cost_time_predictor.schemas import SystemState
        
        state = SystemState(
            available_vram_mb=25000,
            gpu_load_ratio=0.3,
            memory_pressure_ratio=0.2,
        )
        
        assert state.available_vram_mb == 25000
        assert state.gpu_load_ratio == 0.3
    
    def test_prediction_evaluation_creation(self):
        """Test PredictionEvaluation creation from prediction and actuals."""
        from deepthinker.resources.cost_time_predictor.schemas import (
            CostTimePrediction,
            PredictionEvaluation,
        )
        
        prediction = CostTimePrediction(
            wall_time_seconds=100.0,
            gpu_seconds=80.0,
            vram_peak_mb=15000,
            confidence=0.7,
            model_version="v1",
            used_fallback=False,
        )
        
        evaluation = PredictionEvaluation.from_prediction_and_actual(
            timestamp="2024-01-01T00:00:00",
            mission_id="test-mission",
            phase_name="Test Phase",
            phase_type="research",
            prediction=prediction,
            actual_wall_time=120.0,
            actual_gpu_seconds=90.0,
            actual_vram_peak=16000,
        )
        
        assert evaluation.predicted_wall_time_seconds == 100.0
        assert evaluation.actual_wall_time_seconds == 120.0
        assert evaluation.wall_time_error_abs == 20.0
        assert abs(evaluation.wall_time_error_pct - 16.67) < 0.1  # ~16.67%


class TestFeatureEncoder:
    """Tests for feature encoder."""
    
    def test_encoder_initialization(self):
        """Test encoder initializes correctly."""
        from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
        
        encoder = FeatureEncoder()
        assert encoder.VERSION == 1
        assert encoder.feature_dim > 0
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding produces correct output."""
        from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
        
        encoder = FeatureEncoder()
        
        # Test internal method
        result = encoder._one_hot("research", ["research", "design", "synthesis"])
        assert result == [1.0, 0.0, 0.0]
        
        result = encoder._one_hot("design", ["research", "design", "synthesis"])
        assert result == [0.0, 1.0, 0.0]
        
        # Unknown value should produce all zeros
        result = encoder._one_hot("unknown", ["research", "design", "synthesis"])
        assert result == [0.0, 0.0, 0.0]
    
    def test_multi_hot_encoding(self):
        """Test multi-hot encoding produces correct output."""
        from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
        
        encoder = FeatureEncoder()
        
        categories = ["research_council", "planner_council", "coder_council"]
        
        result = encoder._multi_hot(
            ["research_council", "coder_council"],
            categories
        )
        assert result == [1.0, 0.0, 1.0]
        
        # Empty list
        result = encoder._multi_hot([], categories)
        assert result == [0.0, 0.0, 0.0]
    
    def test_encode_produces_correct_dimension(self):
        """Test encoding produces vector of expected dimension."""
        from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
        from deepthinker.resources.cost_time_predictor.schemas import (
            PhaseContext, ExecutionPlan, SystemState
        )
        
        encoder = FeatureEncoder()
        
        ctx = PhaseContext(
            phase_name="Test",
            phase_type="research",
            effort_level="standard",
            mission_time_budget_seconds=3600.0,
            time_remaining_seconds=3000.0,
            iteration_index=0,
        )
        
        plan = ExecutionPlan(
            model_tier="medium",
            model_names=["cogito:14b"],
            councils_invoked=["research_council"],
            consensus_enabled=False,
            max_iterations=3,
            per_call_timeout_seconds=90.0,
            search_enabled=False,
        )
        
        state = SystemState(
            available_vram_mb=30000,
            gpu_load_ratio=0.5,
            memory_pressure_ratio=0.3,
        )
        
        features = encoder.encode(ctx, plan, state)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == encoder.feature_dim
    
    def test_encode_deterministic(self):
        """Test encoding is deterministic (same input -> same output)."""
        from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
        from deepthinker.resources.cost_time_predictor.schemas import (
            PhaseContext, ExecutionPlan, SystemState
        )
        
        encoder = FeatureEncoder()
        
        ctx = PhaseContext(
            phase_name="Test",
            phase_type="design",
            effort_level="thorough",
            mission_time_budget_seconds=1800.0,
            time_remaining_seconds=1500.0,
            iteration_index=2,
        )
        
        plan = ExecutionPlan(
            model_tier="large",
            model_names=["gemma3:27b"],
            councils_invoked=["planner_council", "coder_council"],
            consensus_enabled=True,
            max_iterations=5,
            per_call_timeout_seconds=120.0,
            search_enabled=True,
        )
        
        state = SystemState(
            available_vram_mb=20000,
            gpu_load_ratio=0.7,
            memory_pressure_ratio=0.5,
        )
        
        # Encode twice
        features1 = encoder.encode(ctx, plan, state)
        features2 = encoder.encode(ctx, plan, state)
        
        # Should be identical
        np.testing.assert_array_equal(features1, features2)
    
    def test_get_feature_names(self):
        """Test feature names are generated correctly."""
        from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
        
        encoder = FeatureEncoder()
        names = encoder.get_feature_names()
        
        assert len(names) == encoder.feature_dim
        assert "phase_type_research" in names
        assert "model_tier_medium" in names
        assert "time_budget_normalized" in names


class TestModelRegistry:
    """Tests for model registry."""
    
    def test_registry_initialization(self):
        """Test registry creates directory."""
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            assert registry.base_dir.exists()
    
    def test_list_versions_empty(self):
        """Test listing versions when no models exist."""
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            versions = registry.list_versions()
            assert versions == []
    
    def test_get_next_version(self):
        """Test next version calculation."""
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            
            # First version should be 1
            assert registry.get_next_version() == 1
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        from deepthinker.resources.cost_time_predictor.model_registry import (
            ModelRegistry, ModelMetadata
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            
            # Create a simple mock model
            mock_model = {"type": "test", "weights": [1, 2, 3]}
            
            metadata = ModelMetadata(
                version=1,
                trained_at="2024-01-01T00:00:00",
                dataset_size=100,
                feature_version=1,
                validation_mae={"wall_time": 10.5, "gpu_seconds": 8.0},
                model_type="TestModel",
            )
            
            # Save
            success = registry.save_model(mock_model, metadata)
            assert success
            
            # Check files exist
            assert (registry.base_dir / "model_v1.joblib").exists()
            assert (registry.base_dir / "metadata_v1.json").exists()
            
            # Load
            loaded_model = registry.load_model(1)
            assert loaded_model == mock_model
            
            loaded_metadata = registry.load_metadata(1)
            assert loaded_metadata.version == 1
            assert loaded_metadata.dataset_size == 100
    
    def test_load_latest(self):
        """Test loading the latest model."""
        from deepthinker.resources.cost_time_predictor.model_registry import (
            ModelRegistry, ModelMetadata
        )
        from deepthinker.resources.cost_time_predictor.config import FEATURE_VECTOR_VERSION
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            
            # Save two versions
            for v in [1, 2]:
                model = {"version": v}
                metadata = ModelMetadata(
                    version=v,
                    trained_at=f"2024-01-0{v}T00:00:00",
                    dataset_size=100 * v,
                    feature_version=FEATURE_VECTOR_VERSION,
                    validation_mae={"wall_time": 10.0 / v},
                    model_type="TestModel",
                )
                registry.save_model(model, metadata)
            
            # Load latest
            result = registry.load_latest()
            assert result is not None
            
            model, metadata = result
            assert model["version"] == 2
            assert metadata.version == 2
    
    def test_get_statistics(self):
        """Test getting registry statistics."""
        from deepthinker.resources.cost_time_predictor.model_registry import (
            ModelRegistry, ModelMetadata
        )
        from deepthinker.resources.cost_time_predictor.config import FEATURE_VECTOR_VERSION
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            
            # Save a model
            model = {"test": True}
            metadata = ModelMetadata(
                version=1,
                trained_at="2024-01-01T00:00:00",
                dataset_size=50,
                feature_version=FEATURE_VECTOR_VERSION,
                validation_mae={"wall_time": 15.0},
            )
            registry.save_model(model, metadata)
            
            stats = registry.get_statistics()
            assert stats["total_versions"] == 1
            assert stats["latest_version"] == 1
            assert stats["latest_dataset_size"] == 50


class TestPredictor:
    """Tests for the predictor runtime."""
    
    def test_predictor_initialization_without_model(self):
        """Test predictor initializes correctly without trained model."""
        from deepthinker.resources.cost_time_predictor.predictor import CostTimePredictor
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            predictor = CostTimePredictor(registry=registry)
            
            assert not predictor.has_model
            assert predictor.model_version == "v0-fallback"
    
    def test_fallback_prediction(self):
        """Test fallback prediction when no model is available."""
        from deepthinker.resources.cost_time_predictor.predictor import CostTimePredictor
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        from deepthinker.resources.cost_time_predictor.schemas import (
            PhaseContext, ExecutionPlan, SystemState
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            predictor = CostTimePredictor(registry=registry)
            
            ctx = PhaseContext(
                phase_name="Test",
                phase_type="research",
                effort_level="standard",
                mission_time_budget_seconds=3600.0,
                time_remaining_seconds=3000.0,
                iteration_index=0,
            )
            
            plan = ExecutionPlan(
                model_tier="medium",
                model_names=["cogito:14b"],
                councils_invoked=["research_council"],
                consensus_enabled=False,
                max_iterations=3,
                per_call_timeout_seconds=90.0,
                search_enabled=False,
            )
            
            state = SystemState(
                available_vram_mb=30000,
                gpu_load_ratio=0.5,
                memory_pressure_ratio=0.3,
            )
            
            prediction = predictor.predict(ctx, plan, state)
            
            assert prediction.used_fallback is True
            assert prediction.model_version == "v0-fallback"
            assert prediction.wall_time_seconds > 0
            assert prediction.gpu_seconds > 0
            assert prediction.vram_peak_mb > 0
            assert prediction.confidence < 1.0
    
    def test_vram_clamping(self):
        """Test that VRAM predictions are clamped to safe bounds."""
        from deepthinker.resources.cost_time_predictor.predictor import CostTimePredictor
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        from deepthinker.resources.cost_time_predictor.schemas import (
            PhaseContext, ExecutionPlan, SystemState
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            predictor = CostTimePredictor(registry=registry)
            
            ctx = PhaseContext(
                phase_name="Test",
                phase_type="synthesis",
                effort_level="thorough",
                mission_time_budget_seconds=3600.0,
                time_remaining_seconds=3000.0,
                iteration_index=0,
            )
            
            plan = ExecutionPlan(
                model_tier="xlarge",  # Would normally need lots of VRAM
                model_names=["llama3:70b"],
                councils_invoked=["synthesis_council"],
                consensus_enabled=False,
                max_iterations=1,
                per_call_timeout_seconds=120.0,
                search_enabled=False,
            )
            
            # Very limited VRAM
            state = SystemState(
                available_vram_mb=5000,  # Only 5GB available
                gpu_load_ratio=0.8,
                memory_pressure_ratio=0.7,
            )
            
            prediction = predictor.predict(ctx, plan, state)
            
            # VRAM should be clamped to available - safety margin (2GB)
            assert prediction.vram_peak_mb <= 3000  # 5000 - 2000
    
    def test_time_clamping(self):
        """Test that time predictions respect remaining time."""
        from deepthinker.resources.cost_time_predictor.predictor import CostTimePredictor
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        from deepthinker.resources.cost_time_predictor.schemas import (
            PhaseContext, ExecutionPlan, SystemState
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            predictor = CostTimePredictor(registry=registry)
            
            # Very little time remaining
            ctx = PhaseContext(
                phase_name="Test",
                phase_type="synthesis",
                effort_level="thorough",
                mission_time_budget_seconds=3600.0,
                time_remaining_seconds=180.0,  # Only 3 minutes left
                iteration_index=5,
            )
            
            plan = ExecutionPlan(
                model_tier="large",
                model_names=["gemma3:27b"],
                councils_invoked=["synthesis_council"],
                consensus_enabled=False,
                max_iterations=1,
                per_call_timeout_seconds=120.0,
                search_enabled=False,
            )
            
            state = SystemState(
                available_vram_mb=30000,
                gpu_load_ratio=0.3,
                memory_pressure_ratio=0.2,
            )
            
            prediction = predictor.predict(ctx, plan, state)
            
            # Time should be clamped to remaining - synthesis reserve (120s)
            assert prediction.wall_time_seconds <= 60.0  # 180 - 120
    
    def test_get_model_info(self):
        """Test getting model info."""
        from deepthinker.resources.cost_time_predictor.predictor import CostTimePredictor
        from deepthinker.resources.cost_time_predictor.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir) / "models")
            predictor = CostTimePredictor(registry=registry)
            
            info = predictor.get_model_info()
            
            assert "has_model" in info
            assert "model_version" in info
            assert "feature_encoder_version" in info
            assert "feature_dim" in info


class TestEvaluationLogger:
    """Tests for evaluation logger."""
    
    def test_logger_initialization(self):
        """Test logger creates directory."""
        from deepthinker.resources.cost_time_predictor.eval_logger import EvaluationLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "eval" / "test.jsonl"
            logger = EvaluationLogger(log_path=log_path)
            
            assert log_path.parent.exists()
    
    def test_log_evaluation(self):
        """Test logging an evaluation."""
        from deepthinker.resources.cost_time_predictor.eval_logger import EvaluationLogger
        from deepthinker.resources.cost_time_predictor.schemas import CostTimePrediction
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "eval.jsonl"
            logger = EvaluationLogger(log_path=log_path)
            
            prediction = CostTimePrediction(
                wall_time_seconds=100.0,
                gpu_seconds=80.0,
                vram_peak_mb=15000,
                confidence=0.7,
                model_version="v1",
                used_fallback=False,
            )
            
            logger.log_evaluation(
                mission_id="test-mission",
                phase_name="Test Phase",
                phase_type="research",
                prediction=prediction,
                actual_wall_time=120.0,
                actual_gpu_seconds=90.0,
                actual_vram_peak=16000,
            )
            
            # Check file exists and has content
            assert log_path.exists()
            
            with open(log_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["mission_id"] == "test-mission"
            assert record["predicted_wall_time_seconds"] == 100.0
            assert record["actual_wall_time_seconds"] == 120.0
    
    def test_read_evaluations(self):
        """Test reading evaluations from log."""
        from deepthinker.resources.cost_time_predictor.eval_logger import EvaluationLogger
        from deepthinker.resources.cost_time_predictor.schemas import CostTimePrediction
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "eval.jsonl"
            logger = EvaluationLogger(log_path=log_path)
            
            # Log multiple evaluations
            for i in range(3):
                prediction = CostTimePrediction(
                    wall_time_seconds=100.0 + i * 10,
                    gpu_seconds=80.0,
                    vram_peak_mb=15000,
                    confidence=0.7,
                    model_version="v1",
                    used_fallback=False,
                )
                
                logger.log_evaluation(
                    mission_id=f"mission-{i}",
                    phase_name="Test Phase",
                    phase_type="research" if i < 2 else "synthesis",
                    prediction=prediction,
                    actual_wall_time=120.0,
                    actual_gpu_seconds=90.0,
                    actual_vram_peak=16000,
                )
            
            # Read all
            evaluations = list(logger.read_evaluations())
            assert len(evaluations) == 3
            
            # Read filtered
            research_evals = list(logger.read_evaluations({"phase_type": "research"}))
            assert len(research_evals) == 2
    
    def test_get_statistics(self):
        """Test getting evaluation statistics."""
        from deepthinker.resources.cost_time_predictor.eval_logger import EvaluationLogger
        from deepthinker.resources.cost_time_predictor.schemas import CostTimePrediction
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "eval.jsonl"
            logger = EvaluationLogger(log_path=log_path)
            
            # Log some evaluations
            for i in range(5):
                prediction = CostTimePrediction(
                    wall_time_seconds=100.0,
                    gpu_seconds=80.0,
                    vram_peak_mb=15000,
                    confidence=0.7,
                    model_version="v1",
                    used_fallback=(i % 2 == 0),  # Alternate
                )
                
                logger.log_evaluation(
                    mission_id=f"mission-{i}",
                    phase_name="Test Phase",
                    phase_type="research",
                    prediction=prediction,
                    actual_wall_time=110.0 + i * 5,  # Varying actual
                    actual_gpu_seconds=90.0,
                    actual_vram_peak=16000,
                )
            
            stats = logger.get_statistics()
            
            assert stats["total_evaluations"] == 5
            assert stats["ml_predictions"] == 2
            assert stats["fallback_predictions"] == 3
            assert "wall_time_mae" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

