#!/usr/bin/env python3
"""
Training Pipeline for Cost & Time Predictor.

Trains ML models on historical PhaseOutcome data to predict
execution costs (wall time, GPU seconds, VRAM peak).

Usage:
    python train_predictor.py --min-samples 20
    python train_predictor.py --force  # Train even with fewer samples
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from deepthinker.resources.cost_time_predictor.schemas import (
    PhaseContext,
    ExecutionPlan,
    SystemState,
)
from deepthinker.resources.cost_time_predictor.config import (
    PREDICTOR_CONFIG,
    FEATURE_VECTOR_VERSION,
)
from deepthinker.resources.cost_time_predictor.feature_encoder import FeatureEncoder
from deepthinker.resources.cost_time_predictor.model_registry import (
    ModelRegistry,
    ModelMetadata,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Load PhaseOutcome records and encode as features/targets.
    
    Returns:
        Tuple of (X features, Y targets, raw_records)
        X shape: (n_samples, n_features)
        Y shape: (n_samples, 3) for [wall_time, gpu_seconds, vram_peak]
    """
    from deepthinker.orchestration.orchestration_store import OrchestrationStore
    
    encoder = FeatureEncoder()
    store = OrchestrationStore()
    
    X_list: List[np.ndarray] = []
    Y_list: List[List[float]] = []
    raw_records: List[dict] = []
    
    skipped_count = 0
    
    for outcome in store.read_outcomes():
        try:
            # Skip records with missing or invalid data
            if outcome.wall_time_seconds <= 0:
                skipped_count += 1
                continue
            
            # Reconstruct PhaseContext from outcome
            constraints = outcome.constraints or {}
            time_budget_minutes = constraints.get("time_budget_minutes", 60)
            
            ctx = PhaseContext(
                phase_name=outcome.phase_name,
                phase_type=outcome.phase_type,
                effort_level=outcome.effort_level,
                mission_time_budget_seconds=time_budget_minutes * 60,
                time_remaining_seconds=outcome.time_remaining_at_start,
                iteration_index=0  # Not stored in current PhaseOutcome
            )
            
            # Reconstruct ExecutionPlan from outcome
            models_used = outcome.models_used or []
            model_names = [m[0] for m in models_used if len(m) >= 1]
            model_tier = models_used[0][1] if models_used and len(models_used[0]) >= 2 else "medium"
            
            # Infer max_iterations and timeout from constraints
            max_iterations = constraints.get("max_iterations", 10)
            
            plan = ExecutionPlan(
                model_tier=model_tier,
                model_names=model_names,
                councils_invoked=outcome.councils_invoked or [],
                consensus_enabled=outcome.consensus_executed,
                max_iterations=max_iterations,
                per_call_timeout_seconds=90.0,  # Default timeout
                search_enabled=constraints.get("enable_internet", False),
            )
            
            # SystemState - not available in historical data, use reasonable defaults
            # We estimate based on the actual VRAM usage
            vram_peak = outcome.vram_peak_mb or 15000
            state = SystemState(
                available_vram_mb=max(vram_peak + 5000, 30000),  # Assume headroom existed
                gpu_load_ratio=0.5,  # Unknown, use middle value
                memory_pressure_ratio=0.3,  # Assume moderate pressure
            )
            
            # Encode features
            features = encoder.encode(ctx, plan, state)
            
            # Targets: [wall_time, gpu_seconds, vram_peak]
            targets = [
                outcome.wall_time_seconds,
                outcome.gpu_seconds if outcome.gpu_seconds > 0 else outcome.wall_time_seconds * 0.7,
                outcome.vram_peak_mb if outcome.vram_peak_mb > 0 else 15000,
            ]
            
            X_list.append(features)
            Y_list.append(targets)
            raw_records.append(outcome.to_dict())
            
        except Exception as e:
            logger.warning(f"Failed to process outcome: {e}")
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} invalid records")
    
    if not X_list:
        return np.array([]), np.array([]), []
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    logger.info(f"Loaded {len(X)} training samples with {X.shape[1]} features")
    
    return X, Y, raw_records


def train(
    min_samples: int = 20,
    force: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Optional[dict]:
    """
    Train a cost/time prediction model.
    
    Args:
        min_samples: Minimum samples required to train
        force: Train even with fewer samples
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with training results, or None if training failed
    """
    logger.info("=" * 60)
    logger.info("Cost & Time Predictor Training")
    logger.info("=" * 60)
    
    # Load data
    X, Y, raw_records = load_training_data()
    
    if len(X) == 0:
        logger.error("No training data available")
        return None
    
    # Check minimum samples
    if len(X) < min_samples and not force:
        logger.error(
            f"Insufficient data: {len(X)} samples < {min_samples} required. "
            f"Use --force to train anyway."
        )
        return None
    
    if len(X) < min_samples:
        logger.warning(f"Training with only {len(X)} samples (minimum recommended: {min_samples})")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    if len(X) >= 5:  # Need at least 5 samples to split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
    else:
        # Not enough to split - use all for training
        X_train, Y_train = X, Y
        X_val, Y_val = X, Y
        logger.warning("Not enough samples to split - using same data for train/val")
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Select model based on data size
    use_xgboost_threshold = PREDICTOR_CONFIG.get("use_xgboost_threshold", 100)
    
    if len(X) >= use_xgboost_threshold:
        try:
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor
            
            logger.info("Using XGBoost (dataset size >= threshold)")
            base_model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                verbosity=0,
            )
            model = MultiOutputRegressor(base_model)
            model_type = "XGBoost"
            
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            
            base_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
            )
            model = MultiOutputRegressor(base_model)
            model_type = "RandomForest"
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        logger.info("Using RandomForest (dataset size < threshold)")
        base_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
        )
        model = MultiOutputRegressor(base_model)
        model_type = "RandomForest"
    
    # Train
    logger.info("Training model...")
    model.fit(X_train, Y_train)
    
    # Validate
    preds = model.predict(X_val)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = {
        "wall_time": float(mean_absolute_error(Y_val[:, 0], preds[:, 0])),
        "gpu_seconds": float(mean_absolute_error(Y_val[:, 1], preds[:, 1])),
        "vram_peak": float(mean_absolute_error(Y_val[:, 2], preds[:, 2])),
    }
    
    rmse = {
        "wall_time": float(np.sqrt(mean_squared_error(Y_val[:, 0], preds[:, 0]))),
        "gpu_seconds": float(np.sqrt(mean_squared_error(Y_val[:, 1], preds[:, 1]))),
        "vram_peak": float(np.sqrt(mean_squared_error(Y_val[:, 2], preds[:, 2]))),
    }
    
    logger.info("Validation Results:")
    logger.info(f"  Wall Time MAE: {mae['wall_time']:.2f}s, RMSE: {rmse['wall_time']:.2f}s")
    logger.info(f"  GPU Seconds MAE: {mae['gpu_seconds']:.2f}s, RMSE: {rmse['gpu_seconds']:.2f}s")
    logger.info(f"  VRAM Peak MAE: {mae['vram_peak']:.0f}MB, RMSE: {rmse['vram_peak']:.0f}MB")
    
    # Save model
    registry = ModelRegistry()
    version = registry.get_next_version()
    
    metadata = ModelMetadata(
        version=version,
        trained_at=datetime.utcnow().isoformat(),
        dataset_size=len(X),
        feature_version=FEATURE_VECTOR_VERSION,
        validation_mae=mae,
        model_type=model_type,
        training_config={
            "test_size": test_size,
            "random_state": random_state,
            "min_samples": min_samples,
        },
    )
    
    if registry.save_model(model, metadata):
        logger.info(f"Saved model v{version} to {registry.base_dir}")
    else:
        logger.error("Failed to save model")
        return None
    
    # Log feature importances if available
    try:
        encoder = FeatureEncoder()
        feature_names = encoder.get_feature_names()
        
        # Get importances from first estimator (wall_time)
        if hasattr(model.estimators_[0], 'feature_importances_'):
            importances = model.estimators_[0].feature_importances_
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 Feature Importances (wall_time):")
            for name, imp in importance_pairs[:10]:
                logger.info(f"  {name}: {imp:.4f}")
    except Exception as e:
        logger.debug(f"Could not log feature importances: {e}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return {
        "version": version,
        "dataset_size": len(X),
        "model_type": model_type,
        "validation_mae": mae,
        "validation_rmse": rmse,
        "feature_version": FEATURE_VECTOR_VERSION,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train Cost & Time Predictor model"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum samples required to train (default: 20)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even with fewer than min-samples"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    result = train(
        min_samples=args.min_samples,
        force=args.force,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

