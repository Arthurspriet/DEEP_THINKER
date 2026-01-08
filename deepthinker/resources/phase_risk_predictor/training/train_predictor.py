#!/usr/bin/env python3
"""
Training Pipeline for Phase Risk Predictor.

Trains ML models on historical PhaseOutcome data to predict
execution risk (retry probability, expected retries, failure mode).

Usage:
    python train_predictor.py --min-samples 30
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

from deepthinker.resources.phase_risk_predictor.schemas import (
    PhaseRiskContext,
    PhaseRiskExecutionPlan,
    PhaseRiskSystemState,
)
from deepthinker.resources.phase_risk_predictor.config import (
    PREDICTOR_CONFIG,
    FEATURE_VECTOR_VERSION,
    KNOWN_FAILURE_MODES,
)
from deepthinker.resources.phase_risk_predictor.feature_encoder import PhaseRiskFeatureEncoder
from deepthinker.resources.phase_risk_predictor.model_registry import (
    PhaseRiskModelRegistry,
    ModelMetadata,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def infer_failure_mode(outcome) -> str:
    """
    Infer failure mode from PhaseOutcome.
    
    Args:
        outcome: PhaseOutcome record
        
    Returns:
        Failure mode string
    """
    # Check quality score for low_quality
    if outcome.quality_score is not None and outcome.quality_score < 0.3:
        return "low_quality"
    
    # Check wall time vs expected for timeout
    # If phase took significantly longer than expected, might be timeout-related
    if outcome.wall_time_seconds > 300:  # More than 5 minutes
        return "timeout"
    
    # Check arbiter output for hints
    if outcome.arbiter_raw_output:
        arbiter_lower = outcome.arbiter_raw_output.lower()
        if "hallucin" in arbiter_lower:
            return "hallucination"
        if "incoherent" in arbiter_lower or "inconsistent" in arbiter_lower:
            return "incoherent"
        if "low quality" in arbiter_lower or "poor" in arbiter_lower:
            return "low_quality"
    
    return "unknown"


def infer_retry_count(outcome, phase_rounds: dict) -> int:
    """
    Infer retry count from outcome and phase tracking.
    
    Args:
        outcome: PhaseOutcome record
        phase_rounds: Dictionary of phase name to round count
        
    Returns:
        Number of retries (rounds - 1)
    """
    # Check if we have phase_rounds info
    rounds = phase_rounds.get(outcome.phase_name, 1)
    return max(0, rounds - 1)


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Load PhaseOutcome records and encode as features/targets.
    
    Returns:
        Tuple of (X features, Y targets, raw_records)
        X shape: (n_samples, n_features)
        Y shape: (n_samples, 3) for [retry_probability, expected_retries, failure_mode_index]
    """
    from deepthinker.orchestration.orchestration_store import OrchestrationStore
    
    encoder = PhaseRiskFeatureEncoder()
    store = OrchestrationStore()
    
    X_list: List[np.ndarray] = []
    Y_list: List[List[float]] = []
    raw_records: List[dict] = []
    
    skipped_count = 0
    
    # Group outcomes by mission to get phase_rounds info
    mission_outcomes = {}
    for outcome in store.read_outcomes():
        if outcome.mission_id not in mission_outcomes:
            mission_outcomes[outcome.mission_id] = []
        mission_outcomes[outcome.mission_id].append(outcome)
    
    # Process outcomes
    for mission_id, outcomes in mission_outcomes.items():
        # Build phase_rounds from mission outcomes
        phase_rounds = {}
        for outcome in outcomes:
            if outcome.phase_name not in phase_rounds:
                phase_rounds[outcome.phase_name] = 0
            phase_rounds[outcome.phase_name] += 1
        
        for idx, outcome in enumerate(outcomes):
            try:
                # Skip records with missing or invalid data
                if outcome.wall_time_seconds <= 0:
                    skipped_count += 1
                    continue
                
                # Reconstruct PhaseRiskContext from outcome
                constraints = outcome.constraints or {}
                time_budget_minutes = constraints.get("time_budget_minutes", 60)
                
                # Calculate retry count from phase_rounds
                retry_count = infer_retry_count(outcome, phase_rounds)
                
                ctx = PhaseRiskContext(
                    phase_name=outcome.phase_name,
                    phase_type=outcome.phase_type,
                    effort_level=outcome.effort_level,
                    iteration_index=idx,
                    retry_count_so_far=0,  # At prediction time, retries hadn't happened yet
                    mission_time_remaining_seconds=outcome.time_remaining_at_start,
                )
                
                # Reconstruct PhaseRiskExecutionPlan from outcome
                models_used = outcome.models_used or []
                model_names = [m[0] for m in models_used if len(m) >= 1]
                model_tier = models_used[0][1] if models_used and len(models_used[0]) >= 2 else "medium"
                
                max_iterations = constraints.get("max_iterations", 10)
                
                plan = PhaseRiskExecutionPlan(
                    model_tier=model_tier,
                    model_names=model_names,
                    councils_invoked=outcome.councils_invoked or [],
                    consensus_enabled=outcome.consensus_executed,
                    search_enabled=constraints.get("enable_internet", False),
                    max_iterations=max_iterations,
                )
                
                # SystemState - not available in historical data, use reasonable defaults
                vram_peak = outcome.vram_peak_mb or 15000
                state = PhaseRiskSystemState(
                    available_vram_mb=max(vram_peak + 5000, 30000),
                    gpu_load_ratio=0.5,
                    memory_pressure_ratio=0.3,
                )
                
                # Encode features
                features = encoder.encode(ctx, plan, state)
                
                # Determine failure mode
                failure_mode = infer_failure_mode(outcome)
                failure_mode_idx = KNOWN_FAILURE_MODES.index(failure_mode) if failure_mode in KNOWN_FAILURE_MODES else len(KNOWN_FAILURE_MODES) - 1
                
                # Calculate retry probability (1 if retry occurred, 0 otherwise)
                retry_probability = 1.0 if retry_count > 0 else 0.0
                
                # Targets: [retry_probability, expected_retries, failure_mode_index]
                targets = [
                    retry_probability,
                    float(retry_count),
                    float(failure_mode_idx),
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
    min_samples: int = 30,
    force: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Optional[dict]:
    """
    Train a phase risk prediction model.
    
    Args:
        min_samples: Minimum samples required to train
        force: Train even with fewer samples
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with training results, or None if training failed
    """
    logger.info("=" * 60)
    logger.info("Phase Risk Predictor Training")
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
    use_xgboost_threshold = PREDICTOR_CONFIG.get("use_xgboost_threshold", 120)
    
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
    
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    
    # Compute MAE for retry predictions
    retry_mae = float(mean_absolute_error(Y_val[:, 1], preds[:, 1]))
    
    # Compute AUC for retry probability (binary classification)
    try:
        # Clamp predictions to valid probability range
        retry_prob_preds = np.clip(preds[:, 0], 0, 1)
        retry_auc = float(roc_auc_score(Y_val[:, 0], retry_prob_preds))
    except ValueError:
        # AUC can fail if all samples have same class
        retry_auc = 0.5
        logger.warning("Could not compute AUC (possibly all samples same class)")
    
    # Failure mode accuracy (classification)
    failure_mode_preds = np.round(preds[:, 2]).astype(int)
    failure_mode_preds = np.clip(failure_mode_preds, 0, len(KNOWN_FAILURE_MODES) - 1)
    failure_mode_accuracy = float(np.mean(failure_mode_preds == Y_val[:, 2].astype(int)))
    
    validation_metrics = {
        "retry_mae": retry_mae,
        "retry_auc": retry_auc,
        "failure_mode_accuracy": failure_mode_accuracy,
    }
    
    logger.info("Validation Results:")
    logger.info(f"  Retry Count MAE: {retry_mae:.3f}")
    logger.info(f"  Retry Probability AUC: {retry_auc:.3f}")
    logger.info(f"  Failure Mode Accuracy: {failure_mode_accuracy:.3f}")
    
    # Save model
    registry = PhaseRiskModelRegistry()
    version = registry.get_next_version()
    
    metadata = ModelMetadata(
        version=version,
        trained_at=datetime.utcnow().isoformat(),
        dataset_size=len(X),
        feature_version=FEATURE_VECTOR_VERSION,
        validation_metrics=validation_metrics,
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
        encoder = PhaseRiskFeatureEncoder()
        feature_names = encoder.get_feature_names()
        
        # Get importances from first estimator (retry_probability)
        if hasattr(model.estimators_[0], 'feature_importances_'):
            importances = model.estimators_[0].feature_importances_
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 Feature Importances (retry prediction):")
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
        "validation_metrics": validation_metrics,
        "feature_version": FEATURE_VECTOR_VERSION,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train Phase Risk Predictor model"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum samples required to train (default: 30)"
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

