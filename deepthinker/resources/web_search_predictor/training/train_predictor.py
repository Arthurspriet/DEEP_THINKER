#!/usr/bin/env python3
"""
Training Pipeline for Web Search Predictor.

Trains ML models on historical PhaseOutcome data to predict
web search necessity (search_required, expected_queries, hallucination_risk).

Usage:
    python train_predictor.py --min-samples 40
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

from deepthinker.resources.web_search_predictor.schemas import (
    WebSearchContext,
    WebSearchExecutionPlan,
    WebSearchSystemState,
)
from deepthinker.resources.web_search_predictor.config import (
    PREDICTOR_CONFIG,
    FEATURE_VECTOR_VERSION,
)
from deepthinker.resources.web_search_predictor.feature_encoder import (
    WebSearchFeatureEncoder,
    contains_dates,
    contains_named_entities,
    contains_factual_claims,
)
from deepthinker.resources.web_search_predictor.model_registry import (
    WebSearchModelRegistry,
    ModelMetadata,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def infer_hallucination_occurred(outcome) -> bool:
    """
    Infer whether hallucination occurred from PhaseOutcome.
    
    Args:
        outcome: PhaseOutcome record
        
    Returns:
        True if hallucination indicators are present
    """
    # Check arbiter output for hallucination mentions
    if outcome.arbiter_raw_output:
        arbiter_lower = outcome.arbiter_raw_output.lower()
        if "hallucin" in arbiter_lower:
            return True
        if "factual error" in arbiter_lower or "incorrect fact" in arbiter_lower:
            return True
        if "unverified claim" in arbiter_lower:
            return True
    
    # Low quality score might indicate hallucination
    if outcome.quality_score is not None and outcome.quality_score < 0.3:
        return True
    
    return False


def infer_search_used(outcome) -> Tuple[bool, int]:
    """
    Infer whether web search was used and how many queries.
    
    Args:
        outcome: PhaseOutcome record
        
    Returns:
        Tuple of (search_used, num_queries)
    """
    constraints = outcome.constraints or {}
    
    # Check if search was enabled
    search_enabled = constraints.get("enable_internet", False) or constraints.get("allow_internet", False)
    
    if not search_enabled:
        return False, 0
    
    # Check arbiter output for search indicators
    if outcome.arbiter_raw_output:
        arbiter_lower = outcome.arbiter_raw_output.lower()
        if "web search" in arbiter_lower or "searched" in arbiter_lower:
            # Try to count queries from output
            # Default to 2 if we know search was used but can't count
            return True, 2
    
    # If search was enabled for research phases, assume it was used
    if outcome.phase_type.lower() in ["research", "deep_analysis"]:
        return True, 2
    
    return False, 0


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Load PhaseOutcome records and encode as features/targets.
    
    Returns:
        Tuple of (X features, Y targets, raw_records)
        X shape: (n_samples, n_features)
        Y shape: (n_samples, 3) for [search_probability, expected_queries, hallucination_risk]
    """
    from deepthinker.orchestration.orchestration_store import OrchestrationStore
    
    encoder = WebSearchFeatureEncoder()
    store = OrchestrationStore()
    
    X_list: List[np.ndarray] = []
    Y_list: List[List[float]] = []
    raw_records: List[dict] = []
    
    skipped_count = 0
    
    # Process outcomes
    for outcome in store.read_outcomes():
        try:
            # Skip records with missing or invalid data
            if outcome.wall_time_seconds <= 0:
                skipped_count += 1
                continue
            
            # Reconstruct WebSearchContext from outcome
            constraints = outcome.constraints or {}
            
            # Analyze content for search-relevant patterns
            # Use arbiter output as proxy for phase content
            content_to_analyze = outcome.arbiter_raw_output or ""
            
            ctx = WebSearchContext(
                phase_name=outcome.phase_name,
                phase_type=outcome.phase_type,
                effort_level=outcome.effort_level,
                iteration_index=0,  # Not tracked in outcome
                prompt_token_count=outcome.tokens_consumed // 2,  # Rough estimate
                contains_dates=contains_dates(content_to_analyze),
                contains_named_entities=contains_named_entities(content_to_analyze),
                contains_factual_claims=contains_factual_claims(content_to_analyze),
            )
            
            # Reconstruct WebSearchExecutionPlan from outcome
            models_used = outcome.models_used or []
            model_names = [m[0] for m in models_used if len(m) >= 1]
            model_tier = models_used[0][1] if models_used and len(models_used[0]) >= 2 else "medium"
            
            search_enabled = constraints.get("enable_internet", False) or constraints.get("allow_internet", False)
            max_iterations = constraints.get("max_iterations", 10)
            
            plan = WebSearchExecutionPlan(
                model_tier=model_tier,
                model_names=model_names,
                councils_invoked=outcome.councils_invoked or [],
                consensus_enabled=outcome.consensus_executed,
                search_enabled_by_planner=search_enabled,
                max_iterations=max_iterations,
            )
            
            # SystemState
            state = WebSearchSystemState(
                available_time_seconds=outcome.time_remaining_at_start,
            )
            
            # Encode features
            features = encoder.encode(ctx, plan, state)
            
            # Determine targets
            search_used, num_queries = infer_search_used(outcome)
            hallucination_occurred = infer_hallucination_occurred(outcome)
            
            # Targets: [search_probability, expected_queries, hallucination_risk]
            targets = [
                1.0 if search_used else 0.0,
                float(num_queries),
                1.0 if hallucination_occurred else 0.0,
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
    min_samples: int = 40,
    force: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Optional[dict]:
    """
    Train a web search prediction model.
    
    Args:
        min_samples: Minimum samples required to train
        force: Train even with fewer samples
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with training results, or None if training failed
    """
    logger.info("=" * 60)
    logger.info("Web Search Predictor Training")
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
    use_xgboost_threshold = PREDICTOR_CONFIG.get("use_xgboost_threshold", 150)
    
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
    
    # Compute metrics
    # Search prediction AUC (binary classification)
    try:
        search_prob_preds = np.clip(preds[:, 0], 0, 1)
        search_auc = float(roc_auc_score(Y_val[:, 0], search_prob_preds))
    except ValueError:
        search_auc = 0.5
        logger.warning("Could not compute search AUC (possibly all samples same class)")
    
    # Query count MAE
    query_mae = float(mean_absolute_error(Y_val[:, 1], preds[:, 1]))
    
    # Hallucination prediction AUC
    try:
        halluc_prob_preds = np.clip(preds[:, 2], 0, 1)
        hallucination_auc = float(roc_auc_score(Y_val[:, 2], halluc_prob_preds))
    except ValueError:
        hallucination_auc = 0.5
        logger.warning("Could not compute hallucination AUC (possibly all samples same class)")
    
    # Hallucination MAE (for regression view)
    hallucination_mae = float(mean_absolute_error(Y_val[:, 2], np.clip(preds[:, 2], 0, 1)))
    
    validation_metrics = {
        "search_auc": search_auc,
        "query_mae": query_mae,
        "hallucination_auc": hallucination_auc,
        "hallucination_mae": hallucination_mae,
    }
    
    logger.info("Validation Results:")
    logger.info(f"  Search Required AUC: {search_auc:.3f}")
    logger.info(f"  Query Count MAE: {query_mae:.3f}")
    logger.info(f"  Hallucination Risk AUC: {hallucination_auc:.3f}")
    logger.info(f"  Hallucination Risk MAE: {hallucination_mae:.3f}")
    
    # Save model
    registry = WebSearchModelRegistry()
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
        encoder = WebSearchFeatureEncoder()
        feature_names = encoder.get_feature_names()
        
        # Get importances from first estimator (search_probability)
        if hasattr(model.estimators_[0], 'feature_importances_'):
            importances = model.estimators_[0].feature_importances_
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 Feature Importances (search prediction):")
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
        description="Train Web Search Predictor model"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=40,
        help="Minimum samples required to train (default: 40)"
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

