"""
Configuration for ML Influence Monitoring.

Central configuration for the influence monitoring and governance layer.
All settings are designed to be non-invasive and can be disabled entirely.
"""

from pathlib import Path
from typing import Dict, Any

# Base storage path for observability data
OBSERVABILITY_BASE_PATH = Path("kb/observability")

# Storage paths for influence monitoring
ML_INFLUENCE_EVENTS_PATH = OBSERVABILITY_BASE_PATH / "ml_influence_events.jsonl"
ML_INFLUENCE_METRICS_PATH = OBSERVABILITY_BASE_PATH / "ml_influence_metrics.jsonl"
ML_DRIFT_ALERTS_PATH = OBSERVABILITY_BASE_PATH / "ml_drift_alerts.jsonl"

# Main configuration
INFLUENCE_MONITORING_CONFIG: Dict[str, Any] = {
    # Master switch - if False, all monitoring is disabled
    "enabled": True,
    
    # Drift detection settings
    "drift_detection_window": 50,  # Number of events to analyze for drift
    "alert_severity_threshold": 0.7,  # Min severity (0-1) to emit alert
    
    # Metrics computation
    "metrics_flush_interval_minutes": 30,  # How often to compute and flush metrics
    "metrics_retention_days": 90,  # How long to retain computed metrics
    
    # Event logging
    "max_events_per_file": 100000,  # Rotate log after this many events
    "log_prediction_details": True,  # Include full prediction summary in events
    
    # Silent influence detection thresholds
    "correlation_alert_threshold": 0.75,  # Correlation above this triggers alert
    "entropy_decrease_threshold": 0.2,  # Entropy decrease above this is suspicious
    
    # Known predictor names for validation
    "known_predictors": [
        "cost_time",
        "phase_risk", 
        "web_search",
    ],
    
    # Predictor modes
    "valid_modes": [
        "shadow",    # Prediction only, no influence
        "advisory",  # Prediction shown to planner, not enforced
        "active",    # Prediction directly affects decisions
    ],
}


def is_monitoring_enabled() -> bool:
    """Check if influence monitoring is enabled."""
    return INFLUENCE_MONITORING_CONFIG.get("enabled", False)


def get_events_path() -> Path:
    """Get the path for influence events log."""
    return ML_INFLUENCE_EVENTS_PATH


def get_metrics_path() -> Path:
    """Get the path for computed metrics log."""
    return ML_INFLUENCE_METRICS_PATH


def get_alerts_path() -> Path:
    """Get the path for drift alerts log."""
    return ML_DRIFT_ALERTS_PATH


