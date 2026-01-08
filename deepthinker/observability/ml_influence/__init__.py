"""
ML Influence Monitoring & Governance Layer.

A centralized monitoring system that tracks, quantifies, and audits
how ML predictors influence DeepThinker's architecture, decisions,
and outcomes over time.

This module provides:
- Unified event schema for all predictors
- Thread-safe influence tracking
- Offline metrics computation
- Silent influence / drift detection
- Human-readable governance reports

Safety Invariants:
- Passive observation only - no control over execution
- No modification of planner or agent logic
- Append-only logs
- System functions with monitoring disabled
"""

from .schemas import MLInfluenceEvent, MLDriftAlert
from .config import INFLUENCE_MONITORING_CONFIG
from .influence_tracker import MLInfluenceTracker, get_influence_tracker
from .metrics import MetricsEngine
from .drift_detection import DriftDetector
from .reporter import MLInfluenceReporter

__all__ = [
    # Schemas
    "MLInfluenceEvent",
    "MLDriftAlert",
    # Config
    "INFLUENCE_MONITORING_CONFIG",
    # Core components
    "MLInfluenceTracker",
    "get_influence_tracker",
    "MetricsEngine",
    "DriftDetector",
    "MLInfluenceReporter",
]


