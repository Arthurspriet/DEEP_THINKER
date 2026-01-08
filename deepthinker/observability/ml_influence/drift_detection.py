"""
Drift Detection for ML Influence Monitoring.

Detects silent influence, coupling, and distribution shifts in predictor behavior.
This is the critical safety component that answers:
"Is ML starting to steer the system without us explicitly allowing it?"

Detection Categories:
- influence_leak: Shadow mode predictor appears to affect decisions
- confidence_drift: Confidence distribution shifting unexpectedly  
- distribution_shift: Prediction distribution changing over time
- correlation_spike: Sudden correlation between predictions and decisions
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import MLInfluenceEvent, MLDriftAlert
from .config import (
    get_events_path,
    get_alerts_path,
    INFLUENCE_MONITORING_CONFIG,
)
from .metrics import MetricsEngine

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect drift and silent influence in ML predictor behavior.
    
    Analyzes event streams to identify concerning patterns:
    - Predictions correlating with decisions in shadow mode
    - Confidence distributions shifting over time
    - Decision entropy decreasing after predictor introduction
    """
    
    def __init__(
        self,
        events_path: Optional[Path] = None,
        alerts_path: Optional[Path] = None,
    ):
        """
        Initialize the drift detector.
        
        Args:
            events_path: Path to influence events log
            alerts_path: Path to drift alerts output
        """
        self._events_path = events_path or get_events_path()
        self._alerts_path = alerts_path or get_alerts_path()
        self._metrics_engine = MetricsEngine(events_path=self._events_path)
    
    def detect_all(
        self,
        window_size: Optional[int] = None,
    ) -> List[MLDriftAlert]:
        """
        Run all drift detection algorithms.
        
        Args:
            window_size: Number of recent events to analyze
                        (default from config)
        
        Returns:
            List of detected drift alerts
        """
        window_size = window_size or INFLUENCE_MONITORING_CONFIG.get(
            "drift_detection_window", 50
        )
        severity_threshold = INFLUENCE_MONITORING_CONFIG.get(
            "alert_severity_threshold", 0.7
        )
        
        alerts = []
        
        # Load recent events
        events = list(self._metrics_engine.read_events(limit=window_size * 2))
        if len(events) < 10:
            logger.debug("[DriftDetector] Not enough events for drift detection")
            return alerts
        
        # Run detection for each predictor
        known_predictors = INFLUENCE_MONITORING_CONFIG.get("known_predictors", [])
        
        for predictor_name in known_predictors:
            pred_events = [e for e in events if e.predictor_name == predictor_name]
            if len(pred_events) < 5:
                continue
            
            # 1. Influence leak detection (shadow mode affecting decisions)
            leak_alert = self._detect_influence_leak(pred_events)
            if leak_alert and leak_alert.severity >= severity_threshold:
                alerts.append(leak_alert)
            
            # 2. Confidence drift detection
            conf_alert = self._detect_confidence_drift(pred_events)
            if conf_alert and conf_alert.severity >= severity_threshold:
                alerts.append(conf_alert)
            
            # 3. Distribution shift detection
            dist_alert = self._detect_distribution_shift(pred_events)
            if dist_alert and dist_alert.severity >= severity_threshold:
                alerts.append(dist_alert)
        
        # 4. Cross-predictor correlation detection
        corr_alerts = self._detect_correlation_spikes(events)
        for alert in corr_alerts:
            if alert.severity >= severity_threshold:
                alerts.append(alert)
        
        return alerts
    
    def _detect_influence_leak(
        self,
        events: List[MLInfluenceEvent],
    ) -> Optional[MLDriftAlert]:
        """
        Detect if shadow mode predictions are leaking into decisions.
        
        Looks for correlation between prediction values and final decisions
        when the predictor is supposed to be in shadow mode (no influence).
        """
        # Filter to shadow mode events only
        shadow_events = [e for e in events if e.predictor_mode == "shadow"]
        
        if len(shadow_events) < 5:
            return None
        
        # Check if decisions are diverging in correlation with predictions
        # This requires both original and final decisions to be present
        analyzable = [
            e for e in shadow_events
            if e.planner_original_decision and e.planner_final_decision
        ]
        
        if len(analyzable) < 5:
            return None
        
        # Count how often final differs from original
        divergence_count = sum(
            1 for e in analyzable
            if e.planner_original_decision != e.planner_final_decision
        )
        divergence_rate = divergence_count / len(analyzable)
        
        # In pure shadow mode, divergence should be near zero
        # If it's high, predictions might be leaking into decisions
        if divergence_rate > 0.1:  # More than 10% divergence is suspicious
            severity = min(1.0, divergence_rate * 2)  # Scale to severity
            
            predictor_name = shadow_events[0].predictor_name
            timestamps = sorted(e.timestamp for e in shadow_events)
            
            return MLDriftAlert.create(
                predictor_name=predictor_name,
                drift_type="influence_leak",
                severity=severity,
                evidence={
                    "divergence_rate": divergence_rate,
                    "divergence_count": divergence_count,
                    "analyzable_count": len(analyzable),
                    "shadow_mode_events": len(shadow_events),
                },
                window_start=timestamps[0],
                window_end=timestamps[-1],
                event_count=len(shadow_events),
                recommendations=[
                    "Investigate how shadow predictions are reaching the planner",
                    "Check for direct coupling between predictor and decision logic",
                    "Consider adding isolation barriers",
                ],
            )
        
        return None
    
    def _detect_confidence_drift(
        self,
        events: List[MLInfluenceEvent],
    ) -> Optional[MLDriftAlert]:
        """
        Detect if confidence distribution is drifting over time.
        
        Compares confidence statistics between first and second half of window.
        """
        if len(events) < 10:
            return None
        
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Split into halves
        mid = len(sorted_events) // 2
        first_half = sorted_events[:mid]
        second_half = sorted_events[mid:]
        
        # Compute confidence statistics
        first_confs = [e.confidence for e in first_half]
        second_confs = [e.confidence for e in second_half]
        
        first_mean = np.mean(first_confs)
        second_mean = np.mean(second_confs)
        first_std = np.std(first_confs)
        second_std = np.std(second_confs)
        
        # Check for significant drift
        mean_drift = abs(second_mean - first_mean)
        std_drift = abs(second_std - first_std)
        
        # Combined drift score
        drift_score = mean_drift + std_drift * 0.5
        
        if drift_score > 0.15:  # Significant drift
            severity = min(1.0, drift_score * 3)
            
            predictor_name = events[0].predictor_name
            timestamps = sorted(e.timestamp for e in events)
            
            return MLDriftAlert.create(
                predictor_name=predictor_name,
                drift_type="confidence_drift",
                severity=severity,
                evidence={
                    "first_half_mean": float(first_mean),
                    "second_half_mean": float(second_mean),
                    "first_half_std": float(first_std),
                    "second_half_std": float(second_std),
                    "mean_drift": float(mean_drift),
                    "std_drift": float(std_drift),
                    "drift_score": float(drift_score),
                },
                window_start=timestamps[0],
                window_end=timestamps[-1],
                event_count=len(events),
                recommendations=[
                    "Review recent model updates or retraining",
                    "Check for changes in input data distribution",
                    "Consider recalibrating confidence thresholds",
                ],
            )
        
        return None
    
    def _detect_distribution_shift(
        self,
        events: List[MLInfluenceEvent],
    ) -> Optional[MLDriftAlert]:
        """
        Detect if prediction distribution is shifting.
        
        Uses entropy analysis to detect changes in prediction patterns.
        """
        if len(events) < 10:
            return None
        
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Split into halves
        mid = len(sorted_events) // 2
        first_half = sorted_events[:mid]
        second_half = sorted_events[mid:]
        
        # Compute fallback rate (proxy for distribution)
        first_fallback = sum(1 for e in first_half if e.used_fallback) / len(first_half)
        second_fallback = sum(1 for e in second_half if e.used_fallback) / len(second_half)
        
        fallback_shift = abs(second_fallback - first_fallback)
        
        # Compute mode distribution shift
        def mode_distribution(events_list):
            modes = defaultdict(int)
            for e in events_list:
                modes[e.predictor_mode] += 1
            total = len(events_list)
            return {k: v / total for k, v in modes.items()}
        
        first_modes = mode_distribution(first_half)
        second_modes = mode_distribution(second_half)
        
        # Calculate mode shift
        all_modes = set(first_modes.keys()) | set(second_modes.keys())
        mode_shift = sum(
            abs(first_modes.get(m, 0) - second_modes.get(m, 0))
            for m in all_modes
        ) / 2  # Normalize
        
        # Combined shift score
        shift_score = fallback_shift * 0.6 + mode_shift * 0.4
        
        if shift_score > 0.2:  # Significant shift
            severity = min(1.0, shift_score * 2)
            
            predictor_name = events[0].predictor_name
            timestamps = sorted(e.timestamp for e in events)
            
            return MLDriftAlert.create(
                predictor_name=predictor_name,
                drift_type="distribution_shift",
                severity=severity,
                evidence={
                    "first_fallback_rate": float(first_fallback),
                    "second_fallback_rate": float(second_fallback),
                    "fallback_shift": float(fallback_shift),
                    "first_mode_dist": first_modes,
                    "second_mode_dist": second_modes,
                    "mode_shift": float(mode_shift),
                    "shift_score": float(shift_score),
                },
                window_start=timestamps[0],
                window_end=timestamps[-1],
                event_count=len(events),
                recommendations=[
                    "Investigate changes in workload patterns",
                    "Check for model degradation",
                    "Review fallback trigger conditions",
                ],
            )
        
        return None
    
    def _detect_correlation_spikes(
        self,
        events: List[MLInfluenceEvent],
    ) -> List[MLDriftAlert]:
        """
        Detect sudden correlation spikes between predictors.
        
        Looks for unexpected coupling between different predictors.
        """
        alerts = []
        
        if len(events) < 20:
            return alerts
        
        # Group events by phase
        phase_events: Dict[str, List[MLInfluenceEvent]] = defaultdict(list)
        for event in events:
            phase_key = f"{event.mission_id}:{event.phase_name}"
            phase_events[phase_key].append(event)
        
        # Find phases with multiple predictors
        multi_predictor_phases = {
            k: v for k, v in phase_events.items()
            if len(set(e.predictor_name for e in v)) > 1
        }
        
        if len(multi_predictor_phases) < 5:
            return alerts
        
        # Analyze confidence correlation between predictors
        predictor_confidences: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        for phase_key, phase_evts in multi_predictor_phases.items():
            for event in phase_evts:
                predictor_confidences[event.predictor_name].append(
                    (phase_key, event.confidence)
                )
        
        # Check for high correlation between predictor pairs
        predictors = list(predictor_confidences.keys())
        correlation_threshold = INFLUENCE_MONITORING_CONFIG.get(
            "correlation_alert_threshold", 0.75
        )
        
        for i, pred1 in enumerate(predictors):
            for pred2 in predictors[i + 1:]:
                # Find common phases
                phases1 = {p for p, _ in predictor_confidences[pred1]}
                phases2 = {p for p, _ in predictor_confidences[pred2]}
                common_phases = phases1 & phases2
                
                if len(common_phases) < 5:
                    continue
                
                # Get confidence values for common phases
                conf1 = {p: c for p, c in predictor_confidences[pred1] if p in common_phases}
                conf2 = {p: c for p, c in predictor_confidences[pred2] if p in common_phases}
                
                # Compute correlation
                common_list = list(common_phases)
                vals1 = [conf1[p] for p in common_list]
                vals2 = [conf2[p] for p in common_list]
                
                if len(vals1) < 5:
                    continue
                
                correlation = np.corrcoef(vals1, vals2)[0, 1]
                
                if abs(correlation) > correlation_threshold:
                    severity = min(1.0, abs(correlation))
                    
                    timestamps = sorted(e.timestamp for e in events)
                    
                    alert = MLDriftAlert.create(
                        predictor_name=f"{pred1}+{pred2}",
                        drift_type="correlation_spike",
                        severity=severity,
                        evidence={
                            "predictor_pair": [pred1, pred2],
                            "correlation": float(correlation),
                            "common_phases": len(common_phases),
                            "correlation_threshold": correlation_threshold,
                        },
                        window_start=timestamps[0],
                        window_end=timestamps[-1],
                        event_count=len(events),
                        recommendations=[
                            f"Investigate coupling between {pred1} and {pred2}",
                            "Check for shared input features or dependencies",
                            "Consider if correlation is expected or concerning",
                        ],
                    )
                    alerts.append(alert)
        
        return alerts
    
    def run_detection_and_log(
        self,
        window_size: Optional[int] = None,
    ) -> List[MLDriftAlert]:
        """
        Run drift detection and log any alerts to file.
        
        Args:
            window_size: Number of events to analyze
            
        Returns:
            List of detected alerts
        """
        alerts = self.detect_all(window_size=window_size)
        
        if not alerts:
            logger.debug("[DriftDetector] No drift alerts detected")
            return alerts
        
        # Log alerts to file
        try:
            self._alerts_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._alerts_path, "a", encoding="utf-8") as f:
                for alert in alerts:
                    f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
            
            logger.warning(
                f"[DriftDetector] Detected {len(alerts)} drift alerts: "
                f"{[a.drift_type for a in alerts]}"
            )
            
        except Exception as e:
            logger.warning(f"[DriftDetector] Failed to log alerts: {e}")
        
        return alerts
    
    def get_recent_alerts(
        self,
        limit: int = 10,
    ) -> List[MLDriftAlert]:
        """
        Get recent drift alerts from the log.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts (newest first)
        """
        if not self._alerts_path.exists():
            return []
        
        alerts = []
        try:
            with open(self._alerts_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            alerts.append(MLDriftAlert.from_dict(data))
                        except (json.JSONDecodeError, TypeError):
                            continue
            
            # Sort by timestamp descending and limit
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            return alerts[:limit]
            
        except Exception as e:
            logger.warning(f"[DriftDetector] Failed to read alerts: {e}")
            return []


