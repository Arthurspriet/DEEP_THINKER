"""
Metrics Engine for ML Influence Monitoring.

Computes derived influence metrics from event logs, either offline
or periodically during system operation.

Metric Categories:
- Predictor-Level: frequency, fallback rate, confidence statistics
- Influence-Level: decision divergence, outcome correlation
- System-Level: predictor coverage, overlap analysis
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .schemas import MLInfluenceEvent, MLInfluenceMetrics
from .config import (
    get_events_path,
    get_metrics_path,
    INFLUENCE_MONITORING_CONFIG,
)

logger = logging.getLogger(__name__)


class MetricsEngine:
    """
    Compute influence metrics from event logs.
    
    Provides both real-time and batch computation of metrics
    for governance reporting and drift detection.
    """
    
    def __init__(
        self,
        events_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ):
        """
        Initialize the metrics engine.
        
        Args:
            events_path: Path to influence events log
            metrics_path: Path to computed metrics output
        """
        self._events_path = events_path or get_events_path()
        self._metrics_path = metrics_path or get_metrics_path()
    
    def read_events(
        self,
        filter_predictor: Optional[str] = None,
        filter_mission: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[MLInfluenceEvent]:
        """
        Read events from the log with optional filtering.
        
        Args:
            filter_predictor: Only return events for this predictor
            filter_mission: Only return events for this mission
            start_time: Only events after this ISO timestamp
            end_time: Only events before this ISO timestamp
            limit: Maximum number of events to return
            
        Yields:
            MLInfluenceEvent instances matching filters
        """
        if not self._events_path.exists():
            return
        
        count = 0
        try:
            with open(self._events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Apply filters
                        if filter_predictor and data.get("predictor_name") != filter_predictor:
                            continue
                        if filter_mission and data.get("mission_id") != filter_mission:
                            continue
                        if start_time and data.get("timestamp", "") < start_time:
                            continue
                        if end_time and data.get("timestamp", "") > end_time:
                            continue
                        
                        yield MLInfluenceEvent.from_dict(data)
                        count += 1
                        
                        if limit and count >= limit:
                            return
                            
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        logger.debug(f"[MetricsEngine] Failed to parse event: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"[MetricsEngine] Failed to read events: {e}")
    
    def compute_predictor_metrics(
        self,
        predictor_name: str,
        events: Optional[List[MLInfluenceEvent]] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics for a specific predictor.
        
        Args:
            predictor_name: Name of the predictor
            events: Pre-loaded events (if None, reads from file)
            
        Returns:
            Dictionary of computed metrics
        """
        if events is None:
            events = list(self.read_events(filter_predictor=predictor_name))
        
        if not events:
            return {
                "predictor_name": predictor_name,
                "prediction_count": 0,
                "error": "no_events",
            }
        
        # Basic counts
        total = len(events)
        fallback_count = sum(1 for e in events if e.used_fallback)
        
        # Confidence statistics
        confidences = [e.confidence for e in events]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        std_confidence = float(np.std(confidences)) if len(confidences) > 1 else 0.0
        
        # Confidence distribution
        confidence_buckets = {
            "low_0_0.3": sum(1 for c in confidences if c < 0.3),
            "medium_0.3_0.7": sum(1 for c in confidences if 0.3 <= c < 0.7),
            "high_0.7_1.0": sum(1 for c in confidences if c >= 0.7),
        }
        
        # Decision divergence (when we have both original and final decisions)
        divergence_count = 0
        divergence_analyzable = 0
        for event in events:
            if event.planner_original_decision and event.planner_final_decision:
                divergence_analyzable += 1
                if event.planner_original_decision != event.planner_final_decision:
                    divergence_count += 1
        
        # Mode distribution
        mode_counts = defaultdict(int)
        for event in events:
            mode_counts[event.predictor_mode] += 1
        
        # Version distribution
        version_counts = defaultdict(int)
        for event in events:
            version_counts[event.predictor_version] += 1
        
        # Time range
        timestamps = sorted(e.timestamp for e in events)
        
        return {
            "predictor_name": predictor_name,
            "prediction_count": total,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / total if total > 0 else 0.0,
            "avg_confidence": avg_confidence,
            "std_confidence": std_confidence,
            "confidence_distribution": confidence_buckets,
            "divergence_analyzable": divergence_analyzable,
            "divergence_count": divergence_count,
            "divergence_rate": divergence_count / divergence_analyzable if divergence_analyzable > 0 else None,
            "mode_distribution": dict(mode_counts),
            "version_distribution": dict(version_counts),
            "time_range": {
                "earliest": timestamps[0] if timestamps else None,
                "latest": timestamps[-1] if timestamps else None,
            },
        }
    
    def compute_system_metrics(
        self,
        events: Optional[List[MLInfluenceEvent]] = None,
    ) -> Dict[str, Any]:
        """
        Compute system-wide metrics across all predictors.
        
        Args:
            events: Pre-loaded events (if None, reads from file)
            
        Returns:
            Dictionary of system-wide metrics
        """
        if events is None:
            events = list(self.read_events())
        
        if not events:
            return {
                "total_events": 0,
                "error": "no_events",
            }
        
        # Group by predictor
        by_predictor: Dict[str, List[MLInfluenceEvent]] = defaultdict(list)
        for event in events:
            by_predictor[event.predictor_name].append(event)
        
        # Group by phase (mission_id + phase_name)
        phases_with_predictions: set = set()
        for event in events:
            phase_key = f"{event.mission_id}:{event.phase_name}"
            phases_with_predictions.add(phase_key)
        
        # Predictor overlap analysis
        # Count how often multiple predictors fire for the same phase
        phase_predictors: Dict[str, set] = defaultdict(set)
        for event in events:
            phase_key = f"{event.mission_id}:{event.phase_name}"
            phase_predictors[phase_key].add(event.predictor_name)
        
        overlap_counts = defaultdict(int)
        for phase_key, predictors in phase_predictors.items():
            num_predictors = len(predictors)
            overlap_counts[num_predictors] += 1
        
        # Compute per-predictor summaries
        predictor_summaries = {}
        for pred_name, pred_events in by_predictor.items():
            predictor_summaries[pred_name] = {
                "count": len(pred_events),
                "fallback_rate": sum(1 for e in pred_events if e.used_fallback) / len(pred_events),
                "avg_confidence": float(np.mean([e.confidence for e in pred_events])),
            }
        
        # Time range
        timestamps = sorted(e.timestamp for e in events)
        
        return {
            "total_events": len(events),
            "unique_phases": len(phases_with_predictions),
            "predictors_active": list(by_predictor.keys()),
            "predictor_summaries": predictor_summaries,
            "overlap_distribution": dict(overlap_counts),
            "phases_with_multiple_predictors": sum(
                1 for p in phase_predictors.values() if len(p) > 1
            ),
            "time_range": {
                "earliest": timestamps[0] if timestamps else None,
                "latest": timestamps[-1] if timestamps else None,
            },
        }
    
    def compute_overlap_matrix(
        self,
        events: Optional[List[MLInfluenceEvent]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute predictor co-occurrence matrix.
        
        Shows how often pairs of predictors fire on the same phase.
        
        Returns:
            Nested dict: matrix[pred1][pred2] = count of co-occurrences
        """
        if events is None:
            events = list(self.read_events())
        
        # Group predictors by phase
        phase_predictors: Dict[str, set] = defaultdict(set)
        for event in events:
            phase_key = f"{event.mission_id}:{event.phase_name}"
            phase_predictors[phase_key].add(event.predictor_name)
        
        # Get all predictor names
        all_predictors = set()
        for predictors in phase_predictors.values():
            all_predictors.update(predictors)
        
        # Build matrix
        matrix: Dict[str, Dict[str, int]] = {
            p1: {p2: 0 for p2 in all_predictors}
            for p1 in all_predictors
        }
        
        for predictors in phase_predictors.values():
            pred_list = list(predictors)
            for i, p1 in enumerate(pred_list):
                for p2 in pred_list[i:]:
                    matrix[p1][p2] += 1
                    if p1 != p2:
                        matrix[p2][p1] += 1
        
        return matrix
    
    def flush_metrics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> bool:
        """
        Compute and persist metrics snapshot.
        
        Args:
            period_start: Start of analysis period (ISO timestamp)
            period_end: End of analysis period (ISO timestamp)
            
        Returns:
            True if metrics were written successfully
        """
        try:
            # Load events for period
            events = list(self.read_events(
                start_time=period_start,
                end_time=period_end,
            ))
            
            if not events:
                logger.debug("[MetricsEngine] No events to compute metrics from")
                return False
            
            # Compute system metrics
            system_metrics = self.compute_system_metrics(events)
            
            # Compute per-predictor metrics
            predictor_metrics = {}
            for predictor_name in INFLUENCE_MONITORING_CONFIG.get("known_predictors", []):
                pred_events = [e for e in events if e.predictor_name == predictor_name]
                if pred_events:
                    predictor_metrics[predictor_name] = self.compute_predictor_metrics(
                        predictor_name, pred_events
                    )
            
            # Create metrics snapshot
            snapshot = {
                "timestamp": datetime.utcnow().isoformat(),
                "period_start": period_start or (events[0].timestamp if events else None),
                "period_end": period_end or (events[-1].timestamp if events else None),
                "system_metrics": system_metrics,
                "predictor_metrics": predictor_metrics,
                "overlap_matrix": self.compute_overlap_matrix(events),
            }
            
            # Ensure directory exists
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to metrics log
            with open(self._metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            
            logger.info(
                f"[MetricsEngine] Flushed metrics: {len(events)} events, "
                f"{len(predictor_metrics)} predictors"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[MetricsEngine] Failed to flush metrics: {e}")
            return False
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently computed metrics snapshot.
        
        Returns:
            Latest metrics dict or None if no metrics exist
        """
        if not self._metrics_path.exists():
            return None
        
        try:
            # Read last line
            with open(self._metrics_path, "r", encoding="utf-8") as f:
                last_line = None
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                
                if last_line:
                    return json.loads(last_line)
                return None
                
        except Exception as e:
            logger.warning(f"[MetricsEngine] Failed to read latest metrics: {e}")
            return None


