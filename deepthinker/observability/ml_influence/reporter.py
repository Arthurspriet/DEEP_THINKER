"""
Reporter for ML Influence Monitoring.

Generates human-readable governance reports for ML predictor influence.
Provides summaries, alerts, and readiness assessments.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import MLInfluenceEvent, MLDriftAlert, MLInfluenceMetrics
from .config import (
    get_events_path,
    get_metrics_path,
    get_alerts_path,
    INFLUENCE_MONITORING_CONFIG,
)
from .metrics import MetricsEngine
from .drift_detection import DriftDetector

logger = logging.getLogger(__name__)


class MLInfluenceReporter:
    """
    Generate human-readable governance reports.
    
    Provides:
    - Per-predictor influence summaries
    - System-wide ML footprint analysis
    - Drift alerts summary
    - Readiness score for advisory mode activation
    """
    
    def __init__(
        self,
        events_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
        alerts_path: Optional[Path] = None,
    ):
        """
        Initialize the reporter.
        
        Args:
            events_path: Path to influence events
            metrics_path: Path to computed metrics
            alerts_path: Path to drift alerts
        """
        self._events_path = events_path or get_events_path()
        self._metrics_path = metrics_path or get_metrics_path()
        self._alerts_path = alerts_path or get_alerts_path()
        
        self._metrics_engine = MetricsEngine(
            events_path=self._events_path,
            metrics_path=self._metrics_path,
        )
        self._drift_detector = DriftDetector(
            events_path=self._events_path,
            alerts_path=self._alerts_path,
        )
    
    def generate_predictor_report(
        self,
        predictor_name: str,
    ) -> Dict[str, Any]:
        """
        Generate a detailed report for a specific predictor.
        
        Args:
            predictor_name: Name of the predictor (cost_time, phase_risk, web_search)
            
        Returns:
            Dictionary containing predictor analysis
        """
        # Get metrics
        metrics = self._metrics_engine.compute_predictor_metrics(predictor_name)
        
        if metrics.get("error") == "no_events":
            return {
                "predictor_name": predictor_name,
                "status": "no_data",
                "message": f"No events recorded for predictor '{predictor_name}'",
            }
        
        # Get recent events for trend analysis
        events = list(self._metrics_engine.read_events(
            filter_predictor=predictor_name,
            limit=100,
        ))
        
        # Compute health indicators
        health_score = self._compute_predictor_health(metrics, events)
        
        # Get recent alerts for this predictor
        recent_alerts = [
            a for a in self._drift_detector.get_recent_alerts(limit=20)
            if predictor_name in a.predictor_name
        ]
        
        return {
            "predictor_name": predictor_name,
            "status": "healthy" if health_score > 0.7 else "warning" if health_score > 0.4 else "critical",
            "health_score": health_score,
            "metrics": metrics,
            "recent_alerts": [a.to_dict() for a in recent_alerts[:5]],
            "recommendations": self._generate_recommendations(metrics, recent_alerts),
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generate a system-wide ML influence report.
        
        Returns:
            Dictionary containing full system analysis
        """
        # Get system metrics
        system_metrics = self._metrics_engine.compute_system_metrics()
        
        if system_metrics.get("error") == "no_events":
            return {
                "status": "no_data",
                "message": "No ML influence events recorded yet",
                "generated_at": datetime.utcnow().isoformat(),
            }
        
        # Get all recent alerts
        recent_alerts = self._drift_detector.get_recent_alerts(limit=20)
        
        # Compute overlap matrix
        overlap_matrix = self._metrics_engine.compute_overlap_matrix()
        
        # Generate per-predictor summaries
        predictor_reports = {}
        for predictor_name in INFLUENCE_MONITORING_CONFIG.get("known_predictors", []):
            report = self.generate_predictor_report(predictor_name)
            if report.get("status") != "no_data":
                predictor_reports[predictor_name] = report
        
        # Compute overall system health
        system_health = self._compute_system_health(
            system_metrics, predictor_reports, recent_alerts
        )
        
        return {
            "status": "healthy" if system_health > 0.7 else "warning" if system_health > 0.4 else "critical",
            "system_health_score": system_health,
            "system_metrics": system_metrics,
            "predictor_reports": predictor_reports,
            "overlap_matrix": overlap_matrix,
            "recent_alerts": [a.to_dict() for a in recent_alerts[:10]],
            "advisory_mode_readiness": self._compute_advisory_readiness(
                system_metrics, predictor_reports, recent_alerts
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def generate_weekly_report(self) -> str:
        """
        Generate a human-readable weekly report.
        
        Returns:
            Markdown-formatted report string
        """
        system_report = self.generate_system_report()
        
        if system_report.get("status") == "no_data":
            return "# ML Influence Weekly Report\n\nNo data available for reporting."
        
        lines = [
            "# ML Influence Weekly Report",
            f"\n**Generated:** {system_report['generated_at']}",
            f"\n**System Health:** {system_report['status'].upper()} ({system_report['system_health_score']:.2f})",
            "",
            "## Overview",
            "",
        ]
        
        # System metrics summary
        sm = system_report.get("system_metrics", {})
        lines.extend([
            f"- **Total Events:** {sm.get('total_events', 0)}",
            f"- **Unique Phases:** {sm.get('unique_phases', 0)}",
            f"- **Active Predictors:** {', '.join(sm.get('predictors_active', []))}",
            "",
        ])
        
        # Advisory mode readiness
        readiness = system_report.get("advisory_mode_readiness", {})
        lines.extend([
            "## Advisory Mode Readiness",
            "",
            f"**Overall Score:** {readiness.get('overall_score', 0):.2f}/1.00",
            "",
        ])
        
        for criterion, details in readiness.get("criteria", {}).items():
            status_emoji = "✅" if details.get("passed") else "❌"
            lines.append(f"- {status_emoji} **{criterion}:** {details.get('message', 'N/A')}")
        
        lines.append("")
        
        # Per-predictor summary
        lines.extend([
            "## Predictor Summary",
            "",
        ])
        
        for pred_name, pred_report in system_report.get("predictor_reports", {}).items():
            metrics = pred_report.get("metrics", {})
            lines.extend([
                f"### {pred_name}",
                "",
                f"- **Status:** {pred_report.get('status', 'unknown').upper()}",
                f"- **Predictions:** {metrics.get('prediction_count', 0)}",
                f"- **Fallback Rate:** {metrics.get('fallback_rate', 0):.1%}",
                f"- **Avg Confidence:** {metrics.get('avg_confidence', 0):.2f}",
                "",
            ])
        
        # Alerts summary
        alerts = system_report.get("recent_alerts", [])
        if alerts:
            lines.extend([
                "## Recent Alerts",
                "",
            ])
            for alert in alerts[:5]:
                severity_indicator = "🔴" if alert["severity"] > 0.8 else "🟡" if alert["severity"] > 0.5 else "🟢"
                lines.append(
                    f"- {severity_indicator} **{alert['drift_type']}** ({alert['predictor_name']}): "
                    f"severity {alert['severity']:.2f}"
                )
            lines.append("")
        else:
            lines.extend([
                "## Recent Alerts",
                "",
                "No drift alerts detected. ✅",
                "",
            ])
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            "",
        ])
        
        all_recommendations = set()
        for pred_report in system_report.get("predictor_reports", {}).values():
            all_recommendations.update(pred_report.get("recommendations", []))
        
        if all_recommendations:
            for rec in list(all_recommendations)[:5]:
                lines.append(f"- {rec}")
        else:
            lines.append("- No action items at this time")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _compute_predictor_health(
        self,
        metrics: Dict[str, Any],
        events: List[MLInfluenceEvent],
    ) -> float:
        """Compute health score for a predictor (0-1)."""
        if not events:
            return 0.5  # Neutral if no data
        
        score = 1.0
        
        # Penalize high fallback rate (indicates model issues)
        fallback_rate = metrics.get("fallback_rate", 0)
        if fallback_rate > 0.5:
            score -= 0.3
        elif fallback_rate > 0.3:
            score -= 0.1
        
        # Penalize low confidence
        avg_confidence = metrics.get("avg_confidence", 0.5)
        if avg_confidence < 0.4:
            score -= 0.2
        elif avg_confidence < 0.6:
            score -= 0.1
        
        # Penalize high confidence variance (unstable)
        std_confidence = metrics.get("std_confidence", 0)
        if std_confidence > 0.3:
            score -= 0.2
        elif std_confidence > 0.2:
            score -= 0.1
        
        # Penalize unexpected divergence in shadow mode
        divergence_rate = metrics.get("divergence_rate")
        if divergence_rate is not None and divergence_rate > 0.1:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _compute_system_health(
        self,
        system_metrics: Dict[str, Any],
        predictor_reports: Dict[str, Dict[str, Any]],
        recent_alerts: List[MLDriftAlert],
    ) -> float:
        """Compute overall system health score (0-1)."""
        score = 1.0
        
        # Average predictor health
        if predictor_reports:
            avg_predictor_health = sum(
                r.get("health_score", 0.5) for r in predictor_reports.values()
            ) / len(predictor_reports)
            score = (score + avg_predictor_health) / 2
        
        # Penalize for recent high-severity alerts
        high_severity_alerts = sum(1 for a in recent_alerts if a.severity > 0.7)
        if high_severity_alerts > 3:
            score -= 0.3
        elif high_severity_alerts > 1:
            score -= 0.15
        elif high_severity_alerts > 0:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _compute_advisory_readiness(
        self,
        system_metrics: Dict[str, Any],
        predictor_reports: Dict[str, Dict[str, Any]],
        recent_alerts: List[MLDriftAlert],
    ) -> Dict[str, Any]:
        """
        Compute readiness score for enabling advisory mode.
        
        Advisory mode should only be enabled when:
        - Predictors have sufficient track record
        - No active drift alerts
        - Consistent prediction accuracy
        - No silent influence detected
        """
        criteria = {}
        
        # 1. Sufficient data volume
        total_events = system_metrics.get("total_events", 0)
        data_sufficient = total_events >= 100
        criteria["data_volume"] = {
            "passed": data_sufficient,
            "message": f"{total_events} events recorded (minimum: 100)",
            "score": min(1.0, total_events / 100),
        }
        
        # 2. No high-severity alerts
        high_severity = sum(1 for a in recent_alerts if a.severity > 0.7)
        no_critical_alerts = high_severity == 0
        criteria["no_critical_alerts"] = {
            "passed": no_critical_alerts,
            "message": f"{high_severity} high-severity alerts" if high_severity else "No high-severity alerts",
            "score": 1.0 if high_severity == 0 else max(0.0, 1.0 - high_severity * 0.25),
        }
        
        # 3. No influence leaks detected
        leak_alerts = [a for a in recent_alerts if a.drift_type == "influence_leak"]
        no_leaks = len(leak_alerts) == 0
        criteria["no_influence_leaks"] = {
            "passed": no_leaks,
            "message": f"{len(leak_alerts)} leak alerts" if leak_alerts else "No influence leaks detected",
            "score": 1.0 if no_leaks else 0.0,
        }
        
        # 4. Healthy predictors
        healthy_predictors = sum(
            1 for r in predictor_reports.values()
            if r.get("health_score", 0) > 0.6
        )
        total_predictors = len(predictor_reports)
        all_healthy = healthy_predictors == total_predictors and total_predictors > 0
        criteria["predictor_health"] = {
            "passed": all_healthy,
            "message": f"{healthy_predictors}/{total_predictors} predictors healthy",
            "score": healthy_predictors / max(1, total_predictors),
        }
        
        # 5. Low fallback rates
        avg_fallback = 0.0
        if predictor_reports:
            avg_fallback = sum(
                r.get("metrics", {}).get("fallback_rate", 0)
                for r in predictor_reports.values()
            ) / len(predictor_reports)
        low_fallback = avg_fallback < 0.3
        criteria["low_fallback_rate"] = {
            "passed": low_fallback,
            "message": f"Average fallback rate: {avg_fallback:.1%}",
            "score": max(0.0, 1.0 - avg_fallback),
        }
        
        # Compute overall readiness
        overall_score = sum(c["score"] for c in criteria.values()) / len(criteria)
        all_passed = all(c["passed"] for c in criteria.values())
        
        return {
            "ready": all_passed and overall_score > 0.8,
            "overall_score": overall_score,
            "criteria": criteria,
            "recommendation": (
                "System is ready for advisory mode activation"
                if all_passed and overall_score > 0.8
                else "Address failing criteria before enabling advisory mode"
            ),
        }
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        alerts: List[MLDriftAlert],
    ) -> List[str]:
        """Generate recommendations based on metrics and alerts."""
        recommendations = []
        
        # High fallback rate
        if metrics.get("fallback_rate", 0) > 0.3:
            recommendations.append(
                "High fallback rate detected - consider retraining the model "
                "or reviewing confidence thresholds"
            )
        
        # Low confidence
        if metrics.get("avg_confidence", 0.5) < 0.5:
            recommendations.append(
                "Low average confidence - review model calibration "
                "and training data coverage"
            )
        
        # From alerts
        alert_types = set(a.drift_type for a in alerts)
        if "influence_leak" in alert_types:
            recommendations.append(
                "CRITICAL: Shadow mode influence leak detected - "
                "investigate coupling between predictor and planner"
            )
        if "confidence_drift" in alert_types:
            recommendations.append(
                "Confidence distribution drifting - consider "
                "recalibration or model refresh"
            )
        if "distribution_shift" in alert_types:
            recommendations.append(
                "Prediction distribution shift detected - "
                "review workload patterns and model inputs"
            )
        
        return recommendations


