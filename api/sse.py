"""
Server-Sent Events (SSE) Manager for DeepThinker.

Provides real-time event streaming for mission updates.
"""

import asyncio
import json
from typing import Dict, Set, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class SSEEvent:
    """A single SSE event."""
    event_type: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def format(self) -> str:
        """Format as SSE message."""
        payload = {
            "type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(payload)}\n\n"


class SSEManager:
    """
    Manages SSE connections and event broadcasting.
    
    Supports multiple subscribers per mission with async event streaming.
    """
    
    def __init__(self):
        self._queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def subscribe(self, mission_id: str) -> AsyncGenerator[str, None]:
        """
        Subscribe to events for a mission.
        
        Yields SSE-formatted event strings.
        """
        queue: asyncio.Queue = asyncio.Queue()
        
        async with self._lock:
            self._queues[mission_id].add(queue)
        
        try:
            # Send initial connection event
            yield SSEEvent(
                event_type="connected",
                data={"mission_id": mission_id, "message": "SSE connection established"}
            ).format()
            
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event.format()
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            async with self._lock:
                self._queues[mission_id].discard(queue)
                if not self._queues[mission_id]:
                    del self._queues[mission_id]
    
    async def publish(self, mission_id: str, event: SSEEvent) -> int:
        """
        Publish an event to all subscribers of a mission.
        
        Returns the number of subscribers notified.
        """
        async with self._lock:
            queues = self._queues.get(mission_id, set()).copy()
        
        import logging
        logger = logging.getLogger(__name__)
        
        for queue in queues:
            try:
                await queue.put(event)
            except asyncio.QueueFull:
                logger.warning(f"SSE queue full for mission {mission_id}, dropping event")
            except Exception as e:
                logger.debug(f"Failed to publish SSE event: {e}")
        
        return len(queues)
    
    async def publish_phase_started(self, mission_id: str, phase_name: str, phase_index: int):
        """Publish phase_started event."""
        await self.publish(mission_id, SSEEvent(
            event_type="phase_started",
            data={"phase_name": phase_name, "phase_index": phase_index}
        ))
    
    async def publish_phase_completed(self, mission_id: str, phase_name: str, phase_index: int, artifacts: Dict[str, Any] = None):
        """Publish phase_completed event."""
        await self.publish(mission_id, SSEEvent(
            event_type="phase_completed",
            data={
                "phase_name": phase_name,
                "phase_index": phase_index,
                "artifacts": artifacts or {}
            }
        ))
    
    async def publish_council_started(self, mission_id: str, council_name: str, models: list):
        """Publish council_started event."""
        await self.publish(mission_id, SSEEvent(
            event_type="council_started",
            data={"council_name": council_name, "models": models}
        ))
    
    async def publish_council_completed(self, mission_id: str, council_name: str, success: bool, duration_s: float = None):
        """Publish council_completed event."""
        await self.publish(mission_id, SSEEvent(
            event_type="council_completed",
            data={
                "council_name": council_name,
                "success": success,
                "duration_s": duration_s
            }
        ))
    
    async def publish_artifact_generated(self, mission_id: str, artifact_name: str, artifact_type: str):
        """Publish artifact_generated event."""
        await self.publish(mission_id, SSEEvent(
            event_type="artifact_generated",
            data={"artifact_name": artifact_name, "artifact_type": artifact_type}
        ))
    
    async def publish_log_added(self, mission_id: str, log_message: str, level: str = "info"):
        """Publish log_added event."""
        await self.publish(mission_id, SSEEvent(
            event_type="log_added",
            data={"message": log_message, "level": level}
        ))
    
    async def publish_mission_completed(self, mission_id: str, status: str, final_artifacts: Dict[str, Any] = None):
        """Publish mission_completed event."""
        await self.publish(mission_id, SSEEvent(
            event_type="mission_completed",
            data={
                "status": status,
                "final_artifacts": final_artifacts or {}
            }
        ))
    
    async def publish_meta_update(self, mission_id: str, phase_name: str, meta_result: Dict[str, Any]):
        """Publish meta_update event for meta-cognition layer updates."""
        await self.publish(mission_id, SSEEvent(
            event_type="meta_update",
            data={
                "phase_name": phase_name,
                "reflection": meta_result.get("reflection", {}),
                "hypotheses": meta_result.get("hypotheses", {}),
                "debate": meta_result.get("debate", {}),
                "revision": meta_result.get("revision", {}),
                "success": meta_result.get("success", False),
            }
        ))
    
    # =========================================================================
    # RICH EXECUTION EVENTS (for frontend live panels)
    # =========================================================================
    
    async def publish_step_execution(
        self,
        mission_id: str,
        step_name: str,
        step_type: str,
        model_used: str,
        duration_s: float,
        status: str,
        attempts: int = 1,
        output_preview: str = None,
        pivot_suggestion: str = None,
        error: str = None,
        artifacts: Dict[str, str] = None
    ):
        """Publish step_execution event with rich execution details."""
        await self.publish(mission_id, SSEEvent(
            event_type="step_execution",
            data={
                "step_name": step_name,
                "step_type": step_type,
                "model_used": model_used,
                "duration_s": duration_s,
                "status": status,
                "attempts": attempts,
                "output_preview": output_preview[:500] if output_preview and len(output_preview) > 500 else output_preview,
                "pivot_suggestion": pivot_suggestion,
                "error": error,
                "artifacts": list(artifacts.keys()) if artifacts else []
            }
        ))
    
    async def publish_model_selection(
        self,
        mission_id: str,
        phase_name: str,
        models: list,
        reason: str,
        downgraded: bool = False,
        wait_for_capacity: bool = False,
        fallback_models: list = None,
        phase_importance: float = 0.5,
        estimated_vram: int = 0,
        time_remaining: float = 0,
        total_time: float = 0,
        gpu_stats: Dict[str, Any] = None
    ):
        """Publish model_selection event with supervisor reasoning."""
        await self.publish(mission_id, SSEEvent(
            event_type="model_selection",
            data={
                "phase_name": phase_name,
                "models": models,
                "reason": reason,
                "downgraded": downgraded,
                "wait_for_capacity": wait_for_capacity,
                "fallback_models": fallback_models or [],
                "phase_importance": phase_importance,
                "estimated_vram": estimated_vram,
                "time_remaining": time_remaining,
                "total_time": total_time,
                "time_percent": (time_remaining / total_time * 100) if total_time > 0 else 0,
                "gpu_stats": gpu_stats or {}
            }
        ))
    
    async def publish_consensus_result(
        self,
        mission_id: str,
        council_name: str,
        mechanism: str,
        agreement_score: float,
        model_outputs: Dict[str, Dict[str, Any]] = None,
        final_decision: str = None
    ):
        """Publish consensus_result event with council agreement details."""
        # Truncate model outputs for transmission
        truncated_outputs = {}
        if model_outputs:
            for model, output in model_outputs.items():
                truncated_outputs[model] = {
                    "success": output.get("success", False),
                    "preview": output.get("output", "")[:300] if output.get("output") else None,
                    "duration_s": output.get("duration_s"),
                    "tokens": output.get("tokens")
                }
        
        await self.publish(mission_id, SSEEvent(
            event_type="consensus_result",
            data={
                "council_name": council_name,
                "mechanism": mechanism,
                "agreement_score": agreement_score,
                "model_outputs": truncated_outputs,
                "final_decision": final_decision[:200] if final_decision and len(final_decision) > 200 else final_decision
            }
        ))
    
    async def publish_resource_update(
        self,
        mission_id: str,
        gpu_available: int = 0,
        gpu_total: int = 0,
        vram_used_mb: int = 0,
        vram_total_mb: int = 0,
        time_remaining: float = 0,
        total_time: float = 0,
        active_models: list = None
    ):
        """Publish resource_update event with GPU/time utilization."""
        await self.publish(mission_id, SSEEvent(
            event_type="resource_update",
            data={
                "gpu_available": gpu_available,
                "gpu_total": gpu_total,
                "gpu_utilization": ((gpu_total - gpu_available) / gpu_total * 100) if gpu_total > 0 else 0,
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "vram_utilization": (vram_used_mb / vram_total_mb * 100) if vram_total_mb > 0 else 0,
                "time_remaining": time_remaining,
                "total_time": total_time,
                "time_utilization": ((total_time - time_remaining) / total_time * 100) if total_time > 0 else 0,
                "active_models": active_models or []
            }
        ))
    
    async def publish_model_execution(
        self,
        mission_id: str,
        model_name: str,
        success: bool,
        duration_s: float = None,
        tokens_in: int = None,
        tokens_out: int = None,
        output_preview: str = None,
        error: str = None
    ):
        """Publish model_execution event for individual model calls."""
        await self.publish(mission_id, SSEEvent(
            event_type="model_execution",
            data={
                "model_name": model_name,
                "success": success,
                "duration_s": duration_s,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "output_preview": output_preview[:300] if output_preview and len(output_preview) > 300 else output_preview,
                "error": error
            }
        ))
    
    async def publish_multi_view_analysis(
        self,
        mission_id: str,
        agreement: float,
        optimist_confidence: float,
        skeptic_confidence: float,
        optimist_opportunities: list = None,
        skeptic_risks: list = None,
        high_agreement: bool = False,
        confidence_gap: float = None
    ):
        """Publish multi_view_analysis event with optimist/skeptic comparison."""
        await self.publish(mission_id, SSEEvent(
            event_type="multi_view_analysis",
            data={
                "agreement": agreement,
                "optimist_confidence": optimist_confidence,
                "skeptic_confidence": skeptic_confidence,
                "optimist_opportunities": optimist_opportunities or [],
                "skeptic_risks": skeptic_risks or [],
                "high_agreement": high_agreement,
                "confidence_gap": confidence_gap if confidence_gap is not None else abs(optimist_confidence - skeptic_confidence)
            }
        ))
    
    async def publish_phase_metrics(
        self,
        mission_id: str,
        phase_name: str = None,
        difficulty_score: float = 0.0,
        uncertainty_score: float = 0.0,
        progress_score: float = 0.0,
        novelty_score: float = 0.0,
        confidence_score: float = 0.0
    ):
        """Publish phase_metrics event with phase performance metrics."""
        await self.publish(mission_id, SSEEvent(
            event_type="phase_metrics",
            data={
                "phase_name": phase_name,
                "difficulty_score": difficulty_score,
                "uncertainty_score": uncertainty_score,
                "progress_score": progress_score,
                "novelty_score": novelty_score,
                "confidence_score": confidence_score
            }
        ))
    
    def get_subscriber_count(self, mission_id: str) -> int:
        """Get the number of subscribers for a mission."""
        return len(self._queues.get(mission_id, set()))
    
    # =========================================================================
    # GOVERNANCE & EPISTEMIC EVENTS (new for frontend overhaul)
    # =========================================================================
    
    async def publish_governance_update(
        self,
        mission_id: str,
        phase_name: str,
        verdict: str,
        violations: int = 0,
        violation_details: list = None,
        retry_count: int = 0,
        max_retries: int = 3,
        retry_reason: str = None,
        force_web_search: bool = False,
        epistemic_risk_score: float = 0.0
    ):
        """Publish governance_update event with verdict, violations, and retry info."""
        await self.publish(mission_id, SSEEvent(
            event_type="governance_update",
            data={
                "phase_name": phase_name,
                "verdict": verdict,  # "PASS", "BLOCK", "WARN"
                "violations": violations,
                "violation_details": violation_details or [],
                "retry_count": retry_count,
                "max_retries": max_retries,
                "retry_reason": retry_reason,
                "force_web_search": force_web_search,
                "epistemic_risk_score": epistemic_risk_score
            }
        ))
    
    async def publish_research_progress(
        self,
        mission_id: str,
        phase_name: str,
        iteration: int,
        completeness: float,
        should_continue: bool,
        context_delta: str = None,
        web_searches_performed: int = 0,
        key_points_count: int = 0,
        gaps_count: int = 0,
        confidence_score: float = 0.0
    ):
        """Publish research_progress event with iteration details and completeness."""
        await self.publish(mission_id, SSEEvent(
            event_type="research_progress",
            data={
                "phase_name": phase_name,
                "iteration": iteration,
                "completeness": completeness,
                "completeness_percent": completeness * 100,
                "should_continue": should_continue,
                "context_delta": context_delta,
                "web_searches_performed": web_searches_performed,
                "key_points_count": key_points_count,
                "gaps_count": gaps_count,
                "confidence_score": confidence_score
            }
        ))
    
    async def publish_ml_governance(
        self,
        mission_id: str,
        system_health: float,
        predictor_status: Dict[str, Dict[str, Any]] = None,
        advisory_readiness: float = 0.0,
        advisory_ready: bool = False,
        alerts: list = None
    ):
        """Publish ml_governance event with predictor health and alerts."""
        await self.publish(mission_id, SSEEvent(
            event_type="ml_governance",
            data={
                "system_health": system_health,
                "system_health_percent": system_health * 100,
                "predictor_status": predictor_status or {},
                "advisory_readiness": advisory_readiness,
                "advisory_ready": advisory_ready,
                "alerts": alerts or [],
                "alerts_count": len(alerts) if alerts else 0
            }
        ))
    
    async def publish_synthesis_iteration(
        self,
        mission_id: str,
        iteration: int,
        content_preview: str = None,
        quality_score: float = 0.0,
        word_count: int = 0,
        sections_count: int = 0,
        is_final: bool = False
    ):
        """Publish synthesis_iteration event with iteration content and metrics."""
        await self.publish(mission_id, SSEEvent(
            event_type="synthesis_iteration",
            data={
                "iteration": iteration,
                "content_preview": content_preview[:500] if content_preview and len(content_preview) > 500 else content_preview,
                "quality_score": quality_score,
                "word_count": word_count,
                "sections_count": sections_count,
                "is_final": is_final
            }
        ))
    
    async def publish_epistemic_update(
        self,
        mission_id: str,
        phase_name: str,
        unresolved_questions: list = None,
        evidence_requests: list = None,
        focus_areas: list = None,
        claims_count: int = 0,
        verified_claims: int = 0,
        grounding_score: float = 0.0
    ):
        """Publish epistemic_update event with questions, evidence, and focus areas."""
        await self.publish(mission_id, SSEEvent(
            event_type="epistemic_update",
            data={
                "phase_name": phase_name,
                "unresolved_questions": (unresolved_questions or [])[:10],  # Limit to 10
                "unresolved_count": len(unresolved_questions) if unresolved_questions else 0,
                "evidence_requests": (evidence_requests or [])[:10],
                "evidence_count": len(evidence_requests) if evidence_requests else 0,
                "focus_areas": (focus_areas or [])[:10],
                "focus_count": len(focus_areas) if focus_areas else 0,
                "claims_count": claims_count,
                "verified_claims": verified_claims,
                "grounding_score": grounding_score
            }
        ))
    
    async def publish_deep_analysis_update(
        self,
        mission_id: str,
        scenarios: list = None,
        stress_tests: list = None,
        tradeoffs: list = None,
        robustness_score: float = 0.0,
        failure_modes: list = None,
        recommendations: list = None
    ):
        """Publish deep_analysis_update event with scenarios, stress tests, and tradeoffs."""
        # Truncate long lists and content
        def truncate_list(items, max_items=5, max_chars=200):
            if not items:
                return []
            result = []
            for item in items[:max_items]:
                if isinstance(item, str) and len(item) > max_chars:
                    result.append(item[:max_chars] + "...")
                elif isinstance(item, dict):
                    # Truncate description fields in dicts
                    truncated = {k: (v[:max_chars] + "..." if isinstance(v, str) and len(v) > max_chars else v) 
                                for k, v in item.items()}
                    result.append(truncated)
                else:
                    result.append(item)
            return result
        
        await self.publish(mission_id, SSEEvent(
            event_type="deep_analysis_update",
            data={
                "scenarios": truncate_list(scenarios),
                "scenarios_count": len(scenarios) if scenarios else 0,
                "stress_tests": truncate_list(stress_tests),
                "stress_tests_count": len(stress_tests) if stress_tests else 0,
                "tradeoffs": truncate_list(tradeoffs),
                "tradeoffs_count": len(tradeoffs) if tradeoffs else 0,
                "robustness_score": robustness_score,
                "failure_modes": truncate_list(failure_modes),
                "failure_modes_count": len(failure_modes) if failure_modes else 0,
                "recommendations": truncate_list(recommendations)
            }
        ))
    
    async def publish_phase_error(
        self,
        mission_id: str,
        phase_name: str,
        error_type: str,
        error_message: str,
        phase_index: int = 0,
        retry_available: bool = False,
        suggestions: list = None,
        context: Dict[str, Any] = None
    ):
        """Publish phase_error event with error details and recovery suggestions."""
        await self.publish(mission_id, SSEEvent(
            event_type="phase_error",
            data={
                "phase_name": phase_name,
                "phase_index": phase_index,
                "error_type": error_type,
                "error_message": error_message[:500] if error_message and len(error_message) > 500 else error_message,
                "retry_available": retry_available,
                "suggestions": (suggestions or [])[:5],
                "context": context or {}
            }
        ))
    
    async def publish_supervisor_decision(
        self,
        mission_id: str,
        phase_name: str,
        models: list,
        decision_type: str = "model_selection",
        downgraded: bool = False,
        downgrade_reason: str = None,
        fallback_models: list = None,
        gpu_pressure: str = "normal",
        estimated_vram_mb: int = 0,
        wait_for_capacity: bool = False,
        phase_importance: float = 0.5,
        temperature: float = 0.7,
        parallelism: int = 1
    ):
        """Publish supervisor_decision event with detailed model selection info."""
        await self.publish(mission_id, SSEEvent(
            event_type="supervisor_decision",
            data={
                "phase_name": phase_name,
                "models": models,
                "decision_type": decision_type,
                "downgraded": downgraded,
                "downgrade_reason": downgrade_reason,
                "fallback_models": fallback_models or [],
                "gpu_pressure": gpu_pressure,
                "estimated_vram_mb": estimated_vram_mb,
                "wait_for_capacity": wait_for_capacity,
                "phase_importance": phase_importance,
                "temperature": temperature,
                "parallelism": parallelism
            }
        ))
    
    # =========================================================================
    # ALIGNMENT CONTROL LAYER EVENTS (Gap 3)
    # =========================================================================
    
    async def publish_alignment_update(
        self,
        mission_id: str,
        phase_name: str,
        alignment_score: float,
        warning: bool = False,
        correction: bool = False,
        action_applied: str = None,
        cusum_neg: float = 0.0,
        drift_delta: float = 0.0,
        consecutive_triggers: int = 0
    ):
        """
        Publish alignment_update event for real-time alignment monitoring.
        
        Alignment Control Layer (Gap 3): Frontend can display live alignment
        trajectory, warning states, and correction events.
        
        Args:
            mission_id: Mission identifier
            phase_name: Current phase name
            alignment_score: Current alignment score (a_t, 0-1)
            warning: Whether warning threshold was crossed
            correction: Whether correction threshold was crossed (triggered)
            action_applied: Name of corrective action applied, if any
            cusum_neg: CUSUM negative drift statistic
            drift_delta: Drift delta from previous phase (d_t)
            consecutive_triggers: Number of consecutive correction triggers
        """
        await self.publish(mission_id, SSEEvent(
            event_type="alignment_update",
            data={
                "phase_name": phase_name,
                "alignment_score": alignment_score,
                "alignment_percent": alignment_score * 100,
                "warning": warning,
                "correction": correction,
                "action_applied": action_applied,
                "cusum_neg": cusum_neg,
                "drift_delta": drift_delta,
                "consecutive_triggers": consecutive_triggers,
                "status": "correction" if correction else ("warning" if warning else "healthy")
            }
        ))
    
    async def publish_alignment_warning(
        self,
        mission_id: str,
        phase_name: str,
        alignment_score: float,
        message: str,
        threshold: float
    ):
        """
        Publish alignment_warning event when warning threshold is crossed.
        
        This is a distinct event from alignment_update for explicit warning notifications.
        """
        await self.publish(mission_id, SSEEvent(
            event_type="alignment_warning",
            data={
                "phase_name": phase_name,
                "alignment_score": alignment_score,
                "threshold": threshold,
                "message": message
            }
        ))
    
    async def publish_alignment_correction(
        self,
        mission_id: str,
        phase_name: str,
        action: str,
        reason: str,
        alignment_score: float,
        consecutive_triggers: int
    ):
        """
        Publish alignment_correction event when a corrective action is applied.
        
        This is a distinct event from alignment_update for explicit correction notifications.
        """
        await self.publish(mission_id, SSEEvent(
            event_type="alignment_correction",
            data={
                "phase_name": phase_name,
                "action": action,
                "reason": reason,
                "alignment_score": alignment_score,
                "consecutive_triggers": consecutive_triggers
            }
        ))


# Global SSE manager instance
sse_manager = SSEManager()

