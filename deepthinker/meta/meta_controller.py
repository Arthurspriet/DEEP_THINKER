"""
Meta Controller for DeepThinker Meta-Cognition Engine.

Orchestrates all meta-cognition components: reflection, hypothesis management,
internal debate, plan revision, and the ReasoningSupervisor.

Enhanced in 2.0 with:
- ReasoningSupervisor integration for depth contracts and metrics
- Conditional multi-view reasoning (Optimist + Skeptic)
- Context summarization for later phases
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .reflection import ReflectionEngine
from .hypotheses import HypothesisManager
from .debate import DebateEngine
from .plan_revision import PlanReviser
from .supervisor import (
    ReasoningSupervisor,
    PhaseMetrics,
    DepthContract,
    DeepeningPlan,
    LoopDetection,
)

if TYPE_CHECKING:
    from ..models.model_pool import ModelPool
    from ..missions.mission_types import MissionState

# Multi-view councils import
try:
    from ..councils.multi_view import OptimistCouncil, SkepticCouncil
    MULTIVIEW_AVAILABLE = True
except ImportError:
    MULTIVIEW_AVAILABLE = False
    OptimistCouncil = None
    SkepticCouncil = None

# Verbose logging
try:
    from ..cli import verbose_logger
    VERBOSE_AVAILABLE = True
except ImportError:
    VERBOSE_AVAILABLE = False
    verbose_logger = None

logger = logging.getLogger(__name__)


class MetaController:
    """
    Orchestrates the meta-cognition layer for DeepThinker missions.
    
    After each phase completes, the MetaController:
    1. Runs reflection on the phase output
    2. Updates hypotheses based on reflection
    3. Runs internal debate on hypotheses
    4. Revises the plan based on all insights
    5. Computes phase metrics via ReasoningSupervisor
    6. Creates depth contracts for future phases
    7. Conditionally triggers multi-view analysis
    
    This provides deeper reasoning for long-running missions (1h+).
    """
    
    def __init__(
        self,
        model_pool: "ModelPool",
        enable_multiview: bool = True,
        enable_supervisor: bool = True,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the meta controller.
        
        Args:
            model_pool: Model pool for LLM calls
            enable_multiview: Whether to enable multi-view councils
            enable_supervisor: Whether to enable ReasoningSupervisor
            ollama_base_url: Ollama server URL
        """
        self.model_pool = model_pool
        self.ollama_base_url = ollama_base_url
        
        # Initialize sub-engines
        self.reflection_engine = ReflectionEngine(model_pool)
        self.hypothesis_manager = HypothesisManager(model_pool)
        self.debate_engine = DebateEngine(model_pool)
        self.plan_reviser = PlanReviser(model_pool)
        
        # Initialize ReasoningSupervisor
        self.enable_supervisor = enable_supervisor
        self.supervisor: Optional[ReasoningSupervisor] = None
        if enable_supervisor:
            self.supervisor = ReasoningSupervisor(
                model_pool=model_pool,
                ollama_base_url=ollama_base_url
            )
        
        # Initialize multi-view councils
        self.enable_multiview = enable_multiview and MULTIVIEW_AVAILABLE
        self.optimist_council: Optional["OptimistCouncil"] = None
        self.skeptic_council: Optional["SkepticCouncil"] = None
        if self.enable_multiview:
            self._setup_multiview_councils()
        
        self._initialized_missions: set = set()
        
        # Cache for depth contracts
        self._depth_contracts: Dict[str, DepthContract] = {}
    
    def _setup_multiview_councils(self) -> None:
        """Set up multi-view councils (Optimist + Skeptic)."""
        if not MULTIVIEW_AVAILABLE:
            return
        
        try:
            self.optimist_council = OptimistCouncil(
                model_pool=self.model_pool,
                ollama_base_url=self.ollama_base_url
            )
            self.skeptic_council = SkepticCouncil(
                model_pool=self.model_pool,
                ollama_base_url=self.ollama_base_url
            )
            logger.debug("Multi-view councils initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize multi-view councils: {e}")
            self.enable_multiview = False
    
    def _ensure_state_initialized(self, state: "MissionState") -> None:
        """Ensure all meta-cognition state fields are initialized."""
        # Initialize hypotheses
        if not hasattr(state, 'hypotheses') or not state.hypotheses:
            state.hypotheses = {
                "active": [],
                "rejected": [],
                "evidence": {},
                "confidence": {},
            }
        
        # Initialize updated_plan
        if not hasattr(state, 'updated_plan') or not state.updated_plan:
            state.updated_plan = {
                "new_subgoals": [],
                "invalidated_goals": [],
                "priority_adjustments": [],
                "missing_data": [],
                "revision_history": [],
            }
        
        # Initialize next_actions
        if not hasattr(state, 'next_actions') or state.next_actions is None:
            state.next_actions = []
        
        # Initialize meta_traces
        if not hasattr(state, 'meta_traces') or not state.meta_traces:
            state.meta_traces = {
                "phases_processed": [],
                "total_reflections": 0,
                "total_debates": 0,
                "total_revisions": 0,
                "depth_contracts": {},
                "phase_metrics": {},
                "multiview_results": {},
            }
        
        # Ensure new fields exist in meta_traces
        if "depth_contracts" not in state.meta_traces:
            state.meta_traces["depth_contracts"] = {}
        if "phase_metrics" not in state.meta_traces:
            state.meta_traces["phase_metrics"] = {}
        if "multiview_results" not in state.meta_traces:
            state.meta_traces["multiview_results"] = {}
        
        # Initialize work_summary sections
        if "reflection" not in state.work_summary:
            state.work_summary["reflection"] = {}
        if "debate" not in state.work_summary:
            state.work_summary["debate"] = {}
        if "meta" not in state.work_summary:
            state.work_summary["meta"] = {}
        if "multiview" not in state.work_summary:
            state.work_summary["multiview"] = {}
        
        # Initialize context_summaries and recent_outputs
        if not hasattr(state, 'context_summaries'):
            state.context_summaries = {}
        if not hasattr(state, 'recent_outputs'):
            state.recent_outputs = []
        if not hasattr(state, 'iteration_count'):
            state.iteration_count = 0
    
    def initialize_mission(self, state: "MissionState") -> None:
        """
        Initialize meta-cognition for a new mission.
        
        Generates initial hypotheses based on the objective.
        
        Args:
            state: Mission state to initialize
        """
        if state.mission_id in self._initialized_missions:
            return
        
        self._ensure_state_initialized(state)
        
        try:
            # Generate initial hypotheses
            self.hypothesis_manager.generate_initial_hypotheses(
                objective=state.objective,
                state=state
            )
            
            self._initialized_missions.add(state.mission_id)
            state.log("Meta-cognition engine initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize meta-cognition: {e}")
    
    def process_phase(
        self,
        phase_name: str,
        council_output: Any,
        state: "MissionState"
    ) -> Dict[str, Any]:
        """
        Process meta-cognition after a phase completes.
        
        Sequence:
        1. Reflect on phase output
        2. Update hypotheses
        3. Run internal debate
        4. Revise plan
        5. Analyze with ReasoningSupervisor (metrics + depth contract)
        6. Conditionally run multi-view councils
        7. Summarize context for later phases
        
        Args:
            phase_name: Name of the completed phase
            council_output: Output from the phase's council/execution
            state: Current mission state
            
        Returns:
            Meta-cognition results containing:
            - reflection: Reflection analysis
            - hypotheses: Updated hypothesis summary
            - debate: Debate results summary
            - revision: Plan revision summary
            - phase_metrics: ReasoningSupervisor metrics
            - depth_contract: Contract for future phases
            - multiview: Multi-view results (if triggered)
            - success: Whether meta-cognition completed
        """
        result = {
            "phase_name": phase_name,
            "reflection": {},
            "hypotheses": {},
            "debate": {},
            "revision": {},
            "phase_metrics": {},
            "depth_contract": {},
            "multiview": {},
            "success": False,
        }
        
        try:
            # Ensure state is initialized
            self._ensure_state_initialized(state)
            
            # Initialize if first phase
            if state.mission_id not in self._initialized_missions:
                self.initialize_mission(state)
            
            # Track recent outputs for loop detection
            output_str = self._format_output(council_output)
            if hasattr(state, 'recent_outputs'):
                state.recent_outputs.append(output_str)
                # Keep only last 10
                if len(state.recent_outputs) > 10:
                    state.recent_outputs = state.recent_outputs[-10:]
            
            # 1. REFLECTION
            reflection = self.reflection_engine.reflect_on_phase_output(
                phase_name=phase_name,
                output=council_output,
                state=state
            )
            result["reflection"] = {
                "assumptions_count": len(reflection.get("assumptions", [])),
                "weaknesses_count": len(reflection.get("weaknesses", [])),
                "questions_count": len(reflection.get("questions", [])),
            }
            state.meta_traces["total_reflections"] += 1
            
            # 2. HYPOTHESIS UPDATE
            updated_hypotheses = self.hypothesis_manager.update_hypotheses(
                reflection=reflection,
                council_output=council_output,
                state=state
            )
            
            # Reject/strengthen based on evidence
            self.hypothesis_manager.reject_or_strengthen(state)
            
            # Get hypothesis summary
            hyp_summary = self.hypothesis_manager.export_summary(state)
            result["hypotheses"] = hyp_summary
            
            # 3. INTERNAL DEBATE
            # Only run debate if there are active hypotheses
            active_hypotheses = state.hypotheses.get("active", [])
            debate_results = {}
            if active_hypotheses:
                debate_results = self.debate_engine.run_internal_debate(
                    phase_name=phase_name,
                    hypotheses=active_hypotheses,
                    context=output_str,
                    state=state
                )
                result["debate"] = {
                    "contradictions_count": len(debate_results.get("contradictions_found", [])),
                    "adjustments_count": len(debate_results.get("confidence_adjustments", {})),
                    "ran_debate": True,
                }
                state.meta_traces["total_debates"] += 1
            else:
                result["debate"] = {"ran_debate": False}
            
            # 4. PLAN REVISION
            revision = self.plan_reviser.revise_plan(
                state=state,
                phase_name=phase_name,
                reflection=reflection,
                debate=debate_results,
                hypotheses=hyp_summary
            )
            result["revision"] = {
                "new_subgoals_count": len(revision.get("new_subgoals", [])),
                "next_actions_count": len(revision.get("next_actions", [])),
            }
            state.meta_traces["total_revisions"] += 1
            
            # 5. REASONING SUPERVISOR ANALYSIS
            phase_metrics = None
            depth_contract = None
            if self.supervisor:
                try:
                    # Analyze phase output
                    phase_metrics = self.supervisor.analyze_phase_output(
                        phase_name=phase_name,
                        output=council_output,
                        state=state
                    )
                    result["phase_metrics"] = phase_metrics.to_dict()
                    state.meta_traces["phase_metrics"][phase_name] = phase_metrics.to_dict()
                    
                    # Create depth contract
                    time_remaining = state.remaining_minutes()
                    depth_contract = self.supervisor.create_depth_contract(
                        metrics=phase_metrics,
                        phase_name=phase_name,
                        time_remaining=time_remaining
                    )
                    result["depth_contract"] = depth_contract.to_dict()
                    state.meta_traces["depth_contracts"][phase_name] = depth_contract.to_dict()
                    self._depth_contracts[phase_name] = depth_contract
                    
                    # Log to verbose logger
                    if VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
                        if hasattr(verbose_logger, 'log_supervisor_metrics'):
                            verbose_logger.log_supervisor_metrics(
                                phase_metrics,
                                mission_id=state.mission_id,
                                phase_name=phase_name
                            )
                        if hasattr(verbose_logger, 'log_depth_contract'):
                            verbose_logger.log_depth_contract(depth_contract, phase_name)
                    
                except Exception as e:
                    logger.warning(f"ReasoningSupervisor failed: {e}")
            
            # 6. CONDITIONAL MULTI-VIEW
            if phase_metrics and self.enable_multiview and self.supervisor:
                try:
                    should_multiview = self.supervisor.should_run_multiview(
                        metrics=phase_metrics,
                        state=state
                    )
                    
                    if should_multiview:
                        multiview_result = self._run_multiview_councils(
                            state=state,
                            phase_name=phase_name,
                            content=output_str
                        )
                        result["multiview"] = multiview_result
                        state.meta_traces["multiview_results"][phase_name] = multiview_result
                        state.work_summary["multiview"][phase_name] = multiview_result
                        
                        # Log trigger reason
                        if VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
                            if hasattr(verbose_logger, 'log_multiview_triggered'):
                                reason = f"difficulty={phase_metrics.difficulty_score:.2f}, uncertainty={phase_metrics.uncertainty_score:.2f}"
                                verbose_logger.log_multiview_triggered(reason, phase_metrics)
                        
                        state.log(f"Multi-view analysis triggered for phase '{phase_name}'")
                    
                except Exception as e:
                    logger.warning(f"Multi-view analysis failed: {e}")
            
            # 7. CONTEXT SUMMARIZATION
            if self.supervisor:
                try:
                    context_summary = self.supervisor.summarize_context(state)
                    if hasattr(state, 'context_summaries'):
                        state.context_summaries[phase_name] = context_summary
                except Exception as e:
                    logger.warning(f"Context summarization failed: {e}")
            
            # Track processed phase
            state.meta_traces["phases_processed"].append(phase_name)
            
            result["success"] = True
            state.log(f"Meta-cognition completed for phase '{phase_name}'")
            
        except Exception as e:
            logger.error(f"Meta-cognition failed for phase '{phase_name}': {e}")
            result["error"] = str(e)
            state.log(f"Meta-cognition error for phase '{phase_name}': {e}")
        
        return result
    
    def _run_multiview_councils(
        self,
        state: "MissionState",
        phase_name: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Run multi-view councils (Optimist + Skeptic) on phase content.
        
        Args:
            state: Current mission state
            phase_name: Name of the phase
            content: Content to analyze
            
        Returns:
            Dictionary with optimist_output, skeptic_output, agreement_score
        """
        result = {
            "optimist": {},
            "skeptic": {},
            "agreement_score": 0.0,
            "ran_multiview": False,
        }
        
        if not self.enable_multiview:
            return result
        
        # Truncate content for efficiency
        content_truncated = content[:4000] if len(content) > 4000 else content
        
        # Run OptimistCouncil
        if self.optimist_council:
            try:
                from ..councils.multi_view.optimist_council import OptimistContext
                opt_context = OptimistContext(
                    objective=state.objective,
                    content=content_truncated,
                    iteration=getattr(state, 'iteration_count', 1)
                )
                opt_result = self.optimist_council.execute(opt_context)
                
                if opt_result.success and opt_result.output:
                    result["optimist"] = {
                        "opportunities": getattr(opt_result.output, 'opportunities', []),
                        "strengths": getattr(opt_result.output, 'strengths', []),
                        "confidence": getattr(opt_result.output, 'confidence', 0.5),
                        "reasoning": getattr(opt_result.output, 'reasoning', ''),
                    }
            except Exception as e:
                logger.warning(f"OptimistCouncil failed: {e}")
        
        # Run SkepticCouncil
        if self.skeptic_council:
            try:
                from ..councils.multi_view.skeptic_council import SkepticContext
                skep_context = SkepticContext(
                    objective=state.objective,
                    content=content_truncated,
                    iteration=getattr(state, 'iteration_count', 1)
                )
                skep_result = self.skeptic_council.execute(skep_context)
                
                if skep_result.success and skep_result.output:
                    result["skeptic"] = {
                        "risks": getattr(skep_result.output, 'risks', []),
                        "weaknesses": getattr(skep_result.output, 'weaknesses', []),
                        "confidence": getattr(skep_result.output, 'confidence', 0.5),
                        "reasoning": getattr(skep_result.output, 'reasoning', ''),
                    }
            except Exception as e:
                logger.warning(f"SkepticCouncil failed: {e}")
        
        # Compute agreement score
        if result["optimist"] and result["skeptic"]:
            opt_conf = result["optimist"].get("confidence", 0.5)
            skep_conf = result["skeptic"].get("confidence", 0.5)
            result["agreement_score"] = 1.0 - abs(opt_conf - skep_conf)
            result["ran_multiview"] = True
            
            # Log multi-view comparison
            if VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
                if hasattr(verbose_logger, 'log_multi_view_comparison'):
                    # Create mock objects for logging
                    class MockOutput:
                        pass
                    
                    opt_mock = MockOutput()
                    opt_mock.confidence = opt_conf
                    opt_mock.opportunities = result["optimist"].get("opportunities", [])
                    
                    skep_mock = MockOutput()
                    skep_mock.confidence = skep_conf
                    skep_mock.risks = result["skeptic"].get("risks", [])
                    
                    verbose_logger.log_multi_view_comparison(
                        opt_mock, 
                        skep_mock,
                        mission_id=state.mission_id
                    )
        
        return result
    
    def get_depth_contract(self, phase_name: str) -> Optional[DepthContract]:
        """
        Get the depth contract for a phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            DepthContract if available, None otherwise
        """
        return self._depth_contracts.get(phase_name)
    
    def get_phase_metrics(self, phase_name: str, state: "MissionState") -> Optional[PhaseMetrics]:
        """
        Get the latest phase metrics from state.
        
        Args:
            phase_name: Name of the phase
            state: Current mission state
            
        Returns:
            PhaseMetrics if available, None otherwise
        """
        metrics_dict = state.meta_traces.get("phase_metrics", {}).get(phase_name)
        if metrics_dict:
            return PhaseMetrics(**metrics_dict)
        return None
    
    def _format_output(self, output: Any) -> str:
        """Format council output for context."""
        if isinstance(output, dict):
            parts = []
            for key, value in list(output.items())[:5]:
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "..."
                parts.append(f"{key}: {value_str}")
            return "\n".join(parts)
        elif isinstance(output, str):
            return output[:2000] + "..." if len(output) > 2000 else output
        else:
            output_str = str(output)
            return output_str[:2000] + "..." if len(output_str) > 2000 else output_str
    
    def get_meta_summary(self, state: "MissionState") -> Dict[str, Any]:
        """
        Get a summary of all meta-cognition activity for the mission.
        
        Args:
            state: Current mission state
            
        Returns:
            Summary dictionary
        """
        self._ensure_state_initialized(state)
        
        return {
            "phases_processed": len(state.meta_traces.get("phases_processed", [])),
            "total_reflections": state.meta_traces.get("total_reflections", 0),
            "total_debates": state.meta_traces.get("total_debates", 0),
            "total_revisions": state.meta_traces.get("total_revisions", 0),
            "active_hypotheses": len(state.hypotheses.get("active", [])),
            "rejected_hypotheses": len(state.hypotheses.get("rejected", [])),
            "pending_actions": len(state.next_actions),
            "new_subgoals": len(state.updated_plan.get("new_subgoals", [])),
        }
    
    def should_continue_iteration(self, state: "MissionState") -> bool:
        """
        Determine if the mission should continue iterating.
        
        Uses ReasoningSupervisor if available, otherwise falls back to heuristics.
        
        Based on:
        - ReasoningSupervisor convergence analysis (if available)
        - Hypothesis confidence levels
        - Outstanding questions
        - Time remaining
        - Loop detection
        
        Args:
            state: Current mission state
            
        Returns:
            True if iteration should continue
        """
        self._ensure_state_initialized(state)
        
        # Check time
        if state.remaining_minutes() < 1.0:
            return False
        
        # Use ReasoningSupervisor if available
        if self.supervisor:
            try:
                metrics = self.supervisor.analyze_mission_state(state)
                
                # Log mission iteration info
                if VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    if hasattr(verbose_logger, 'log_mission_iteration'):
                        iteration = getattr(state, 'iteration_count', 0)
                        verbose_logger.log_mission_iteration(iteration, metrics)
                
                # Check for loops
                loop_detection = self.supervisor.detect_loops(state)
                if loop_detection.loop_detected:
                    if VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
                        if hasattr(verbose_logger, 'log_loop_detected'):
                            verbose_logger.log_loop_detected(loop_detection)
                    state.log(f"Loop detected: {loop_detection.recommendation}")
                
                # Supervisor decides
                should_stop = self.supervisor.should_stop_mission(metrics)
                return not should_stop
                
            except Exception as e:
                logger.warning(f"ReasoningSupervisor iteration check failed: {e}")
                # Fall through to heuristic method
        
        # Fallback: Heuristic-based checks
        
        # Check hypothesis confidence
        active = state.hypotheses.get("active", [])
        if active:
            avg_confidence = sum(h.get("confidence", 0.5) for h in active) / len(active)
            # Continue if confidence is low
            if avg_confidence < 0.6:
                return True
        
        # Check if there are unresolved questions
        reflections = state.work_summary.get("reflection", {})
        total_questions = sum(
            len(r.get("questions", []))
            for r in reflections.values()
            if isinstance(r, dict)
        )
        
        # Continue if many questions remain
        if total_questions > 5:
            return True
        
        return False
    
    def plan_deepening(self, state: "MissionState") -> Optional[DeepeningPlan]:
        """
        Get a deepening plan from the ReasoningSupervisor.
        
        Args:
            state: Current mission state
            
        Returns:
            DeepeningPlan if supervisor is available and recommends deepening
        """
        if not self.supervisor:
            return None
        
        try:
            metrics = self.supervisor.analyze_mission_state(state)
            plan = self.supervisor.plan_deepening(state, metrics)
            
            # Log deepening plan
            if plan.has_work and VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
                if hasattr(verbose_logger, 'log_deepening_plan'):
                    verbose_logger.log_deepening_plan(plan)
            
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to get deepening plan: {e}")
            return None
    
    def reset_for_new_mission(self, mission_id: str) -> None:
        """
        Reset controller state for a new mission.
        
        Args:
            mission_id: ID of the new mission
        """
        self._depth_contracts.clear()
        
        if self.supervisor:
            self.supervisor.reset()
        
        # Remove from initialized set if present
        self._initialized_missions.discard(mission_id)

