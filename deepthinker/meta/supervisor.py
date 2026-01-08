"""
ReasoningSupervisor for DeepThinker 2.0.

Central brain for meta-level decisions including:
- Phase output analysis (difficulty, uncertainty, progress)
- Mission-level convergence detection
- Multi-view triggering decisions
- Deepening plan generation
- Loop detection and context summarization
- Depth contract creation

Enhanced with CognitiveSpine integration:
- Pre-phase validation
- Fallback policies for error recovery
- Contraction mode for resource exhaustion
- Automatic memory compression

Convergence Logic (v2.0):
- PRIMARY criteria: unresolved_questions == 0, novelty_delta < threshold, progress_delta < epsilon
- SECONDARY (informational): confidence_score, agreement_score
- Confidence is logged but NOT a blocking stop condition
"""

import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState
    from ..models.model_pool import ModelPool
    from ..core.cognitive_spine import CognitiveSpine

logger = logging.getLogger(__name__)


# =============================================================================
# Convergence Thresholds (Tunable)
# =============================================================================

# Primary convergence criteria thresholds
NOVELTY_THRESHOLD = 0.15          # Max novelty delta to allow stop
PROGRESS_EPSILON = 0.05           # Max progress delta to allow stop
MAX_UNRESOLVED_QUESTIONS = 0      # Must be zero to stop

# Secondary (informational) thresholds
MIN_CONFIDENCE_WARNING = 0.5      # Log warning below this (but don't block)
MULTIVIEW_DISAGREEMENT_WARNING = 0.3  # Log warning above this


@dataclass
class PhaseMetrics:
    """
    Metrics computed for a single phase output.
    
    Attributes:
        difficulty_score: How complex/difficult the content is (0-1)
        uncertainty_score: How uncertain/ambiguous the output is (0-1)
        progress_score: How much progress was made (0-1)
        novelty_score: How novel this output is vs previous (0-1)
        confidence_score: Confidence in the output quality (0-1)
        contradiction_count: Number of contradictions detected
        questions_remaining: Unresolved questions count
    """
    difficulty_score: float = 0.5
    uncertainty_score: float = 0.5
    progress_score: float = 0.5
    novelty_score: float = 0.5
    confidence_score: float = 0.5
    contradiction_count: int = 0
    questions_remaining: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def is_high_difficulty(self) -> bool:
        """Check if difficulty is high (>0.7)."""
        return self.difficulty_score > 0.7
    
    @property
    def is_high_uncertainty(self) -> bool:
        """Check if uncertainty is high (>0.7)."""
        return self.uncertainty_score > 0.7
    
    @property
    def needs_deeper_analysis(self) -> bool:
        """Check if this phase needs deeper analysis."""
        return (self.is_high_difficulty and self.is_high_uncertainty) or \
               self.contradiction_count > 2 or \
               self.confidence_score < 0.4


@dataclass
class MissionMetrics:
    """
    Aggregated metrics for the entire mission.
    
    Attributes:
        avg_difficulty: Average difficulty across phases
        avg_uncertainty: Average uncertainty across phases
        overall_progress: Overall mission progress (0-1)
        convergence_score: How converged the mission is (0-1, higher=more converged)
        iteration_count: Number of mission iterations completed
        time_remaining_minutes: Time remaining in minutes
        stagnation_count: Consecutive iterations with low improvement
        loop_detected: Whether a reasoning loop was detected
        phase_metrics: Per-phase metrics
    """
    avg_difficulty: float = 0.5
    avg_uncertainty: float = 0.5
    overall_progress: float = 0.0
    convergence_score: float = 0.0
    iteration_count: int = 0
    time_remaining_minutes: float = 0.0
    stagnation_count: int = 0
    loop_detected: bool = False
    phase_metrics: Dict[str, PhaseMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "avg_difficulty": self.avg_difficulty,
            "avg_uncertainty": self.avg_uncertainty,
            "overall_progress": self.overall_progress,
            "convergence_score": self.convergence_score,
            "iteration_count": self.iteration_count,
            "time_remaining_minutes": self.time_remaining_minutes,
            "stagnation_count": self.stagnation_count,
            "loop_detected": self.loop_detected,
        }
        result["phase_metrics"] = {
            k: v.to_dict() for k, v in self.phase_metrics.items()
        }
        return result


@dataclass
class DepthContract:
    """
    Contract specifying how deep a council/phase should go.
    
    Attributes:
        exploration_depth: How much exploration to do (0=minimal, 1=exhaustive)
        max_rounds: Maximum rounds for this phase/council
        model_tier: Suggested model tier ("small", "medium", "large", "xlarge")
        allow_alternatives: Whether to explore alternative approaches
        focus_areas: Specific areas to focus on
        skip_areas: Areas to skip or de-prioritize
        time_budget_minutes: Time budget for this contract
    """
    exploration_depth: float = 0.5
    max_rounds: int = 2
    model_tier: str = "medium"
    allow_alternatives: bool = True
    focus_areas: List[str] = field(default_factory=list)
    skip_areas: List[str] = field(default_factory=list)
    time_budget_minutes: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DeepeningPlan:
    """
    Plan for deepening analysis after a phase pass.
    
    Attributes:
        run_researcher: Whether to re-run researcher council
        run_planner: Whether to re-run planner council
        run_coder: Whether to re-run coder council
        run_evaluator: Whether to re-run evaluator council
        run_simulation: Whether to re-run simulation council
        focus_areas: Areas to focus deepening on
        max_deepening_rounds: Maximum rounds of deepening
        reason: Explanation for this deepening plan
    """
    run_researcher: bool = False
    run_planner: bool = False
    run_coder: bool = False
    run_evaluator: bool = False
    run_simulation: bool = False
    focus_areas: List[str] = field(default_factory=list)
    max_deepening_rounds: int = 1
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def has_work(self) -> bool:
        """Check if this plan has any work to do."""
        return any([
            self.run_researcher,
            self.run_planner,
            self.run_coder,
            self.run_evaluator,
            self.run_simulation
        ])


@dataclass
class ConvergenceResult:
    """
    Result of strict convergence check.
    
    Attributes:
        can_stop: Whether mission can stop iterating
        reason: Reason for the decision
        blocking_criteria: List of criteria that blocked convergence
        secondary_warnings: Non-blocking warnings (e.g., low confidence)
        unresolved_count: Number of unresolved questions
        novelty_delta: Change in novelty from previous iteration
        progress_delta: Change in progress from previous iteration
        confidence: Confidence score (informational only)
    """
    can_stop: bool
    reason: str
    blocking_criteria: List[str] = field(default_factory=list)
    secondary_warnings: List[str] = field(default_factory=list)
    unresolved_count: int = 0
    novelty_delta: float = 0.0
    progress_delta: float = 0.0
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoopDetection:
    """
    Result of loop detection analysis.
    
    Attributes:
        loop_detected: Whether a loop was detected
        similarity_score: Similarity to previous outputs (0-1)
        stagnation_count: Consecutive low-novelty iterations
        recommendation: What to do about the loop
        details: Additional details about the detection
    """
    loop_detected: bool = False
    similarity_score: float = 0.0
    stagnation_count: int = 0
    recommendation: str = ""
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ReasoningSupervisor:
    """
    Central brain for meta-level reasoning decisions.
    
    Analyzes phase outputs, mission state, and decides:
    - Whether to continue iterating
    - Whether to trigger multi-view analysis
    - What to deepen and how
    - Whether loops are occurring
    - How to summarize context
    
    Uses heuristics and optional LLM analysis for decisions.
    """
    
    # Thresholds for decision making
    CONVERGENCE_THRESHOLD = 0.85
    STAGNATION_THRESHOLD = 3
    MULTIVIEW_DIFFICULTY_THRESHOLD = 0.6
    MULTIVIEW_UNCERTAINTY_THRESHOLD = 0.6
    LOOP_SIMILARITY_THRESHOLD = 0.9
    MIN_PROGRESS_PER_ITERATION = 0.05
    
    def __init__(
        self,
        model_pool: Optional["ModelPool"] = None,
        use_llm_analysis: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        cognitive_spine: Optional["CognitiveSpine"] = None
    ):
        """
        Initialize the ReasoningSupervisor.
        
        Args:
            model_pool: Optional model pool for LLM-based analysis
            use_llm_analysis: Whether to use LLM for deeper analysis
            ollama_base_url: Ollama server URL
            cognitive_spine: Optional CognitiveSpine for validation and fallbacks
        """
        self.model_pool = model_pool
        self.use_llm_analysis = use_llm_analysis
        self.ollama_base_url = ollama_base_url
        self._cognitive_spine: Optional["CognitiveSpine"] = cognitive_spine
        
        # History tracking
        self._output_hashes: List[str] = []
        self._phase_metrics_history: Dict[str, List[PhaseMetrics]] = {}
        self._mission_metrics_history: List[MissionMetrics] = []
        
        # Fallback state tracking
        self._fallback_attempts: Dict[str, int] = {}
        self._contraction_mode: bool = False
    
    def analyze_phase_output(
        self,
        phase_name: str,
        output: Any,
        state: "MissionState"
    ) -> PhaseMetrics:
        """
        Analyze a phase's output to compute metrics.
        
        Uses heuristics to estimate difficulty, uncertainty, progress,
        and novelty of the output.
        
        Args:
            phase_name: Name of the phase
            output: The phase output (artifacts dict or council output)
            state: Current mission state
            
        Returns:
            PhaseMetrics with computed scores
        """
        metrics = PhaseMetrics()
        
        try:
            # Convert output to string for analysis
            output_str = self._output_to_string(output)
            
            # Compute difficulty based on content complexity
            metrics.difficulty_score = self._estimate_difficulty(output_str, state)
            
            # Compute uncertainty from hedging language and question marks
            metrics.uncertainty_score = self._estimate_uncertainty(output_str)
            
            # Compute progress based on artifacts and content growth
            metrics.progress_score = self._estimate_progress(phase_name, output, state)
            
            # Compute novelty by comparing to previous outputs
            metrics.novelty_score = self._estimate_novelty(output_str)
            
            # Estimate confidence from output structure
            metrics.confidence_score = self._estimate_confidence(output_str)
            
            # Count contradictions and remaining questions
            metrics.contradiction_count = self._count_contradictions(output_str, state)
            metrics.questions_remaining = self._count_questions(output_str)
            
            # Store in history
            if phase_name not in self._phase_metrics_history:
                self._phase_metrics_history[phase_name] = []
            self._phase_metrics_history[phase_name].append(metrics)
            
            logger.debug(f"Phase '{phase_name}' metrics: {metrics.to_dict()}")
            
        except Exception as e:
            logger.warning(f"Error analyzing phase output: {e}")
        
        return metrics
    
    def analyze_mission_state(self, state: "MissionState") -> MissionMetrics:
        """
        Analyze overall mission state to compute aggregated metrics.
        
        Args:
            state: Current mission state
            
        Returns:
            MissionMetrics with aggregated scores
        """
        metrics = MissionMetrics()
        
        try:
            # Get time remaining
            metrics.time_remaining_minutes = state.remaining_minutes()
            metrics.iteration_count = getattr(state, 'iteration_count', 0)
            
            # Aggregate phase metrics
            phase_difficulties = []
            phase_uncertainties = []
            phase_progress = []
            
            for phase in state.phases:
                if phase.status == "completed":
                    # Get latest metrics for this phase
                    phase_metrics = self._get_latest_phase_metrics(phase.name)
                    if phase_metrics:
                        metrics.phase_metrics[phase.name] = phase_metrics
                        phase_difficulties.append(phase_metrics.difficulty_score)
                        phase_uncertainties.append(phase_metrics.uncertainty_score)
                        phase_progress.append(phase_metrics.progress_score)
            
            # Compute averages
            if phase_difficulties:
                metrics.avg_difficulty = sum(phase_difficulties) / len(phase_difficulties)
            if phase_uncertainties:
                metrics.avg_uncertainty = sum(phase_uncertainties) / len(phase_uncertainties)
            if phase_progress:
                metrics.overall_progress = sum(phase_progress) / len(phase_progress)
            
            # Compute convergence score
            metrics.convergence_score = self._compute_convergence(state, metrics)
            
            # Check for stagnation
            metrics.stagnation_count = self._compute_stagnation_count(metrics)
            
            # Check for loops
            loop_detection = self.detect_loops(state)
            metrics.loop_detected = loop_detection.loop_detected
            
            # Store in history
            self._mission_metrics_history.append(metrics)
            
            logger.debug(f"Mission metrics: convergence={metrics.convergence_score:.2f}, "
                        f"progress={metrics.overall_progress:.2f}")
            
        except Exception as e:
            logger.warning(f"Error analyzing mission state: {e}")
        
        return metrics
    
    def should_stop_mission(self, metrics: MissionMetrics) -> bool:
        """
        Determine if the mission should stop iterating.
        
        Based on:
        - Convergence score (high = stop)
        - Time remaining (low = stop)
        - Stagnation (high = stop)
        - Loop detection (detected = consider stopping)
        - Overall progress (high = stop)
        
        Args:
            metrics: Current mission metrics
            
        Returns:
            True if mission should stop, False if should continue
        """
        # Stop if no time left
        if metrics.time_remaining_minutes < 1.0:
            logger.info("Stopping: time exhausted")
            return True
        
        # Stop if converged
        if metrics.convergence_score >= self.CONVERGENCE_THRESHOLD:
            logger.info(f"Stopping: converged ({metrics.convergence_score:.2f})")
            return True
        
        # Stop if stagnating for too long
        if metrics.stagnation_count >= self.STAGNATION_THRESHOLD:
            logger.info(f"Stopping: stagnation ({metrics.stagnation_count} iterations)")
            return True
        
        # Stop if loop detected and stagnating
        if metrics.loop_detected and metrics.stagnation_count >= 2:
            logger.info("Stopping: loop detected with stagnation")
            return True
        
        # Stop if progress is very high
        if metrics.overall_progress >= 0.95:
            logger.info(f"Stopping: high progress ({metrics.overall_progress:.2f})")
            return True
        
        # Continue iterating
        return False
    
    def check_convergence(
        self,
        state: "MissionState",
        evaluator_output: Any = None,
        multiview_disagreement: float = 0.0,
        updated_plan: Any = None
    ) -> bool:
        """
        Comprehensive convergence check for mission iteration.
        
        Convergence is ONLY allowed if ALL of the following are true:
        - evaluator.missing_info is empty
        - evaluator.questions is empty
        - confidence_score >= 0.70
        - updated_plan.new_subgoals is empty
        - multiview_disagreement < 0.25
        
        Args:
            state: Current mission state
            evaluator_output: Latest evaluator output (EvaluationResult)
            multiview_disagreement: Disagreement score from optimist/skeptic (0-1)
            updated_plan: Latest workflow plan from planner
            
        Returns:
            True if convergence criteria are met, False otherwise
        """
        # Check evaluator feedback
        if evaluator_output is not None:
            # Check for missing info
            if hasattr(evaluator_output, 'missing_info') and evaluator_output.missing_info:
                logger.info(f"Cannot converge: {len(evaluator_output.missing_info)} missing info items")
                return False
            
            # Check for unresolved questions
            if hasattr(evaluator_output, 'questions') and evaluator_output.questions:
                logger.info(f"Cannot converge: {len(evaluator_output.questions)} unresolved questions")
                return False
            
            # Check confidence threshold
            if hasattr(evaluator_output, 'confidence_score'):
                if evaluator_output.confidence_score < 0.70:
                    logger.info(f"Cannot converge: low confidence ({evaluator_output.confidence_score:.2f} < 0.70)")
                    return False
            
            # Check for critical missing flag
            if hasattr(evaluator_output, 'critical_missing') and evaluator_output.critical_missing:
                logger.info("Cannot converge: critical information is missing")
                return False
        
        # Check updated plan for pending work
        if updated_plan is not None:
            if hasattr(updated_plan, 'new_subgoals') and updated_plan.new_subgoals:
                logger.info(f"Cannot converge: {len(updated_plan.new_subgoals)} pending subgoals")
                return False
            
            if hasattr(updated_plan, 'has_pending_work') and updated_plan.has_pending_work():
                logger.info("Cannot converge: plan has pending work")
                return False
        
        # Check multiview disagreement
        if multiview_disagreement > 0.25:
            logger.info(f"Cannot converge: high multiview disagreement ({multiview_disagreement:.2f} > 0.25)")
            return False
        
        # All criteria passed - convergence allowed
        logger.info("Convergence criteria met")
        return True
    
    def check_convergence_strict(
        self,
        state: "MissionState",
        evaluator_output: Any = None,
        current_metrics: Optional[PhaseMetrics] = None,
        multiview_disagreement: float = 0.0
    ) -> ConvergenceResult:
        """
        STRICT convergence check prioritizing unresolved questions over confidence.
        
        PRIMARY CRITERIA (ALL must pass for STOP):
        1. unresolved_questions == 0
        2. novelty_delta < NOVELTY_THRESHOLD
        3. progress_delta < PROGRESS_EPSILON
        
        SECONDARY (informational only, logged but not blocking):
        - confidence_score (warning if < 0.5)
        - multiview_agreement
        
        Args:
            state: Current mission state
            evaluator_output: Latest evaluator output (EvaluationResult)
            current_metrics: Current phase metrics for delta computation
            multiview_disagreement: Disagreement from optimist/skeptic
            
        Returns:
            ConvergenceResult with detailed analysis
        """
        blocking = []
        warnings = []
        
        # === PRIMARY CRITERIA ===
        
        # 1. Unresolved questions must be zero
        unresolved_count = 0
        if evaluator_output is not None:
            # Check questions field
            if hasattr(evaluator_output, 'questions') and evaluator_output.questions:
                unresolved_count += len(evaluator_output.questions)
            
            # Check missing_info field
            if hasattr(evaluator_output, 'missing_info') and evaluator_output.missing_info:
                unresolved_count += len(evaluator_output.missing_info)
            
            # Check critical_missing flag
            if hasattr(evaluator_output, 'critical_missing') and evaluator_output.critical_missing:
                unresolved_count += 1  # At least one critical item
        
        if unresolved_count > MAX_UNRESOLVED_QUESTIONS:
            blocking.append(f"unresolved_questions={unresolved_count}")
        
        # 2. Novelty delta must be below threshold
        novelty_delta = 0.0
        if current_metrics is not None:
            novelty_delta = current_metrics.novelty_score
            # Compare to previous if available
            if hasattr(self, '_prev_novelty'):
                novelty_delta = abs(current_metrics.novelty_score - self._prev_novelty)
        
        if novelty_delta > NOVELTY_THRESHOLD:
            blocking.append(f"novelty_delta={novelty_delta:.2f} > {NOVELTY_THRESHOLD}")
        
        # 3. Progress delta must be below epsilon (converged)
        progress_delta = 0.0
        if current_metrics is not None:
            progress_delta = current_metrics.progress_score
            if hasattr(self, '_prev_progress'):
                progress_delta = abs(current_metrics.progress_score - self._prev_progress)
        
        if progress_delta > PROGRESS_EPSILON:
            blocking.append(f"progress_delta={progress_delta:.2f} > {PROGRESS_EPSILON}")
        
        # === SECONDARY CRITERIA (informational only) ===
        
        confidence = 0.5
        if evaluator_output is not None and hasattr(evaluator_output, 'confidence_score'):
            confidence = evaluator_output.confidence_score
            
            if confidence < MIN_CONFIDENCE_WARNING:
                warnings.append(f"low_confidence={confidence:.2f}")
        
        if multiview_disagreement > MULTIVIEW_DISAGREEMENT_WARNING:
            warnings.append(f"multiview_disagreement={multiview_disagreement:.2f}")
        
        # === DECISION ===
        
        can_stop = len(blocking) == 0
        
        if can_stop:
            reason = "convergence_achieved"
            logger.info(
                f"Convergence ALLOWED: unresolved={unresolved_count}, "
                f"novelty_delta={novelty_delta:.2f}, progress_delta={progress_delta:.2f}"
            )
        else:
            reason = "blocked"
            logger.info(
                f"Convergence BLOCKED: {blocking}"
            )
        
        # Log secondary warnings
        if warnings:
            logger.debug(f"Convergence warnings (non-blocking): {warnings}")
        
        # Store metrics for next comparison
        if current_metrics is not None:
            self._prev_novelty = current_metrics.novelty_score
            self._prev_progress = current_metrics.progress_score
        
        return ConvergenceResult(
            can_stop=can_stop,
            reason=reason,
            blocking_criteria=blocking,
            secondary_warnings=warnings,
            unresolved_count=unresolved_count,
            novelty_delta=novelty_delta,
            progress_delta=progress_delta,
            confidence=confidence,
        )
    
    def should_force_iteration(
        self,
        state: "MissionState",
        evaluator_output: Any = None
    ) -> bool:
        """
        Check if another iteration should be forced based on time/confidence.
        
        Forces iteration if:
        - time_remaining > 0.25 * total_time_budget
        - AND confidence_score < 0.7
        
        Args:
            state: Current mission state
            evaluator_output: Latest evaluator output
            
        Returns:
            True if another iteration should be forced
        """
        time_remaining = state.remaining_minutes()
        total_time = state.constraints.time_budget_minutes
        
        # Check time ratio
        if total_time > 0:
            time_ratio = time_remaining / total_time
        else:
            time_ratio = 0
        
        if time_ratio <= 0.25:
            return False
        
        # Check confidence
        if evaluator_output is not None:
            if hasattr(evaluator_output, 'confidence_score'):
                if evaluator_output.confidence_score < 0.70:
                    logger.info(
                        f"Forcing iteration: time available ({time_ratio:.0%}) "
                        f"and low confidence ({evaluator_output.confidence_score:.2f})"
                    )
                    return True
        
        return False
    
    def should_run_multiview(
        self,
        metrics: PhaseMetrics,
        state: "MissionState"
    ) -> bool:
        """
        Determine if multi-view analysis (Optimist + Skeptic) should run.
        
        Triggered when:
        - High difficulty AND high uncertainty
        - Contradictions detected
        - Multiple iterations without convergence
        - Strategic mission type (inferred from objective)
        
        Args:
            metrics: Phase metrics
            state: Current mission state
            
        Returns:
            True if multi-view should run
        """
        # Check difficulty + uncertainty threshold
        if (metrics.difficulty_score >= self.MULTIVIEW_DIFFICULTY_THRESHOLD and
            metrics.uncertainty_score >= self.MULTIVIEW_UNCERTAINTY_THRESHOLD):
            logger.debug("Multi-view triggered: high difficulty + uncertainty")
            return True
        
        # Check contradictions
        if metrics.contradiction_count >= 2:
            logger.debug("Multi-view triggered: contradictions detected")
            return True
        
        # Check if many questions remain
        if metrics.questions_remaining >= 5:
            logger.debug("Multi-view triggered: many unresolved questions")
            return True
        
        # Check mission type for strategic keywords
        objective_lower = state.objective.lower()
        strategic_keywords = [
            "strategic", "analysis", "evaluation", "comparison",
            "decision", "recommendation", "assessment", "review"
        ]
        if any(kw in objective_lower for kw in strategic_keywords):
            # For strategic missions, lower the threshold
            if metrics.difficulty_score >= 0.5 and metrics.uncertainty_score >= 0.5:
                logger.debug("Multi-view triggered: strategic mission")
                return True
        
        # Check iteration count - trigger after 2+ iterations if still uncertain
        iteration_count = getattr(state, 'iteration_count', 0)
        if iteration_count >= 2 and metrics.uncertainty_score >= 0.5:
            logger.debug("Multi-view triggered: multiple iterations with uncertainty")
            return True
        
        return False
    
    def plan_deepening(
        self,
        state: "MissionState",
        metrics: MissionMetrics
    ) -> DeepeningPlan:
        """
        Create a plan for what to deepen after a mission iteration.
        
        Based on:
        - Which phases have high uncertainty
        - Which phases have low progress
        - Time remaining
        - Contradiction areas
        
        Args:
            state: Current mission state
            metrics: Current mission metrics
            
        Returns:
            DeepeningPlan specifying what to re-run
        """
        plan = DeepeningPlan()
        
        # Check time - don't deepen if time is very low
        if metrics.time_remaining_minutes < 2.0:
            plan.reason = "Insufficient time for deepening"
            return plan
        
        # Identify phases needing deepening
        high_uncertainty_phases = []
        low_progress_phases = []
        
        for phase_name, phase_metrics in metrics.phase_metrics.items():
            if phase_metrics.uncertainty_score >= 0.6:
                high_uncertainty_phases.append(phase_name)
            if phase_metrics.progress_score < 0.4:
                low_progress_phases.append(phase_name)
        
        # Decide which councils to re-run
        needs_deepening = set(high_uncertainty_phases + low_progress_phases)
        
        for phase_name in needs_deepening:
            phase_lower = phase_name.lower()
            
            if any(kw in phase_lower for kw in ["research", "recon", "gather"]):
                plan.run_researcher = True
                plan.focus_areas.append(f"Deepen research on: {phase_name}")
            
            if any(kw in phase_lower for kw in ["plan", "design", "strategy"]):
                plan.run_planner = True
                plan.focus_areas.append(f"Refine plan for: {phase_name}")
            
            if any(kw in phase_lower for kw in ["implement", "code", "build"]):
                plan.run_coder = True
                plan.focus_areas.append(f"Improve implementation: {phase_name}")
            
            if any(kw in phase_lower for kw in ["test", "eval", "validate"]):
                plan.run_evaluator = True
                plan.focus_areas.append(f"Strengthen evaluation: {phase_name}")
            
            if any(kw in phase_lower for kw in ["simul", "scenario"]):
                plan.run_simulation = True
                plan.focus_areas.append(f"Explore scenarios: {phase_name}")
        
        # If nothing specific, run evaluator for overall quality check
        if not plan.has_work and metrics.overall_progress < 0.7:
            plan.run_evaluator = True
            plan.focus_areas.append("General quality improvement")
        
        # Set rounds based on time remaining
        if metrics.time_remaining_minutes >= 10:
            plan.max_deepening_rounds = 2
        else:
            plan.max_deepening_rounds = 1
        
        # Generate reason
        if plan.has_work:
            plan.reason = f"Deepening for: {', '.join(needs_deepening)}"
        else:
            plan.reason = "No deepening needed"
        
        return plan
    
    def detect_loops(self, state: "MissionState") -> LoopDetection:
        """
        Detect if the mission is stuck in a reasoning loop.
        
        Checks:
        - Hash similarity of recent outputs
        - Stagnation in quality scores
        - Repetitive structure patterns
        
        Args:
            state: Current mission state
            
        Returns:
            LoopDetection with analysis results
        """
        detection = LoopDetection()
        
        try:
            # Get recent outputs from state
            recent_outputs = getattr(state, 'recent_outputs', [])
            
            if len(recent_outputs) < 2:
                return detection
            
            # Compute hash similarity between recent outputs
            recent_hashes = [
                hashlib.md5(str(o).encode()).hexdigest()[:16]
                for o in recent_outputs[-5:]
            ]
            
            # Check for identical hashes
            unique_hashes = set(recent_hashes)
            if len(unique_hashes) < len(recent_hashes) * 0.5:
                detection.loop_detected = True
                detection.similarity_score = 1.0 - (len(unique_hashes) / len(recent_hashes))
                detection.recommendation = "Break loop: try alternative approach or force plan revision"
                detection.details = f"Found {len(recent_hashes) - len(unique_hashes)} duplicate outputs"
            
            # Check content similarity (simplified)
            if len(recent_outputs) >= 2:
                last_two = [str(o) for o in recent_outputs[-2:]]
                # Simple Jaccard similarity on words
                words1 = set(last_two[0].lower().split())
                words2 = set(last_two[1].lower().split())
                if words1 and words2:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0
                    
                    detection.similarity_score = max(detection.similarity_score, similarity)
                    
                    if similarity >= self.LOOP_SIMILARITY_THRESHOLD:
                        detection.loop_detected = True
                        detection.recommendation = "High similarity: inject novelty or shift focus"
                        detection.details = f"Content similarity: {similarity:.1%}"
            
            # Check stagnation in mission metrics history
            if len(self._mission_metrics_history) >= 3:
                recent_progress = [m.overall_progress for m in self._mission_metrics_history[-3:]]
                progress_delta = max(recent_progress) - min(recent_progress)
                
                if progress_delta < self.MIN_PROGRESS_PER_ITERATION:
                    detection.stagnation_count += 1
                    if detection.stagnation_count >= 2:
                        detection.loop_detected = True
                        detection.recommendation = "Stagnation: escalate to larger model or force synthesis"
            
        except Exception as e:
            logger.warning(f"Error in loop detection: {e}")
        
        return detection
    
    def summarize_context(self, state: "MissionState") -> str:
        """
        Create a concise summary of the current context for later phases.
        
        Summarizes:
        - Key findings from completed phases
        - Current hypotheses
        - Outstanding questions
        - Main conclusions so far
        
        Args:
            state: Current mission state
            
        Returns:
            Context summary string
        """
        parts = [f"# Mission Context Summary\n\nObjective: {state.objective}\n"]
        
        # Summarize completed phases
        parts.append("\n## Phase Summaries\n")
        for phase in state.phases:
            if phase.status == "completed" and phase.artifacts:
                parts.append(f"\n### {phase.name}")
                # Get key artifacts (truncated)
                for key, value in list(phase.artifacts.items())[:3]:
                    if not key.startswith("_"):
                        truncated = str(value)[:300] + "..." if len(str(value)) > 300 else str(value)
                        parts.append(f"\n- {key}: {truncated}")
        
        # Include active hypotheses if available
        hypotheses = getattr(state, 'hypotheses', {})
        active = hypotheses.get('active', [])
        if active:
            parts.append("\n\n## Active Hypotheses")
            for h in active[:5]:
                if isinstance(h, dict):
                    parts.append(f"\n- {h.get('hypothesis', h)}")
                else:
                    parts.append(f"\n- {h}")
        
        # Include meta-level insights
        meta_traces = getattr(state, 'meta_traces', {})
        if meta_traces.get('total_reflections', 0) > 0:
            parts.append(f"\n\n## Meta-Cognition Stats")
            parts.append(f"\n- Reflections: {meta_traces.get('total_reflections', 0)}")
            parts.append(f"- Debates: {meta_traces.get('total_debates', 0)}")
            parts.append(f"- Plan revisions: {meta_traces.get('total_revisions', 0)}")
        
        # Include iteration count
        iteration_count = getattr(state, 'iteration_count', 0)
        if iteration_count > 0:
            parts.append(f"\n\n## Iteration: {iteration_count}")
        
        # Time info
        remaining = state.remaining_minutes()
        parts.append(f"\n\n## Time Remaining: {remaining:.1f} minutes")
        
        return "".join(parts)
    
    def create_depth_contract(
        self,
        metrics: PhaseMetrics,
        phase_name: str,
        time_remaining: float = 10.0
    ) -> DepthContract:
        """
        Create a depth contract for a phase based on its metrics.
        
        Higher difficulty/uncertainty -> deeper exploration
        Lower time remaining -> shallower exploration
        
        Args:
            metrics: Phase metrics
            phase_name: Name of the phase
            time_remaining: Time remaining in minutes
            
        Returns:
            DepthContract specifying depth parameters
        """
        contract = DepthContract()
        
        # Base exploration depth on difficulty + uncertainty
        base_depth = (metrics.difficulty_score + metrics.uncertainty_score) / 2
        
        # Adjust for time pressure
        time_factor = min(1.0, time_remaining / 10.0)  # Full depth if >= 10 min
        contract.exploration_depth = base_depth * time_factor
        
        # Set max rounds
        if contract.exploration_depth >= 0.7:
            contract.max_rounds = 3
        elif contract.exploration_depth >= 0.4:
            contract.max_rounds = 2
        else:
            contract.max_rounds = 1
        
        # Time budget is proportional to exploration depth
        contract.time_budget_minutes = min(time_remaining * 0.3, 5.0 + contract.exploration_depth * 5.0)
        
        # Set model tier
        if metrics.difficulty_score >= 0.8 or metrics.uncertainty_score >= 0.8:
            contract.model_tier = "large"
        elif metrics.difficulty_score >= 0.5 or metrics.uncertainty_score >= 0.5:
            contract.model_tier = "medium"
        else:
            contract.model_tier = "small"
        
        # Allow alternatives if high uncertainty
        contract.allow_alternatives = metrics.uncertainty_score >= 0.5
        
        # Focus areas based on phase name
        phase_lower = phase_name.lower()
        if "research" in phase_lower:
            contract.focus_areas = ["sources", "evidence", "context"]
        elif "design" in phase_lower or "plan" in phase_lower:
            contract.focus_areas = ["alternatives", "trade-offs", "risks"]
        elif "implement" in phase_lower or "code" in phase_lower:
            contract.focus_areas = ["correctness", "edge cases", "quality"]
        elif "test" in phase_lower or "eval" in phase_lower:
            contract.focus_areas = ["coverage", "failure modes", "validation"]
        elif "synthesis" in phase_lower:
            contract.focus_areas = ["coherence", "completeness", "actionability"]
        
        return contract
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _output_to_string(self, output: Any) -> str:
        """Convert output to string for analysis."""
        if isinstance(output, dict):
            parts = []
            for key, value in output.items():
                if not str(key).startswith("_"):
                    parts.append(f"{key}: {str(value)[:500]}")
            return "\n".join(parts)
        return str(output)
    
    def _estimate_difficulty(self, text: str, state: "MissionState") -> float:
        """
        Estimate content difficulty based on heuristics.
        
        Higher difficulty indicators:
        - Technical jargon
        - Complex sentence structures
        - Multiple competing factors
        - Conditional/nuanced statements
        """
        score = 0.5  # Base score
        
        text_lower = text.lower()
        
        # Technical complexity indicators
        technical_words = [
            "however", "although", "nevertheless", "whereas",
            "complexity", "trade-off", "tradeoff", "nuance",
            "depends on", "conditional", "multi-factor",
            "uncertainty", "ambiguous", "unclear"
        ]
        tech_count = sum(1 for w in technical_words if w in text_lower)
        score += min(0.3, tech_count * 0.05)
        
        # Length and depth (longer = potentially more complex)
        word_count = len(text.split())
        if word_count > 1000:
            score += 0.1
        if word_count > 2000:
            score += 0.1
        
        # Check objective complexity
        objective_lower = state.objective.lower()
        complex_keywords = ["analyze", "evaluate", "compare", "strategic", "comprehensive"]
        if any(kw in objective_lower for kw in complex_keywords):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _estimate_uncertainty(self, text: str) -> float:
        """
        Estimate uncertainty from hedging language and question marks.
        """
        score = 0.3  # Base score
        
        text_lower = text.lower()
        
        # Hedging language
        hedging_words = [
            "might", "may", "could", "possibly", "perhaps",
            "uncertain", "unclear", "not sure", "don't know",
            "it seems", "appears to", "likely", "unlikely",
            "possibly", "potentially", "presumably"
        ]
        hedge_count = sum(1 for w in hedging_words if w in text_lower)
        score += min(0.4, hedge_count * 0.05)
        
        # Question marks
        question_count = text.count("?")
        score += min(0.2, question_count * 0.02)
        
        # "TODO", "TBD", etc.
        if "todo" in text_lower or "tbd" in text_lower or "to be determined" in text_lower:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _estimate_progress(
        self,
        phase_name: str,
        output: Any,
        state: "MissionState"
    ) -> float:
        """
        Estimate progress made in this phase.
        """
        score = 0.5  # Base score
        
        # Check if we have artifacts
        if isinstance(output, dict):
            artifact_count = len([k for k in output.keys() if not str(k).startswith("_")])
            score += min(0.3, artifact_count * 0.05)
        
        # Check phase status
        for phase in state.phases:
            if phase.name == phase_name:
                if phase.status == "completed":
                    score += 0.2
                break
        
        # Check for concrete outputs (code blocks, lists, etc.)
        output_str = str(output)
        if "```" in output_str:  # Code blocks
            score += 0.1
        if output_str.count("\n- ") >= 3:  # Lists
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _estimate_novelty(self, text: str) -> float:
        """
        Estimate novelty compared to previous outputs.
        """
        # Compute hash of current output
        current_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check against history
        if current_hash in self._output_hashes:
            return 0.1  # Very low novelty - seen before
        
        # Add to history
        self._output_hashes.append(current_hash)
        if len(self._output_hashes) > 50:
            self._output_hashes = self._output_hashes[-50:]
        
        # Compare content similarity with recent outputs
        if len(self._output_hashes) >= 2:
            # Simple check: word overlap with previous
            return 0.7  # Default to medium-high novelty
        
        return 0.9  # First output is novel
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence in output quality.
        """
        score = 0.5  # Base score
        
        text_lower = text.lower()
        
        # Positive confidence indicators
        confident_words = [
            "clearly", "definitely", "certainly", "confirmed",
            "verified", "proven", "established", "demonstrates"
        ]
        conf_count = sum(1 for w in confident_words if w in text_lower)
        score += min(0.3, conf_count * 0.05)
        
        # Negative confidence indicators (reduce score)
        uncertain_words = ["error", "failed", "unable", "cannot", "warning"]
        unc_count = sum(1 for w in uncertain_words if w in text_lower)
        score -= min(0.3, unc_count * 0.1)
        
        return min(1.0, max(0.0, score))
    
    def _count_contradictions(self, text: str, state: "MissionState") -> int:
        """
        Count potential contradictions in output.
        """
        count = 0
        
        text_lower = text.lower()
        
        # Contradiction indicators
        contradiction_phrases = [
            "however, ", "on the other hand", "contradicts",
            "inconsistent with", "conflicts with", "but also",
            "alternatively", "in contrast"
        ]
        
        for phrase in contradiction_phrases:
            count += text_lower.count(phrase)
        
        # Check meta_traces for debate contradictions
        meta_traces = getattr(state, 'meta_traces', {})
        debate_stats = meta_traces.get('debate', {})
        if isinstance(debate_stats, dict):
            count += debate_stats.get('contradictions_found', 0)
        
        return count
    
    def _count_questions(self, text: str) -> int:
        """
        Count unresolved questions.
        """
        # Simple heuristic: count question marks
        return text.count("?")
    
    def _get_latest_phase_metrics(self, phase_name: str) -> Optional[PhaseMetrics]:
        """
        Get the latest metrics for a phase.
        """
        history = self._phase_metrics_history.get(phase_name, [])
        return history[-1] if history else None
    
    def _compute_convergence(
        self,
        state: "MissionState",
        metrics: MissionMetrics
    ) -> float:
        """
        Compute overall convergence score.
        
        Higher when:
        - Many phases completed
        - High confidence across phases
        - Low uncertainty
        - High progress
        """
        score = 0.0
        
        # Phase completion ratio
        completed = len([p for p in state.phases if p.status == "completed"])
        total = len(state.phases)
        if total > 0:
            score += 0.3 * (completed / total)
        
        # Inverse of uncertainty
        score += 0.2 * (1.0 - metrics.avg_uncertainty)
        
        # Progress
        score += 0.3 * metrics.overall_progress
        
        # Average confidence from phase metrics
        confidences = [pm.confidence_score for pm in metrics.phase_metrics.values()]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            score += 0.2 * avg_conf
        
        return min(1.0, max(0.0, score))
    
    def _compute_stagnation_count(self, current_metrics: MissionMetrics) -> int:
        """
        Count consecutive iterations with minimal improvement.
        """
        if len(self._mission_metrics_history) < 2:
            return 0
        
        count = 0
        for i in range(len(self._mission_metrics_history) - 1, 0, -1):
            prev = self._mission_metrics_history[i - 1]
            curr = self._mission_metrics_history[i]
            
            progress_delta = curr.overall_progress - prev.overall_progress
            if progress_delta < self.MIN_PROGRESS_PER_ITERATION:
                count += 1
            else:
                break
        
        return count
    
    # =========================================================================
    # CognitiveSpine Integration & Fallback Policies
    # =========================================================================
    
    def set_cognitive_spine(self, spine: "CognitiveSpine") -> None:
        """
        Set the CognitiveSpine instance.
        
        Args:
            spine: CognitiveSpine instance
        """
        self._cognitive_spine = spine
    
    def validate_phase_context(
        self,
        phase_name: str,
        context: Any,
        previous_output: Optional[Any] = None
    ) -> Tuple[bool, Any, List[str]]:
        """
        Validate context before phase execution.
        
        Uses CognitiveSpine for validation and auto-correction.
        
        Args:
            phase_name: Name of the phase
            context: Context for the phase
            previous_output: Output from previous phase
            
        Returns:
            Tuple of (is_valid, corrected_context, warnings)
        """
        if self._cognitive_spine is None:
            return True, context, []
        
        return self._cognitive_spine.validate_phase_boundary(
            phase_name, context, previous_output
        )
    
    def get_fallback_action(
        self,
        error_type: str,
        council_name: str,
        state: Optional["MissionState"] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Determine fallback action for a failure.
        
        Fallback policies:
        - schema_mismatch: Reconstruct minimal context, retry
        - consensus_missing: Run single best model
        - output_too_long: Compress and retry
        - time_exhausted: Skip to synthesis
        - resource_exceeded: Compress memory, retry with smaller model
        
        Args:
            error_type: Type of error that occurred
            council_name: Name of the failing council
            state: Current mission state
            
        Returns:
            Tuple of (action, action_params)
        """
        # Track attempts to prevent infinite loops
        attempt_key = f"{council_name}_{error_type}"
        attempts = self._fallback_attempts.get(attempt_key, 0)
        self._fallback_attempts[attempt_key] = attempts + 1
        
        # Max 2 fallback attempts per error type
        if attempts >= 2:
            logger.warning(f"Max fallback attempts reached for {attempt_key}")
            return "skip", {"reason": "max_fallback_attempts"}
        
        if error_type == "schema_mismatch":
            return "reconstruct_minimal", {
                "council": council_name,
                "action": "Create minimal valid context and retry"
            }
        
        elif error_type == "consensus_missing":
            return "single_model", {
                "council": council_name,
                "action": "Run with single best model instead of consensus"
            }
        
        elif error_type == "output_too_long":
            return "compress_retry", {
                "council": council_name,
                "max_chars": 5000,
                "action": "Compress output and retry"
            }
        
        elif error_type == "time_exhausted":
            return "force_synthesis", {
                "council": council_name,
                "action": "Skip to synthesis with compressed memory"
            }
        
        elif error_type == "resource_exceeded":
            return "downgrade_retry", {
                "council": council_name,
                "model_tier": "small",
                "action": "Compress memory, retry with smaller model"
            }
        
        else:
            return "skip", {"reason": f"Unknown error type: {error_type}"}
    
    def apply_fallback(
        self,
        action: str,
        params: Dict[str, Any],
        context: Any,
        state: Optional["MissionState"] = None
    ) -> Tuple[Any, bool]:
        """
        Apply a fallback action.
        
        Args:
            action: Fallback action to apply
            params: Action parameters
            context: Current context
            state: Current mission state
            
        Returns:
            Tuple of (modified_context, should_retry)
        """
        logger.info(f"Applying fallback: {action} - {params.get('action', '')}")
        
        if action == "reconstruct_minimal":
            if self._cognitive_spine is not None:
                council_name = params.get("council", "")
                objective = ""
                if hasattr(context, 'objective'):
                    objective = context.objective
                elif isinstance(context, dict):
                    objective = context.get('objective', '')
                
                new_context = self._cognitive_spine.create_minimal_context(
                    council_name, objective
                )
                return new_context, True
        
        elif action == "single_model":
            # Flag for council to use single model mode
            if isinstance(context, dict):
                context["_use_single_model"] = True
            return context, True
        
        elif action == "compress_retry":
            max_chars = params.get("max_chars", 5000)
            if self._cognitive_spine is not None and isinstance(context, dict):
                compressed = self._cognitive_spine.compress_context(context, max_chars)
                return compressed, True
            return context, True
        
        elif action == "force_synthesis":
            # Enter contraction mode and skip to synthesis
            self.enter_contraction_mode(state)
            return context, False  # Don't retry, proceed to synthesis
        
        elif action == "downgrade_retry":
            # Compress and flag for smaller model
            if self._cognitive_spine is not None and isinstance(context, dict):
                compressed = self._cognitive_spine.compress_context(context, 3000)
                compressed["_model_tier"] = "small"
                return compressed, True
            return context, True
        
        return context, False
    
    def enter_contraction_mode(
        self,
        state: Optional["MissionState"] = None
    ) -> bool:
        """
        Enter contraction mode when resources are exhausted.
        
        In contraction mode:
        - Skip non-essential phases
        - Compress all memory
        - Force synthesis with minimal context
        - Use smallest available models
        
        Args:
            state: Current mission state
            
        Returns:
            True if contraction mode was entered
        """
        if self._contraction_mode:
            return True
        
        self._contraction_mode = True
        logger.warning("ReasoningSupervisor entering contraction mode")
        
        if self._cognitive_spine is not None:
            self._cognitive_spine.enter_contraction_mode()
        
        # Log the trigger
        time_remaining = 0.0
        if state is not None:
            time_remaining = state.remaining_minutes()
        
        logger.info(f"Contraction mode: time_remaining={time_remaining:.1f}min")
        
        return True
    
    def is_contraction_mode(self) -> bool:
        """Check if in contraction mode."""
        return self._contraction_mode
    
    def should_skip_phase(
        self,
        phase_name: str,
        state: "MissionState"
    ) -> Tuple[bool, str]:
        """
        Determine if a phase should be skipped.
        
        Skips non-essential phases in contraction mode.
        
        Args:
            phase_name: Name of the phase
            state: Current mission state
            
        Returns:
            Tuple of (should_skip, reason)
        """
        if not self._contraction_mode:
            return False, ""
        
        # Essential phases that should never be skipped
        essential_phases = ["synthesis", "report", "final", "conclusion"]
        name_lower = phase_name.lower()
        
        if any(kw in name_lower for kw in essential_phases):
            return False, ""
        
        # Skip optional phases in contraction mode
        optional_phases = [
            "deep_analysis", "simulation", "testing", 
            "refinement", "iteration", "validation"
        ]
        
        if any(kw in name_lower for kw in optional_phases):
            return True, "Contraction mode - skipping optional phase"
        
        # Check time - skip if very low
        if state.remaining_minutes() < 1.0:
            return True, "Time exhausted"
        
        return False, ""
    
    def get_compressed_context_for_synthesis(
        self,
        state: "MissionState"
    ) -> Dict[str, Any]:
        """
        Create compressed context for emergency synthesis.
        
        Used when forcing synthesis in contraction mode.
        
        Args:
            state: Current mission state
            
        Returns:
            Compressed synthesis context
        """
        # Gather artifacts from all completed phases
        all_findings = []
        for phase in state.phases:
            if phase.status == "completed" and phase.artifacts:
                for key, value in phase.artifacts.items():
                    if not key.startswith('_'):
                        # Truncate large values
                        if len(str(value)) > 500:
                            all_findings.append(f"{key}: {str(value)[:500]}...")
                        else:
                            all_findings.append(f"{key}: {value}")
        
        # Compress findings
        findings_text = "\n".join(all_findings)
        if len(findings_text) > 3000:
            findings_text = findings_text[:3000] + "\n[Truncated due to contraction mode]"
        
        return {
            "objective": state.objective,
            "prior_findings": findings_text,
            "unresolved_issues": [],  # Skip unresolved in contraction
            "structural_gaps": [],
            "iteration": 1,
            "max_iterations": 1,  # Single iteration in contraction
            "_contraction_mode": True,
        }
    
    def reset(self) -> None:
        """
        Reset supervisor state for a new mission.
        """
        self._output_hashes.clear()
        self._phase_metrics_history.clear()
        self._mission_metrics_history.clear()
        self._fallback_attempts.clear()
        self._contraction_mode = False


