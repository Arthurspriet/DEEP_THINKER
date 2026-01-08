"""
Council Workflow Runner for DeepThinker 2.0.

Orchestrates the council-based multi-LLM workflow with non-deterministic
branching, multiple candidate solutions, and council disagreement resolution.

Enhanced for autonomous execution:
- Time-aware iteration with convergence detection
- Multi-view reasoning (optimist + skeptic councils)
- All councils activated by default

Flow:
    meta_planning -> planner_council -> researcher_council
    -> coder_council -> evaluator_council -> simulator_council
    -> [optimist_council + skeptic_council] -> arbiter -> executor
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .state_manager import CouncilStateManager, council_state_manager, CouncilPhase
from .iteration_manager import IterationManager

# Verbose logging integration
try:
    from ..cli import verbose_logger
    VERBOSE_LOGGER_AVAILABLE = True
except ImportError:
    VERBOSE_LOGGER_AVAILABLE = False
    verbose_logger = None

from ..meta_planner import MetaPlanner
from ..arbiter import Arbiter, CouncilOutput
from ..councils import (
    PlannerCouncil,
    ResearcherCouncil,
    CoderCouncil,
    EvaluatorCouncil,
    SimulationCouncil,
)

# Multi-view councils
try:
    from ..councils.multi_view import OptimistCouncil, SkepticCouncil
    MULTIVIEW_AVAILABLE = True
except ImportError:
    MULTIVIEW_AVAILABLE = False
    OptimistCouncil = None
    SkepticCouncil = None

from ..councils.planner_council.planner_council import PlannerContext
from ..councils.researcher_council.researcher_council import ResearchContext
from ..councils.coder_council.coder_council import CoderContext
from ..councils.evaluator_council.evaluator_council import EvaluatorContext
from ..councils.simulation_council.simulation_council import SimulationContext
from ..models.council_model_config import CouncilConfig, DEFAULT_COUNCIL_CONFIG
from ..execution.data_config import DataConfig
from ..execution.simulation_config import SimulationConfig


@dataclass
class CouncilWorkflowConfig:
    """Configuration for council workflow execution."""
    
    max_iterations: int = 5  # Increased for autonomous mode
    quality_threshold: float = 7.0
    council_config: CouncilConfig = field(default_factory=lambda: DEFAULT_COUNCIL_CONFIG)
    data_config: Optional[DataConfig] = None
    simulation_config: Optional[SimulationConfig] = None
    ollama_base_url: str = "http://localhost:11434"
    verbose: bool = False
    
    # Time-aware iteration settings
    time_budget_seconds: Optional[float] = None  # None = no time limit
    min_iteration_time_seconds: float = 30.0
    convergence_threshold: float = 0.02
    consecutive_convergence_count: int = 2
    
    # Autonomous mode settings
    force_all_councils: bool = True
    enable_multiview: bool = True


@dataclass
class CouncilWorkflowResult:
    """Result from council workflow execution."""
    
    workflow_id: str
    objective: str
    success: bool
    final_code: str
    final_evaluation: Any
    quality_score: float
    iterations_completed: int
    meta_plan: Any
    council_outputs: Dict[str, Any] = field(default_factory=dict)
    simulation_findings: Any = None
    arbiter_decision: Any = None
    error: Optional[str] = None


class CouncilWorkflowRunner:
    """
    Orchestrates the DeepThinker 2.0 council-based workflow.
    
    Manages meta-planning, council execution, consensus building,
    and arbiter-based final decision making.
    
    Enhanced for autonomous execution:
    - Time-aware iteration with automatic convergence detection
    - Multi-view reasoning (OptimistCouncil + SkepticCouncil)
    - All councils activated by default for deep analysis
    """
    
    def __init__(self, config: Optional[CouncilWorkflowConfig] = None):
        """
        Initialize council workflow runner.
        
        Args:
            config: Workflow configuration
        """
        self.config = config or CouncilWorkflowConfig()
        self.state_manager = council_state_manager
        
        # Enhanced iteration manager with time/convergence awareness
        self.iteration_manager = IterationManager(
            max_iterations=self.config.max_iterations,
            quality_threshold=self.config.quality_threshold,
            convergence_threshold=self.config.convergence_threshold,
            min_iterations=2,
            time_budget_seconds=self.config.time_budget_seconds,
            min_iteration_time_seconds=self.config.min_iteration_time_seconds,
            consecutive_convergence_count=self.config.consecutive_convergence_count
        )
        
        # Initialize components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize all workflow components."""
        base_url = self.config.ollama_base_url
        
        # Meta-planner with force_all_councils setting
        self.meta_planner = MetaPlanner(
            ollama_base_url=base_url,
            force_all_councils=self.config.force_all_councils
        )
        self.arbiter = Arbiter(ollama_base_url=base_url)
        
        # Core councils
        self.planner_council = PlannerCouncil(ollama_base_url=base_url)
        self.researcher_council = ResearcherCouncil(ollama_base_url=base_url)
        self.coder_council = CoderCouncil(ollama_base_url=base_url)
        self.evaluator_council = EvaluatorCouncil(
            ollama_base_url=base_url,
            quality_threshold=self.config.quality_threshold
        )
        self.simulation_council = SimulationCouncil(ollama_base_url=base_url)
        
        # Multi-view councils
        self.optimist_council = None
        self.skeptic_council = None
        
        if self.config.enable_multiview and MULTIVIEW_AVAILABLE:
            self.optimist_council = OptimistCouncil(ollama_base_url=base_url)
            self.skeptic_council = SkepticCouncil(ollama_base_url=base_url)
    
    def run(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CouncilWorkflowResult:
        """
        Execute the complete council-based workflow.
        
        Args:
            objective: Primary task objective
            context: Additional context
            
        Returns:
            CouncilWorkflowResult with final output
        """
        # Start workflow tracking
        workflow_id = self.state_manager.start_workflow(
            objective=objective,
            max_iterations=self.config.max_iterations
        )
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 DeepThinker 2.0 Council Workflow Started")
            print(f"   Workflow ID: {workflow_id}")
            print(f"   Objective: {objective[:100]}...")
            print(f"{'='*60}\n")
        
        try:
            # Phase 1: Meta-Planning
            meta_plan = self._run_meta_planning(objective, context)
            self.state_manager.set_meta_plan(vars(meta_plan))
            
            if self.config.verbose:
                print(f"\n📋 Meta-Plan: {meta_plan.councils_to_activate}")
                print(f"   Iterations: {meta_plan.iteration_count}")
                print(f"   Exploration: {meta_plan.exploration_level}")
            
            # Update iteration manager with meta-plan settings
            self.iteration_manager.max_iterations = min(
                meta_plan.iteration_count,
                self.config.max_iterations
            )
            
            # Phase 2: Planner Council
            workflow_plan = None
            if "planner" in meta_plan.councils_to_activate:
                # Verbose logging: context flow
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_context_flow("MetaPlanner", "PlannerCouncil", {
                        "objective": objective,
                        "councils_to_activate": meta_plan.councils_to_activate
                    })
                workflow_plan = self._run_planner_council(objective, context)
            
            # Phase 3: Researcher Council
            research_findings = None
            if "researcher" in meta_plan.councils_to_activate:
                # Verbose logging: context flow
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_context_flow("PlannerCouncil", "ResearcherCouncil", {
                        "objective": objective,
                        "has_workflow_plan": workflow_plan is not None
                    })
                research_findings = self._run_researcher_council(
                    objective, 
                    workflow_plan
                )
            
            # Phase 4-6: Iterative Code Generation, Evaluation, Simulation
            final_result = self._run_iterative_phase(
                objective=objective,
                context=context,
                workflow_plan=workflow_plan,
                research_findings=research_findings,
                meta_plan=meta_plan
            )
            
            # Complete workflow
            self.state_manager.complete_workflow(final_output=final_result)
            
            return CouncilWorkflowResult(
                workflow_id=workflow_id,
                objective=objective,
                success=True,
                final_code=final_result.get("code", ""),
                final_evaluation=final_result.get("evaluation"),
                quality_score=final_result.get("quality_score", 0.0),
                iterations_completed=final_result.get("iterations", 0),
                meta_plan=meta_plan,
                council_outputs=final_result.get("council_outputs", {}),
                simulation_findings=final_result.get("simulation"),
                arbiter_decision=final_result.get("arbiter_decision")
            )
            
        except Exception as e:
            self.state_manager.complete_workflow(error=str(e))
            
            return CouncilWorkflowResult(
                workflow_id=workflow_id,
                objective=objective,
                success=False,
                final_code="",
                final_evaluation=None,
                quality_score=0.0,
                iterations_completed=0,
                meta_plan=None,
                error=str(e)
            )
    
    def _run_meta_planning(
        self,
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Run meta-planning phase."""
        if self.config.verbose:
            print("\n🧠 Phase 1: Meta-Planning...")
        
        # Verbose logging: meta-planner council
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.meta_planner.__class__)
            verbose_logger.log_context_received("Meta-Planning Input", {"objective": objective, "context": context})
        
        self.state_manager.start_council_execution(
            council_name="meta_planner",
            phase=CouncilPhase.META_PLANNING,
            models=[self.config.council_config.meta_planner_model],
            consensus_method="single_model"
        )
        
        meta_plan = self.meta_planner.plan(objective, context)
        
        self.state_manager.complete_council_execution(
            council_name="meta_planner",
            output=meta_plan,
            consensus_confidence=1.0
        )
        
        # Verbose logging: meta-plan output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("MetaPlanner", success=True)
            verbose_logger.log_context_output("Meta-Plan", vars(meta_plan) if hasattr(meta_plan, '__dict__') else meta_plan)
        
        return meta_plan
    
    def _run_planner_council(
        self,
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Run planner council phase."""
        if self.config.verbose:
            print("\n📋 Phase 2: Planner Council...")
        
        # Verbose logging: planner council
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.planner_council.__class__)
            verbose_logger.log_council_activated(
                "PlannerCouncil",
                {"objective": objective, "context": context},
                models=self.config.council_config.planner.get_model_names()
            )
        
        self.state_manager.start_council_execution(
            council_name="planner",
            phase=CouncilPhase.PLANNER_COUNCIL,
            models=self.config.council_config.planner.get_model_names(),
            consensus_method="weighted_blend"
        )
        
        result = self.planner_council.plan(
            objective=objective,
            context=context,
            data_config=self.config.data_config,
            simulation_config=self.config.simulation_config,
            max_iterations=self.config.max_iterations,
            quality_threshold=self.config.quality_threshold
        )
        
        confidence = 1.0
        if result.consensus_details and hasattr(result.consensus_details, 'confidence'):
            confidence = result.consensus_details.confidence
        
        self.state_manager.complete_council_execution(
            council_name="planner",
            output=result.output,
            consensus_confidence=confidence
        )
        
        # Verbose logging: planner output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("PlannerCouncil", success=result.success)
            if result.output:
                verbose_logger.log_context_output("Planner Output", result.output)
        
        return result.output
    
    def _run_researcher_council(
        self,
        objective: str,
        workflow_plan: Any
    ) -> Any:
        """Run researcher council phase."""
        if self.config.verbose:
            print("\n🔍 Phase 3: Researcher Council...")
        
        # Verbose logging: researcher council
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.researcher_council.__class__)
            verbose_logger.log_council_activated(
                "ResearcherCouncil",
                {"objective": objective, "workflow_plan": str(workflow_plan)[:200] if workflow_plan else None},
                models=self.config.council_config.researcher.get_model_names()
            )
        
        self.state_manager.start_council_execution(
            council_name="researcher",
            phase=CouncilPhase.RESEARCHER_COUNCIL,
            models=self.config.council_config.researcher.get_model_names(),
            consensus_method="voting"
        )
        
        planner_requirements = None
        if workflow_plan and hasattr(workflow_plan, 'get_agent_requirements'):
            planner_requirements = workflow_plan.get_agent_requirements("researcher")
        
        result = self.researcher_council.research(
            objective=objective,
            planner_requirements=planner_requirements
        )
        
        confidence = 1.0
        if result.consensus_details and hasattr(result.consensus_details, 'confidence'):
            confidence = result.consensus_details.confidence
        
        self.state_manager.complete_council_execution(
            council_name="researcher",
            output=result.output,
            consensus_confidence=confidence
        )
        
        # Verbose logging: researcher output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("ResearcherCouncil", success=result.success)
            if result.output:
                verbose_logger.log_context_output("Research Output", result.output)
        
        return result.output
    
    def _run_iterative_phase(
        self,
        objective: str,
        context: Optional[Dict[str, Any]],
        workflow_plan: Any,
        research_findings: Any,
        meta_plan: Any
    ) -> Dict[str, Any]:
        """
        Run the iterative code generation, evaluation, simulation loop.
        
        Enhanced with:
        - Time-aware iteration (stops when time budget exhausted)
        - Convergence detection (stops when improvement < threshold)
        - Multi-view councils (optimist + skeptic perspectives)
        """
        current_code = None
        current_evaluation = None
        council_outputs = {}
        optimist_output = None
        skeptic_output = None
        
        # Reset and start iteration manager with time tracking
        self.iteration_manager.reset()
        self.iteration_manager.start()
        
        while True:
            # Start iteration
            iteration = self.iteration_manager.start_iteration()
            self.state_manager.start_iteration(iteration)
            
            # Calculate time info for logging
            time_remaining = self.iteration_manager.time_remaining()
            time_str = f" ({time_remaining:.0f}s remaining)" if time_remaining < float('inf') else ""
            
            if self.config.verbose:
                print(f"\n📝 Iteration {iteration}/{self.iteration_manager.max_iterations}{time_str}")
            
            # Coder Council
            if "coder" in meta_plan.councils_to_activate:
                # Verbose logging: context flow
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    source = "EvaluatorCouncil" if current_evaluation else "ResearcherCouncil"
                    verbose_logger.log_context_flow(source, "CoderCouncil", {
                        "iteration": iteration,
                        "has_previous_code": current_code is not None,
                        "has_feedback": current_evaluation is not None
                    })
                code_result = self._run_coder_council(
                    objective=objective,
                    context=context,
                    research_findings=research_findings,
                    workflow_plan=workflow_plan,
                    previous_code=current_code,
                    evaluation_feedback=current_evaluation,
                    iteration=iteration
                )
                current_code = code_result.code if hasattr(code_result, 'code') else str(code_result)
                council_outputs["coder"] = code_result
            
            # Evaluator Council
            if "evaluator" in meta_plan.councils_to_activate and current_code:
                # Verbose logging: context flow
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_context_flow("CoderCouncil", "EvaluatorCouncil", {
                        "iteration": iteration,
                        "code_length": len(current_code)
                    })
                eval_result = self._run_evaluator_council(
                    objective=objective,
                    code=current_code,
                    iteration=iteration
                )
                current_evaluation = eval_result
                council_outputs["evaluator"] = eval_result
            
            # Multi-view councils (optimist + skeptic) - run in parallel conceptually
            if self.config.enable_multiview and current_code:
                optimist_output, skeptic_output = self._run_multiview_councils(
                    objective=objective,
                    content=current_code,
                    evaluation=current_evaluation,
                    iteration=iteration
                )
                if optimist_output:
                    council_outputs["optimist"] = optimist_output
                if skeptic_output:
                    council_outputs["skeptic"] = skeptic_output
            
            # Get quality score
            quality_score = 0.0
            passed = False
            if current_evaluation:
                if hasattr(current_evaluation, 'quality_score'):
                    quality_score = current_evaluation.quality_score
                    passed = current_evaluation.passed
            
            if self.config.verbose:
                print(f"   Quality: {quality_score}/10 {'✅' if passed else '❌'}")
                if self.iteration_manager.time_budget_seconds:
                    remaining = self.iteration_manager.time_remaining()
                    elapsed = self.iteration_manager.time_elapsed()
                    print(f"   Time: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
            
            # Record iteration with multi-view outputs
            self.iteration_manager.record_iteration(
                code=current_code or "",
                quality_score=quality_score,
                passed=passed,
                council_outputs=council_outputs,
                optimist_output=optimist_output,
                skeptic_output=skeptic_output
            )
            
            self.state_manager.update_iteration_results(
                code=current_code or "",
                quality_score=quality_score,
                passed=passed
            )
            
            # Check if we should continue
            should_continue, reason = self.iteration_manager.should_continue()
            
            if self.config.verbose:
                print(f"   Status: {reason}")
            
            if not should_continue:
                break
        
        # Verbose logging: final iteration table
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            self.iteration_manager.log_final_summary()
        
        # Simulation Council (after iterations complete)
        simulation_findings = None
        if "simulation" in meta_plan.councils_to_activate and current_code:
            # Verbose logging: context flow
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_context_flow("EvaluatorCouncil", "SimulationCouncil", {
                    "iterations_completed": self.iteration_manager.current_iteration,
                    "final_quality": current_evaluation.quality_score if hasattr(current_evaluation, 'quality_score') else None
                })
            simulation_findings = self._run_simulation_council(
                code=current_code,
                objective=objective
            )
            council_outputs["simulation"] = simulation_findings
        
        # Arbiter - Final Decision
        # Verbose logging: context flow
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_context_flow("SimulationCouncil" if simulation_findings else "EvaluatorCouncil", "Arbiter", {
                "councils_with_output": list(council_outputs.keys())
            })
        arbiter_decision = self._run_arbiter(
            objective=objective,
            council_outputs=council_outputs
        )
        
        # Get best iteration result
        best = self.iteration_manager.get_best_iteration()
        
        return {
            "code": current_code,
            "evaluation": current_evaluation,
            "quality_score": best.quality_score if best else 0.0,
            "iterations": self.iteration_manager.current_iteration,
            "council_outputs": council_outputs,
            "simulation": simulation_findings,
            "arbiter_decision": arbiter_decision
        }
    
    def _run_coder_council(
        self,
        objective: str,
        context: Optional[Dict[str, Any]],
        research_findings: Any,
        workflow_plan: Any,
        previous_code: Optional[str],
        evaluation_feedback: Any,
        iteration: int
    ) -> Any:
        """Run coder council for code generation or revision."""
        if self.config.verbose:
            action = "Revising" if previous_code else "Generating"
            print(f"   💻 {action} code...")
        
        # Verbose logging: coder council
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.coder_council.__class__)
            verbose_logger.log_council_activated(
                "CoderCouncil",
                {
                    "objective": objective[:200] if objective else None,
                    "iteration": iteration,
                    "has_previous_code": previous_code is not None,
                    "has_feedback": evaluation_feedback is not None
                },
                models=self.config.council_config.coder.get_model_names()
            )
        
        self.state_manager.start_council_execution(
            council_name="coder",
            phase=CouncilPhase.CODER_COUNCIL,
            models=self.config.council_config.coder.get_model_names(),
            consensus_method="critique_exchange"
        )
        
        planner_requirements = None
        if workflow_plan and hasattr(workflow_plan, 'get_agent_requirements'):
            planner_requirements = workflow_plan.get_agent_requirements("coder")
        
        research_str = None
        if research_findings:
            if hasattr(research_findings, 'raw_output'):
                research_str = research_findings.raw_output
            else:
                research_str = str(research_findings)
        
        if previous_code and evaluation_feedback:
            # Revision
            eval_str = str(evaluation_feedback)
            result = self.coder_council.revise_code(
                objective=objective,
                previous_code=previous_code,
                evaluation_feedback=eval_str,
                iteration=iteration,
                context=context
            )
        else:
            # Initial generation
            result = self.coder_council.generate_code(
                objective=objective,
                context=context,
                research_findings=research_str,
                planner_requirements=planner_requirements,
                data_config=self.config.data_config
            )
        
        confidence = 1.0
        if result.consensus_details and hasattr(result.consensus_details, 'consensus_reached'):
            confidence = 0.9 if result.consensus_details.consensus_reached else 0.7
        
        self.state_manager.complete_council_execution(
            council_name="coder",
            output=result.output,
            consensus_confidence=confidence
        )
        
        # Verbose logging: coder output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("CoderCouncil", success=result.success)
            if result.output:
                # Log code preview
                code = result.output.code if hasattr(result.output, 'code') else str(result.output)
                verbose_logger.log_context_output("Code Output", {"code_preview": code[:300] + "..." if len(code) > 300 else code})
        
        return result.output
    
    def _run_evaluator_council(
        self,
        objective: str,
        code: str,
        iteration: int
    ) -> Any:
        """Run evaluator council for code assessment."""
        if self.config.verbose:
            print(f"   📊 Evaluating code...")
        
        # Verbose logging: evaluator council
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.evaluator_council.__class__)
            verbose_logger.log_council_activated(
                "EvaluatorCouncil",
                {"objective": objective[:200] if objective else None, "iteration": iteration, "code_length": len(code)},
                models=self.config.council_config.evaluator.get_model_names()
            )
        
        self.state_manager.start_council_execution(
            council_name="evaluator",
            phase=CouncilPhase.EVALUATOR_COUNCIL,
            models=self.config.council_config.evaluator.get_model_names(),
            consensus_method="weighted_blend"
        )
        
        result = self.evaluator_council.evaluate(
            objective=objective,
            code=code,
            iteration=iteration
        )
        
        confidence = 1.0
        if result.consensus_details:
            # WeightedBlend doesn't have explicit confidence, use 0.85
            confidence = 0.85
        
        self.state_manager.complete_council_execution(
            council_name="evaluator",
            output=result.output,
            consensus_confidence=confidence
        )
        
        # Verbose logging: evaluator output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("EvaluatorCouncil", success=result.success)
            if result.output:
                verbose_logger.log_context_output("Evaluation Output", result.output)
        
        return result.output
    
    def _run_simulation_council(
        self,
        code: str,
        objective: str
    ) -> Any:
        """Run simulation council for edge-case testing."""
        if self.config.verbose:
            print(f"\n🔬 Running Simulation Council...")
        
        # Verbose logging: simulation council
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_requirements(self.simulation_council.__class__)
            verbose_logger.log_council_activated(
                "SimulationCouncil",
                {"objective": objective[:200] if objective else None, "code_length": len(code)},
                models=self.config.council_config.simulation.get_model_names()
            )
        
        self.state_manager.start_council_execution(
            council_name="simulation",
            phase=CouncilPhase.SIMULATION_COUNCIL,
            models=self.config.council_config.simulation.get_model_names(),
            consensus_method="semantic_distance"
        )
        
        result = self.simulation_council.simulate(
            code=code,
            objective=objective
        )
        
        confidence = 1.0
        if result.consensus_details and hasattr(result.consensus_details, 'confidence'):
            confidence = result.consensus_details.confidence
        
        self.state_manager.complete_council_execution(
            council_name="simulation",
            output=result.output,
            consensus_confidence=confidence
        )
        
        # Verbose logging: simulation output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("SimulationCouncil", success=result.success)
            if result.output:
                verbose_logger.log_context_output("Simulation Output", result.output)
        
        return result.output
    
    def _run_multiview_councils(
        self,
        objective: str,
        content: str,
        evaluation: Any,
        iteration: int
    ) -> tuple:
        """
        Run optimist and skeptic councils for multi-view reasoning.
        
        Args:
            objective: Task objective
            content: Content to evaluate (code, analysis, etc.)
            evaluation: Prior evaluation results
            iteration: Current iteration number
            
        Returns:
            Tuple of (optimist_output, skeptic_output)
        """
        optimist_output = None
        skeptic_output = None
        
        if not MULTIVIEW_AVAILABLE:
            return optimist_output, skeptic_output
        
        # Prepare prior evaluation string
        prior_eval_str = None
        if evaluation:
            if hasattr(evaluation, 'raw_output'):
                prior_eval_str = evaluation.raw_output
            else:
                prior_eval_str = str(evaluation)
        
        # Run Optimist Council
        if self.optimist_council and "optimist" in getattr(self, '_active_councils', ['optimist']):
            if self.config.verbose:
                print(f"   🌟 Running OptimistCouncil...")
            
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_council_activated(
                    "OptimistCouncil",
                    {"objective": objective[:200], "iteration": iteration},
                    models=self.config.council_config.evaluator.get_model_names()
                )
            
            self.state_manager.start_council_execution(
                council_name="optimist",
                phase=CouncilPhase.EVALUATOR_COUNCIL,  # Uses evaluator phase slot
                models=self.config.council_config.evaluator.get_model_names(),
                consensus_method="weighted_blend"
            )
            
            try:
                result = self.optimist_council.evaluate(
                    objective=objective,
                    content=content,
                    prior_evaluations=prior_eval_str,
                    iteration=iteration
                )
                optimist_output = result.output if result.success else None
                
                self.state_manager.complete_council_execution(
                    council_name="optimist",
                    output=optimist_output,
                    consensus_confidence=0.8
                )
                
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_council_execution_complete("OptimistCouncil", success=result.success)
            except Exception as e:
                if self.config.verbose:
                    print(f"   ⚠️ OptimistCouncil error: {e}")
        
        # Run Skeptic Council
        if self.skeptic_council and "skeptic" in getattr(self, '_active_councils', ['skeptic']):
            if self.config.verbose:
                print(f"   🔍 Running SkepticCouncil...")
            
            if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                verbose_logger.log_council_activated(
                    "SkepticCouncil",
                    {"objective": objective[:200], "iteration": iteration},
                    models=self.config.council_config.evaluator.get_model_names()
                )
            
            self.state_manager.start_council_execution(
                council_name="skeptic",
                phase=CouncilPhase.EVALUATOR_COUNCIL,
                models=self.config.council_config.evaluator.get_model_names(),
                consensus_method="weighted_blend"
            )
            
            try:
                result = self.skeptic_council.evaluate(
                    objective=objective,
                    content=content,
                    prior_evaluations=prior_eval_str,
                    iteration=iteration
                )
                skeptic_output = result.output if result.success else None
                
                self.state_manager.complete_council_execution(
                    council_name="skeptic",
                    output=skeptic_output,
                    consensus_confidence=0.8
                )
                
                if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
                    verbose_logger.log_council_execution_complete("SkepticCouncil", success=result.success)
            except Exception as e:
                if self.config.verbose:
                    print(f"   ⚠️ SkepticCouncil error: {e}")
        
        # Log multi-view comparison if both available
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            if optimist_output and skeptic_output:
                if hasattr(verbose_logger, 'log_multi_view_comparison'):
                    verbose_logger.log_multi_view_comparison(optimist_output, skeptic_output)
        
        return optimist_output, skeptic_output
    
    def _run_arbiter(
        self,
        objective: str,
        council_outputs: Dict[str, Any]
    ) -> Any:
        """Run arbiter for final decision."""
        if self.config.verbose:
            print(f"\n⚖️ Running Arbiter...")
        
        # Verbose logging: arbiter
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_activated(
                "Arbiter",
                {"objective": objective[:200] if objective else None, "councils": list(council_outputs.keys())},
                models=[self.config.council_config.arbiter_model]
            )
        
        self.state_manager.start_council_execution(
            council_name="arbiter",
            phase=CouncilPhase.ARBITRATION,
            models=[self.config.council_config.arbiter_model],
            consensus_method="single_model"
        )
        
        # Convert council outputs to CouncilOutput objects
        outputs = []
        for name, output in council_outputs.items():
            outputs.append(CouncilOutput(
                council_name=name,
                output=output,
                confidence=0.8
            ))
        
        decision = self.arbiter.arbitrate(
            council_outputs=outputs,
            objective=objective
        )
        
        self.state_manager.complete_council_execution(
            council_name="arbiter",
            output=decision,
            consensus_confidence=decision.confidence
        )
        
        # Verbose logging: arbiter output
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_council_execution_complete("Arbiter", success=True)
            verbose_logger.log_context_output("Arbiter Decision", {
                "confidence": decision.confidence,
                "final_output": decision.final_output[:500] if hasattr(decision, 'final_output') and decision.final_output else None
            })
        
        return decision


def run_council_workflow(
    objective: str,
    context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 5,
    quality_threshold: float = 7.0,
    data_config: Optional[DataConfig] = None,
    simulation_config: Optional[SimulationConfig] = None,
    ollama_base_url: str = "http://localhost:11434",
    verbose: bool = False,
    time_budget_seconds: Optional[float] = None,
    force_all_councils: bool = True,
    enable_multiview: bool = True
) -> CouncilWorkflowResult:
    """
    Convenience function to run the DeepThinker 2.0 council workflow.
    
    Args:
        objective: Primary task objective
        context: Additional context
        max_iterations: Maximum refinement iterations
        quality_threshold: Minimum quality score to stop
        data_config: Dataset configuration
        simulation_config: Simulation configuration
        ollama_base_url: Ollama server URL
        verbose: Enable verbose output
        time_budget_seconds: Time budget for iteration (None = no limit)
        force_all_councils: Activate all councils (autonomous mode)
        enable_multiview: Enable optimist/skeptic councils
        
    Returns:
        CouncilWorkflowResult with final output
    """
    config = CouncilWorkflowConfig(
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        data_config=data_config,
        simulation_config=simulation_config,
        ollama_base_url=ollama_base_url,
        verbose=verbose,
        time_budget_seconds=time_budget_seconds,
        force_all_councils=force_all_councils,
        enable_multiview=enable_multiview
    )
    
    runner = CouncilWorkflowRunner(config)
    return runner.run(objective, context)

