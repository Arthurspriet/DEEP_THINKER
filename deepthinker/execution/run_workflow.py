"""
Workflow orchestration for the DeepThinker multi-agent system.

Enhanced with optional governance support:
- Enable governance checks after each iteration
- Uses SafetyCore registry for module availability
- Graceful degradation if governance modules unavailable
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import json
import logging

from crewai import Crew, Process, Task

from ..agents import create_coder_agent, create_evaluator_agent, create_simulator_agent, create_executor_agent, create_websearch_agent, create_planner_agent
from ..tasks import create_code_task, create_evaluate_task, create_revise_task, create_simulate_task, create_execute_task, create_research_task, create_planning_task
from ..tools import WebSearchTool
from ..evaluation import EvaluationResultParser, CombinedEvaluationResult, MetricResult
from ..models import OllamaLoader, LiteLLMMonitor, AgentModelConfig
from .data_config import DataConfig
from .code_executor import CodeExecutor
from .docker_executor import DockerExecutor
from .metric_computer import MetricComputer
from .simulation_config import SimulationConfig
from .simulation_runner import SimulationRunner
from .agent_state_manager import agent_state_manager, AgentPhase
from .plan_config import WorkflowPlan, WorkflowPlanParser

logger = logging.getLogger(__name__)

# =============================================================================
# SafetyCore Integration for Optional Governance
# =============================================================================

# Try to import SafetyCore for governance access
try:
    from ..core.safety_registry import safety, SAFETY_CORE_AVAILABLE
    SAFETY_CORE_AVAILABLE = True
except ImportError:
    SAFETY_CORE_AVAILABLE = False
    safety = None

# Direct imports for governance (fallback if SafetyCore unavailable)
try:
    from ..governance import NormativeController, NormativeVerdict, VerdictStatus
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    NormativeController = None
    NormativeVerdict = None
    VerdictStatus = None


def _get_governance_controller():
    """
    Get governance controller using SafetyCore or direct import.
    
    Returns:
        NormativeController instance or None if unavailable
    """
    # Try SafetyCore first
    if SAFETY_CORE_AVAILABLE and safety is not None:
        if safety.is_available("governance"):
            try:
                module = safety.get("governance")
                if module and hasattr(module, 'NormativeController'):
                    return module.NormativeController()
            except Exception as e:
                logger.debug(f"Failed to get governance via SafetyCore: {e}")
    
    # Fall back to direct import
    if GOVERNANCE_AVAILABLE and NormativeController is not None:
        try:
            return NormativeController()
        except Exception as e:
            logger.debug(f"Failed to create NormativeController: {e}")
    
    return None


@dataclass
class IterationConfig:
    """
    Configuration for iterative code refinement.
    
    Attributes:
        max_iterations: Maximum number of refinement cycles
        quality_threshold: Minimum quality score (0-10) to stop iterating
        enabled: Whether iteration is enabled
    """
    
    max_iterations: int = 3
    quality_threshold: float = 7.0
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0 <= self.quality_threshold <= 10:
            raise ValueError("quality_threshold must be between 0 and 10")


@dataclass
class ResearchConfig:
    """
    Configuration for web research phase.
    
    Attributes:
        enabled: Whether to run web research before code generation
        max_results: Maximum search results per query
        timeout: Timeout in seconds for search requests
    """
    
    enabled: bool = True
    max_results: int = 5
    timeout: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_results < 1:
            raise ValueError("max_results must be at least 1")
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1")


@dataclass
class PlanningConfig:
    """
    Configuration for planning phase.
    
    Attributes:
        enabled: Whether to run planning before execution
        save_plan: Whether to save plan to file
        plan_output_path: Path to save plan JSON
    """
    
    enabled: bool = True
    save_plan: bool = False
    plan_output_path: Optional[str] = None


@dataclass
class GovernanceConfig:
    """
    Configuration for governance integration in workflow.
    
    Enables opt-in governance checks from the mission path.
    
    Attributes:
        enabled: Whether to run governance checks
        check_after_each_iteration: Run governance check after each iteration
        fail_on_block: Raise exception if governance blocks execution
        warn_threshold: Minimum score to log warning
        block_threshold: Minimum score to block execution
    """
    
    enabled: bool = False  # Off by default for backward compatibility
    check_after_each_iteration: bool = True
    fail_on_block: bool = False
    warn_threshold: float = 0.3
    block_threshold: float = 0.7


class WorkflowRunner:
    """
    Orchestrates the DeepThinker multi-agent workflow.
    
    Manages agent initialization, task execution, and iterative refinement.
    
    Enhanced with optional governance support:
    - Enable via governance_config parameter
    - Applies NormativeController checks after each iteration
    - Logs violations and can optionally block on severe issues
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        iteration_config: Optional[IterationConfig] = None,
        data_config: Optional[DataConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
        research_config: Optional[ResearchConfig] = None,
        planning_config: Optional[PlanningConfig] = None,
        agent_model_config: Optional[AgentModelConfig] = None,
        governance_config: Optional[GovernanceConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize workflow runner.
        
        Args:
            model_name: Ollama model to use (legacy, used if agent_model_config not provided)
            iteration_config: Configuration for iterative refinement
            data_config: Configuration for dataset-based evaluation
            simulation_config: Configuration for post-training simulation
            research_config: Configuration for web research phase
            planning_config: Configuration for planning phase
            agent_model_config: Configuration for agent-specific models
            governance_config: Configuration for governance integration (opt-in)
            verbose: Whether to print detailed progress
        """
        self.model_name = model_name
        self.iteration_config = iteration_config or IterationConfig()
        self.data_config = data_config
        self.simulation_config = simulation_config or SimulationConfig.create_disabled()
        self.research_config = research_config or ResearchConfig()
        self.planning_config = planning_config or PlanningConfig()
        self.governance_config = governance_config or GovernanceConfig()
        self.verbose = verbose
        
        # Initialize model loader with agent-specific configuration
        self.model_loader = OllamaLoader(agent_model_config=agent_model_config)
        
        # Will be initialized on first run
        self.agents = None
        self.iteration_history: List[Dict[str, Any]] = []
        self.workflow_id: Optional[str] = None
        self.research_findings: Optional[str] = None
        self.workflow_plan: Optional[WorkflowPlan] = None
        
        # Governance integration
        self._governance_controller = None
        self._governance_violations: List[Dict[str, Any]] = []
        if self.governance_config.enabled:
            self._governance_controller = _get_governance_controller()
            if self._governance_controller is None:
                if self.verbose:
                    print("⚠️  Governance enabled but controller unavailable")
                logger.warning("Governance enabled but NormativeController unavailable")
            else:
                if self.verbose:
                    print("✓ Governance enabled")
                logger.info("Governance integration enabled")
    
    def run(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete workflow.
        
        Args:
            objective: Primary task objective
            context: Additional context for code generation
            scenarios: Simulation scenarios to test
            
        Returns:
            Dictionary containing final code, evaluation, and iteration history
        """
        self._initialize_agents()
        
        # Start workflow tracking
        self.workflow_id = agent_state_manager.start_workflow(
            objective=objective,
            model_name=self.model_name,
            max_iterations=self.iteration_config.max_iterations
        )
        
        if self.verbose:
            print(f"🚀 Starting DeepThinker workflow: {objective}")
            print(f"   Workflow ID: {self.workflow_id}")
            print(f"   Model: {self.model_name}")
            print(f"   Max iterations: {self.iteration_config.max_iterations}")
            print(f"   Quality threshold: {self.iteration_config.quality_threshold}")
        
        try:
            # Run planning phase if enabled
            if self.planning_config.enabled:
                self.workflow_plan = self._run_planning_phase(objective, context)
                
                if self.verbose:
                    print("\n" + self.workflow_plan.summary())
                
                # Save plan if requested
                if self.planning_config.save_plan and self.planning_config.plan_output_path:
                    self._save_plan_to_file(self.workflow_plan, self.planning_config.plan_output_path)
            
            # Run iterative code generation and evaluation
            code_result = self._run_iterative_code_evaluation_phase(objective, context)
            
            # Run simulation phase if enabled
            simulation_summary = None
            simulation_report = None
            
            if self.simulation_config.is_enabled():
                simulation_summary, simulation_report = self._run_simulation_phase_with_execution(
                    code_result["final_code"]
                )
            elif scenarios:
                # Legacy support: if scenarios provided but no simulation_config
                simulation_report = self._run_simulation_phase(
                    code_result["final_code"],
                    scenarios
                )
            
            # Complete workflow tracking
            agent_state_manager.complete_workflow()
            
            return {
                "workflow_id": self.workflow_id,
                "objective": objective,
                "workflow_plan": self.workflow_plan,
                "final_code": code_result["final_code"],
                "final_evaluation": code_result["final_evaluation"],
                "iteration_history": self.iteration_history,
                "simulation_summary": simulation_summary,
                "simulation_report": simulation_report,
                "iterations_completed": code_result["iterations_completed"],
                "quality_score": code_result["quality_score"]
            }
        except Exception as e:
            # Mark workflow as failed
            agent_state_manager.complete_workflow(error=str(e))
            raise
    
    def _initialize_agents(self):
        """Initialize all agents with their optimized LLM instances."""
        if self.agents is not None:
            return
        
        # Create agent-specific LLM instances using optimized models
        planner_llm = self.model_loader.create_planner_llm()
        coder_llm = self.model_loader.create_coder_llm()
        evaluator_llm = self.model_loader.create_evaluator_llm()
        simulator_llm = self.model_loader.create_simulator_llm()
        websearch_llm = self.model_loader.create_websearch_llm()
        executor_llm = self.model_loader.create_executor_llm()
        
        # Create websearch tool
        websearch_tool = WebSearchTool(
            max_results=self.research_config.max_results,
            timeout=self.research_config.timeout
        )
        
        # Create agents
        self.agents = {
            "planner": create_planner_agent(planner_llm),
            "coder": create_coder_agent(coder_llm),
            "evaluator": create_evaluator_agent(evaluator_llm),
            "simulator": create_simulator_agent(simulator_llm),
            "executor": create_executor_agent(executor_llm),
            "researcher": create_websearch_agent(websearch_llm, tools=[websearch_tool])
        }
        
        if self.verbose:
            print(f"🤖 Initialized agents with optimized models:")
            print(f"   Planner: {self.model_loader.agent_config.planner_model}")
            print(f"   WebSearch: {self.model_loader.agent_config.websearch_model}")
            print(f"   Coder: {self.model_loader.agent_config.coder_model}")
            print(f"   Evaluator: {self.model_loader.agent_config.evaluator_model}")
            print(f"   Simulator: {self.model_loader.agent_config.simulator_model}")
            print(f"   Executor: {self.model_loader.agent_config.executor_model}")
    
    def _run_iterative_code_evaluation_phase(
        self,
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run iterative code generation and evaluation with refinement loop.
        
        Args:
            objective: Code generation objective
            context: Additional context
            
        Returns:
            Dictionary with final code, evaluation, and iteration count
        """
        current_code = None
        current_evaluation = None
        previous_metrics = None
        score_history = []
        
        # Run research phase before first iteration if enabled (and if plan recommends it)
        should_research = self.research_config.enabled
        if self.workflow_plan:
            should_research = should_research and self.workflow_plan.is_agent_enabled("research")
        
        if should_research:
            self.research_findings = self._run_research_phase(objective, context)
            # Add research findings to context
            if context is None:
                context = {}
            context["research_findings"] = self.research_findings
        
        for iteration in range(self.iteration_config.max_iterations):
            if self.verbose:
                print(f"\n📝 Iteration {iteration + 1}/{self.iteration_config.max_iterations}")
            
            # Start iteration tracking
            agent_state_manager.start_iteration(iteration + 1)
            
            # Generate or revise code
            if iteration == 0:
                current_code = self._run_code_generation_phase(objective, context, iteration + 1)
            else:
                current_code = self._run_revision_phase(
                    objective,
                    current_code,
                    current_evaluation,
                    previous_metrics,
                    iteration
                )
            
            # Execute on dataset if configured
            metric_result = None
            if self.data_config and self.data_config.is_enabled():
                metric_result = self._execute_code_on_dataset(current_code)
                previous_metrics = metric_result
            
            # Evaluate code
            current_evaluation = self._run_evaluation_phase(
                objective,
                current_code,
                metric_result,
                iteration + 1
            )
            
            # Track history
            iteration_data = {
                "iteration": iteration + 1,
                "code": current_code,
                "evaluation": {
                    "quality_score": current_evaluation.quality_score,
                    "passed": current_evaluation.passed,
                    "issues": [{"severity": i.severity, "description": i.description} 
                              for i in current_evaluation.issues],
                    "recommendations": current_evaluation.recommendations,
                },
                "metrics": metric_result.metrics if metric_result else None
            }
            self.iteration_history.append(iteration_data)
            
            # Update state manager with iteration results
            agent_state_manager.update_iteration_results(
                code=current_code,
                quality_score=current_evaluation.quality_score,
                passed=current_evaluation.passed,
                issues=[{"severity": i.severity, "description": i.description} 
                       for i in current_evaluation.issues],
                recommendations=current_evaluation.recommendations,
                metrics=metric_result.metrics if metric_result else None
            )
            
            score_history.append(current_evaluation.quality_score)
            
            if self.verbose:
                print(f"   Quality: {current_evaluation.quality_score}/10")
                print(f"   Status: {'✅ PASSED' if current_evaluation.passed else '❌ NEEDS WORK'}")
            
            # Run governance check if enabled
            if self.governance_config.enabled and self.governance_config.check_after_each_iteration:
                governance_result = self._run_governance_check(
                    iteration=iteration + 1,
                    code=current_code,
                    evaluation=current_evaluation,
                    objective=objective
                )
                if governance_result and governance_result.get("blocked", False):
                    if self.governance_config.fail_on_block:
                        raise RuntimeError(
                            f"Governance blocked iteration {iteration + 1}: "
                            f"{governance_result.get('reason', 'Unknown')}"
                        )
                    if self.verbose:
                        print(f"   ⚠️  Governance: {governance_result.get('status', 'WARN')}")
            
            # Check termination conditions
            if not self.iteration_config.enabled:
                break
            
            if current_evaluation.quality_score >= self.iteration_config.quality_threshold:
                if self.verbose:
                    print(f"✨ Quality threshold reached!")
                break
            
            # Anti-loop: stop if no significant improvement
            if len(score_history) >= 3:
                recent_improvement = score_history[-1] - score_history[-3]
                if recent_improvement < 0.1:
                    if self.verbose:
                        print("⚠️  No significant improvement, stopping iterations")
                    break
        
        return {
            "final_code": current_code,
            "final_evaluation": current_evaluation,
            "iterations_completed": len(self.iteration_history),
            "quality_score": current_evaluation.quality_score,
            "governance_violations": self._governance_violations if self._governance_violations else None
        }
    
    def _run_governance_check(
        self,
        iteration: int,
        code: str,
        evaluation: Any,
        objective: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run governance check on iteration output.
        
        Uses NormativeController from mission path to validate outputs.
        
        Args:
            iteration: Current iteration number
            code: Generated code
            evaluation: Evaluation result
            objective: Task objective
            
        Returns:
            Dictionary with governance result or None if unavailable
        """
        if self._governance_controller is None:
            return None
        
        try:
            # Build phase output for governance evaluation
            phase_output = {
                "code": code[:5000] if code else "",  # Limit for evaluation
                "objective": objective,
                "iteration": iteration,
                "quality_score": getattr(evaluation, 'quality_score', 0),
                "issues": [
                    {"severity": i.severity, "description": i.description}
                    for i in getattr(evaluation, 'issues', [])
                ],
            }
            
            # Evaluate with governance controller
            verdict = self._governance_controller.evaluate(
                phase_name=f"code_iteration_{iteration}",
                phase_output=phase_output,
                mission_state=None  # No mission state in workflow path
            )
            
            # Process verdict
            result = {
                "iteration": iteration,
                "status": verdict.status.value if hasattr(verdict.status, 'value') else str(verdict.status),
                "violations": len(verdict.violations) if hasattr(verdict, 'violations') else 0,
                "blocked": False,
                "reason": None
            }
            
            # Check for block status
            if hasattr(verdict, 'status'):
                status_str = str(verdict.status).upper()
                if "BLOCK" in status_str:
                    result["blocked"] = True
                    result["reason"] = "Governance policy violation"
            
            # Log violations
            if hasattr(verdict, 'violations') and verdict.violations:
                for violation in verdict.violations[:3]:  # Log first 3
                    violation_info = {
                        "iteration": iteration,
                        "type": getattr(violation, 'rule_id', 'unknown'),
                        "severity": getattr(violation, 'severity', 0)
                    }
                    self._governance_violations.append(violation_info)
                    logger.debug(f"Governance violation: {violation_info}")
            
            if self.verbose and result["violations"] > 0:
                print(f"   📋 Governance: {result['violations']} violation(s)")
            
            return result
            
        except Exception as e:
            logger.warning(f"Governance check failed: {e}")
            return {"error": str(e), "blocked": False}
    
    def _run_code_generation_phase(
        self,
        objective: str,
        context: Optional[Dict[str, Any]],
        iteration: int
    ) -> str:
        """Generate initial code."""
        # Set agent context for monitoring
        logger = LiteLLMMonitor.get_logger()
        if logger:
            logger.set_agent_context("coder", iteration)
        
        # Add planner requirements to context if available
        enhanced_context = context.copy() if context else {}
        if self.workflow_plan:
            coder_requirements = self.workflow_plan.get_agent_requirements("coder")
            if coder_requirements:
                enhanced_context["planner_requirements"] = coder_requirements
        
        # Track agent execution start
        task_description = f"Generate code for: {objective[:100]}"
        agent_state_manager.start_agent_execution(
            agent_name="coder",
            phase=AgentPhase.CODE_GENERATION,
            input_text=task_description,
            metadata={"context": enhanced_context, "iteration": iteration}
        )
        
        task = create_code_task(
            self.agents["coder"],
            objective,
            enhanced_context,
            self.data_config
        )
        
        crew = Crew(
            agents=[self.agents["coder"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        code = str(result)
        
        # Track agent execution completion
        agent_state_manager.complete_agent_execution(
            agent_name="coder",
            output_text=code,
            llm_metrics=self._get_recent_llm_metrics()
        )
        
        return code
    
    def _run_research_phase(
        self,
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Run web research phase to gather information before code generation.
        
        Args:
            objective: Code generation objective
            context: Additional context
            
        Returns:
            Research findings as a string
        """
        if self.verbose:
            print(f"\n🔍 Running Research Phase...")
        
        # Add planner requirements to context if available
        enhanced_context = context.copy() if context else {}
        if self.workflow_plan:
            researcher_requirements = self.workflow_plan.get_agent_requirements("researcher")
            if researcher_requirements:
                enhanced_context["planner_requirements"] = researcher_requirements
        
        # Set agent context for monitoring
        logger = LiteLLMMonitor.get_logger()
        if logger:
            logger.set_agent_context("researcher", 0)
        
        # Track agent execution start
        agent_state_manager.start_agent_execution(
            agent_name="researcher",
            phase=AgentPhase.RESEARCH,
            input_text=f"Research for: {objective[:100]}",
            metadata={"context": enhanced_context}
        )
        
        task = create_research_task(
            self.agents["researcher"],
            objective,
            enhanced_context
        )
        
        crew = Crew(
            agents=[self.agents["researcher"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        research_findings = str(result)
        
        # Track agent execution completion
        agent_state_manager.complete_agent_execution(
            agent_name="researcher",
            output_text=research_findings,
            llm_metrics=self._get_recent_llm_metrics()
        )
        
        if self.verbose:
            print(f"✅ Research Phase Complete")
        
        return research_findings
    
    def _run_planning_phase(
        self,
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> WorkflowPlan:
        """
        Run planning phase to create execution strategy.
        
        Args:
            objective: Primary objective
            context: Additional context
            
        Returns:
            Parsed WorkflowPlan object
        """
        if self.verbose:
            print(f"\n📋 Running Planning Phase...")
        
        # Set agent context for monitoring
        logger = LiteLLMMonitor.get_logger()
        if logger:
            logger.set_agent_context("planner", 0)
        
        # Track agent execution start
        agent_state_manager.start_agent_execution(
            agent_name="planner",
            phase=AgentPhase.PLANNING,
            input_text=f"Plan for: {objective[:100]}",
            metadata={"context": context}
        )
        
        task = create_planning_task(
            self.agents["planner"],
            objective,
            context,
            self.data_config,
            self.simulation_config,
            self.iteration_config
        )
        
        crew = Crew(
            agents=[self.agents["planner"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        plan_text = str(result)
        
        # Parse the plan
        workflow_plan = WorkflowPlanParser.parse(plan_text)
        
        # Track agent execution completion
        agent_state_manager.complete_agent_execution(
            agent_name="planner",
            output_text=plan_text,
            llm_metrics=self._get_recent_llm_metrics()
        )
        
        if self.verbose:
            print(f"✅ Planning Phase Complete")
        
        return workflow_plan
    
    def _save_plan_to_file(self, plan: WorkflowPlan, output_path: str) -> None:
        """Save workflow plan to JSON file."""
        import json
        from pathlib import Path
        
        plan_data = {
            "objective_analysis": plan.objective_analysis,
            "workflow_strategy": plan.workflow_strategy,
            "agent_requirements": plan.agent_requirements,
            "success_criteria": plan.success_criteria,
            "iteration_strategy": plan.iteration_strategy,
            "raw_plan": plan.raw_plan
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        if self.verbose:
            print(f"💾 Plan saved to: {output_path}")
    
    def _run_revision_phase(
        self,
        objective: str,
        previous_code: str,
        evaluation: Any,
        metric_result: Optional[MetricResult],
        iteration: int
    ) -> str:
        """Revise code based on evaluation feedback."""
        # Set agent context for monitoring
        logger = LiteLLMMonitor.get_logger()
        if logger:
            logger.set_agent_context("coder", iteration + 1)
        
        # Track agent execution start
        feedback_preview = f"Score: {evaluation.quality_score}, Issues: {len(evaluation.issues)}"
        agent_state_manager.start_agent_execution(
            agent_name="coder",
            phase=AgentPhase.CODE_REVISION,
            input_text=f"Revise code based on feedback: {feedback_preview}",
            metadata={"iteration": iteration + 1, "previous_score": evaluation.quality_score}
        )
        
        task = create_revise_task(
            self.agents["coder"],
            objective,
            previous_code,
            evaluation,
            iteration,
            metric_result,
            self.iteration_history[-2].get("metrics") if len(self.iteration_history) >= 2 else None
        )
        
        crew = Crew(
            agents=[self.agents["coder"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        code = str(result)
        
        # Track agent execution completion
        agent_state_manager.complete_agent_execution(
            agent_name="coder",
            output_text=code,
            llm_metrics=self._get_recent_llm_metrics()
        )
        
        return code
    
    def _run_evaluation_phase(
        self,
        objective: str,
        code: str,
        metric_result: Optional[MetricResult],
        iteration: int
    ) -> Any:
        """Evaluate generated code."""
        # Set agent context for monitoring
        logger = LiteLLMMonitor.get_logger()
        if logger:
            logger.set_agent_context("evaluator", iteration)
        
        # Track agent execution start
        agent_state_manager.start_agent_execution(
            agent_name="evaluator",
            phase=AgentPhase.EVALUATION,
            input_text=f"Evaluate code for: {objective[:100]}",
            metadata={"iteration": iteration, "code_length": len(code)}
        )
        
        task = create_evaluate_task(
            self.agents["evaluator"],
            objective,
            code,
            metric_result
        )
        
        crew = Crew(
            agents=[self.agents["evaluator"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        eval_text = str(result)
        
        # Parse evaluation result
        parser = EvaluationResultParser(quality_threshold=self.iteration_config.quality_threshold)
        evaluation = parser.parse(eval_text)
        
        # Track agent execution completion
        agent_state_manager.complete_agent_execution(
            agent_name="evaluator",
            output_text=eval_text,
            llm_metrics=self._get_recent_llm_metrics()
        )
        
        # Create combined result if metrics available
        if metric_result and self.data_config:
            combined_eval = CombinedEvaluationResult(
                quality_score=evaluation.quality_score,
                passed=evaluation.passed,
                issues=evaluation.issues,
                recommendations=evaluation.recommendations,
                strengths=evaluation.strengths,
                raw_output=evaluation.raw_output,
                metrics=metric_result,
                metric_weight=self.data_config.metric_weight
            )
            combined_eval.compute_combined_score()
            return combined_eval
        
        return evaluation
    
    def _execute_code_on_dataset(
        self,
        code: str,
        profile: Optional[Any] = None
    ) -> Optional[MetricResult]:
        """Execute code on configured dataset."""
        if not self.data_config or not self.data_config.is_enabled():
            return None
        
        try:
            # Choose executor based on configuration
            if self.data_config.execution_backend == "docker":
                if not DockerExecutor.is_available():
                    if self.verbose:
                        print("⚠️  Docker not available, falling back to subprocess executor")
                    executor = CodeExecutor
                    exec_result, y_test, y_pred = executor.execute_model_on_data(
                        code, self.data_config
                    )
                else:
                    # Initialize Docker executor with profile support
                    docker_config = self.data_config.docker_config
                    
                    # Use execution profile if specified, otherwise use legacy config
                    from .profile_registry import get_default_registry
                    from .execution_profile import ExecutionProfile
                    
                    if docker_config and docker_config.execution_profile:
                        # Use profile-based executor
                        registry = get_default_registry()
                        exec_profile = registry.get_profile(docker_config.execution_profile)
                        executor_instance = DockerExecutor(
                            profile=exec_profile,
                            auto_build_image=docker_config.auto_build_image
                        )
                    else:
                        # Legacy mode: construct from old parameters
                        executor_instance = DockerExecutor(
                            image_name=docker_config.image_name if docker_config else "deepthinker-sandbox:latest",
                            memory_limit=docker_config.memory_limit if docker_config else "512m",
                            cpu_limit=docker_config.cpu_limit if docker_config else 1.0,
                            timeout=self.data_config.execution_timeout,
                            enable_security_scanning=docker_config.enable_security_scanning if docker_config else True,
                            auto_build_image=docker_config.auto_build_image if docker_config else True
                        )
                    
                    if self.verbose:
                        profile_name = executor_instance.profile.name if hasattr(executor_instance, 'profile') else "legacy"
                        print(f"🔒 Using Docker executor (profile: {profile_name})")
                    
                    # Use profile if available, otherwise None (uses instance profile)
                    exec_profile_param = executor_instance.profile if hasattr(executor_instance, 'profile') else None
                    exec_result, y_test, y_pred = executor_instance.execute_model_on_data(
                        code, self.data_config, profile=exec_profile_param
                    )
                    executor_instance.cleanup()
            else:
                # Use subprocess executor (default)
                exec_result, y_test, y_pred = CodeExecutor.execute_model_on_data(
                    code, self.data_config
                )
            
            if not exec_result.success:
                return MetricResult(
                    task_type=self.data_config.task_type,
                    execution_result=exec_result,
                    num_samples=0
                )
            
            # Compute metrics
            metrics = MetricComputer.compute_metrics(
                y_test,
                y_pred,
                self.data_config.task_type
            )
            
            return MetricResult(
                task_type=self.data_config.task_type,
                metrics=metrics,
                execution_result=exec_result,
                num_samples=len(y_test)
            )
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Metric computation failed: {e}")
            return None
    
    def _get_recent_llm_metrics(self) -> Dict[str, Any]:
        """Get metrics from the most recent LLM call."""
        logger = LiteLLMMonitor.get_logger()
        if not logger:
            return {}
        
        # Return basic metrics from logger
        # Note: This is a simplified version; in production you'd track per-call metrics
        return {
            "total_tokens": 0,  # Would be populated by actual call
            "latency_seconds": 0,
            "cost_usd": 0
        }
    
    def _run_simulation_phase_with_execution(
        self,
        code: str
    ) -> tuple:
        """
        Run full simulation phase with execution and LLM analysis.
        
        Args:
            code: Final model code to simulate
            
        Returns:
            Tuple of (SimulationSummary, simulation_report_str)
        """
        if self.verbose:
            print(f"\n🔬 Starting Simulation Phase...")
        
        # Create simulation runner
        runner = SimulationRunner(
            base_data_config=self.data_config,
            execution_timeout=self.data_config.execution_timeout if self.data_config else 30,
            verbose=self.verbose
        )
        
        # Execute all scenarios
        simulation_summary = runner.run_scenarios(code, self.simulation_config)
        
        if self.verbose:
            print(f"\n📊 Simulation execution complete. Generating analysis report...")
        
        # Generate LLM-powered analysis report
        task = create_simulate_task(
            self.agents["simulator"],
            code,
            simulation_summary
        )
        
        crew = Crew(
            agents=[self.agents["simulator"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        simulation_report = str(result)
        
        if self.verbose:
            print(f"\n✅ Simulation phase complete!")
        
        return simulation_summary, simulation_report
    
    def _run_simulation_phase(
        self,
        code: str,
        scenarios: List[str]
    ) -> str:
        """Run scenario-based simulation (legacy support)."""
        # This method is kept for backward compatibility
        # Convert old-style scenario list to simple task
        task = Task(
            description=f"""
Test the following code against scenarios: {', '.join(scenarios)}

Code:
```python
{code}
```

Provide analysis of how the code would behave in each scenario.
""",
            expected_output="Scenario analysis report",
            agent=self.agents["simulator"]
        )
        
        crew = Crew(
            agents=[self.agents["simulator"]],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        return str(result)


def run_deepthinker_workflow(
    objective: str,
    context: Optional[Dict[str, Any]] = None,
    scenarios: Optional[List[str]] = None,
    model_name: str = "deepseek-r1:8b",
    iteration_config: Optional[IterationConfig] = None,
    data_config: Optional[DataConfig] = None,
    simulation_config: Optional[SimulationConfig] = None,
    research_config: Optional[ResearchConfig] = None,
    planning_config: Optional[PlanningConfig] = None,
    agent_model_config: Optional[AgentModelConfig] = None,
    governance_config: Optional[GovernanceConfig] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run the DeepThinker workflow.
    
    Args:
        objective: Primary task objective
        context: Additional context for code generation
        scenarios: Simulation scenarios to test (legacy)
        model_name: Ollama model to use (legacy, used if agent_model_config not provided)
        iteration_config: Configuration for iterative refinement
        data_config: Configuration for dataset-based evaluation
        simulation_config: Configuration for post-training simulation
        research_config: Configuration for web research phase
        planning_config: Configuration for planning phase
        agent_model_config: Configuration for agent-specific models
        governance_config: Configuration for governance integration (opt-in)
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary containing workflow results
    """
    runner = WorkflowRunner(
        model_name=model_name,
        iteration_config=iteration_config,
        data_config=data_config,
        simulation_config=simulation_config,
        research_config=research_config,
        planning_config=planning_config,
        agent_model_config=agent_model_config,
        governance_config=governance_config,
        verbose=verbose
    )
    
    return runner.run(objective, context, scenarios)

