#!/usr/bin/env python3
"""
DeepThinker - Main CLI Entry Point

Command-line interface for the DeepThinker multi-agent autonomous AI system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import click

# Configure environment for Ollama/LiteLLM integration
os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

from deepthinker.execution import (
    run_deepthinker_workflow,
    IterationConfig,
    DataConfig,
    SimulationConfig,
    ResearchConfig,
    PlanningConfig
)
from deepthinker.execution.data_config import DockerConfig
from deepthinker.models import (
    enable_monitoring as start_llm_monitoring,
    print_monitoring_summary,
    AgentModelConfig
)
from deepthinker.cli import configure_verbose_logging, verbose_logger


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    DeepThinker - Long-running multi-agent autonomous AI system.
    
    Uses CrewAI and Ollama for complex, multi-phase AI tasks.
    """
    pass


@cli.command()
@click.argument("objective", type=str)
@click.option(
    "--context-file",
    type=click.Path(exists=True),
    help="JSON file with additional context for code generation"
)
@click.option(
    "--scenarios-file",
    type=click.Path(exists=True),
    help="JSON file with simulation scenarios"
)
@click.option(
    "--model",
    default="deepseek-r1:8b",
    help="Ollama model to use (default: deepseek-r1:8b)"
)
@click.option(
    "--model-all",
    type=str,
    help="Override all agent models with this model (overrides individual agent models)"
)
@click.option(
    "--planner-model",
    type=str,
    help="Model for Planner agent (default: cogito:14b)"
)
@click.option(
    "--websearch-model",
    type=str,
    help="Model for WebSearch agent (default: gemma3:12b)"
)
@click.option(
    "--coder-model",
    type=str,
    help="Model for Coder agent (default: deepseek-r1:8b)"
)
@click.option(
    "--evaluator-model",
    type=str,
    help="Model for Evaluator agent (default: gemma3:27b)"
)
@click.option(
    "--simulator-model",
    type=str,
    help="Model for Simulator agent (default: mistral:instruct)"
)
@click.option(
    "--executor-model",
    type=str,
    help="Model for Executor agent (default: llama3.2:3b)"
)
@click.option(
    "--max-iterations",
    type=int,
    default=3,
    help="Maximum number of code refinement iterations (default: 3)"
)
@click.option(
    "--quality-threshold",
    type=float,
    default=7.0,
    help="Minimum quality score (0-10) to stop iterations (default: 7.0)"
)
@click.option(
    "--no-iteration",
    is_flag=True,
    help="Disable iterative refinement (single pass only)"
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Path to dataset file (CSV/JSON) for metric-based evaluation"
)
@click.option(
    "--task-type",
    type=click.Choice(["classification", "regression"]),
    help="Type of ML task (required with --data-path)"
)
@click.option(
    "--target-column",
    type=str,
    help="Name of target column in dataset (default: last column)"
)
@click.option(
    "--test-split",
    type=float,
    default=0.2,
    help="Fraction of data for testing (default: 0.2)"
)
@click.option(
    "--metric-weight",
    type=float,
    default=0.5,
    help="Weight for metrics in combined score 0-1 (default: 0.5)"
)
@click.option(
    "--simulation-mode",
    type=click.Choice(["none", "basic", "scenarios"]),
    default="none",
    help="Simulation mode: none (disabled), basic (validation set), scenarios (config file)"
)
@click.option(
    "--simulation-config",
    type=click.Path(exists=True),
    help="Path to simulation config JSON (required for --simulation-mode=scenarios)"
)
@click.option(
    "--validation-split",
    type=float,
    default=0.1,
    help="Validation split ratio for basic simulation mode (default: 0.1)"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for results (JSON format)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--verbose-full",
    is_flag=True,
    help="Enable verbose output without truncation (full system prompts, outputs, code)"
)
@click.option(
    "--enable-monitoring",
    is_flag=True,
    default=True,
    help="Enable LiteLLM monitoring (default: True)"
)
@click.option(
    "--monitoring-verbose",
    is_flag=True,
    help="Enable verbose monitoring output"
)
@click.option(
    "--execution-backend",
    type=click.Choice(["subprocess", "docker"]),
    default="subprocess",
    help="Code execution backend: subprocess (default) or docker (secure)"
)
@click.option(
    "--docker-memory",
    type=str,
    default="512m",
    help="Docker memory limit (e.g., 512m, 1g) - only with --execution-backend=docker"
)
@click.option(
    "--docker-cpu",
    type=float,
    default=1.0,
    help="Docker CPU limit (e.g., 1.0 = 1 core) - only with --execution-backend=docker"
)
@click.option(
    "--enable-security-scan",
    is_flag=True,
    default=True,
    help="Enable pre-execution security scanning (default: True)"
)
@click.option(
    "--enable-research",
    is_flag=True,
    default=True,
    help="Enable web research phase before code generation (default: True)"
)
@click.option(
    "--no-research",
    is_flag=True,
    help="Disable web research phase"
)
@click.option(
    "--max-search-results",
    type=int,
    default=5,
    help="Maximum search results per query in research phase (default: 5)"
)
@click.option(
    "--search-timeout",
    type=int,
    default=10,
    help="Timeout in seconds for web search requests (default: 10)"
)
@click.option(
    "--enable-planning",
    is_flag=True,
    default=True,
    help="Enable planning phase before execution (default: True)"
)
@click.option(
    "--no-planning",
    is_flag=True,
    help="Disable planning phase"
)
@click.option(
    "--save-plan",
    type=click.Path(),
    help="Save workflow plan to JSON file"
)
def run(
    objective: str,
    context_file: Optional[str],
    scenarios_file: Optional[str],
    model: str,
    model_all: Optional[str],
    planner_model: Optional[str],
    websearch_model: Optional[str],
    coder_model: Optional[str],
    evaluator_model: Optional[str],
    simulator_model: Optional[str],
    executor_model: Optional[str],
    max_iterations: int,
    quality_threshold: float,
    no_iteration: bool,
    data_path: Optional[str],
    task_type: Optional[str],
    target_column: Optional[str],
    test_split: float,
    metric_weight: float,
    simulation_mode: str,
    simulation_config: Optional[str],
    validation_split: float,
    output: Optional[str],
    verbose: bool,
    verbose_full: bool,
    enable_monitoring: bool,
    monitoring_verbose: bool,
    execution_backend: str,
    docker_memory: str,
    docker_cpu: float,
    enable_security_scan: bool,
    enable_research: bool,
    no_research: bool,
    max_search_results: int,
    search_timeout: int,
    enable_planning: bool,
    no_planning: bool,
    save_plan: Optional[str]
):
    """
    Run the DeepThinker workflow with the given OBJECTIVE.
    
    Example:
        deepthinker run "Create a binary search tree class"
        
        deepthinker run "Build a decision tree classifier" \\
            --data-path data/iris.csv \\
            --task-type classification \\
            --max-iterations 5
    """
    try:
        # Configure verbose logging
        # --verbose-full implies --verbose
        effective_verbose = verbose or verbose_full
        configure_verbose_logging(enabled=effective_verbose, full_mode=verbose_full)
        
        # Initialize LiteLLM monitoring
        if enable_monitoring:
            start_llm_monitoring(
                log_dir="logs",
                verbose=monitoring_verbose or effective_verbose,
                enable_console_output=False,
                ollama_api_base=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
            )
        # Load context if provided
        context = None
        if context_file:
            with open(context_file, 'r') as f:
                context = json.load(f)
        
        # Load scenarios if provided
        scenarios = None
        if scenarios_file:
            with open(scenarios_file, 'r') as f:
                scenarios_data = json.load(f)
                scenarios = scenarios_data.get("scenarios", [])
        
        # Validate data config
        data_config = None
        if data_path:
            if not task_type:
                click.echo("❌ Error: --task-type is required when using --data-path", err=True)
                sys.exit(1)
            
            # Create Docker config if using Docker backend
            docker_config = None
            if execution_backend == "docker":
                docker_config = DockerConfig(
                    memory_limit=docker_memory,
                    cpu_limit=docker_cpu,
                    enable_security_scanning=enable_security_scan,
                    auto_build_image=True
                )
            
            data_config = DataConfig(
                data_path=data_path,
                task_type=task_type,
                target_column=target_column,
                test_split_ratio=test_split,
                metric_weight=metric_weight,
                execution_backend=execution_backend,
                docker_config=docker_config
            )
        
        # Create iteration config
        iteration_config = IterationConfig(
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            enabled=not no_iteration
        )
        
        # Create simulation config
        sim_config = None
        if simulation_mode == "none":
            sim_config = SimulationConfig.create_disabled()
        elif simulation_mode == "basic":
            if not data_path:
                click.echo("❌ Error: --data-path is required for simulation mode 'basic'", err=True)
                sys.exit(1)
            sim_config = SimulationConfig.create_basic_mode(validation_split=validation_split)
        elif simulation_mode == "scenarios":
            if not simulation_config:
                click.echo("❌ Error: --simulation-config is required for simulation mode 'scenarios'", err=True)
                sys.exit(1)
            if not data_path:
                click.echo("❌ Error: --data-path is required for simulation mode 'scenarios'", err=True)
                sys.exit(1)
            try:
                sim_config = SimulationConfig.from_json_file(simulation_config)
            except Exception as e:
                click.echo(f"❌ Error loading simulation config: {e}", err=True)
                sys.exit(1)
        
        # Create research config
        research_enabled = enable_research and not no_research
        research_cfg = ResearchConfig(
            enabled=research_enabled,
            max_results=max_search_results,
            timeout=search_timeout
        )
        
        # Create planning config
        planning_enabled = enable_planning and not no_planning
        planning_cfg = PlanningConfig(
            enabled=planning_enabled,
            save_plan=bool(save_plan),
            plan_output_path=save_plan
        )
        
        # Create agent model config
        agent_model_cfg = AgentModelConfig()
        
        # Apply individual model overrides if specified
        if planner_model:
            agent_model_cfg.planner_model = planner_model
        if websearch_model:
            agent_model_cfg.websearch_model = websearch_model
        if coder_model:
            agent_model_cfg.coder_model = coder_model
        if evaluator_model:
            agent_model_cfg.evaluator_model = evaluator_model
        if simulator_model:
            agent_model_cfg.simulator_model = simulator_model
        if executor_model:
            agent_model_cfg.executor_model = executor_model
        
        # Override all if --model-all specified
        if model_all:
            agent_model_cfg = agent_model_cfg.override_all(model_all)
        
        # Display configuration
        click.echo("🚀 DeepThinker Starting...")
        click.echo(f"   Objective: {objective}")
        click.echo(f"   Model (legacy): {model}")
        if effective_verbose or model_all or any([planner_model, websearch_model, coder_model, evaluator_model, simulator_model, executor_model]):
            click.echo(f"\n   Agent-Specific Models:")
            click.echo(f"     Planner: {agent_model_cfg.planner_model}")
            click.echo(f"     WebSearch: {agent_model_cfg.websearch_model}")
            click.echo(f"     Coder: {agent_model_cfg.coder_model}")
            click.echo(f"     Evaluator: {agent_model_cfg.evaluator_model}")
            click.echo(f"     Simulator: {agent_model_cfg.simulator_model}")
            click.echo(f"     Executor: {agent_model_cfg.executor_model}")
        if iteration_config.enabled:
            click.echo(f"   Max Iterations: {iteration_config.max_iterations}")
            click.echo(f"   Quality Threshold: {iteration_config.quality_threshold}/10")
        else:
            click.echo("   Iteration: Disabled (single pass)")
        
        if planning_cfg.enabled:
            click.echo("   Planning: Enabled")
            if planning_cfg.save_plan:
                click.echo(f"   Plan Output: {planning_cfg.plan_output_path}")
        else:
            click.echo("   Planning: Disabled")
        
        if research_cfg.enabled:
            click.echo(f"   Web Research: Enabled (max {research_cfg.max_results} results)")
        else:
            click.echo("   Web Research: Disabled")
        
        if data_config:
            click.echo(f"   Dataset: {data_config.data_path}")
            click.echo(f"   Task Type: {data_config.task_type}")
        
        if sim_config and sim_config.is_enabled():
            click.echo(f"   Simulation Mode: {sim_config.mode}")
            if sim_config.mode == "basic":
                click.echo(f"   Validation Split: {sim_config.validation_split}")
            elif sim_config.mode == "scenarios":
                click.echo(f"   Scenarios: {len(sim_config.scenarios)}")
        
        click.echo()
        
        # Run workflow
        result = run_deepthinker_workflow(
            objective=objective,
            context=context,
            scenarios=scenarios,
            model_name=model,
            iteration_config=iteration_config,
            data_config=data_config,
            simulation_config=sim_config,
            research_config=research_cfg,
            planning_config=planning_cfg,
            agent_model_config=agent_model_cfg,
            verbose=effective_verbose
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("✅ WORKFLOW COMPLETE")
        click.echo("="*60)
        
        # Show plan summary if available
        if result.get('workflow_plan'):
            click.echo("\n" + result['workflow_plan'].summary())
        
        click.echo(f"\nIterations: {result['iterations_completed']}")
        click.echo(f"Final Quality Score: {result['quality_score']:.1f}/10")
        
        if result['final_evaluation'].passed:
            click.echo("Status: ✅ PASSED")
        else:
            click.echo("Status: ⚠️  NEEDS IMPROVEMENT")
        
        # Show final code
        click.echo("\n" + "-"*60)
        click.echo("FINAL CODE")
        click.echo("-"*60)
        click.echo(result['final_code'])
        click.echo()
        
        # Show simulation results if available
        if result.get('simulation_summary'):
            click.echo("\n" + "="*60)
            click.echo("SIMULATION RESULTS")
            click.echo("="*60)
            click.echo(result['simulation_summary'].summary())
            click.echo()
        
        if result.get('simulation_report'):
            click.echo("\n" + "="*60)
            click.echo("SIMULATION ANALYSIS REPORT")
            click.echo("="*60)
            click.echo(result['simulation_report'])
            click.echo()
        
        # Save to output file if requested
        if output:
            output_path = Path(output)
            
            # Prepare result for JSON serialization
            json_result = {
                "objective": result["objective"],
                "final_code": result["final_code"],
                "iterations_completed": result["iterations_completed"],
                "quality_score": result["quality_score"],
                "final_evaluation": {
                    "quality_score": result["final_evaluation"].quality_score,
                    "passed": result["final_evaluation"].passed
                },
                "iteration_history": result["iteration_history"]
            }
            
            # Add workflow plan if available
            if result.get("workflow_plan"):
                plan = result["workflow_plan"]
                json_result["workflow_plan"] = {
                    "objective_analysis": plan.objective_analysis,
                    "workflow_strategy": plan.workflow_strategy,
                    "agent_requirements": plan.agent_requirements,
                    "success_criteria": plan.success_criteria,
                    "iteration_strategy": plan.iteration_strategy
                }
            
            # Add simulation results if available
            if result.get("simulation_summary"):
                sim_summary = result["simulation_summary"]
                json_result["simulation_summary"] = {
                    "total_scenarios": sim_summary.total_scenarios,
                    "successful_scenarios": sim_summary.successful_scenarios,
                    "overall_success": sim_summary.overall_success,
                    "scenarios": [
                        {
                            "name": sr.scenario_name,
                            "description": sr.scenario_description,
                            "success": sr.success,
                            "metrics": sr.metrics,
                            "num_samples": sr.num_samples,
                            "error_message": sr.error_message
                        }
                        for sr in sim_summary.scenario_results
                    ],
                    "cross_scenario_analysis": sim_summary.cross_scenario_analysis
                }
            
            if result.get("simulation_report"):
                json_result["simulation_report"] = result["simulation_report"]
            
            with open(output_path, 'w') as f:
                json.dump(json_result, f, indent=2, default=str)
            click.echo(f"💾 Results saved to: {output_path}")
        
        # Print monitoring summary if enabled
        if enable_monitoring:
            print_monitoring_summary()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--model",
    default="deepseek-r1:8b",
    help="Model to test (default: deepseek-r1:8b)"
)
def test_connection(model: str):
    """
    Test connection to Ollama server.
    """
    from deepthinker.models import OllamaLoader
    
    click.echo(f"Testing connection to Ollama with model: {model}")
    
    try:
        loader = OllamaLoader()
        click.echo(f"Connecting to: {loader.base_url}")
        
        if loader.validate_connection():
            click.echo("✅ Connection successful!")
        else:
            click.echo("❌ Connection failed - is Ollama running?")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_models():
    """
    List available Ollama models.
    """
    from deepthinker.models import OllamaLoader
    
    try:
        loader = OllamaLoader()
        models = loader.list_available_models()
        
        click.echo("Available Ollama models:")
        for model in models:
            click.echo(f"  - {model}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# MISSION COMMANDS
# =============================================================================

@cli.group()
def mission():
    """
    Mission management commands for long-running autonomous tasks.
    
    Missions are time-bounded, multi-phase operations that use all councils
    to accomplish complex objectives like deep analysis or end-to-end application builds.
    """
    pass


def _create_orchestrator():
    """Create a MissionOrchestrator with all councils initialized."""
    from deepthinker.missions import MissionOrchestrator, MissionStore
    from deepthinker.councils.planner_council.planner_council import PlannerCouncil
    from deepthinker.councils.researcher_council.researcher_council import ResearcherCouncil
    from deepthinker.councils.coder_council.coder_council import CoderCouncil
    from deepthinker.councils.evaluator_council.evaluator_council import EvaluatorCouncil
    from deepthinker.councils.simulation_council.simulation_council import SimulationCouncil
    from deepthinker.arbiter.arbiter import Arbiter
    
    ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    
    return MissionOrchestrator(
        planner_council=PlannerCouncil(ollama_base_url=ollama_url),
        researcher_council=ResearcherCouncil(ollama_base_url=ollama_url),
        coder_council=CoderCouncil(ollama_base_url=ollama_url),
        evaluator_council=EvaluatorCouncil(ollama_base_url=ollama_url),
        simulation_council=SimulationCouncil(ollama_base_url=ollama_url),
        arbiter=Arbiter(ollama_base_url=ollama_url),
        store=MissionStore()
    )


@mission.command("start")
@click.option(
    "--objective",
    "-o",
    required=True,
    help="The mission objective (what you want to accomplish)"
)
@click.option(
    "--time",
    "-t",
    "time_budget",
    type=int,
    required=True,
    help="Time budget in minutes"
)
@click.option(
    "--allow-internet/--no-internet",
    default=True,
    help="Allow internet/web research (default: True)"
)
@click.option(
    "--allow-code-exec/--no-code-exec",
    "allow_code_execution",
    default=True,
    help="Allow code execution (default: True)"
)
@click.option(
    "--max-iterations",
    type=int,
    default=100,
    help="Maximum iterations per phase (default: 100)"
)
@click.option(
    "--notes",
    type=str,
    help="Additional notes or constraints for the mission"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--verbose-full",
    is_flag=True,
    help="Enable verbose output without truncation (full system prompts, outputs, code)"
)
def mission_start(
    objective: str,
    time_budget: int,
    allow_internet: bool,
    allow_code_execution: bool,
    max_iterations: int,
    notes: Optional[str],
    verbose: bool,
    verbose_full: bool
):
    """
    Start a new long-running mission.
    
    Examples:
    
        deepthinker mission start -o "Analyze the situation in Ukraine" -t 60
        
        deepthinker mission start -o "Build a FastAPI + React app" -t 600 --allow-code-exec
        
        deepthinker mission start -o "Research best practices for X" -t 30 --no-code-exec
    """
    from deepthinker.missions.mission_types import build_constraints_from_time_budget, infer_effort_level
    
    try:
        # Configure verbose logging
        # --verbose-full implies --verbose
        effective_verbose = verbose or verbose_full
        configure_verbose_logging(enabled=effective_verbose, full_mode=verbose_full)
        
        # Infer effort level for display
        effort = infer_effort_level(time_budget)
        
        click.echo("🚀 Starting Mission Engine...")
        click.echo(f"   Objective: {objective}")
        click.echo(f"   Time Budget: {time_budget} minutes")
        click.echo(f"   Effort Level: {effort.value.upper()}")
        click.echo(f"   Internet: {'Enabled' if allow_internet else 'Disabled'}")
        click.echo(f"   Code Execution: {'Enabled' if allow_code_execution else 'Disabled'}")
        click.echo()
        
        # Create constraints using effort-based builder
        constraints = build_constraints_from_time_budget(
            time_budget_minutes=time_budget,
            allow_code=allow_code_execution,
            allow_internet=allow_internet,
            notes=notes,
            max_iterations=max_iterations,
        )
        
        # Create orchestrator and mission
        orchestrator = _create_orchestrator()
        state = orchestrator.create_mission(objective, constraints)
        
        click.echo(f"📋 Mission Created: {state.mission_id}")
        click.echo(f"   Phases: {len(state.phases)}")
        for i, phase in enumerate(state.phases, 1):
            click.echo(f"     {i}. {phase.name}")
        click.echo()
        
        # Run the mission
        click.echo("▶️  Running mission...")
        click.echo("-" * 60)
        
        def heartbeat(s):
            if verbose:
                phase = s.current_phase()
                if phase:
                    click.echo(f"   [{s.remaining_minutes():.1f}m remaining] Phase: {phase.name} ({phase.status})")
        
        final_state = orchestrator.run_until_complete_or_timeout(
            state.mission_id,
            heartbeat_callback=heartbeat if verbose else None
        )
        
        # Display results
        click.echo("-" * 60)
        click.echo()
        click.echo("=" * 60)
        click.echo(f"✅ MISSION {final_state.status.upper()}")
        click.echo("=" * 60)
        click.echo(final_state.summary())
        click.echo()
        
        # Verbose output: show detailed phase information
        if verbose:
            click.echo("=" * 60)
            click.echo("PHASE DETAILS")
            click.echo("=" * 60)
            
            # Phase keywords for council detection
            phase_council_map = {
                "research": ["recon", "context", "situation", "research", "gather", "sources", "analysis", "investigate"],
                "planner": ["design", "architecture", "plan", "strategy", "approach", "requirements", "synthesis", "report"],
                "coder": ["implementation", "coding", "build", "develop", "code", "implement", "create"],
                "evaluator": ["evaluation", "quality", "review", "assess"],
                "simulator": ["testing", "simulation", "validation", "test", "verify", "edge", "stress"]
            }
            
            for phase in final_state.phases:
                duration = phase.duration_seconds()
                duration_str = f"{duration:.1f}s" if duration else "N/A"
                
                # Determine which councils were likely used
                phase_lower = phase.name.lower()
                councils_used = []
                for council, keywords in phase_council_map.items():
                    if any(kw in phase_lower for kw in keywords):
                        councils_used.append(council)
                if not councils_used:
                    councils_used = ["research"]  # Default
                
                click.echo(f"\n📋 {phase.name}")
                click.echo(f"   Status: {phase.status} | Duration: {duration_str} | Iterations: {phase.iterations}")
                click.echo(f"   Councils: {', '.join(councils_used)}")
                
                if phase.artifacts:
                    artifact_keys = list(phase.artifacts.keys())
                    click.echo(f"   Artifacts: {', '.join(artifact_keys)}")
            
            click.echo()
        
        # Show final artifacts (internal dict)
        if final_state.final_artifacts:
            click.echo("📄 Consolidated Artifacts:")
            for key, value in final_state.final_artifacts.items():
                click.echo(f"\n--- {key} ---")
                # Truncate long outputs for display
                if len(value) > 2000:
                    click.echo(value[:2000] + "\n... [truncated]")
                else:
                    click.echo(value)
        
        # Show output deliverables (generated files)
        click.echo()
        if final_state.output_deliverables:
            click.echo("📦 Final Deliverables:")
            for artifact in final_state.output_deliverables:
                format_name = artifact.format.value.upper()
                click.echo(f"  - [{format_name:<20}] {artifact.path}")
        else:
            click.echo("📦 Final Deliverables: None generated")
        
        click.echo()
        click.echo(f"💾 Mission saved: {final_state.mission_id}")
        click.echo("   Use 'deepthinker mission status --id <mission_id>' to view details")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@mission.command("status")
@click.option(
    "--id",
    "mission_id",
    required=True,
    help="Mission ID to check status"
)
@click.option(
    "--show-logs",
    is_flag=True,
    help="Show mission execution logs"
)
@click.option(
    "--show-artifacts",
    is_flag=True,
    help="Show phase artifacts"
)
def mission_status(mission_id: str, show_logs: bool, show_artifacts: bool):
    """
    Check the status of an existing mission.
    """
    from deepthinker.missions import MissionStore
    
    try:
        store = MissionStore()
        
        if not store.exists(mission_id):
            click.echo(f"❌ Mission not found: {mission_id}", err=True)
            sys.exit(1)
        
        state = store.load(mission_id)
        
        click.echo("=" * 60)
        click.echo("MISSION STATUS")
        click.echo("=" * 60)
        click.echo(state.summary())
        
        if show_artifacts:
            click.echo()
            click.echo("=" * 60)
            click.echo("PHASE ARTIFACTS")
            click.echo("=" * 60)
            for phase in state.phases:
                if phase.artifacts:
                    click.echo(f"\n--- {phase.name} ---")
                    for key, value in phase.artifacts.items():
                        truncated = value[:500] + "..." if len(value) > 500 else value
                        click.echo(f"  {key}: {truncated}")
        
        if show_logs:
            click.echo()
            click.echo("=" * 60)
            click.echo("EXECUTION LOGS")
            click.echo("=" * 60)
            for log in state.logs[-50:]:  # Show last 50 logs
                click.echo(log)
        
        if state.final_artifacts:
            click.echo()
            click.echo("=" * 60)
            click.echo("CONSOLIDATED ARTIFACTS")
            click.echo("=" * 60)
            for key in state.final_artifacts:
                click.echo(f"  - {key}")
        
        # Show final deliverables (output files)
        click.echo()
        click.echo("=" * 60)
        click.echo("FINAL DELIVERABLES")
        click.echo("=" * 60)
        if state.output_deliverables:
            for artifact in state.output_deliverables:
                format_name = artifact.format.value.upper()
                desc = f" - {artifact.description}" if artifact.description else ""
                click.echo(f"  - {format_name:<20} {artifact.path}{desc}")
        else:
            click.echo("  No final deliverables yet.")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@mission.command("resume")
@click.option(
    "--id",
    "mission_id",
    required=True,
    help="Mission ID to resume"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def mission_resume(mission_id: str, verbose: bool):
    """
    Resume a previously started mission.
    
    Missions can be resumed if they haven't completed, failed, or expired.
    """
    try:
        click.echo(f"▶️  Resuming mission: {mission_id}")
        
        orchestrator = _create_orchestrator()
        
        def heartbeat(s):
            if verbose:
                phase = s.current_phase()
                if phase:
                    click.echo(f"   [{s.remaining_minutes():.1f}m remaining] Phase: {phase.name} ({phase.status})")
        
        final_state = orchestrator.resume_mission(mission_id)
        
        click.echo()
        click.echo("=" * 60)
        click.echo(f"✅ MISSION {final_state.status.upper()}")
        click.echo("=" * 60)
        click.echo(final_state.summary())
        
    except FileNotFoundError:
        click.echo(f"❌ Mission not found: {mission_id}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@mission.command("list")
@click.option(
    "--status",
    type=click.Choice(["all", "pending", "running", "completed", "failed", "expired", "aborted"]),
    default="all",
    help="Filter by status"
)
def mission_list(status: str):
    """
    List all missions in the store.
    """
    from deepthinker.missions import MissionStore
    
    try:
        store = MissionStore()
        missions = store.list_missions_with_status()
        
        if not missions:
            click.echo("No missions found.")
            return
        
        # Filter by status if specified
        if status != "all":
            missions = [m for m in missions if m["status"] == status]
        
        if not missions:
            click.echo(f"No missions with status '{status}' found.")
            return
        
        click.echo("=" * 80)
        click.echo("MISSIONS")
        click.echo("=" * 80)
        click.echo(f"{'ID':<38} {'Status':<10} {'Remaining':<10} Objective")
        click.echo("-" * 80)
        
        for m in missions:
            remaining = f"{m['remaining_minutes']:.0f}m" if m['remaining_minutes'] > 0 else "expired"
            click.echo(f"{m['mission_id']:<38} {m['status']:<10} {remaining:<10} {m['objective']}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@mission.command("abort")
@click.option(
    "--id",
    "mission_id",
    required=True,
    help="Mission ID to abort"
)
@click.option(
    "--reason",
    default="User requested abort",
    help="Reason for aborting the mission"
)
def mission_abort(mission_id: str, reason: str):
    """
    Abort a running mission.
    """
    try:
        orchestrator = _create_orchestrator()
        state = orchestrator.abort_mission(mission_id, reason)
        
        click.echo(f"🛑 Mission aborted: {mission_id}")
        click.echo(f"   Reason: {reason}")
        click.echo(f"   Final Status: {state.status}")
        
    except FileNotFoundError:
        click.echo(f"❌ Mission not found: {mission_id}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# CONTEXT INSPECTION COMMANDS
# =============================================================================

@cli.group()
def context():
    """
    Context inspection commands for debugging and understanding workflows.
    
    Inspect council configurations, context schemas, mission artifacts,
    and execution history.
    """
    pass


@context.command("councils")
@click.option(
    "--council",
    "-c",
    type=click.Choice(["planner", "researcher", "coder", "evaluator", "simulator", "all"]),
    default="all",
    help="Which council to inspect (default: all)"
)
@click.option(
    "--show-prompt",
    is_flag=True,
    help="Show full system prompts"
)
def context_councils(council: str, show_prompt: bool):
    """
    Inspect council configurations and context schemas.
    
    Shows:
    - Council system prompts
    - Expected input/output schemas
    - Context dataclass structures
    """
    from deepthinker.cli import configure_verbose_logging
    
    # Enable verbose logging for inspection
    configure_verbose_logging(enabled=True, full_mode=show_prompt)
    
    ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    
    councils_to_inspect = []
    
    if council == "all" or council == "planner":
        from deepthinker.councils.planner_council.planner_council import PlannerCouncil
        councils_to_inspect.append(("PlannerCouncil", PlannerCouncil(ollama_base_url=ollama_url)))
    
    if council == "all" or council == "researcher":
        from deepthinker.councils.researcher_council.researcher_council import ResearcherCouncil
        councils_to_inspect.append(("ResearcherCouncil", ResearcherCouncil(ollama_base_url=ollama_url)))
    
    if council == "all" or council == "coder":
        from deepthinker.councils.coder_council.coder_council import CoderCouncil
        councils_to_inspect.append(("CoderCouncil", CoderCouncil(ollama_base_url=ollama_url)))
    
    if council == "all" or council == "evaluator":
        from deepthinker.councils.evaluator_council.evaluator_council import EvaluatorCouncil
        councils_to_inspect.append(("EvaluatorCouncil", EvaluatorCouncil(ollama_base_url=ollama_url)))
    
    if council == "all" or council == "simulator":
        from deepthinker.councils.simulation_council.simulation_council import SimulationCouncil
        councils_to_inspect.append(("SimulationCouncil", SimulationCouncil(ollama_base_url=ollama_url)))
    
    click.echo("=" * 60)
    click.echo("📋 Council Configuration Inspector")
    click.echo("=" * 60)
    
    for name, council_instance in councils_to_inspect:
        verbose_logger.log_council_requirements(council_instance.__class__)
        click.echo()


@context.command("mission")
@click.option(
    "--id",
    "mission_id",
    required=True,
    help="Mission ID to inspect"
)
@click.option(
    "--phases",
    is_flag=True,
    help="Show phase artifacts in detail"
)
@click.option(
    "--iterations",
    is_flag=True,
    help="Show iteration history"
)
def context_mission(mission_id: str, phases: bool, iterations: bool):
    """
    Inspect a mission's context and artifacts.
    
    Shows:
    - Phase artifacts and relationships
    - Context accumulation between phases
    - Iteration history with quality scores
    """
    from deepthinker.missions import MissionStore
    from deepthinker.cli import configure_verbose_logging
    
    # Enable verbose logging
    configure_verbose_logging(enabled=True, full_mode=False)
    
    store = MissionStore()
    
    if not store.exists(mission_id):
        click.echo(f"❌ Mission not found: {mission_id}", err=True)
        sys.exit(1)
    
    state = store.load(mission_id)
    
    click.echo("=" * 60)
    click.echo(f"📋 Mission Context Inspector: {mission_id[:8]}...")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"Objective: {state.objective}")
    click.echo(f"Status: {state.status}")
    click.echo(f"Phases: {len(state.phases)}")
    click.echo()
    
    if phases:
        verbose_logger.log_mission_artifact_summary(state.phases)
    
    if iterations:
        # Show iteration-like info from phase iterations
        click.echo()
        click.echo("=" * 60)
        click.echo("📊 Phase Iteration Summary")
        click.echo("=" * 60)
        for phase in state.phases:
            duration = phase.duration_seconds() or 0
            click.echo(f"  {phase.name}: {phase.iterations} iterations, {duration:.1f}s")
    
    # Show context flow
    click.echo()
    click.echo("=" * 60)
    click.echo("➡️  Context Flow")
    click.echo("=" * 60)
    
    for i, phase in enumerate(state.phases):
        if i > 0:
            prev_phase = state.phases[i-1]
            prev_artifacts = list(prev_phase.artifacts.keys()) if prev_phase.artifacts else []
            click.echo(f"  {prev_phase.name} → {phase.name}")
            if prev_artifacts:
                click.echo(f"    Fields passed: {', '.join(prev_artifacts)}")


@context.command("state")
def context_state():
    """
    Show current workflow state from state managers.
    
    Displays:
    - Current workflow status
    - Active council/phase
    - Token usage metrics
    """
    from deepthinker.workflow.state_manager import council_state_manager
    from deepthinker.execution.agent_state_manager import agent_state_manager
    from deepthinker.cli import configure_verbose_logging
    
    # Enable verbose logging
    configure_verbose_logging(enabled=True, full_mode=False)
    
    click.echo("=" * 60)
    click.echo("📸 State Manager Snapshot")
    click.echo("=" * 60)
    
    # Council state
    council_workflow = council_state_manager.get_current_workflow()
    if council_workflow:
        click.echo()
        click.echo("Council Workflow State:")
        click.echo(f"  Workflow ID: {council_workflow.get('workflow_id', 'N/A')}")
        click.echo(f"  Status: {council_workflow.get('status', 'N/A')}")
        click.echo(f"  Current Phase: {council_workflow.get('current_phase', 'N/A')}")
        click.echo(f"  Current Council: {council_workflow.get('current_council', 'N/A')}")
        click.echo(f"  Iteration: {council_workflow.get('current_iteration', 0)}/{council_workflow.get('max_iterations', 0)}")
    else:
        click.echo()
        click.echo("No active council workflow.")
    
    # Agent state
    agent_workflow = agent_state_manager.get_current_workflow()
    if agent_workflow:
        click.echo()
        click.echo("Agent Workflow State:")
        click.echo(f"  Workflow ID: {agent_workflow.get('workflow_id', 'N/A')}")
        click.echo(f"  Status: {agent_workflow.get('status', 'N/A')}")
        click.echo(f"  Current Agent: {agent_workflow.get('current_agent', 'N/A')}")
        click.echo(f"  Iteration: {agent_workflow.get('current_iteration', 0)}")
    else:
        click.echo()
        click.echo("No active agent workflow.")
    
    # Show metrics summary
    metrics = agent_state_manager.get_agent_metrics_summary()
    if any(m['calls'] > 0 for m in metrics.values()):
        click.echo()
        click.echo("Agent Metrics Summary:")
        for agent, stats in metrics.items():
            if stats['calls'] > 0:
                click.echo(f"  {agent}: {stats['calls']} calls, {stats['total_tokens']} tokens, ${stats['total_cost']:.4f}")


if __name__ == "__main__":
    cli()
