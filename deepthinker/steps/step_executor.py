"""
Step Executor for DeepThinker 2.0 Step Engine.

Executes individual steps using single specialized models, with optional
evaluator reflection for quality control and retry logic.

This is the core execution engine that sits BELOW councils:
- Councils handle strategy and planning (which steps to do)
- StepExecutor handles the actual doing (executing each step)
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from .step_types import (
    StepDefinition,
    StepExecutionContext,
    StepResult,
    StepEvaluationResult,
    StepStatus,
)

# Sprint 1-2: Metrics Integration for Tool Tracking
try:
    from ..metrics import get_metrics_config, get_tool_tracker, ToolUsageRecord
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    get_metrics_config = None
    get_tool_tracker = None
    ToolUsageRecord = None

if TYPE_CHECKING:
    from ..models.model_pool import ModelPool
    from ..councils.evaluator_council.evaluator_council import EvaluatorCouncil
    from ..arbiter.arbiter import Arbiter


# Default model mapping by step type
# These are sensible defaults that can be overridden by step.preferred_model
DEFAULT_MODEL_BY_STEP_TYPE: Dict[str, str] = {
    "research": "gemma3:12b",
    "analysis": "cogito:14b",
    "design": "gemma3:27b",
    "coding": "deepseek-r1:8b",
    "testing": "mistral:instruct",
    "synthesis": "gemma3:27b",
    "meta": "gemma3:27b",
}

# Temperature mapping by step type
DEFAULT_TEMPERATURE_BY_STEP_TYPE: Dict[str, float] = {
    "research": 0.5,
    "analysis": 0.3,
    "design": 0.6,
    "coding": 0.2,
    "testing": 0.7,
    "synthesis": 0.4,
    "meta": 0.5,
}


class StepExecutor:
    """
    Executes a single step using a single model, with optional evaluator reflection.
    
    The StepExecutor is responsible for:
    1. Choosing the appropriate model for a step
    2. Building the execution prompt with full context
    3. Running the model and handling retries
    4. Optionally consulting EvaluatorCouncil for quality validation
    5. Returning a structured StepResult
    
    Unlike councils, which use multiple models and consensus, the StepExecutor
    uses a single model per step for efficiency. Quality control comes from
    optional evaluator reflection, not from multi-model consensus.
    """
    
    def __init__(
        self,
        model_pool: "ModelPool",
        evaluator_council: Optional["EvaluatorCouncil"] = None,
        arbiter: Optional["Arbiter"] = None,
        enable_reflection: bool = True,
        reflection_threshold: float = 6.0,
    ):
        """
        Initialize the StepExecutor.
        
        Args:
            model_pool: ModelPool instance for running models
            evaluator_council: Optional EvaluatorCouncil for step validation
            arbiter: Optional Arbiter for complex decisions (not typically used for steps)
            enable_reflection: Whether to use evaluator reflection for quality control
            reflection_threshold: Minimum quality score to pass without retry (0-10)
        """
        self.model_pool = model_pool
        self.evaluator = evaluator_council
        self.arbiter = arbiter
        self.enable_reflection = enable_reflection
        self.reflection_threshold = reflection_threshold
    
    def _check_and_consume_reanchor_prompt(
        self,
        mission_state: Optional[Any]
    ) -> Optional[str]:
        """
        Check for and consume a re-anchor prompt from alignment controller state.
        
        Alignment Control Layer (Gap 2): When drift is detected, the controller
        injects a micro re-anchor prompt that should be prepended to the system
        prompt for step execution.
        
        The prompt is single-use: consumed (cleared) after retrieval.
        
        Args:
            mission_state: Optional mission state with alignment_controller_state
            
        Returns:
            Re-anchor prompt string if present, None otherwise
        """
        if mission_state is None:
            return None
        
        # Check for alignment_controller_state with reanchor_prompt
        controller_state = getattr(mission_state, "alignment_controller_state", None)
        if not controller_state or not isinstance(controller_state, dict):
            return None
        
        reanchor_prompt = controller_state.get("reanchor_prompt")
        if not reanchor_prompt:
            return None
        
        # Consume the prompt (single-use)
        controller_state.pop("reanchor_prompt", None)
        controller_state.pop("reanchor_prompt_phase", None)
        
        return reanchor_prompt
    
    def _choose_model_for_step(self, step: StepDefinition) -> str:
        """
        Choose the best model to execute this step.
        
        Priority:
        1. If step.preferred_model is set and appears valid, use it
        2. Otherwise, use the default model for the step's type
        3. Fall back to gemma3:12b as a safe default
        
        Args:
            step: The step definition
            
        Returns:
            Model name string
        """
        # Priority 1: Explicit preferred model
        if step.preferred_model:
            return step.preferred_model
        
        # Priority 2: Default for step type
        step_type = step.step_type.lower()
        if step_type in DEFAULT_MODEL_BY_STEP_TYPE:
            return DEFAULT_MODEL_BY_STEP_TYPE[step_type]
        
        # Priority 3: Safe fallback
        return "gemma3:12b"
    
    def _get_temperature_for_step(self, step: StepDefinition) -> float:
        """
        Get the appropriate temperature for this step type.
        
        Args:
            step: The step definition
            
        Returns:
            Temperature value (0.0 to 1.0)
        """
        step_type = step.step_type.lower()
        return DEFAULT_TEMPERATURE_BY_STEP_TYPE.get(step_type, 0.5)
    
    def _build_system_prompt(self, step: StepDefinition) -> str:
        """
        Build a system prompt tailored to the step type.
        
        Args:
            step: The step definition
            
        Returns:
            System prompt string
        """
        step_type = step.step_type.lower()
        
        base_prompt = """You are an expert AI assistant executing a specific step in a larger mission.
Focus on completing the assigned task thoroughly and precisely.
Be concrete, actionable, and detailed in your output."""
        
        type_specific = {
            "research": """
You are a research specialist. Your task is to:
- Gather relevant information and context
- Identify key facts, sources, and insights
- Synthesize findings into actionable intelligence
- Be thorough but focused on what's relevant to the objective""",
            
            "analysis": """
You are an analytical specialist. Your task is to:
- Analyze data, situations, or information provided
- Identify patterns, trends, and key insights
- Draw logical conclusions supported by evidence
- Present analysis clearly with structured reasoning""",
            
            "design": """
You are a design and architecture specialist. Your task is to:
- Create well-structured designs and plans
- Consider constraints, requirements, and trade-offs
- Propose clear, implementable solutions
- Document design decisions and rationale""",
            
            "coding": """
You are a coding specialist. Your task is to:
- Write clean, correct, well-documented code
- Follow best practices and coding standards
- Handle edge cases and error conditions
- Keep code simple and maintainable
- Include helpful comments where appropriate""",
            
            "testing": """
You are a testing and validation specialist. Your task is to:
- Design comprehensive test scenarios
- Identify edge cases and potential failure modes
- Validate correctness and robustness
- Document test results and findings""",
            
            "synthesis": """
You are a synthesis specialist. Your task is to:
- Consolidate information from multiple sources
- Create coherent summaries and conclusions
- Produce clear, well-organized deliverables
- Ensure consistency and completeness""",
            
            "meta": """
You are a meta-level reasoning specialist. Your task is to:
- Reflect on processes and approaches
- Identify improvements and optimizations
- Make strategic recommendations
- Think about the bigger picture""",
        }
        
        specific = type_specific.get(step_type, "")
        
        return f"{base_prompt}\n{specific}"
    
    def _build_prompt(
        self,
        step: StepDefinition,
        ctx: StepExecutionContext,
    ) -> str:
        """
        Build the full execution prompt for a step.
        
        The prompt includes:
        - Mission context and objective
        - Current phase information
        - Step-specific instructions
        - Previous steps summary
        - Relevant shared artifacts
        - Tool availability information
        
        Args:
            step: The step definition
            ctx: Execution context with mission/phase information
            
        Returns:
            Complete prompt string
        """
        # Build tools section
        tools_section = ""
        if step.tools:
            tools_list = ", ".join(step.tools)
            tools_section = f"""
## AVAILABLE TOOLS
You have access to: {tools_list}

Use these tools as needed to complete the step.
"""
        
        # Build constraints section
        constraints_section = ""
        if ctx.constraints_notes:
            constraints_section = f"""
## CONSTRAINTS
{ctx.constraints_notes}
"""
        
        # Build previous steps section
        prev_section = ""
        if ctx.previous_steps_summary:
            prev_section = f"""
## PREVIOUS STEPS IN THIS PHASE
{ctx.previous_steps_summary}

Build upon the work done in previous steps.
"""
        
        # Build artifacts section
        artifacts_section = ""
        if ctx.shared_artifacts:
            artifacts_list = []
            for name, content in ctx.shared_artifacts.items():
                # Truncate long artifacts
                truncated = content[:1500] + "..." if len(content) > 1500 else content
                artifacts_list.append(f"### {name}\n{truncated}")
            if artifacts_list:
                artifacts_section = f"""
## RELEVANT ARTIFACTS FROM PRIOR WORK
{chr(10).join(artifacts_list)}
"""
        
        # Build time awareness section
        time_section = ""
        if ctx.remaining_time_minutes < 10:
            time_section = f"""
## TIME CONSTRAINT
Only {ctx.remaining_time_minutes:.1f} minutes remaining. Be efficient and prioritize key deliverables.
"""
        
        prompt = f"""# MISSION CONTEXT
**Mission Objective:** {ctx.mission_objective}

# CURRENT PHASE: {ctx.phase_name}
{ctx.phase_description}

# YOUR STEP: {step.name}
**Step Type:** {step.step_type}

## STEP DESCRIPTION
{step.description}
{tools_section}
{constraints_section}
{prev_section}
{artifacts_section}
{time_section}

## INSTRUCTIONS
Complete this step thoroughly. Provide your output in a clear, structured format.
If you produce code, wrap it in appropriate code blocks.
If you produce analysis or findings, organize them with clear headings.

## YOUR OUTPUT:
"""
        
        return prompt
    
    def _evaluate_step_output(
        self,
        step: StepDefinition,
        output: str,
        ctx: StepExecutionContext,
    ) -> StepEvaluationResult:
        """
        Evaluate the quality of a step's output using the EvaluatorCouncil.
        
        This is optional reflection that happens after each step to determine
        if the output is good enough or needs to be retried.
        
        Args:
            step: The step definition
            output: The output from the step execution
            ctx: Execution context
            
        Returns:
            StepEvaluationResult with pass/fail and quality score
        """
        if self.evaluator is None:
            # No evaluator available, accept the output
            return StepEvaluationResult(
                passed=True,
                quality_score=7.0,
                recommendations=["No evaluator configured - output accepted"]
            )
        
        try:
            # Build a focused evaluation context
            from ..councils.evaluator_council.evaluator_council import EvaluatorContext
            
            eval_objective = f"""Evaluate the output of step '{step.name}' (type: {step.step_type}).

Step was supposed to: {step.description}

The step is part of phase '{ctx.phase_name}' in mission: {ctx.mission_objective}

Evaluate whether the output adequately accomplishes the step's objective.
Score from 0-10 where 7+ is acceptable."""
            
            eval_context = EvaluatorContext(
                objective=eval_objective,
                code=output,  # Using code field to hold the step output
                quality_threshold=self.reflection_threshold,
            )
            
            result = self.evaluator.execute(eval_context)
            
            if result.success and result.output:
                evaluation = result.output
                passed = getattr(evaluation, 'passed', False)
                score = getattr(evaluation, 'quality_score', 5.0)
                issues = getattr(evaluation, 'issues', [])
                recs = getattr(evaluation, 'recommendations', [])
                
                # Format issues as strings if they're objects
                issue_strs = []
                for issue in issues:
                    if hasattr(issue, 'description'):
                        issue_strs.append(str(issue.description))
                    else:
                        issue_strs.append(str(issue))
                
                return StepEvaluationResult(
                    passed=passed or score >= self.reflection_threshold,
                    quality_score=score,
                    issues=issue_strs,
                    recommendations=recs,
                )
            else:
                # Evaluation failed, accept the output with lower confidence
                return StepEvaluationResult(
                    passed=True,
                    quality_score=5.0,
                    recommendations=["Evaluation failed - output accepted with lower confidence"]
                )
                
        except Exception as e:
            # On error, accept the output
            return StepEvaluationResult(
                passed=True,
                quality_score=5.0,
                recommendations=[f"Evaluation error: {str(e)} - output accepted"]
            )
    
    def execute_step(
        self,
        step: StepDefinition,
        ctx: StepExecutionContext,
        mission_state: Optional[Any] = None,
    ) -> StepResult:
        """
        Execute a single step with retry logic and optional evaluation.
        
        The execution loop:
        1. Choose model for the step
        2. Build prompt with full context
        3. Execute the model
        4. Optionally evaluate the output
        5. If evaluation fails and attempts remain, retry
        6. Return StepResult with final outcome
        
        Args:
            step: The step definition to execute
            ctx: Execution context with mission/phase information
            mission_state: Optional mission state for alignment prompt injection
            
        Returns:
            StepResult with status, output, and execution details
        """
        model_name = self._choose_model_for_step(step)
        temperature = self._get_temperature_for_step(step)
        
        # Check for alignment re-anchor prompt injection (Gap 2)
        reanchor_prompt = self._check_and_consume_reanchor_prompt(mission_state)
        
        attempt = 0
        started = datetime.utcnow()
        
        last_error: Optional[str] = None
        output: str = ""
        artifacts: Dict[str, str] = {}
        notes: list = []
        pivot_suggestion: Optional[str] = None
        
        step.mark_running()
        
        while attempt < step.max_attempts:
            attempt += 1
            notes.append(f"[Attempt {attempt}/{step.max_attempts}] Using model: {model_name}")
            
            # Build prompts
            system_prompt = self._build_system_prompt(step)
            
            # Alignment Control Layer (Gap 2): Prepend re-anchor prompt if present
            if reanchor_prompt and attempt == 1:
                system_prompt = f"{reanchor_prompt}\n\n{system_prompt}"
                notes.append("[Alignment] Re-anchor prompt injected")
            
            prompt = self._build_prompt(step, ctx)
            
            try:
                # Execute the model
                output = self.model_pool.run_single(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                
                if not output:
                    last_error = "Model returned empty output"
                    notes.append(f"[Attempt {attempt}] Empty output from model")
                    continue
                
                notes.append(f"[Attempt {attempt}] Got output ({len(output)} chars)")
                
                # Extract any code artifacts from output
                artifacts = self._extract_artifacts(output, step.step_type)
                
                # Evaluate if enabled
                if self.enable_reflection and self.evaluator is not None:
                    eval_result = self._evaluate_step_output(step, output, ctx)
                    notes.append(f"[Attempt {attempt}] Eval score: {eval_result.quality_score:.1f}")
                    
                    if not eval_result.passed:
                        last_error = f"Evaluator rejected output (score: {eval_result.quality_score:.1f})"
                        
                        if eval_result.issues:
                            notes.append(f"[Attempt {attempt}] Issues: {', '.join(eval_result.issues[:3])}")
                        
                        if eval_result.pivot_suggestion:
                            pivot_suggestion = eval_result.pivot_suggestion
                        
                        # Continue to next attempt if we haven't exhausted retries
                        if attempt < step.max_attempts:
                            continue
                    else:
                        # Passed evaluation
                        notes.append(f"[Attempt {attempt}] Evaluation passed")
                        last_error = None
                        break
                else:
                    # No evaluation, accept first non-empty output
                    last_error = None
                    break
                    
            except Exception as e:
                last_error = f"Execution error: {str(e)}"
                notes.append(f"[Attempt {attempt}] Error: {str(e)}")
                
                if attempt >= step.max_attempts:
                    break
        
        ended = datetime.utcnow()
        step_duration_ms = (ended - started).total_seconds() * 1000
        
        # Determine final status
        if last_error is None and output:
            status: StepStatus = "completed"
        else:
            status = "failed"
        
        result = StepResult(
            status=status,
            started_at=started,
            ended_at=ended,
            output=output,
            artifacts=artifacts,
            notes=notes,
            pivot_suggestion=pivot_suggestion,
            error=last_error,
            model_used=model_name,
            attempts=attempt,
        )
        
        # === Sprint 1-2: Tool Tracking at Step Boundary ===
        # TODO: @tracked_tool decorator for individual tools when instrumented
        if METRICS_AVAILABLE and get_tool_tracker is not None:
            try:
                metrics_config = get_metrics_config()
                if metrics_config and metrics_config.tool_track_sample_rate > 0:
                    tracker = get_tool_tracker(metrics_config)
                    
                    # Record tool usage for any tools specified in the step
                    if step.tools:
                        tracker.record_step(
                            step_id=step.name,
                            step_result=result,
                            tools_used=step.tools,
                            step_latency_ms=step_duration_ms,
                        )
            except Exception:
                pass  # Don't fail step execution on tracking error
        
        # Update step with result
        if status == "completed":
            step.mark_completed(result)
        else:
            step.mark_failed(result)
        
        return result
    
    def _extract_artifacts(self, output: str, step_type: str) -> Dict[str, str]:
        """
        Extract structured artifacts from step output.
        
        Looks for code blocks, JSON, and other structured content
        that should be saved as named artifacts.
        
        Args:
            output: The raw step output
            step_type: Type of step (affects what to look for)
            
        Returns:
            Dictionary of artifact name -> content
        """
        import re
        artifacts: Dict[str, str] = {}
        
        # Extract code blocks
        code_pattern = r'```(\w+)?\n(.*?)```'
        code_matches = re.findall(code_pattern, output, re.DOTALL)
        
        for i, (lang, code) in enumerate(code_matches):
            lang = lang.lower() if lang else "code"
            artifact_name = f"{step_type}_{lang}_{i}" if i > 0 else f"{step_type}_{lang}"
            artifacts[artifact_name] = code.strip()
        
        # For coding steps, also save the primary code artifact
        if step_type == "coding" and code_matches:
            # Save the first Python/code block as main code
            for lang, code in code_matches:
                if lang and lang.lower() in ("python", "py"):
                    artifacts["code"] = code.strip()
                    break
            else:
                # No Python found, use first code block
                artifacts["code"] = code_matches[0][1].strip()
        
        return artifacts
    
    def execute_step_simple(
        self,
        step: StepDefinition,
        ctx: StepExecutionContext,
    ) -> StepResult:
        """
        Execute a step without evaluation/retry - just run once.
        
        Useful for simple steps where reflection overhead isn't worth it.
        
        Args:
            step: The step definition
            ctx: Execution context
            
        Returns:
            StepResult from single execution attempt
        """
        # Temporarily disable reflection
        original_reflection = self.enable_reflection
        self.enable_reflection = False
        
        # Set max_attempts to 1
        original_attempts = step.max_attempts
        step.max_attempts = 1
        
        try:
            return self.execute_step(step, ctx)
        finally:
            self.enable_reflection = original_reflection
            step.max_attempts = original_attempts

