"""
Meta-Planner Agent for DeepThinker 2.0.

The highest-level strategist that orchestrates councils and determines
the optimal workflow configuration for each objective.

Updated for autonomous multi-council execution:
- Forces activation of ALL councils by default
- Includes multi-view councils (optimist, skeptic)
- Respects PlannerCouncil requirements as mandatory
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    USE_CHAT_OLLAMA = True
except ImportError:
    from langchain_community.llms import Ollama
    USE_CHAT_OLLAMA = False

from ..models.council_model_config import META_PLANNER_MODEL


# All available councils including multi-view
ALL_COUNCILS = [
    "planner", "researcher", "coder", "evaluator", 
    "simulation", "optimist", "skeptic"
]

# Core councils that are always activated
MANDATORY_COUNCILS = [
    "planner", "researcher", "evaluator", "optimist", "skeptic"
]


@dataclass
class MetaPlanDecision:
    """
    Decision from the meta-planner.
    
    Attributes:
        councils_to_activate: Ordered list of councils to execute
        iteration_count: Recommended number of iterations
        exploration_level: Creativity/exploration setting (0-1)
        parallel_councils: Councils that can run in parallel
        skip_reasons: Reasons for skipping any councils
        strategy_notes: Additional strategic notes
        raw_output: Raw LLM output
        planner_required_councils: Councils required by PlannerCouncil
        force_all_councils: Whether all councils were forced on
    """
    
    councils_to_activate: List[str] = field(default_factory=list)
    iteration_count: int = 5  # Increased default for autonomous mode
    exploration_level: float = 0.5
    parallel_councils: List[List[str]] = field(default_factory=list)
    skip_reasons: Dict[str, str] = field(default_factory=dict)
    strategy_notes: str = ""
    raw_output: str = ""
    planner_required_councils: List[str] = field(default_factory=list)
    force_all_councils: bool = True
    
    @classmethod
    def from_text(
        cls, 
        text: str,
        force_all_councils: bool = True,
        planner_requirements: Optional[List[str]] = None
    ) -> "MetaPlanDecision":
        """
        Parse meta-plan decision from text output.
        
        Args:
            text: Raw LLM output
            force_all_councils: If True, activate all councils regardless of LLM suggestion
            planner_requirements: Councils required by PlannerCouncil (always included)
        """
        councils = []
        skip_reasons = {}
        parallel = []
        iteration_count = 5  # Default for autonomous mode
        exploration_level = 0.5
        strategy_notes = ""
        
        lines = text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect sections
            if 'council' in line_lower and 'activate' in line_lower:
                current_section = 'councils'
            elif 'skip' in line_lower or 'disable' in line_lower:
                current_section = 'skip'
            elif 'parallel' in line_lower:
                current_section = 'parallel'
            elif 'iteration' in line_lower:
                current_section = 'iteration'
                # Try to extract number
                import re
                nums = re.findall(r'\d+', line)
                if nums:
                    iteration_count = max(3, int(nums[0]))  # Minimum 3 iterations
            elif 'exploration' in line_lower or 'creativity' in line_lower:
                current_section = 'exploration'
                import re
                nums = re.findall(r'0\.\d+|\d+', line)
                if nums:
                    val = float(nums[0])
                    exploration_level = val if val <= 1 else val / 10
            elif 'strategy' in line_lower or 'note' in line_lower:
                current_section = 'strategy'
            
            # Extract councils mentioned
            if current_section == 'councils':
                for council in ALL_COUNCILS:
                    if council in line_lower and council not in councils:
                        councils.append(council)
            
            # Extract skip reasons (but may be overridden)
            if current_section == 'skip':
                for council in ALL_COUNCILS:
                    if council in line_lower:
                        reason = line.split(':')[-1].strip() if ':' in line else "Skipped by meta-planner"
                        skip_reasons[council] = reason
            
            # Collect strategy notes
            if current_section == 'strategy':
                if line.strip() and not line.strip().endswith(':'):
                    strategy_notes += line.strip() + " "
        
        # === FORCE ALL COUNCILS MODE ===
        if force_all_councils:
            # Activate all councils, clear skip reasons
            councils = list(ALL_COUNCILS)
            skip_reasons = {}
        else:
            # Even without forcing, always include mandatory councils
            for council in MANDATORY_COUNCILS:
                if council not in councils:
                    councils.append(council)
        
        # === ENFORCE PLANNER REQUIREMENTS ===
        planner_requirements = planner_requirements or []
        for required_council in planner_requirements:
            if required_council not in councils:
                councils.append(required_council)
            # Remove from skip reasons if present
            skip_reasons.pop(required_council, None)
        
        # Ensure proper council ordering
        council_order = ["planner", "researcher", "coder", "evaluator", 
                         "simulation", "optimist", "skeptic"]
        councils = sorted(councils, key=lambda c: council_order.index(c) if c in council_order else 99)
        
        return cls(
            councils_to_activate=councils,
            iteration_count=iteration_count,
            exploration_level=exploration_level,
            parallel_councils=parallel,
            skip_reasons=skip_reasons,
            strategy_notes=strategy_notes.strip(),
            raw_output=text,
            planner_required_councils=planner_requirements,
            force_all_councils=force_all_councils
        )
    
    @classmethod
    def create_full_activation(
        cls,
        iteration_count: int = 5,
        exploration_level: float = 0.5,
        strategy_notes: str = "Full autonomous mode - all councils activated"
    ) -> "MetaPlanDecision":
        """
        Create a decision with all councils activated.
        
        Convenience method for autonomous execution mode.
        """
        return cls(
            councils_to_activate=list(ALL_COUNCILS),
            iteration_count=iteration_count,
            exploration_level=exploration_level,
            parallel_councils=[["optimist", "skeptic"]],  # Multi-view runs in parallel
            skip_reasons={},
            strategy_notes=strategy_notes,
            raw_output="",
            force_all_councils=True
        )


class MetaPlanner:
    """
    The highest-level strategist in DeepThinker 2.0.
    
    Determines:
    - Which councils should be activated
    - Optimal iteration count for the task
    - Exploration/creativity levels
    - Parallel execution opportunities
    
    Updated for autonomous mode:
    - Forces all councils by default
    - Includes multi-view councils (optimist, skeptic)
    - Respects PlannerCouncil requirements as mandatory
    """
    
    def __init__(
        self,
        model_name: str = META_PLANNER_MODEL,
        temperature: float = 0.5,
        ollama_base_url: str = "http://localhost:11434",
        force_all_councils: bool = True
    ):
        """
        Initialize meta-planner.
        
        Args:
            model_name: LLM model to use (defaults to largest available)
            temperature: Sampling temperature
            ollama_base_url: Ollama server URL
            force_all_councils: If True, always activate all councils (autonomous mode)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        self.force_all_councils = force_all_councils
        self._llm = None
        self._system_prompt = self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for meta-planner."""
        return """You are the Meta-Planner, the highest-level strategist in DeepThinker 2.0.

Your role is to analyze objectives and determine the optimal workflow configuration.
You make strategic decisions about:
1. Which councils to activate (planner, researcher, coder, evaluator, simulation, optimist, skeptic)
2. How many iterations of refinement to allow (minimum 3, recommend 5+ for complex tasks)
3. The exploration/creativity level for each task
4. Which councils can run in parallel

Available Councils:
- planner: Strategic planning and task decomposition
- researcher: Information gathering and synthesis
- coder: Code generation with cross-review (also reasoning agent when code execution disabled)
- evaluator: Quality assessment and feedback
- simulation: Edge-case and stress testing
- optimist: Positive interpretation and opportunity identification
- skeptic: Critical analysis and risk identification

For AUTONOMOUS MODE (full reasoning):
- Activate ALL councils for comprehensive analysis
- Use at least 5 iterations for complex tasks
- Multi-view councils (optimist + skeptic) should run in parallel
- Never skip councils for important missions

Your decisions directly impact the depth and quality of autonomous reasoning."""
    
    def _get_llm(self) -> Any:
        """Get or create the LLM instance."""
        if self._llm is None:
            if USE_CHAT_OLLAMA:
                # ChatOllama expects just the model name (e.g., "gemma3:27b")
                self._llm = ChatOllama(
                    model=self.model_name,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature
                )
            else:
                # Legacy Ollama class may need prefix
                prefixed_model = f"ollama/{self.model_name}" if not self.model_name.startswith("ollama") else self.model_name
                self._llm = Ollama(
                    model=prefixed_model,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature
                )
        
        return self._llm
    
    def plan(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        planner_requirements: Optional[List[str]] = None
    ) -> MetaPlanDecision:
        """
        Generate a meta-plan for the given objective.
        
        Args:
            objective: Primary objective
            context: Additional context
            constraints: Optional constraints (time, resources, etc.)
            planner_requirements: Councils required by PlannerCouncil (mandatory)
            
        Returns:
            MetaPlanDecision with workflow configuration
        """
        # In force_all_councils mode, skip LLM call for efficiency
        if self.force_all_councils:
            return self._create_autonomous_plan(objective, constraints, planner_requirements)
        
        # Build prompt
        context_str = ""
        if context:
            context_str = f"\n\nAdditional Context:\n{context}"
        
        constraints_str = ""
        if constraints:
            constraints_str = f"\n\nConstraints:\n{constraints}"
        
        prompt = f"""Analyze the following objective and determine the optimal workflow configuration:

## OBJECTIVE
{objective}
{context_str}
{constraints_str}

## DECISION REQUIRED

### 1. COUNCILS TO ACTIVATE
List which councils should be activated in order:
- planner: Yes/No (reason)
- researcher: Yes/No (reason)
- coder: Yes/No (reason)
- evaluator: Yes/No (reason)
- simulation: Yes/No (reason)
- optimist: Yes/No (reason)
- skeptic: Yes/No (reason)

### 2. ITERATION COUNT
How many refinement iterations? (3-10)
Consider task complexity and quality requirements.
For complex tasks, recommend 5+ iterations.

### 3. EXPLORATION LEVEL
What exploration/creativity level? (0.0-1.0)
- 0.0-0.3: Deterministic, precise
- 0.4-0.6: Balanced
- 0.7-1.0: Creative, exploratory

### 4. PARALLEL OPPORTUNITIES
Which councils can run in parallel?
(optimist and skeptic can always run in parallel)

### 5. STRATEGY NOTES
Any additional strategic considerations.

Provide your analysis and decisions:"""

        try:
            llm = self._get_llm()
            
            if USE_CHAT_OLLAMA:
                messages = [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                full_prompt = f"{self._system_prompt}\n\n{prompt}"
                response = llm.invoke(full_prompt)
                output = str(response)
            
            return MetaPlanDecision.from_text(
                output, 
                force_all_councils=self.force_all_councils,
                planner_requirements=planner_requirements
            )
            
        except Exception as e:
            # Return full activation on error
            return MetaPlanDecision.create_full_activation(
                strategy_notes=f"Full activation fallback (error: {str(e)})"
            )
    
    def _create_autonomous_plan(
        self,
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
        planner_requirements: Optional[List[str]] = None
    ) -> MetaPlanDecision:
        """
        Create a plan with all councils activated for autonomous mode.
        
        This is more efficient than calling the LLM when we know we want
        full council activation.
        """
        # Determine iteration count based on time budget
        iteration_count = 5
        exploration_level = 0.5
        
        if constraints:
            time_budget = constraints.get('time_budget_minutes', 5)
            if time_budget >= 10:
                iteration_count = 8
            elif time_budget >= 5:
                iteration_count = 6
            else:
                iteration_count = 4
            
            # Adjust exploration based on task type
            if constraints.get('allow_code_execution', True):
                exploration_level = 0.4  # More deterministic for code
            else:
                exploration_level = 0.6  # More exploratory for analysis
        
        # All councils in proper order
        councils = list(ALL_COUNCILS)
        
        # Ensure planner requirements are included
        if planner_requirements:
            for req in planner_requirements:
                if req not in councils:
                    councils.append(req)
        
        return MetaPlanDecision(
            councils_to_activate=councils,
            iteration_count=iteration_count,
            exploration_level=exploration_level,
            parallel_councils=[["optimist", "skeptic"]],
            skip_reasons={},
            strategy_notes=f"Autonomous mode: all councils activated for objective: {objective[:100]}...",
            raw_output="",
            planner_required_councils=planner_requirements or [],
            force_all_councils=True
        )
    
    def extract_planner_requirements(self, workflow_plan: Any) -> List[str]:
        """
        Extract required councils from a WorkflowPlan.
        
        Args:
            workflow_plan: WorkflowPlan from PlannerCouncil
            
        Returns:
            List of council names that are required
        """
        required = []
        
        if workflow_plan is None:
            return required
        
        # Try to get agent_requirements
        if hasattr(workflow_plan, 'agent_requirements'):
            requirements = workflow_plan.agent_requirements
            if isinstance(requirements, dict):
                # Check which agents have requirements set
                for agent_name, agent_reqs in requirements.items():
                    if agent_reqs:  # Has non-empty requirements
                        # Map agent names to council names
                        council_map = {
                            "researcher": "researcher",
                            "coder": "coder",
                            "evaluator": "evaluator",
                            "simulator": "simulation",
                            "simulation": "simulation",
                            "executor": "coder",
                        }
                        council = council_map.get(agent_name.lower(), agent_name.lower())
                        if council not in required:
                            required.append(council)
        
        # Also check workflow_strategy
        if hasattr(workflow_plan, 'workflow_strategy'):
            strategy = workflow_plan.workflow_strategy
            if isinstance(strategy, dict):
                for agent, enabled in strategy.items():
                    if enabled:
                        council_map = {
                            "researcher": "researcher",
                            "coder": "coder",
                            "evaluator": "evaluator",
                            "simulator": "simulation",
                        }
                        council = council_map.get(agent.lower(), agent.lower())
                        if council not in required:
                            required.append(council)
        
        return required
    
    def adjust_for_iteration(
        self,
        current_decision: MetaPlanDecision,
        iteration: int,
        quality_score: float,
        issues_count: int,
        time_remaining_seconds: Optional[float] = None
    ) -> MetaPlanDecision:
        """
        Adjust meta-plan based on iteration progress.
        
        In autonomous mode, adjustments are made algorithmically rather than
        via LLM calls for efficiency.
        
        Args:
            current_decision: Current meta-plan decision
            iteration: Current iteration number
            quality_score: Latest quality score
            issues_count: Number of issues found
            time_remaining_seconds: Optional remaining time budget
            
        Returns:
            Adjusted MetaPlanDecision
        """
        # In force_all_councils mode, use rule-based adjustment
        if self.force_all_councils:
            return self._adjust_autonomous(
                current_decision, iteration, quality_score, 
                issues_count, time_remaining_seconds
            )
        
        prompt = f"""Based on iteration progress, should we adjust the workflow?

Current Iteration: {iteration}
Quality Score: {quality_score}/10
Issues Found: {issues_count}
Remaining Iterations: {current_decision.iteration_count - iteration}
Time Remaining: {time_remaining_seconds:.0f}s if time_remaining_seconds else 'Unknown'

Current Configuration:
- Active Councils: {', '.join(current_decision.councils_to_activate)}
- Exploration Level: {current_decision.exploration_level}

Should we:
1. Continue as planned?
2. Stop early (quality sufficient)?
3. Add more iterations?
4. Adjust exploration level?

Provide brief recommendation:"""

        try:
            llm = self._get_llm()
            
            if USE_CHAT_OLLAMA:
                messages = [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                full_prompt = f"{self._system_prompt}\n\n{prompt}"
                response = llm.invoke(full_prompt)
                output = str(response)
            
            # Parse adjustment
            output_lower = output.lower()
            
            adjusted = MetaPlanDecision(
                councils_to_activate=current_decision.councils_to_activate.copy(),
                iteration_count=current_decision.iteration_count,
                exploration_level=current_decision.exploration_level,
                parallel_councils=current_decision.parallel_councils.copy(),
                skip_reasons=current_decision.skip_reasons.copy(),
                strategy_notes=output,
                raw_output=output,
                planner_required_councils=current_decision.planner_required_councils.copy(),
                force_all_councils=current_decision.force_all_councils
            )
            
            # Check for stop recommendation
            if 'stop' in output_lower or 'sufficient' in output_lower:
                adjusted.iteration_count = iteration
            
            # Check for more iterations
            if 'more iteration' in output_lower or 'add iteration' in output_lower:
                adjusted.iteration_count += 1
            
            # Adjust exploration
            if 'increase exploration' in output_lower or 'more creative' in output_lower:
                adjusted.exploration_level = min(1.0, adjusted.exploration_level + 0.2)
            elif 'decrease exploration' in output_lower or 'more precise' in output_lower:
                adjusted.exploration_level = max(0.0, adjusted.exploration_level - 0.2)
            
            return adjusted
            
        except Exception:
            return current_decision
    
    def _adjust_autonomous(
        self,
        current_decision: MetaPlanDecision,
        iteration: int,
        quality_score: float,
        issues_count: int,
        time_remaining_seconds: Optional[float] = None,
        uncertainty: float = 0.5,
        multiview_agreement: float = 0.8,
        research_gaps: Optional[List[str]] = None
    ) -> MetaPlanDecision:
        """
        Rule-based adjustment for autonomous mode.
        
        More efficient than LLM-based adjustment.
        
        Enhanced with deepening criteria:
        - Only deepen if uncertainty > 0.55
        - OR multiview_agreement < 0.85
        - OR research_gaps is non-empty
        
        Args:
            current_decision: Current meta-plan decision
            iteration: Current iteration number
            quality_score: Latest quality score
            issues_count: Number of issues found
            time_remaining_seconds: Remaining time budget
            uncertainty: Uncertainty score (0-1, higher = more uncertain)
            multiview_agreement: Agreement between optimist/skeptic (0-1)
            research_gaps: List of identified research gaps
        """
        adjusted = MetaPlanDecision(
            councils_to_activate=current_decision.councils_to_activate.copy(),
            iteration_count=current_decision.iteration_count,
            exploration_level=current_decision.exploration_level,
            parallel_councils=current_decision.parallel_councils.copy(),
            skip_reasons=current_decision.skip_reasons.copy(),
            strategy_notes=current_decision.strategy_notes,
            raw_output=current_decision.raw_output,
            planner_required_councils=current_decision.planner_required_councils.copy(),
            force_all_councils=current_decision.force_all_councils
        )
        
        remaining_iterations = current_decision.iteration_count - iteration
        research_gaps = research_gaps or []
        
        # Check deepening criteria
        should_deepen = self.should_deepen(
            uncertainty=uncertainty,
            multiview_agreement=multiview_agreement,
            research_gaps=research_gaps,
            quality_score=quality_score
        )
        
        # If deepening criteria not met and quality is good, stop early
        if not should_deepen and quality_score >= 7.0:
            adjusted.iteration_count = iteration
            adjusted.strategy_notes += " [Converged: no deepening needed]"
            return adjusted
        
        # Quality-based adjustments
        if quality_score >= 9.0:
            # Excellent quality - can stop
            adjusted.iteration_count = iteration
            adjusted.strategy_notes += " [Quality threshold exceeded]"
        elif quality_score >= 7.5 and issues_count == 0 and not should_deepen:
            # Good quality with no issues and no deepening needed - stop
            adjusted.iteration_count = iteration
            adjusted.strategy_notes += " [Good quality, converged]"
        elif quality_score < 5.0 and remaining_iterations <= 2 and should_deepen:
            # Poor quality with few iterations left and deepening needed - add more
            adjusted.iteration_count += 2
            adjusted.strategy_notes += " [Poor quality, added iterations]"
        
        # Time-based adjustments
        if time_remaining_seconds is not None:
            if time_remaining_seconds < 30:
                # Very low time - stop
                adjusted.iteration_count = iteration
                adjusted.strategy_notes += " [Time exhausted]"
            elif time_remaining_seconds > 180 and quality_score < 7.0 and should_deepen:
                # Plenty of time, mediocre quality, and deepening needed
                adjusted.iteration_count = max(adjusted.iteration_count, iteration + 2)
                adjusted.strategy_notes += " [Time available, extended iterations]"
        
        # Exploration adjustments based on issues and uncertainty
        if issues_count > 5 and adjusted.exploration_level < 0.7:
            adjusted.exploration_level = min(0.8, adjusted.exploration_level + 0.15)
        elif uncertainty > 0.7 and adjusted.exploration_level < 0.7:
            # High uncertainty - increase exploration
            adjusted.exploration_level = min(0.8, adjusted.exploration_level + 0.2)
        elif issues_count == 0 and iteration > 2 and not should_deepen:
            # No issues, stable - reduce exploration
            adjusted.exploration_level = max(0.2, adjusted.exploration_level - 0.1)
        
        return adjusted
    
    def should_deepen(
        self,
        uncertainty: float = 0.5,
        multiview_agreement: float = 0.8,
        research_gaps: Optional[List[str]] = None,
        quality_score: float = 7.0,
        progress: float = 1.0
    ) -> bool:
        """
        Determine if phase deepening should occur.
        
        Deepening criteria (any one triggers deepening):
        1. Uncertainty > 0.55
        2. Multi-view agreement < 0.85
        3. Research gaps exist
        4. Quality score < 6.0 (low quality)
        5. Progress < 0.5 (stalled)
        
        Args:
            uncertainty: Uncertainty score (0-1, higher = more uncertain)
            multiview_agreement: Agreement between optimist/skeptic (0-1)
            research_gaps: List of identified research gaps
            quality_score: Current quality score (0-10)
            progress: Progress indicator (0-1)
            
        Returns:
            True if deepening should occur
        """
        research_gaps = research_gaps or []
        
        # High uncertainty triggers deepening
        if uncertainty > 0.55:
            return True
        
        # Low multi-view agreement triggers deepening
        if multiview_agreement < 0.85:
            return True
        
        # Non-empty research gaps trigger deepening
        if research_gaps:
            return True
        
        # Low quality triggers deepening
        if quality_score < 6.0:
            return True
        
        # Stalled progress triggers deepening
        if progress < 0.5:
            return True
        
        return False
    
    def get_phase_max_iterations(self, phase_name: str) -> int:
        """
        Get maximum iterations for a specific phase type.
        
        Phase-specific defaults to prevent over-iteration:
        - Recon: 1 (unless gaps remain)
        - Analysis: 1 (default)
        - Deep: 2 (only if criteria met)
        - Synthesis: 1 (always single)
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Maximum iterations for this phase type
        """
        phase_lower = phase_name.lower()
        
        PHASE_MAX_ITERATIONS = {
            "recon": 1,
            "reconnaissance": 1,
            "context": 1,
            "analysis": 1,
            "design": 1,
            "deep": 2,
            "implementation": 2,
            "testing": 1,
            "synthesis": 1,
            "report": 1,
        }
        
        for keyword, max_iter in PHASE_MAX_ITERATIONS.items():
            if keyword in phase_lower:
                return max_iter
        
        return 1  # Default

