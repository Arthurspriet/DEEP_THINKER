"""
Planner Council Implementation for DeepThinker 2.0.

Builds planning prompts and returns WorkflowPlan objects.
Uses weighted blend consensus for strategic plan synthesis.

Extended for Step Engine:
- Can now produce phases with embedded step definitions
- Supports structured JSON output for phase+step planning

Enhanced for Iterative Synthesis:
- SynthesisContext for evolving synthesis state
- Incremental synthesis with gap detection
- Iteration-aware context building
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...models.council_model_config import PLANNER_MODELS, CouncilModelPool
from ...consensus.weighted_blend import WeightedBlendConsensus
from ...execution.plan_config import WorkflowPlan, WorkflowPlanParser
from ...steps.step_types import StepDefinition

logger = logging.getLogger(__name__)


@dataclass
class PlannerContext:
    """Context for planner council execution."""
    
    objective: str
    context: Optional[Dict[str, Any]] = None
    available_agents: List[str] = field(default_factory=lambda: [
        "researcher", "coder", "evaluator", "simulator", "executor"
    ])
    max_iterations: int = 3
    quality_threshold: float = 7.0
    data_config: Optional[Any] = None
    simulation_config: Optional[Any] = None
    # Knowledge context from RAG retrieval
    knowledge_context: Optional[str] = None


@dataclass
class SynthesisContext:
    """
    Context for iterative synthesis phase execution.
    
    Designed to evolve across iterations, preventing identical outputs.
    """
    
    objective: str
    prior_findings: str  # Accumulated findings from research/analysis phases
    unresolved_issues: List[str] = field(default_factory=list)  # Issues from evaluator
    structural_gaps: List[str] = field(default_factory=list)  # Missing sections/content
    recommended_sections: List[str] = field(default_factory=list)  # Sections to include
    evaluator_feedback: List[str] = field(default_factory=list)  # Feedback history
    iteration: int = 1
    max_iterations: int = 3
    # Tracking for evolution
    prior_synthesis_summary: Optional[str] = None  # Summary of previous synthesis
    addressed_issues: List[str] = field(default_factory=list)  # Issues resolved this iteration
    # Knowledge context from RAG retrieval
    knowledge_context: Optional[str] = None
    
    def has_work_remaining(self) -> bool:
        """Check if there are gaps or issues requiring more synthesis."""
        return bool(
            self.unresolved_issues or 
            self.structural_gaps or 
            (self.iteration < self.max_iterations and not self.prior_synthesis_summary)
        )
    
    def get_priority_work(self, limit: int = 5) -> List[str]:
        """Get prioritized work items for this iteration."""
        # Prioritize: structural_gaps > unresolved_issues
        all_work = self.structural_gaps + self.unresolved_issues
        return all_work[:limit]


@dataclass
class SynthesisResult:
    """Result from synthesis iteration."""
    
    content: str  # The synthesized content
    sections_completed: List[str] = field(default_factory=list)
    remaining_gaps: List[str] = field(default_factory=list)
    remaining_issues: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    is_complete: bool = False
    iteration: int = 1
    raw_output: str = ""
    
    @classmethod
    def from_text(cls, text: str, iteration: int = 1) -> "SynthesisResult":
        """Parse synthesis result from LLM output."""
        sections = []
        remaining_gaps = []
        remaining_issues = []
        
        current_section = None
        content_lines = []
        
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            
            # Track section headers
            if line.startswith('## ') or line.startswith('### '):
                section_name = line.lstrip('#').strip()
                if section_name:
                    sections.append(section_name)
            
            # Detect remaining work sections
            if 'remaining gap' in line_lower or 'still missing' in line_lower:
                current_section = 'gaps'
            elif 'remaining issue' in line_lower or 'unresolved' in line_lower:
                current_section = 'issues'
            elif line.strip().startswith(('-', '*', '•')):
                content = line.strip().lstrip('-*•').strip()
                if content:
                    if current_section == 'gaps':
                        remaining_gaps.append(content)
                    elif current_section == 'issues':
                        remaining_issues.append(content)
        
        # Extract confidence
        confidence = 0.7
        conf_match = re.search(r'confidence[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if conf_match:
            try:
                val = float(conf_match.group(1))
                confidence = val if val <= 1.0 else val / 10.0
            except ValueError:
                pass
        
        # Check if complete
        is_complete = (
            not remaining_gaps and 
            not remaining_issues and 
            len(sections) >= 3  # At least 3 sections
        )
        
        return cls(
            content=text,
            sections_completed=sections,
            remaining_gaps=remaining_gaps[:5],
            remaining_issues=remaining_issues[:5],
            confidence_score=confidence,
            is_complete=is_complete,
            iteration=iteration,
            raw_output=text
        )


class PlannerCouncil(BaseCouncil):
    """
    Strategic planning council for workflow orchestration.
    
    Multiple planning models propose workflow strategies,
    which are then blended into a unified plan.
    
    Supports dynamic configuration via CouncilDefinition for
    runtime model/temperature/persona selection.
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        cognitive_spine: Optional[Any] = None,
        council_definition: Optional[Any] = None
    ):
        """
        Initialize planner council.
        
        Args:
            model_pool: Custom model pool (defaults to PLANNER_MODELS)
            consensus_engine: Custom consensus (defaults to WeightedBlendConsensus)
                            If None and cognitive_spine provided, gets from spine
            ollama_base_url: Ollama server URL
            cognitive_spine: Optional CognitiveSpine for validation and consensus
            council_definition: Optional CouncilDefinition for dynamic configuration
        """
        # Use council_definition models if provided and no custom pool
        if model_pool is None:
            if council_definition is not None and council_definition.models:
                model_pool = ModelPool(
                    pool_config=council_definition.get_model_configs(),
                    base_url=ollama_base_url
                )
            else:
                model_pool = ModelPool(
                    pool_config=PLANNER_MODELS,
                    base_url=ollama_base_url
                )
        
        # Get consensus from CognitiveSpine if not provided
        if consensus_engine is None:
            if cognitive_spine is not None:
                consensus_engine = cognitive_spine.get_consensus_engine(
                    "weighted_blend", "planner_council"
                )
            else:
                consensus_engine = WeightedBlendConsensus(
                    ollama_base_url=ollama_base_url
                )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="planner_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition
        )
        
        self._load_default_system_prompt()
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for planner council."""
        self._system_prompt = """You are part of a strategic council of planners.
Your role is to propose a version of the best possible workflow plan.
Be explicit, structured, and independent in your reasoning.

You excel at:
- Analyzing complex objectives and breaking them into subtasks
- Determining which agents are needed and in what order
- Defining clear success criteria and evaluation metrics
- Balancing thoroughness with efficiency
- Adapting plans based on task complexity

Your plans must be structured with clear sections:
1. OBJECTIVE ANALYSIS
2. WORKFLOW STRATEGY (which agents to use)
3. AGENT REQUIREMENTS (specific instructions per agent)
4. SUCCESS CRITERIA
5. ITERATION STRATEGY"""
    
    def build_prompt(
        self,
        planner_context: PlannerContext
    ) -> str:
        """
        Build planning prompt from context.
        
        Args:
            planner_context: Context containing objective and configuration
            
        Returns:
            Prompt string for council members
        """
        # Build context string
        context_str = ""
        if planner_context.context:
            context_str = f"\n\nAdditional Context:\n{planner_context.context}"
        
        # Build knowledge context string (from RAG retrieval)
        knowledge_str = ""
        if planner_context.knowledge_context:
            knowledge_str = f"\n\n## RETRIEVED KNOWLEDGE (use as reference):\n{planner_context.knowledge_context}"
        
        # Build data config info
        data_info = ""
        if planner_context.data_config:
            data_info = f"""
Dataset Configuration:
- Task Type: {getattr(planner_context.data_config, 'task_type', 'unknown')}
- Data Path: {getattr(planner_context.data_config, 'data_path', 'not specified')}
"""
        
        # Build simulation config info
        sim_info = ""
        if planner_context.simulation_config:
            sim_info = f"""
Simulation Configuration:
- Enabled: {getattr(planner_context.simulation_config, 'is_enabled', lambda: False)()}
"""
        
        prompt = f"""Create a comprehensive workflow plan for the following objective:

## OBJECTIVE
{planner_context.objective}
{context_str}
{knowledge_str}

## AVAILABLE AGENTS
{', '.join(planner_context.available_agents)}

## CONFIGURATION
- Maximum Iterations: {planner_context.max_iterations}
- Quality Threshold: {planner_context.quality_threshold}/10
{data_info}
{sim_info}

## INSTRUCTIONS
Create a detailed workflow plan with the following sections:

### 1. OBJECTIVE ANALYSIS
Analyze what needs to be accomplished. Break down the objective into key components.

### 2. WORKFLOW STRATEGY
For each agent, specify:
- Agent Name: Yes/No (whether to use this agent)
Explain the reasoning for each decision.

### 3. AGENT REQUIREMENTS
For each enabled agent, provide specific requirements:
- Researcher: What to research, sources to prioritize
- Coder: Implementation requirements, patterns to follow
- Evaluator: Evaluation criteria, quality metrics
- Simulator: Scenarios to test, edge cases
- Executor: Execution requirements

### 4. SUCCESS CRITERIA
List 3-5 measurable criteria for success.

### 5. ITERATION STRATEGY
Describe the iteration approach:
- Recommended iterations count
- When to stop iterating
- Focus areas for each potential iteration

Produce a complete, actionable plan."""

        return prompt
    
    def postprocess(self, consensus_output: Any) -> WorkflowPlan:
        """
        Postprocess consensus output into WorkflowPlan.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed WorkflowPlan object
        """
        if not consensus_output:
            return WorkflowPlan()
        
        # Parse the plan text into structured WorkflowPlan
        return WorkflowPlanParser.parse(str(consensus_output))
    
    def plan(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        data_config: Optional[Any] = None,
        simulation_config: Optional[Any] = None,
        max_iterations: int = 3,
        quality_threshold: float = 7.0
    ) -> CouncilResult:
        """
        Convenience method to create a workflow plan.
        
        Args:
            objective: Primary objective
            context: Additional context
            data_config: Dataset configuration
            simulation_config: Simulation configuration
            max_iterations: Max refinement iterations
            quality_threshold: Quality threshold for completion
            
        Returns:
            CouncilResult with WorkflowPlan
        """
        planner_context = PlannerContext(
            objective=objective,
            context=context,
            data_config=data_config,
            simulation_config=simulation_config,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold
        )
        
        return self.execute(planner_context)
    
    def build_mission_phases_prompt(
        self,
        objective: str,
        time_budget_minutes: int,
        allow_internet: bool = True,
        allow_code_execution: bool = True,
        notes: Optional[str] = None
    ) -> str:
        """
        Build a prompt for generating mission phases WITH steps.
        
        This is used by MissionOrchestrator to get structured phase+step plans.
        
        Args:
            objective: Mission objective
            time_budget_minutes: Total time budget
            allow_internet: Whether web research is allowed
            allow_code_execution: Whether code execution is allowed
            notes: Additional notes/constraints
            
        Returns:
            Prompt string requesting structured phase+step output
        """
        constraints = []
        if not allow_internet:
            constraints.append("- Internet/web search is NOT allowed")
        if not allow_code_execution:
            constraints.append("- Code execution is NOT allowed (design only)")
        constraints.append(f"- Time budget: {time_budget_minutes} minutes")
        if notes:
            constraints.append(f"- Notes: {notes}")
        
        constraint_str = "\n".join(constraints) if constraints else "No special constraints"
        
        prompt = f"""Plan the phases and steps for this mission.

## OBJECTIVE
{objective}

## CONSTRAINTS
{constraint_str}

## INSTRUCTIONS
Create a structured plan with phases and concrete steps. Each phase should contain
specific steps that can be executed by a single specialized model.

Available step types:
- research: Gather information, search web, collect context
- analysis: Analyze data, draw insights, identify patterns
- design: Create designs, architectures, plans
- coding: Write code, implement solutions
- testing: Test, validate, find edge cases
- synthesis: Combine findings, create summaries, produce deliverables
- meta: Reflect, adjust approach, strategic decisions

## OUTPUT FORMAT
Respond with a JSON structure like this:

```json
{{
  "phases": [
    {{
      "name": "Phase Name",
      "description": "What this phase accomplishes",
      "steps": [
        {{
          "name": "Step name",
          "description": "What this step should do",
          "step_type": "research|analysis|design|coding|testing|synthesis|meta",
          "tools": ["web", "code", "simulation"],
          "preferred_model": "optional model name or null"
        }}
      ]
    }}
  ]
}}
```

Create 3-6 phases depending on complexity. Each phase should have 1-4 steps.
Common phase progression: Reconnaissance -> Analysis -> Design -> Implementation -> Testing -> Synthesis

Produce your JSON plan:"""

        return prompt
    
    def parse_phases_with_steps(
        self,
        raw_output: str
    ) -> List[Tuple[str, str, List[StepDefinition]]]:
        """
        Parse planner output into phases with step definitions.
        
        Args:
            raw_output: Raw output from the planner (may contain JSON)
            
        Returns:
            List of tuples: (phase_name, phase_description, steps)
        """
        phases_data = []
        
        # Try to extract JSON from output
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*"phases"[\s\S]*\}', raw_output)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = None
        
        if json_str:
            try:
                data = json.loads(json_str)
                phases = data.get("phases", [])
                
                for phase in phases:
                    phase_name = phase.get("name", "Unnamed Phase")
                    phase_desc = phase.get("description", "")
                    raw_steps = phase.get("steps", [])
                    
                    steps = []
                    for raw_step in raw_steps:
                        step = StepDefinition(
                            name=raw_step.get("name", "Unnamed Step"),
                            description=raw_step.get("description", ""),
                            step_type=raw_step.get("step_type", "research"),
                            tools=raw_step.get("tools", []),
                            preferred_model=raw_step.get("preferred_model"),
                        )
                        steps.append(step)
                    
                    phases_data.append((phase_name, phase_desc, steps))
                    
            except json.JSONDecodeError:
                pass
        
        # Fallback: parse text format if JSON failed
        if not phases_data:
            phases_data = self._parse_phases_from_text(raw_output)
        
        return phases_data
    
    def _parse_phases_from_text(
        self,
        text: str
    ) -> List[Tuple[str, str, List[StepDefinition]]]:
        """
        Fallback parser for non-JSON phase output.
        
        Looks for patterns like:
        PHASE: Name
        DESCRIPTION: What it does
        STEPS:
        - Step 1 (research): Description
        
        Args:
            text: Raw text to parse
            
        Returns:
            List of tuples: (phase_name, phase_description, steps)
        """
        phases_data = []
        
        # Try PHASE: / DESCRIPTION: format
        phase_pattern = r'PHASE:\s*(.+?)(?:\n|$)'
        desc_pattern = r'DESCRIPTION:\s*(.+?)(?=PHASE:|STEPS:|$)'
        
        phase_matches = re.findall(phase_pattern, text, re.IGNORECASE)
        
        if not phase_matches:
            # Try numbered format: "1. Phase Name"
            phase_matches = re.findall(r'^\d+\.\s*(.+)$', text, re.MULTILINE)
        
        for phase_name in phase_matches[:7]:  # Limit to 7 phases
            phase_name = phase_name.strip()
            
            # Try to find description after this phase
            desc_match = re.search(
                rf'{re.escape(phase_name)}.*?(?:DESCRIPTION:|:)\s*(.+?)(?=PHASE:|STEP|$)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            description = desc_match.group(1).strip() if desc_match else ""
            
            # For fallback, create a single step matching the phase
            step_type = self._infer_step_type(phase_name.lower())
            steps = [
                StepDefinition(
                    name=f"Execute {phase_name}",
                    description=description or f"Complete the {phase_name} phase",
                    step_type=step_type,
                )
            ]
            
            phases_data.append((phase_name, description, steps))
        
        return phases_data
    
    def _infer_step_type(self, phase_name: str) -> str:
        """Infer step type from phase name."""
        name_lower = phase_name.lower()
        
        if any(kw in name_lower for kw in ["research", "recon", "gather", "context"]):
            return "research"
        elif any(kw in name_lower for kw in ["analysis", "analyze", "investigate"]):
            return "analysis"
        elif any(kw in name_lower for kw in ["design", "architect", "plan"]):
            return "design"
        elif any(kw in name_lower for kw in ["implement", "code", "build", "develop"]):
            return "coding"
        elif any(kw in name_lower for kw in ["test", "validation", "simulate"]):
            return "testing"
        elif any(kw in name_lower for kw in ["synthesis", "report", "summary", "final"]):
            return "synthesis"
        else:
            return "research"
    
    def synthesize(
        self,
        context: SynthesisContext
    ) -> CouncilResult:
        """
        Execute iterative synthesis with evolving context.
        
        This method builds upon prior synthesis work, focusing only on
        gaps and issues rather than regenerating the entire output.
        
        Args:
            context: SynthesisContext with accumulated findings and gaps
            
        Returns:
            CouncilResult with SynthesisResult
        """
        prompt = self._build_synthesis_prompt(context)
        
        try:
            # Get responses from model pool
            responses = self.model_pool.run_all(
                prompt=prompt,
                system_prompt=self._get_synthesis_system_prompt()
            )
            
            if not responses:
                return CouncilResult(
                    output=None,
                    raw_outputs={},
                    consensus_details=None,
                    council_name=self.council_name,
                    success=False,
                    error="No responses from model pool"
                )
            
            # Use consensus engine - extract text outputs from ModelOutput objects
            # responses is Dict[str, ModelOutput], need to extract output strings
            text_outputs = []
            for name, model_output in responses.items():
                if model_output.success and model_output.output:
                    text_outputs.append(model_output.output)
            
            if not text_outputs:
                return CouncilResult(
                    output=None,
                    raw_outputs=responses,
                    consensus_details=None,
                    council_name=self.council_name,
                    success=False,
                    error="All models produced empty outputs"
                )
            
            consensus_output = self.consensus.synthesize(text_outputs)
            
            # Parse into SynthesisResult
            result = SynthesisResult.from_text(
                str(consensus_output),
                iteration=context.iteration
            )
            
            logger.info(
                f"Synthesis iteration {context.iteration}: "
                f"sections={len(result.sections_completed)}, "
                f"remaining_gaps={len(result.remaining_gaps)}, "
                f"is_complete={result.is_complete}"
            )
            
            return CouncilResult(
                output=result,
                raw_outputs=responses,
                consensus_details=getattr(self.consensus, 'last_details', None),
                council_name=self.council_name,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return CouncilResult(
                output=None,
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error=str(e)
            )
    
    def _build_synthesis_prompt(self, context: SynthesisContext) -> str:
        """Build prompt for synthesis iteration."""
        # Determine if this is incremental
        is_incremental = context.iteration > 1 and context.prior_synthesis_summary
        
        gaps_str = ""
        if context.structural_gaps:
            gaps_str = "\n## STRUCTURAL GAPS (must address)\n" + "\n".join(
                f"- {gap}" for gap in context.structural_gaps
            )
        
        issues_str = ""
        if context.unresolved_issues:
            issues_str = "\n## UNRESOLVED ISSUES (must address)\n" + "\n".join(
                f"- {issue}" for issue in context.unresolved_issues
            )
        
        feedback_str = ""
        if context.evaluator_feedback:
            feedback_str = "\n## EVALUATOR FEEDBACK\n" + "\n".join(
                f"- {fb}" for fb in context.evaluator_feedback[-5:]  # Last 5 items
            )
        
        sections_str = ""
        if context.recommended_sections:
            sections_str = "\n## RECOMMENDED SECTIONS\n" + "\n".join(
                f"- {section}" for section in context.recommended_sections
            )
        
        if is_incremental:
            prompt = f"""## MODE: INCREMENTAL SYNTHESIS (Iteration {context.iteration})
You are refining an existing synthesis. DO NOT start from scratch.

## OBJECTIVE
{context.objective}

## PRIOR SYNTHESIS SUMMARY
{context.prior_synthesis_summary}
{gaps_str}
{issues_str}
{feedback_str}

## INSTRUCTIONS
1. Address each STRUCTURAL GAP by adding the missing content
2. Resolve each UNRESOLVED ISSUE
3. Incorporate EVALUATOR FEEDBACK
4. Do NOT repeat content that was already synthesized

Provide:

### ADDITIONS
New content to add:
[Your additions here]

### REVISIONS
Content that needs revision:
[Your revisions here]

### REMAINING GAPS (if any)
- [Gap still unaddressed]

### REMAINING ISSUES (if any)
- [Issue still unresolved]

### CONFIDENCE
[Confidence: 0.X] in this synthesis

Focus on closing gaps, not regenerating existing content."""
        else:
            # Initial synthesis
            prompt = f"""## MODE: COMPREHENSIVE SYNTHESIS (Iteration 1)
Create a complete synthesis of the mission findings.

## OBJECTIVE
{context.objective}

## FINDINGS TO SYNTHESIZE
{context.prior_findings[:4000]}
{gaps_str}
{issues_str}
{sections_str}

## INSTRUCTIONS
Create a structured final report with:

### 1. EXECUTIVE SUMMARY
Brief overview of the mission and key outcomes.

### 2. KEY FINDINGS
Most important discoveries and insights.

### 3. DELIVERABLES
Code, designs, recommendations, or other outputs.

### 4. ANALYSIS
Detailed analysis of findings and their implications.

### 5. CONCLUSIONS
Summary conclusions based on the analysis.

### 6. NEXT STEPS
Recommended follow-up actions if applicable.

### REMAINING GAPS (if any)
List any topics that still need more work:
- [Gap 1]

### REMAINING ISSUES (if any)
List any unresolved issues:
- [Issue 1]

### CONFIDENCE
[Confidence: 0.X] in this synthesis

Be comprehensive but focused. Ensure all major topics are covered."""

        return prompt
    
    def _get_synthesis_system_prompt(self) -> str:
        """Get system prompt for synthesis."""
        return """You are an expert report synthesizer.
Your role is to combine research findings, analysis, and outcomes into 
cohesive, well-structured reports.

You excel at:
- Identifying the most important findings
- Organizing content logically
- Writing clear executive summaries
- Highlighting actionable conclusions
- Identifying remaining gaps honestly

Be thorough but concise. Focus on value and clarity."""
    
    def update_synthesis_context(
        self,
        context: SynthesisContext,
        result: SynthesisResult
    ) -> SynthesisContext:
        """
        Update synthesis context based on result for next iteration.
        
        Args:
            context: Current synthesis context
            result: Result from synthesis iteration
            
        Returns:
            Updated SynthesisContext for next iteration
        """
        # Mark addressed issues
        addressed = []
        for issue in context.unresolved_issues:
            # If not in remaining issues, it was addressed
            if issue not in result.remaining_issues:
                addressed.append(issue)
        
        # Create updated context
        return SynthesisContext(
            objective=context.objective,
            prior_findings=context.prior_findings,
            unresolved_issues=result.remaining_issues,
            structural_gaps=result.remaining_gaps,
            recommended_sections=context.recommended_sections,
            evaluator_feedback=context.evaluator_feedback,
            iteration=context.iteration + 1,
            max_iterations=context.max_iterations,
            prior_synthesis_summary=result.content[:2000],  # Summary of what was produced
            addressed_issues=context.addressed_issues + addressed
        )

