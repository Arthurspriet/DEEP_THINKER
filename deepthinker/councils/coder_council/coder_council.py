"""
Coder Council Implementation for DeepThinker 2.0.

Code generation with cross-review and self-critique mechanism
using critique exchange consensus.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...models.council_model_config import CODER_MODELS
from ...consensus.critique_exchange import CritiqueConsensus
from ...execution.provenance import CodeProvenance


@dataclass
class CoderContext:
    """Context for coder council execution."""
    
    objective: str
    context: Optional[Dict[str, Any]] = None
    research_findings: Optional[str] = None
    planner_requirements: Optional[str] = None
    previous_code: Optional[str] = None
    evaluation_feedback: Optional[str] = None
    data_config: Optional[Any] = None
    iteration: int = 1
    # Time-awareness fields for depth adjustment
    time_budget_seconds: Optional[float] = None
    time_remaining_seconds: Optional[float] = None
    
    @property
    def time_pressure(self) -> str:
        """
        Get time pressure level for prompt guidance.
        
        Returns:
            "high" - Limited time, be concise and focus on essentials
            "low" - Ample time, explore thoroughly
            "normal" - Balanced approach
        """
        if self.time_remaining_seconds is None:
            return "normal"
        if self.time_budget_seconds is None or self.time_budget_seconds <= 0:
            return "normal"
        ratio = self.time_remaining_seconds / self.time_budget_seconds
        if ratio < 0.3:
            return "high"
        elif ratio > 0.7:
            return "low"
        return "normal"


@dataclass
class CodeOutput:
    """Structured code output."""
    
    code: str
    language: str
    explanation: Optional[str]
    dependencies: List[str]
    provenance: Optional[CodeProvenance] = None
    
    @classmethod
    def from_text(cls, text: str) -> "CodeOutput":
        """Parse code output from text."""
        code = ""
        explanation = ""
        dependencies = []
        language = "python"
        
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', text, re.DOTALL)
        
        if code_blocks:
            # Use the largest code block
            largest_block = max(code_blocks, key=lambda x: len(x[1]))
            language = largest_block[0] or "python"
            code = largest_block[1].strip()
        else:
            # Try to find code without markdown
            lines = text.strip().split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.strip().startswith(('import ', 'from ', 'def ', 'class ', '@')):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            
            code = '\n'.join(code_lines) if code_lines else text
        
        # Extract dependencies
        import_lines = [
            line for line in code.split('\n')
            if line.strip().startswith(('import ', 'from '))
        ]
        for line in import_lines:
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == 'import':
                    dependencies.append(parts[1].split('.')[0])
                elif parts[0] == 'from':
                    dependencies.append(parts[1].split('.')[0])
        
        # Deduplicate
        dependencies = list(set(dependencies))
        
        # Extract explanation (text before code)
        if code_blocks:
            first_block_start = text.find('```')
            if first_block_start > 0:
                explanation = text[:first_block_start].strip()
        
        # Attach default provenance for LLM-generated code
        provenance = CodeProvenance(
            source="llm_generated",
            trust_score=0.5,
            approval_level=0
        )
        
        return cls(
            code=code,
            language=language,
            explanation=explanation if explanation else None,
            dependencies=dependencies,
            provenance=provenance
        )


class CoderCouncil(BaseCouncil):
    """
    Code generation council with cross-review capabilities.
    
    Multiple coding models generate solutions that are
    critiqued and refined through exchange consensus.
    
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
        Initialize coder council.
        
        Args:
            model_pool: Custom model pool (defaults to CODER_MODELS)
            consensus_engine: Custom consensus (defaults to CritiqueConsensus)
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
                    pool_config=CODER_MODELS,
                    base_url=ollama_base_url
                )
        
        # Get consensus from CognitiveSpine if not provided
        if consensus_engine is None:
            if cognitive_spine is not None:
                consensus_engine = cognitive_spine.get_consensus_engine(
                    "critique_exchange", "coder_council"
                )
            else:
                consensus_engine = CritiqueConsensus(
                    ollama_base_url=ollama_base_url
                )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="coder_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition
        )
        
        self._load_default_system_prompt()
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for coder council."""
        self._system_prompt = """You are part of a coding council of expert programmers.
Your role is to generate high-quality, production-ready code.
Be precise, follow best practices, and ensure robustness.

You excel at:
- Writing clean, readable, and well-documented code
- Implementing efficient algorithms and data structures
- Handling edge cases and error conditions
- Following established patterns and conventions
- Creating maintainable and testable solutions

Your code must:
- Include proper error handling
- Have clear docstrings and comments
- Follow PEP 8 style guidelines (for Python)
- Be type-hinted where appropriate
- Handle potential edge cases"""
    
    def build_prompt(
        self,
        coder_context: CoderContext
    ) -> str:
        """
        Build code generation prompt from context.
        
        Args:
            coder_context: Context containing objective and requirements
            
        Returns:
            Prompt string for council members
        """
        # Build context sections
        context_str = ""
        if coder_context.context:
            context_str = f"\n\nAdditional Context:\n{coder_context.context}"
        
        research_str = ""
        if coder_context.research_findings:
            research_str = f"\n\nResearch Findings:\n{coder_context.research_findings}"
        
        planner_str = ""
        if coder_context.planner_requirements:
            planner_str = f"\n\nPlanner Requirements:\n{coder_context.planner_requirements}"
        
        # Handle revision case
        revision_str = ""
        if coder_context.previous_code and coder_context.evaluation_feedback:
            revision_str = f"""

## REVISION REQUEST (Iteration {coder_context.iteration})

### Previous Code:
```python
{coder_context.previous_code}
```

### Evaluation Feedback:
{coder_context.evaluation_feedback}

Please address the feedback and improve the code.
"""
        
        # Build data config info
        data_info = ""
        if coder_context.data_config:
            data_info = f"""
## Dataset Configuration
- Task Type: {getattr(coder_context.data_config, 'task_type', 'unknown')}
- Target Column: {getattr(coder_context.data_config, 'target_column', 'unknown')}
- Feature Columns: {getattr(coder_context.data_config, 'feature_columns', [])}

The code should implement a class with fit(X, y) and predict(X) methods.
"""
        
        if revision_str:
            # Revision prompt
            prompt = f"""Revise and improve the following code based on evaluation feedback:

## OBJECTIVE
{coder_context.objective}
{context_str}
{research_str}
{planner_str}
{data_info}
{revision_str}

## INSTRUCTIONS
1. Carefully analyze the feedback
2. Address each identified issue
3. Maintain working functionality
4. Improve code quality
5. Output the complete revised code

Provide the complete, improved code in a Python code block."""

        else:
            # Initial generation prompt
            prompt = f"""Generate production-quality Python code for the following objective:

## OBJECTIVE
{coder_context.objective}
{context_str}
{research_str}
{planner_str}
{data_info}

## REQUIREMENTS
1. Write clean, well-documented code
2. Include proper error handling
3. Add type hints where appropriate
4. Handle edge cases
5. Follow best practices

## OUTPUT FORMAT
Provide your solution in a Python code block:
```python
# Your code here
```

Include a brief explanation of your approach before the code."""

        # Add time-aware guidance based on time pressure
        time_guidance = self._get_time_guidance(coder_context)
        if time_guidance:
            prompt += time_guidance

        return prompt
    
    def _get_time_guidance(self, context: CoderContext) -> str:
        """
        Generate time-aware guidance for the prompt.
        
        Args:
            context: Coder context with time pressure info
            
        Returns:
            Time guidance string to append to prompt
        """
        if not hasattr(context, 'time_pressure'):
            return ""
        
        pressure = context.time_pressure
        if pressure == "high":
            return """

## TIME CONSTRAINT
Limited time available for this phase. Adjust your approach:
- Focus on core functionality first
- Use straightforward, proven patterns
- Minimize edge case handling to essential ones
- Keep documentation brief but clear"""
        elif pressure == "low":
            return """

## TIME AVAILABLE
Ample time available for thorough implementation:
- Implement comprehensive error handling
- Add detailed documentation and comments
- Consider performance optimizations
- Handle all edge cases thoroughly"""
        return ""
    
    def postprocess(self, consensus_output: Any) -> CodeOutput:
        """
        Postprocess consensus output into CodeOutput.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed CodeOutput object with provenance attached
        """
        if not consensus_output:
            provenance = CodeProvenance(
                source="llm_generated",
                trust_score=0.5,
                approval_level=0
            )
            return CodeOutput(
                code="",
                language="python",
                explanation=None,
                dependencies=[],
                provenance=provenance
            )
        
        return CodeOutput.from_text(str(consensus_output))
    
    def generate_code(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        research_findings: Optional[str] = None,
        planner_requirements: Optional[str] = None,
        data_config: Optional[Any] = None
    ) -> CouncilResult:
        """
        Convenience method to generate code.
        
        Args:
            objective: Coding objective
            context: Additional context
            research_findings: Research output
            planner_requirements: Requirements from planner
            data_config: Dataset configuration
            
        Returns:
            CouncilResult with CodeOutput
        """
        coder_context = CoderContext(
            objective=objective,
            context=context,
            research_findings=research_findings,
            planner_requirements=planner_requirements,
            data_config=data_config,
            iteration=1
        )
        
        return self.execute(coder_context)
    
    def revise_code(
        self,
        objective: str,
        previous_code: str,
        evaluation_feedback: str,
        iteration: int = 2,
        context: Optional[Dict[str, Any]] = None
    ) -> CouncilResult:
        """
        Convenience method to revise code based on feedback.
        
        Args:
            objective: Original objective
            previous_code: Code to revise
            evaluation_feedback: Feedback from evaluator
            iteration: Current iteration number
            context: Additional context
            
        Returns:
            CouncilResult with revised CodeOutput
        """
        coder_context = CoderContext(
            objective=objective,
            context=context,
            previous_code=previous_code,
            evaluation_feedback=evaluation_feedback,
            iteration=iteration
        )
        
        return self.execute(coder_context)

