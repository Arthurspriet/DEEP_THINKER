"""
Reflection Engine for Meta-Cognition.

Analyzes phase outputs to detect assumptions, weaknesses, missing information,
contradictions, and questions for further investigation.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.model_pool import ModelPool
    from ..missions.mission_types import MissionState

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """
    Generates reflections on phase outputs using LLM analysis.
    
    Detects:
    - Assumptions made during the phase
    - Potential errors or weaknesses
    - Missing information
    - Contradictions with prior work
    - Questions to investigate
    - Suggestions for next iteration
    """
    
    def __init__(self, model_pool: "ModelPool"):
        """
        Initialize the reflection engine.
        
        Args:
            model_pool: Model pool for LLM calls
        """
        self.model_pool = model_pool
        self._system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for reflection analysis."""
        return """You are a critical analyst reviewing the output of a phase in a long-running mission.

Your task is to carefully analyze the provided output and identify:

1. ASSUMPTIONS: What assumptions were made (explicitly or implicitly)?
2. WEAKNESSES: What are potential errors, gaps, or weak points?
3. MISSING_INFO: What information is missing that would strengthen the analysis?
4. CONTRADICTIONS: What contradicts prior work or established facts?
5. QUESTIONS: What questions should be investigated further?
6. SUGGESTIONS: What should be done in the next iteration to improve?

Be thorough but concise. Focus on actionable insights.

Output your analysis in the following format:

ASSUMPTIONS:
- [assumption 1]
- [assumption 2]

WEAKNESSES:
- [weakness 1]
- [weakness 2]

MISSING_INFO:
- [missing info 1]
- [missing info 2]

CONTRADICTIONS:
- [contradiction 1]
- [contradiction 2]

QUESTIONS:
- [question 1]
- [question 2]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]"""
    
    def _get_smallest_model(self) -> str:
        """Get the smallest available model for efficient reflection."""
        models = self.model_pool.get_all_models()
        if not models:
            raise ValueError("No models available in model pool")
        
        # Prefer smaller models for meta-cognition (efficiency)
        small_keywords = ["3b", "7b", "8b", "small", "mini", "tiny"]
        for model in models:
            if any(kw in model.lower() for kw in small_keywords):
                return model
        
        # Fallback to first available model
        return models[0]
    
    def reflect_on_phase_output(
        self,
        phase_name: str,
        output: Any,
        state: "MissionState"
    ) -> Dict[str, List[str]]:
        """
        Generate reflection on a phase's output.
        
        Uses an LLM to analyze the output and detect assumptions,
        weaknesses, missing information, contradictions, and questions.
        
        Args:
            phase_name: Name of the phase being reflected upon
            output: The output from the phase (council output or artifacts)
            state: Current mission state for context
            
        Returns:
            Dictionary with keys: assumptions, weaknesses, missing_info,
            contradictions, questions, suggestions
        """
        # Initialize result with empty lists
        result = {
            "assumptions": [],
            "weaknesses": [],
            "missing_info": [],
            "contradictions": [],
            "questions": [],
            "suggestions": [],
        }
        
        try:
            # Build context from prior phases
            prior_context = self._build_prior_context(state, phase_name)
            
            # Convert output to string
            output_str = self._format_output(output)
            
            # Build the reflection prompt
            prompt = self._build_reflection_prompt(
                phase_name=phase_name,
                output=output_str,
                objective=state.objective,
                prior_context=prior_context
            )
            
            # Get smallest model for efficiency
            model = self._get_smallest_model()
            
            # Run LLM
            response = self.model_pool.run_single(
                model_name=model,
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=0.3  # Low temperature for analytical task
            )
            
            if response:
                result = self._parse_reflection(response)
            
        except Exception as e:
            logger.warning(f"Reflection failed for phase '{phase_name}': {e}")
            # Return empty result on failure (graceful degradation)
        
        # Store reflection in state
        if "reflection" not in state.work_summary:
            state.work_summary["reflection"] = {}
        state.work_summary["reflection"][phase_name] = result
        
        return result
    
    def _build_prior_context(self, state: "MissionState", current_phase: str) -> str:
        """Build context from prior completed phases."""
        context_parts = []
        
        for phase in state.phases:
            if phase.name == current_phase:
                break
            if phase.status == "completed" and phase.artifacts:
                # Include key artifacts from prior phases
                artifacts_summary = []
                for key, value in list(phase.artifacts.items())[:3]:  # Limit to 3 artifacts
                    truncated = value[:500] + "..." if len(value) > 500 else value
                    artifacts_summary.append(f"  {key}: {truncated}")
                if artifacts_summary:
                    context_parts.append(f"[{phase.name}]\n" + "\n".join(artifacts_summary))
        
        return "\n\n".join(context_parts) if context_parts else "No prior phases completed."
    
    def _format_output(self, output: Any) -> str:
        """Format phase output for analysis."""
        if isinstance(output, dict):
            # Format dictionary output
            parts = []
            for key, value in output.items():
                value_str = str(value)
                if len(value_str) > 1000:
                    value_str = value_str[:1000] + "..."
                parts.append(f"{key}: {value_str}")
            return "\n".join(parts)
        elif isinstance(output, str):
            return output[:3000] + "..." if len(output) > 3000 else output
        else:
            output_str = str(output)
            return output_str[:3000] + "..." if len(output_str) > 3000 else output_str
    
    def _build_reflection_prompt(
        self,
        phase_name: str,
        output: str,
        objective: str,
        prior_context: str
    ) -> str:
        """Build the prompt for reflection analysis."""
        return f"""Analyze the following phase output from a mission.

## MISSION OBJECTIVE
{objective}

## PRIOR WORK
{prior_context}

## CURRENT PHASE: {phase_name}

## PHASE OUTPUT
{output}

---

Provide your critical analysis of this phase output. Identify assumptions, weaknesses, missing information, contradictions with prior work, questions to investigate, and suggestions for improvement."""
    
    def _parse_reflection(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response into structured reflection."""
        result = {
            "assumptions": [],
            "weaknesses": [],
            "missing_info": [],
            "contradictions": [],
            "questions": [],
            "suggestions": [],
        }
        
        # Map section headers to result keys
        section_map = {
            "assumptions": "assumptions",
            "weaknesses": "weaknesses",
            "missing_info": "missing_info",
            "missing info": "missing_info",
            "missing information": "missing_info",
            "contradictions": "contradictions",
            "questions": "questions",
            "suggestions": "suggestions",
        }
        
        current_section = None
        lines = response.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            line_lower = line.lower().rstrip(":")
            for header, key in section_map.items():
                if header in line_lower and len(line) < 50:
                    current_section = key
                    break
            else:
                # Not a header - add to current section if we're in one
                if current_section and line.startswith("-"):
                    item = line.lstrip("- ").strip()
                    if item and len(item) > 2:
                        result[current_section].append(item)
                elif current_section and line.startswith("*"):
                    item = line.lstrip("* ").strip()
                    if item and len(item) > 2:
                        result[current_section].append(item)
        
        return result

