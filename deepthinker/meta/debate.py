"""
Debate Engine for Meta-Cognition.

Runs internal debates between LLM personas to refine hypotheses
and identify contradictions through adversarial reasoning.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.model_pool import ModelPool
    from ..missions.mission_types import MissionState

logger = logging.getLogger(__name__)


# Persona definitions for internal debate
PERSONA_OPTIMIST = """You are the OPTIMIST in an internal debate.

Your role is to:
- Defend the current hypotheses and approach
- Highlight the strengths and potential of the current work
- Find ways to address weaknesses constructively
- Propose solutions to overcome identified problems
- Maintain a positive but realistic perspective

Debate style: Constructive, solution-oriented, supportive of progress."""

PERSONA_SKEPTIC = """You are the SKEPTIC in an internal debate.

Your role is to:
- Challenge assumptions and weak reasoning
- Identify potential failure modes and risks
- Question evidence and ask for stronger proof
- Point out logical fallacies or gaps
- Stress-test ideas by considering edge cases

Debate style: Critical, questioning, demanding rigorous evidence."""

PERSONA_ANALYST = """You are the ANALYST in an internal debate.

Your role is to:
- Synthesize the optimist and skeptic perspectives
- Identify the strongest arguments from both sides
- Find middle ground and balanced conclusions
- Propose refined hypotheses that address concerns
- Adjust confidence levels based on the debate

Debate style: Balanced, synthesizing, conclusion-oriented."""


class DebateEngine:
    """
    Runs internal debates between LLM personas.
    
    Uses three personas:
    - Optimist: Defends and strengthens hypotheses
    - Skeptic: Challenges and stress-tests hypotheses
    - Analyst: Synthesizes perspectives and refines conclusions
    """
    
    def __init__(self, model_pool: "ModelPool"):
        """
        Initialize the debate engine.
        
        Args:
            model_pool: Model pool for LLM calls
        """
        self.model_pool = model_pool
    
    def _get_smallest_model(self) -> str:
        """Get the smallest available model for efficiency."""
        models = self.model_pool.get_all_models()
        if not models:
            raise ValueError("No models available in model pool")
        
        small_keywords = ["3b", "7b", "8b", "small", "mini", "tiny"]
        for model in models:
            if any(kw in model.lower() for kw in small_keywords):
                return model
        return models[0]
    
    def run_internal_debate(
        self,
        phase_name: str,
        hypotheses: List[Dict[str, Any]],
        context: str,
        state: "MissionState"
    ) -> Dict[str, Any]:
        """
        Run an internal debate on the current hypotheses.
        
        Args:
            phase_name: Name of the phase being debated
            hypotheses: List of hypothesis dictionaries
            context: Context from the phase output
            state: Current mission state
            
        Returns:
            Dictionary containing:
            - debate_transcript: Full debate text
            - refined_hypotheses: Updated hypotheses after debate
            - contradictions_found: List of identified contradictions
            - confidence_adjustments: Dict of hypothesis_id -> adjustment
        """
        result = {
            "debate_transcript": "",
            "refined_hypotheses": [],
            "contradictions_found": [],
            "confidence_adjustments": {},
        }
        
        try:
            # Format hypotheses for debate
            hypotheses_text = self._format_hypotheses(hypotheses)
            
            if not hypotheses_text:
                return result
            
            model = self._get_smallest_model()
            transcript_parts = []
            
            # Round 1: Optimist defends
            optimist_response = self._run_persona(
                persona="optimist",
                system_prompt=PERSONA_OPTIMIST,
                hypotheses=hypotheses_text,
                context=context,
                phase_name=phase_name,
                prior_debate="",
                model=model
            )
            transcript_parts.append(f"[OPTIMIST]\n{optimist_response}")
            
            # Round 2: Skeptic challenges
            skeptic_response = self._run_persona(
                persona="skeptic",
                system_prompt=PERSONA_SKEPTIC,
                hypotheses=hypotheses_text,
                context=context,
                phase_name=phase_name,
                prior_debate=f"The Optimist argued:\n{optimist_response}",
                model=model
            )
            transcript_parts.append(f"\n[SKEPTIC]\n{skeptic_response}")
            
            # Round 3: Analyst synthesizes
            analyst_response = self._run_persona(
                persona="analyst",
                system_prompt=PERSONA_ANALYST,
                hypotheses=hypotheses_text,
                context=context,
                phase_name=phase_name,
                prior_debate=f"The Optimist argued:\n{optimist_response}\n\nThe Skeptic countered:\n{skeptic_response}",
                model=model
            )
            transcript_parts.append(f"\n[ANALYST]\n{analyst_response}")
            
            # Compile results
            result["debate_transcript"] = "\n".join(transcript_parts)
            
            # Parse analyst response for refinements
            parsed = self._parse_analyst_response(analyst_response, hypotheses)
            result["refined_hypotheses"] = parsed.get("refined_hypotheses", [])
            result["contradictions_found"] = parsed.get("contradictions", [])
            result["confidence_adjustments"] = parsed.get("confidence_adjustments", {})
            
        except Exception as e:
            logger.warning(f"Debate failed for phase '{phase_name}': {e}")
            result["debate_transcript"] = f"Debate skipped due to error: {e}"
        
        # Store in state
        if "debate" not in state.work_summary:
            state.work_summary["debate"] = {}
        state.work_summary["debate"][phase_name] = {
            "transcript_length": len(result["debate_transcript"]),
            "contradictions_count": len(result["contradictions_found"]),
            "adjustments_count": len(result["confidence_adjustments"]),
        }
        
        return result
    
    def _format_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> str:
        """Format hypotheses for debate prompt."""
        if not hypotheses:
            return ""
        
        parts = []
        for i, h in enumerate(hypotheses[:5], 1):  # Limit to 5 hypotheses
            statement = h.get("statement", "Unknown")
            confidence = h.get("confidence", 0.5)
            parts.append(f"{i}. {statement} (Confidence: {confidence:.2f})")
        
        return "\n".join(parts)
    
    def _run_persona(
        self,
        persona: str,
        system_prompt: str,
        hypotheses: str,
        context: str,
        phase_name: str,
        prior_debate: str,
        model: str
    ) -> str:
        """Run a single persona in the debate."""
        prompt = f"""Phase: {phase_name}

CURRENT HYPOTHESES:
{hypotheses}

CONTEXT FROM PHASE OUTPUT:
{context[:1500]}

"""
        if prior_debate:
            prompt += f"""PRIOR DEBATE:
{prior_debate[:1500]}

"""
        
        if persona == "optimist":
            prompt += "Defend these hypotheses. Highlight their strengths and propose ways to address any weaknesses."
        elif persona == "skeptic":
            prompt += "Challenge these hypotheses. What could go wrong? What assumptions are weak? What evidence is missing?"
        else:  # analyst
            prompt += """Synthesize the debate. For each hypothesis:
1. State if it should be STRENGTHENED, WEAKENED, or UNCHANGED
2. Provide a confidence adjustment (-0.2 to +0.2)
3. Note any contradictions found

Format your conclusions as:
HYPOTHESIS 1: [STRENGTHENED/WEAKENED/UNCHANGED] | Adjustment: [+/-0.X]
CONTRADICTIONS: [list any found]
REFINED STATEMENT: [if changed]"""

        response = self.model_pool.run_single(
            model_name=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6
        )
        
        return response if response else f"[{persona.upper()} provided no response]"
    
    def _parse_analyst_response(
        self,
        response: str,
        original_hypotheses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse the analyst's synthesis into structured output."""
        result = {
            "refined_hypotheses": [],
            "contradictions": [],
            "confidence_adjustments": {},
        }
        
        lines = response.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Look for hypothesis assessments
            if line_lower.startswith("hypothesis"):
                # Parse confidence adjustment
                if "adjustment:" in line_lower:
                    try:
                        adj_part = line.split("Adjustment:")[-1].strip()
                        adj_value = adj_part.split()[0].replace("+", "")
                        adjustment = float(adj_value)
                        
                        # Extract hypothesis number
                        import re
                        num_match = re.search(r'hypothesis\s*(\d+)', line_lower)
                        if num_match:
                            h_num = int(num_match.group(1)) - 1
                            if 0 <= h_num < len(original_hypotheses):
                                h_id = original_hypotheses[h_num].get("id", f"hyp_{h_num}")
                                result["confidence_adjustments"][h_id] = adjustment
                    except (ValueError, IndexError):
                        pass
            
            # Look for contradictions
            elif line_lower.startswith("contradiction"):
                contradiction = line.split(":", 1)[-1].strip()
                if contradiction and len(contradiction) > 5:
                    result["contradictions"].append(contradiction)
        
        # Create refined hypotheses
        for i, h in enumerate(original_hypotheses):
            refined = dict(h)
            h_id = h.get("id", f"hyp_{i}")
            if h_id in result["confidence_adjustments"]:
                adj = result["confidence_adjustments"][h_id]
                refined["confidence"] = max(0.0, min(1.0, h.get("confidence", 0.5) + adj))
            result["refined_hypotheses"].append(refined)
        
        return result

