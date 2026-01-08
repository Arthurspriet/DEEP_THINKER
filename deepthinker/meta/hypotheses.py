"""
Hypothesis Manager for Meta-Cognition.

Manages a dynamic DAG of reasoning hypotheses that evolve
as the mission progresses through phases.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..models.model_pool import ModelPool
    from ..missions.mission_types import MissionState

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A single hypothesis in the reasoning graph."""
    id: str
    statement: str
    confidence: float = 0.5  # 0.0 to 1.0
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)  # For DAG structure
    status: str = "active"  # active, rejected, confirmed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "parent_ids": self.parent_ids,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            statement=data.get("statement", ""),
            confidence=data.get("confidence", 0.5),
            evidence_for=data.get("evidence_for", []),
            evidence_against=data.get("evidence_against", []),
            parent_ids=data.get("parent_ids", []),
            status=data.get("status", "active"),
        )


class HypothesisManager:
    """
    Manages hypotheses throughout a mission's lifecycle.
    
    Maintains a dynamic DAG of reasoning where:
    - Hypotheses can be generated, updated, rejected, or strengthened
    - Evidence accumulates across phases
    - Confidence scores evolve based on new information
    """
    
    def __init__(self, model_pool: "ModelPool"):
        """
        Initialize the hypothesis manager.
        
        Args:
            model_pool: Model pool for LLM calls
        """
        self.model_pool = model_pool
        self._hypothesis_counter = 0
        self._system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for hypothesis generation."""
        return """You are a hypothesis generator for a reasoning system.

Given an objective or analysis results, generate clear, testable hypotheses.

Each hypothesis should:
1. Be a specific, falsifiable statement
2. Be relevant to the mission objective
3. Be stated clearly and concisely

Output hypotheses in the following format:
HYPOTHESIS: [statement]
CONFIDENCE: [0.0-1.0]
REASONING: [brief reasoning for this hypothesis]

---

Generate 2-4 relevant hypotheses."""
    
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
    
    def _ensure_state_initialized(self, state: "MissionState") -> None:
        """Ensure hypothesis storage is initialized in state."""
        if not hasattr(state, 'hypotheses') or not state.hypotheses:
            state.hypotheses = {
                "active": [],
                "rejected": [],
                "evidence": {},
                "confidence": {},
            }
    
    def _generate_id(self) -> str:
        """Generate a unique hypothesis ID."""
        self._hypothesis_counter += 1
        return f"hyp_{self._hypothesis_counter}"
    
    def generate_initial_hypotheses(
        self,
        objective: str,
        state: "MissionState"
    ) -> List[Hypothesis]:
        """
        Generate initial hypotheses based on the mission objective.
        
        Args:
            objective: The mission objective
            state: Current mission state
            
        Returns:
            List of generated hypotheses
        """
        self._ensure_state_initialized(state)
        hypotheses = []
        
        try:
            prompt = f"""Generate initial hypotheses for this mission objective:

OBJECTIVE:
{objective}

Consider:
- What are the key assumptions we're making?
- What outcomes are we expecting?
- What factors might affect success?
- What alternative approaches might work?

Generate 2-4 testable hypotheses."""

            model = self._get_smallest_model()
            response = self.model_pool.run_single(
                model_name=model,
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=0.5
            )
            
            if response:
                hypotheses = self._parse_hypotheses(response)
                
                # Store in state
                for h in hypotheses:
                    state.hypotheses["active"].append(h.to_dict())
                    state.hypotheses["confidence"][h.id] = h.confidence
                    
        except Exception as e:
            logger.warning(f"Failed to generate initial hypotheses: {e}")
        
        return hypotheses
    
    def update_hypotheses(
        self,
        reflection: Dict[str, List[str]],
        council_output: Any,
        state: "MissionState"
    ) -> List[Hypothesis]:
        """
        Update hypotheses based on reflection results and council output.
        
        Args:
            reflection: Reflection analysis from ReflectionEngine
            council_output: Output from the phase's council
            state: Current mission state
            
        Returns:
            Updated list of active hypotheses
        """
        self._ensure_state_initialized(state)
        
        try:
            # Extract evidence from reflection
            evidence_for = reflection.get("suggestions", [])
            evidence_against = reflection.get("contradictions", []) + reflection.get("weaknesses", [])
            
            # Update existing hypotheses with new evidence
            active_hypotheses = []
            for h_dict in state.hypotheses.get("active", []):
                h = Hypothesis.from_dict(h_dict)
                
                # Add new evidence
                for evidence in evidence_for:
                    if evidence not in h.evidence_for:
                        h.evidence_for.append(evidence)
                        h.confidence = min(1.0, h.confidence + 0.05)
                
                for evidence in evidence_against:
                    if evidence not in h.evidence_against:
                        h.evidence_against.append(evidence)
                        h.confidence = max(0.0, h.confidence - 0.1)
                
                active_hypotheses.append(h)
                state.hypotheses["confidence"][h.id] = h.confidence
            
            # Generate new hypotheses from questions and missing info
            questions = reflection.get("questions", [])
            missing_info = reflection.get("missing_info", [])
            
            if questions or missing_info:
                new_hypotheses = self._generate_from_gaps(
                    questions=questions,
                    missing_info=missing_info,
                    state=state
                )
                active_hypotheses.extend(new_hypotheses)
            
            # Update state
            state.hypotheses["active"] = [h.to_dict() for h in active_hypotheses]
            
            return active_hypotheses
            
        except Exception as e:
            logger.warning(f"Failed to update hypotheses: {e}")
            return [Hypothesis.from_dict(h) for h in state.hypotheses.get("active", [])]
    
    def _generate_from_gaps(
        self,
        questions: List[str],
        missing_info: List[str],
        state: "MissionState"
    ) -> List[Hypothesis]:
        """Generate new hypotheses from identified gaps."""
        hypotheses = []
        
        if not questions and not missing_info:
            return hypotheses
        
        try:
            gaps_text = ""
            if questions:
                gaps_text += "QUESTIONS:\n" + "\n".join(f"- {q}" for q in questions[:3])
            if missing_info:
                gaps_text += "\n\nMISSING INFO:\n" + "\n".join(f"- {m}" for m in missing_info[:3])
            
            prompt = f"""Based on these identified gaps in our analysis, generate hypotheses:

{gaps_text}

Generate 1-2 hypotheses that could help address these gaps."""

            model = self._get_smallest_model()
            response = self.model_pool.run_single(
                model_name=model,
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=0.5
            )
            
            if response:
                hypotheses = self._parse_hypotheses(response)
                
                # Store new hypotheses
                for h in hypotheses:
                    state.hypotheses["active"].append(h.to_dict())
                    state.hypotheses["confidence"][h.id] = h.confidence
                    
        except Exception as e:
            logger.warning(f"Failed to generate hypotheses from gaps: {e}")
        
        return hypotheses
    
    def reject_or_strengthen(self, state: "MissionState") -> Dict[str, List[str]]:
        """
        Evaluate hypotheses and reject weak ones or strengthen confident ones.
        
        Args:
            state: Current mission state
            
        Returns:
            Dictionary with lists of rejected and strengthened hypothesis IDs
        """
        self._ensure_state_initialized(state)
        
        rejected = []
        strengthened = []
        updated_active = []
        
        for h_dict in state.hypotheses.get("active", []):
            h = Hypothesis.from_dict(h_dict)
            
            # Reject hypotheses with very low confidence
            if h.confidence < 0.2:
                h.status = "rejected"
                state.hypotheses["rejected"].append(h.to_dict())
                rejected.append(h.id)
            # Mark high-confidence hypotheses
            elif h.confidence > 0.8:
                h.status = "confirmed"
                strengthened.append(h.id)
                updated_active.append(h.to_dict())
            else:
                updated_active.append(h.to_dict())
        
        state.hypotheses["active"] = updated_active
        
        return {
            "rejected": rejected,
            "strengthened": strengthened,
        }
    
    def export_summary(self, state: "MissionState") -> Dict[str, Any]:
        """
        Export a summary of the current hypothesis state.
        
        Args:
            state: Current mission state
            
        Returns:
            Summary dictionary
        """
        self._ensure_state_initialized(state)
        
        active = state.hypotheses.get("active", [])
        rejected = state.hypotheses.get("rejected", [])
        
        return {
            "total_active": len(active),
            "total_rejected": len(rejected),
            "active_hypotheses": [
                {
                    "id": h.get("id"),
                    "statement": h.get("statement", "")[:100],
                    "confidence": h.get("confidence", 0.5),
                }
                for h in active[:5]  # Top 5
            ],
            "average_confidence": (
                sum(h.get("confidence", 0.5) for h in active) / len(active)
                if active else 0.0
            ),
            "high_confidence_count": sum(
                1 for h in active if h.get("confidence", 0.5) > 0.7
            ),
        }
    
    def _parse_hypotheses(self, response: str) -> List[Hypothesis]:
        """Parse LLM response into hypothesis objects."""
        hypotheses = []
        
        lines = response.split("\n")
        current_statement = None
        current_confidence = 0.5
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            if line_lower.startswith("hypothesis:"):
                # Save previous hypothesis if exists
                if current_statement:
                    hypotheses.append(Hypothesis(
                        id=self._generate_id(),
                        statement=current_statement,
                        confidence=current_confidence,
                    ))
                
                # Start new hypothesis
                current_statement = line.split(":", 1)[1].strip()
                current_confidence = 0.5
                
            elif line_lower.startswith("confidence:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    # Handle various formats: "0.7", "0.7/1.0", "70%"
                    conf_str = conf_str.replace("%", "").split("/")[0].strip()
                    current_confidence = float(conf_str)
                    if current_confidence > 1.0:
                        current_confidence = current_confidence / 100.0
                except (ValueError, IndexError):
                    pass
        
        # Don't forget the last hypothesis
        if current_statement:
            hypotheses.append(Hypothesis(
                id=self._generate_id(),
                statement=current_statement,
                confidence=current_confidence,
            ))
        
        return hypotheses

