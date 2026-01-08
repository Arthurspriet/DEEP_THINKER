"""
Structured Mission State Storage for DeepThinker Memory System.

Persists complete cognitive state from the meta-cognition engine including
hypotheses, reflections, debates, plan revisions, and supervisor signals.

Storage path: kb/missions/<mission_id>/state.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    HypothesisSchema,
    PhaseOutputSchema,
    SupervisorSignalsSchema,
    ReflectionSchema,
    DebateSchema,
    PlanRevisionSchema,
    CouncilFeedbackSchema,
)

logger = logging.getLogger(__name__)


class StructuredMissionState:
    """
    Persists all meta-engine output for a mission.
    
    Provides structured storage for:
    - Hypotheses with evidence tracking
    - Supervisor signals (difficulty, uncertainty, progress, novelty)
    - Council feedback per phase
    - Meta-cognition results (reflections, debates, plan revisions)
    - Loop detection signals
    - Phase outputs and artifacts
    
    All operations are fail-safe - errors are logged but don't crash.
    """
    
    def __init__(
        self,
        mission_id: str,
        objective: str,
        mission_type: Optional[str] = None,
        time_budget_minutes: Optional[float] = None,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize structured mission state.
        
        Args:
            mission_id: Unique mission identifier
            objective: Mission objective text
            mission_type: Type of mission (research, coding, strategic, etc.)
            time_budget_minutes: Time budget for the mission
            base_dir: Base directory for storage (default: kb/)
        """
        self.mission_id = mission_id
        self.objective = objective
        self.mission_type = mission_type
        self.time_budget_minutes = time_budget_minutes
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Storage configuration
        self.base_dir = base_dir or Path("kb")
        self._state_path = self.base_dir / "missions" / mission_id / "state.json"
        
        # Hypotheses storage
        self.hypotheses: Dict[str, HypothesisSchema] = {}
        
        # Supervisor signals
        self.supervisor_signals = SupervisorSignalsSchema()
        
        # Council feedback
        self.council_feedback: List[CouncilFeedbackSchema] = []
        
        # Meta-cognition storage per phase
        self.meta: Dict[str, Any] = {
            "reflections": {},      # phase_name -> ReflectionSchema
            "debates": {},          # phase_name -> DebateSchema
            "plan_revisions": {},   # phase_name -> PlanRevisionSchema
            "contradictions": {},   # phase_name -> List[str]
        }
        
        # Loop detection signals
        self.loop_signals: Dict[str, Any] = {
            "novelty_scores": [],
            "repetition_hashes": [],
            "warnings": [],
            "loop_detected": False,
        }
        
        # Phase outputs
        self.phase_outputs: Dict[str, PhaseOutputSchema] = {}
        
        # Domain and tags (for retrieval)
        self.domain: Optional[str] = None
        self.tags: List[str] = []
        
        # Iteration tracking
        self.iteration_count: int = 0
        self.total_phases_completed: int = 0
    
    # =========================================================================
    # Hypothesis Management
    # =========================================================================
    
    def add_hypothesis(self, hypothesis: HypothesisSchema) -> None:
        """Add or update a hypothesis."""
        hypothesis.updated_at = datetime.utcnow()
        if hypothesis.created_at is None:
            hypothesis.created_at = hypothesis.updated_at
        self.hypotheses[hypothesis.id] = hypothesis
        self._mark_updated()
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[HypothesisSchema]:
        """Get a hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)
    
    def update_hypothesis_confidence(
        self,
        hypothesis_id: str,
        confidence: float,
        evidence_id: Optional[str] = None,
        is_contradiction: bool = False
    ) -> bool:
        """
        Update hypothesis confidence with optional evidence.
        
        Args:
            hypothesis_id: Hypothesis to update
            confidence: New confidence value (0-1)
            evidence_id: Optional evidence supporting this update
            is_contradiction: Whether the evidence contradicts the hypothesis
            
        Returns:
            True if hypothesis was found and updated
        """
        if hypothesis_id not in self.hypotheses:
            return False
        
        h = self.hypotheses[hypothesis_id]
        h.confidence = max(0.0, min(1.0, confidence))
        h.updated_at = datetime.utcnow()
        
        if evidence_id:
            if is_contradiction:
                if evidence_id not in h.contradiction_ids:
                    h.contradiction_ids.append(evidence_id)
            else:
                if evidence_id not in h.evidence_ids:
                    h.evidence_ids.append(evidence_id)
        
        # Update status based on confidence
        if h.confidence < 0.2:
            h.status = "rejected"
        elif h.confidence > 0.8:
            h.status = "confirmed"
        else:
            h.status = "active"
        
        self._mark_updated()
        return True
    
    def get_active_hypotheses(self) -> List[HypothesisSchema]:
        """Get all active hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == "active"]
    
    def get_confirmed_hypotheses(self) -> List[HypothesisSchema]:
        """Get all confirmed hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == "confirmed"]
    
    def get_rejected_hypotheses(self) -> List[HypothesisSchema]:
        """Get all rejected hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == "rejected"]
    
    # =========================================================================
    # Supervisor Signals
    # =========================================================================
    
    def record_supervisor_signals(
        self,
        difficulty: float,
        uncertainty: float,
        progress: float,
        novelty: float,
        confidence: float,
        model_tier: str = "medium",
        convergence: Optional[float] = None,
    ) -> None:
        """Record supervisor signals from a phase analysis."""
        self.supervisor_signals.add_phase_metrics(
            difficulty=difficulty,
            uncertainty=uncertainty,
            progress=progress,
            novelty=novelty,
            confidence=confidence,
            model_tier=model_tier,
        )
        if convergence is not None:
            self.supervisor_signals.convergence_scores.append(convergence)
        self._mark_updated()
    
    def record_loop_detection(
        self,
        loop_detected: bool,
        stagnation_count: int = 0,
        warning: Optional[str] = None
    ) -> None:
        """Record loop detection signals."""
        self.supervisor_signals.loop_detected = loop_detected
        self.supervisor_signals.stagnation_count = stagnation_count
        self.loop_signals["loop_detected"] = loop_detected
        if warning:
            self.loop_signals["warnings"].append(warning)
        self._mark_updated()
    
    # =========================================================================
    # Council Feedback
    # =========================================================================
    
    def add_council_feedback(self, feedback: CouncilFeedbackSchema) -> None:
        """Add council execution feedback."""
        if feedback.created_at is None:
            feedback.created_at = datetime.utcnow()
        self.council_feedback.append(feedback)
        self._mark_updated()
    
    # =========================================================================
    # Meta-Cognition Records
    # =========================================================================
    
    def add_reflection(self, reflection: ReflectionSchema) -> None:
        """Add reflection for a phase."""
        if reflection.created_at is None:
            reflection.created_at = datetime.utcnow()
        self.meta["reflections"][reflection.phase_name] = reflection
        self._mark_updated()
    
    def add_debate(self, debate: DebateSchema) -> None:
        """Add debate results for a phase."""
        if debate.created_at is None:
            debate.created_at = datetime.utcnow()
        self.meta["debates"][debate.phase_name] = debate
        
        # Track contradictions separately for easy access
        if debate.contradictions_found:
            existing = self.meta["contradictions"].get(debate.phase_name, [])
            existing.extend(debate.contradictions_found)
            self.meta["contradictions"][debate.phase_name] = existing
        
        self._mark_updated()
    
    def add_plan_revision(self, revision: PlanRevisionSchema) -> None:
        """Add plan revision for a phase."""
        if revision.created_at is None:
            revision.created_at = datetime.utcnow()
        self.meta["plan_revisions"][revision.phase_name] = revision
        self._mark_updated()
    
    def get_reflection(self, phase_name: str) -> Optional[ReflectionSchema]:
        """Get reflection for a phase."""
        data = self.meta["reflections"].get(phase_name)
        if data is None:
            return None
        if isinstance(data, ReflectionSchema):
            return data
        return ReflectionSchema(**data)
    
    def get_debate(self, phase_name: str) -> Optional[DebateSchema]:
        """Get debate for a phase."""
        data = self.meta["debates"].get(phase_name)
        if data is None:
            return None
        if isinstance(data, DebateSchema):
            return data
        return DebateSchema(**data)
    
    def get_all_contradictions(self) -> List[str]:
        """Get all contradictions found across phases."""
        all_contradictions = []
        for phase_contradictions in self.meta["contradictions"].values():
            all_contradictions.extend(phase_contradictions)
        return all_contradictions
    
    # =========================================================================
    # Phase Outputs
    # =========================================================================
    
    def add_phase_output(self, output: PhaseOutputSchema) -> None:
        """Add phase output."""
        self.phase_outputs[output.phase_name] = output
        self.total_phases_completed = len(self.phase_outputs)
        self._mark_updated()
    
    def get_phase_output(self, phase_name: str) -> Optional[PhaseOutputSchema]:
        """Get phase output."""
        data = self.phase_outputs.get(phase_name)
        if data is None:
            return None
        if isinstance(data, PhaseOutputSchema):
            return data
        return PhaseOutputSchema(**data)
    
    # =========================================================================
    # Loop Signals
    # =========================================================================
    
    def add_novelty_score(self, score: float) -> None:
        """Add novelty score for loop detection."""
        self.loop_signals["novelty_scores"].append(score)
        self._mark_updated()
    
    def add_repetition_hash(self, hash_value: str) -> None:
        """Add output hash for repetition detection."""
        self.loop_signals["repetition_hashes"].append(hash_value)
        # Keep only last 20 hashes
        if len(self.loop_signals["repetition_hashes"]) > 20:
            self.loop_signals["repetition_hashes"] = self.loop_signals["repetition_hashes"][-20:]
        self._mark_updated()
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "mission_type": self.mission_type,
            "time_budget_minutes": self.time_budget_minutes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "domain": self.domain,
            "tags": self.tags,
            "iteration_count": self.iteration_count,
            "total_phases_completed": self.total_phases_completed,
            
            # Hypotheses
            "hypotheses": {
                k: v.model_dump() for k, v in self.hypotheses.items()
            },
            
            # Supervisor signals
            "supervisor_signals": self.supervisor_signals.model_dump(),
            
            # Council feedback
            "council_feedback": [f.model_dump() for f in self.council_feedback],
            
            # Meta-cognition
            "meta": {
                "reflections": {
                    k: v.model_dump() if isinstance(v, ReflectionSchema) else v
                    for k, v in self.meta["reflections"].items()
                },
                "debates": {
                    k: v.model_dump() if isinstance(v, DebateSchema) else v
                    for k, v in self.meta["debates"].items()
                },
                "plan_revisions": {
                    k: v.model_dump() if isinstance(v, PlanRevisionSchema) else v
                    for k, v in self.meta["plan_revisions"].items()
                },
                "contradictions": self.meta["contradictions"],
            },
            
            # Loop signals
            "loop_signals": self.loop_signals,
            
            # Phase outputs
            "phase_outputs": {
                k: v.model_dump() if isinstance(v, PhaseOutputSchema) else v
                for k, v in self.phase_outputs.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "StructuredMissionState":
        """Create state from dictionary."""
        state = cls(
            mission_id=data["mission_id"],
            objective=data["objective"],
            mission_type=data.get("mission_type"),
            time_budget_minutes=data.get("time_budget_minutes"),
            base_dir=base_dir,
        )
        
        # Restore timestamps
        if data.get("created_at"):
            state.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            state.updated_at = datetime.fromisoformat(data["updated_at"])
        
        # Restore metadata
        state.domain = data.get("domain")
        state.tags = data.get("tags", [])
        state.iteration_count = data.get("iteration_count", 0)
        state.total_phases_completed = data.get("total_phases_completed", 0)
        
        # Restore hypotheses
        for k, v in data.get("hypotheses", {}).items():
            state.hypotheses[k] = HypothesisSchema(**v)
        
        # Restore supervisor signals
        if data.get("supervisor_signals"):
            state.supervisor_signals = SupervisorSignalsSchema(**data["supervisor_signals"])
        
        # Restore council feedback
        for f in data.get("council_feedback", []):
            state.council_feedback.append(CouncilFeedbackSchema(**f))
        
        # Restore meta-cognition
        meta = data.get("meta", {})
        for k, v in meta.get("reflections", {}).items():
            state.meta["reflections"][k] = ReflectionSchema(**v) if isinstance(v, dict) else v
        for k, v in meta.get("debates", {}).items():
            state.meta["debates"][k] = DebateSchema(**v) if isinstance(v, dict) else v
        for k, v in meta.get("plan_revisions", {}).items():
            state.meta["plan_revisions"][k] = PlanRevisionSchema(**v) if isinstance(v, dict) else v
        state.meta["contradictions"] = meta.get("contradictions", {})
        
        # Restore loop signals
        state.loop_signals = data.get("loop_signals", state.loop_signals)
        
        # Restore phase outputs
        for k, v in data.get("phase_outputs", {}).items():
            state.phase_outputs[k] = PhaseOutputSchema(**v) if isinstance(v, dict) else v
        
        return state
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self) -> bool:
        """
        Save state to disk.
        
        Returns:
            True if save succeeded, False otherwise
        """
        try:
            # Ensure directory exists
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write state
            with open(self._state_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            
            logger.debug(f"Saved mission state to {self._state_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save mission state: {e}")
            return False
    
    @classmethod
    def load(
        cls,
        mission_id: str,
        base_dir: Optional[Path] = None
    ) -> Optional["StructuredMissionState"]:
        """
        Load state from disk.
        
        Args:
            mission_id: Mission ID to load
            base_dir: Base directory for storage
            
        Returns:
            StructuredMissionState if found, None otherwise
        """
        try:
            base = base_dir or Path("kb")
            state_path = base / "missions" / mission_id / "state.json"
            
            if not state_path.exists():
                return None
            
            with open(state_path, "r") as f:
                data = json.load(f)
            
            return cls.from_dict(data, base_dir=base)
            
        except Exception as e:
            logger.warning(f"Failed to load mission state for {mission_id}: {e}")
            return None
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _mark_updated(self) -> None:
        """Mark state as updated."""
        self.updated_at = datetime.utcnow()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the mission state."""
        return {
            "hypotheses_active": len(self.get_active_hypotheses()),
            "hypotheses_confirmed": len(self.get_confirmed_hypotheses()),
            "hypotheses_rejected": len(self.get_rejected_hypotheses()),
            "phases_completed": self.total_phases_completed,
            "reflections_count": len(self.meta["reflections"]),
            "debates_count": len(self.meta["debates"]),
            "contradictions_count": len(self.get_all_contradictions()),
            "avg_difficulty": self.supervisor_signals.avg_difficulty,
            "avg_uncertainty": self.supervisor_signals.avg_uncertainty,
            "loop_detected": self.loop_signals.get("loop_detected", False),
        }

