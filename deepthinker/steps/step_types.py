"""
Step Data Types for DeepThinker 2.0 Step Engine.

Defines the core data structures for step-based execution within mission phases.
Steps are the atomic units of work executed by single specialized models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime


# Type alias for step status
StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]

# Type alias for step types
StepType = Literal["research", "analysis", "design", "coding", "testing", "synthesis", "meta"]


@dataclass
class StepExecutionContext:
    """
    Context provided to a step during execution.
    
    Contains all the information a step needs to understand its place
    in the mission and access relevant prior work.
    
    Attributes:
        mission_id: Unique identifier for the parent mission
        mission_objective: The overall mission objective
        phase_name: Name of the current phase
        phase_description: Description of the current phase
        step_index: Zero-based index of this step within the phase
        previous_steps_summary: Summary of outputs from previous steps in this phase
        shared_artifacts: Artifacts from previous phases/steps that may be relevant
        remaining_time_minutes: Time remaining until mission deadline
        constraints_notes: Any mission-level constraints (e.g., no internet, no code execution)
    """
    mission_id: str
    mission_objective: str
    phase_name: str
    phase_description: str
    step_index: int
    previous_steps_summary: str = ""
    shared_artifacts: Dict[str, str] = field(default_factory=dict)
    remaining_time_minutes: float = 60.0
    constraints_notes: str = ""


@dataclass
class StepResult:
    """
    Result from executing a single step.
    
    Captures the outcome, timing, and any suggested pivots or follow-up actions.
    
    Attributes:
        status: Final status of the step execution
        started_at: When execution began
        ended_at: When execution completed
        output: The main output/result from the step
        artifacts: Named artifacts produced by this step (e.g., code, summaries)
        notes: Execution notes and observations
        pivot_suggestion: Optional suggestion to modify the plan (e.g., "insert new step to X")
        error: Error message if status is "failed"
        model_used: Name of the model that executed this step
        attempts: Number of execution attempts made
    """
    status: StepStatus
    started_at: datetime
    ended_at: datetime
    output: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    pivot_suggestion: Optional[str] = None
    error: Optional[str] = None
    model_used: str = ""
    attempts: int = 1
    
    def duration_seconds(self) -> float:
        """Get execution duration in seconds."""
        return (self.ended_at - self.started_at).total_seconds()
    
    def is_success(self) -> bool:
        """Check if the step completed successfully."""
        return self.status == "completed"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "output": self.output[:2000] + "..." if len(self.output) > 2000 else self.output,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "pivot_suggestion": self.pivot_suggestion,
            "error": self.error,
            "model_used": self.model_used,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds()
        }


@dataclass
class StepDefinition:
    """
    Definition of a step to be executed within a phase.
    
    Steps are the atomic units of work in the Step Engine. Each step is
    executed by a single specialized model, not a council.
    
    Attributes:
        name: Short, descriptive name for the step
        description: Detailed description of what this step should accomplish
        step_type: Category of work (research, analysis, design, coding, testing, synthesis, meta)
        tools: List of tools this step may use (e.g., ["web", "code", "simulation"])
        preferred_model: Preferred model for this step (optional, defaults based on step_type)
        max_attempts: Maximum retry attempts before marking as failed
        status: Current execution status
        last_result: Result from the most recent execution attempt
        dependencies: Names of previous steps this step depends on (for ordering validation)
    """
    name: str
    description: str
    step_type: str = "research"  # research | analysis | design | coding | testing | synthesis | meta
    tools: List[str] = field(default_factory=list)
    preferred_model: Optional[str] = None
    max_attempts: int = 3
    status: StepStatus = "pending"
    last_result: Optional[StepResult] = None
    dependencies: List[str] = field(default_factory=list)
    
    def mark_running(self) -> None:
        """Mark this step as currently running."""
        self.status = "running"
    
    def mark_completed(self, result: StepResult) -> None:
        """Mark this step as completed with the given result."""
        self.status = "completed"
        self.last_result = result
    
    def mark_failed(self, result: StepResult) -> None:
        """Mark this step as failed with the given result."""
        self.status = "failed"
        self.last_result = result
    
    def mark_skipped(self, reason: str) -> None:
        """Mark this step as skipped."""
        self.status = "skipped"
        now = datetime.utcnow()
        self.last_result = StepResult(
            status="skipped",
            started_at=now,
            ended_at=now,
            output="",
            notes=[f"Skipped: {reason}"]
        )
    
    def is_terminal(self) -> bool:
        """Check if step is in a terminal state."""
        return self.status in ("completed", "failed", "skipped")
    
    def needs_execution(self) -> bool:
        """Check if step needs to be executed."""
        return self.status in ("pending", "running")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "step_type": self.step_type,
            "tools": self.tools,
            "preferred_model": self.preferred_model,
            "max_attempts": self.max_attempts,
            "status": self.status,
            "last_result": self.last_result.to_dict() if self.last_result else None,
            "dependencies": self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StepDefinition":
        """Create a StepDefinition from a dictionary."""
        return cls(
            name=data.get("name", "Unnamed Step"),
            description=data.get("description", ""),
            step_type=data.get("step_type", "research"),
            tools=data.get("tools", []),
            preferred_model=data.get("preferred_model"),
            max_attempts=data.get("max_attempts", 3),
            status=data.get("status", "pending"),
            dependencies=data.get("dependencies", [])
        )


@dataclass
class StepEvaluationResult:
    """
    Result from evaluating a step's output.
    
    Used by the optional evaluator reflection to assess step quality
    and suggest retries or pivots.
    
    Attributes:
        passed: Whether the step output meets quality standards
        quality_score: Numeric quality score (0-10)
        issues: List of identified issues
        recommendations: Suggestions for improvement
        pivot_suggestion: Optional suggestion to change the plan
    """
    passed: bool
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    pivot_suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "pivot_suggestion": self.pivot_suggestion
        }

