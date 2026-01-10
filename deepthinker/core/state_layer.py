"""
State Abstraction Layer for DeepThinker.

Provides a unified interface for state management across both orchestration
paths with pluggable backends:
- TransientBackend: In-memory state (workflow path)
- PersistentBackend: Disk-persisted state (mission path)

The abstraction provides:
- Common iteration context and history format
- Unified event system for SSE
- Backend-agnostic state operations
- Automatic serialization/deserialization

Usage:
    from deepthinker.core.state_layer import state_layer, StateBackend
    
    # Initialize with transient backend (workflow)
    state_layer.initialize(backend="transient")
    
    # Or with persistent backend (mission)
    state_layer.initialize(backend="persistent", state_dir=".deepthinker_state")
    
    # Create a session
    session_id = state_layer.create_session(
        objective="Build a classifier",
        mode="workflow"
    )
    
    # Track iterations
    state_layer.start_iteration(session_id, 1)
    state_layer.record_output(session_id, "iteration", {...})
    state_layer.complete_iteration(session_id, quality_score=7.5)
"""

import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import queue

logger = logging.getLogger(__name__)


# =============================================================================
# Core Types
# =============================================================================

class SessionMode(str, Enum):
    """Session execution mode."""
    WORKFLOW = "workflow"  # Simple iterative workflow
    MISSION = "mission"    # Multi-phase mission


class SessionStatus(str, Enum):
    """Session status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    EXPIRED = "expired"


class PhaseStatus(str, Enum):
    """Phase status within a session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IterationRecord:
    """
    Record of a single iteration.
    
    Common format used by both orchestration paths.
    """
    
    iteration: int
    started_at: str
    completed_at: Optional[str] = None
    
    # Outputs
    code: Optional[str] = None
    output: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    quality_score: Optional[float] = None
    passed: bool = False
    issues: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tracing
    agent_traces: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PhaseRecord:
    """
    Record of a phase (for mission mode).
    """
    
    name: str
    phase_type: str
    status: str = PhaseStatus.PENDING.value
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Outputs
    artifacts: Dict[str, Any] = field(default_factory=dict)
    output_summary: Optional[str] = None
    
    # Iterations within phase
    iterations: List[IterationRecord] = field(default_factory=list)
    
    # Metrics
    duration_seconds: Optional[float] = None
    council_rounds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["iterations"] = [it.to_dict() for it in self.iterations]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseRecord":
        """Create from dictionary."""
        iterations = [
            IterationRecord.from_dict(it) 
            for it in data.pop("iterations", [])
        ]
        record = cls(**data)
        record.iterations = iterations
        return record


@dataclass
class SessionState:
    """
    Complete state of an execution session.
    
    Works for both workflow and mission modes.
    """
    
    session_id: str
    mode: str  # SessionMode value
    status: str = SessionStatus.PENDING.value
    
    # Core info
    objective: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Constraints
    max_iterations: int = 5
    quality_threshold: float = 7.0
    timeout_minutes: Optional[int] = None
    deadline_at: Optional[str] = None
    
    # Current state
    current_iteration: int = 0
    current_phase_index: int = 0
    current_agent: Optional[str] = None
    
    # Records
    iterations: List[IterationRecord] = field(default_factory=list)
    phases: List[PhaseRecord] = field(default_factory=list)
    
    # Outputs
    final_output: Optional[str] = None
    final_artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    error: Optional[str] = None
    failure_reason: Optional[str] = None
    
    # Logs and events
    logs: List[str] = field(default_factory=list)
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["iterations"] = [it.to_dict() for it in self.iterations]
        data["phases"] = [ph.to_dict() for ph in self.phases]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        iterations = [
            IterationRecord.from_dict(it) 
            for it in data.pop("iterations", [])
        ]
        phases = [
            PhaseRecord.from_dict(ph) 
            for ph in data.pop("phases", [])
        ]
        state = cls(**data)
        state.iterations = iterations
        state.phases = phases
        return state
    
    def add_log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the log."""
        self.event_log.append({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    
    def current_iteration_record(self) -> Optional[IterationRecord]:
        """Get the current iteration record."""
        if not self.iterations:
            return None
        return self.iterations[-1]
    
    def current_phase_record(self) -> Optional[PhaseRecord]:
        """Get the current phase record."""
        if not self.phases or self.current_phase_index >= len(self.phases):
            return None
        return self.phases[self.current_phase_index]


# =============================================================================
# State Backend Interface
# =============================================================================

class StateBackend(ABC):
    """
    Abstract backend for state storage.
    """
    
    @abstractmethod
    def save(self, session_id: str, state: SessionState) -> None:
        """Save session state."""
        pass
    
    @abstractmethod
    def load(self, session_id: str) -> Optional[SessionState]:
        """Load session state."""
        pass
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        pass
    
    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete session state."""
        pass
    
    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        pass
    
    @abstractmethod
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Cleanup old sessions. Returns count deleted."""
        pass


class TransientBackend(StateBackend):
    """
    In-memory state storage.
    
    State is lost when the process exits.
    Suitable for workflow path.
    """
    
    def __init__(self, max_sessions: int = 100):
        self._sessions: Dict[str, SessionState] = {}
        self._max_sessions = max_sessions
        self._lock = threading.RLock()
    
    def save(self, session_id: str, state: SessionState) -> None:
        """Save session state."""
        with self._lock:
            self._sessions[session_id] = state
            self._enforce_limit()
    
    def load(self, session_id: str) -> Optional[SessionState]:
        """Load session state."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        with self._lock:
            return session_id in self._sessions
    
    def delete(self, session_id: str) -> None:
        """Delete session state."""
        with self._lock:
            self._sessions.pop(session_id, None)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        with self._lock:
            return list(self._sessions.keys())
    
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Cleanup old sessions."""
        with self._lock:
            cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
            to_delete = []
            
            for session_id, state in self._sessions.items():
                try:
                    created = datetime.fromisoformat(state.created_at)
                    if created.timestamp() < cutoff:
                        to_delete.append(session_id)
                except Exception:
                    pass
            
            for session_id in to_delete:
                del self._sessions[session_id]
            
            return len(to_delete)
    
    def _enforce_limit(self) -> None:
        """Enforce max session limit by removing oldest."""
        while len(self._sessions) > self._max_sessions:
            # Remove oldest
            oldest_id = min(
                self._sessions.keys(),
                key=lambda sid: self._sessions[sid].created_at
            )
            del self._sessions[oldest_id]


class PersistentBackend(StateBackend):
    """
    Disk-persisted state storage.
    
    State survives process restarts.
    Suitable for mission path.
    """
    
    def __init__(self, state_dir: str = ".deepthinker_state"):
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, SessionState] = {}
        self._lock = threading.RLock()
    
    def _get_path(self, session_id: str) -> Path:
        """Get file path for session."""
        return self._state_dir / f"{session_id}.json"
    
    def save(self, session_id: str, state: SessionState) -> None:
        """Save session state."""
        with self._lock:
            path = self._get_path(session_id)
            data = state.to_dict()
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self._cache[session_id] = state
    
    def load(self, session_id: str) -> Optional[SessionState]:
        """Load session state."""
        with self._lock:
            # Check cache first
            if session_id in self._cache:
                return self._cache[session_id]
            
            path = self._get_path(session_id)
            if not path.exists():
                return None
            
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                state = SessionState.from_dict(data)
                self._cache[session_id] = state
                return state
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return None
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self._get_path(session_id).exists()
    
    def delete(self, session_id: str) -> None:
        """Delete session state."""
        with self._lock:
            path = self._get_path(session_id)
            if path.exists():
                path.unlink()
            self._cache.pop(session_id, None)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        sessions = []
        for path in self._state_dir.glob("*.json"):
            sessions.append(path.stem)
        return sessions
    
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Cleanup old sessions."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        deleted = 0
        
        for session_id in self.list_sessions():
            state = self.load(session_id)
            if state:
                try:
                    created = datetime.fromisoformat(state.created_at)
                    if created.timestamp() < cutoff:
                        self.delete(session_id)
                        deleted += 1
                except Exception:
                    pass
        
        return deleted


# =============================================================================
# State Layer Manager
# =============================================================================

class StateLayer:
    """
    Unified state management layer.
    
    Provides backend-agnostic state operations with event broadcasting.
    """
    
    def __init__(self):
        self._backend: Optional[StateBackend] = None
        self._current_session_id: Optional[str] = None
        self._lock = threading.RLock()
        
        # Event broadcasting
        self._event_queues: List[queue.Queue] = []
        self._event_lock = threading.Lock()
        
        # Callbacks
        self._state_callbacks: List[Callable[[str, SessionState], None]] = []
    
    def initialize(
        self,
        backend: str = "transient",
        state_dir: str = ".deepthinker_state",
        max_sessions: int = 100
    ) -> None:
        """
        Initialize the state layer with a backend.
        
        Args:
            backend: "transient" or "persistent"
            state_dir: Directory for persistent storage
            max_sessions: Max sessions for transient backend
        """
        if backend == "transient":
            self._backend = TransientBackend(max_sessions=max_sessions)
            logger.info("StateLayer initialized with transient backend")
        elif backend == "persistent":
            self._backend = PersistentBackend(state_dir=state_dir)
            logger.info(f"StateLayer initialized with persistent backend at {state_dir}")
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @property
    def backend(self) -> StateBackend:
        """Get the current backend."""
        if self._backend is None:
            # Auto-initialize with transient
            self.initialize("transient")
        return self._backend
    
    def create_session(
        self,
        objective: str,
        mode: str = SessionMode.WORKFLOW.value,
        max_iterations: int = 5,
        quality_threshold: float = 7.0,
        timeout_minutes: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new execution session.
        
        Args:
            objective: Session objective
            mode: "workflow" or "mission"
            max_iterations: Maximum iterations
            quality_threshold: Quality threshold to stop
            timeout_minutes: Optional timeout
            context: Optional context dictionary
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        deadline_at = None
        if timeout_minutes:
            deadline_at = (
                datetime.now() + 
                __import__('datetime').timedelta(minutes=timeout_minutes)
            ).isoformat()
        
        state = SessionState(
            session_id=session_id,
            mode=mode,
            objective=objective,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            timeout_minutes=timeout_minutes,
            deadline_at=deadline_at,
            context=context or {}
        )
        
        with self._lock:
            self.backend.save(session_id, state)
            self._current_session_id = session_id
        
        self._broadcast_event("session_created", {
            "session_id": session_id,
            "mode": mode,
            "objective": objective
        })
        
        return session_id
    
    def start_session(self, session_id: Optional[str] = None) -> None:
        """Mark session as running."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if state:
                state.status = SessionStatus.RUNNING.value
                state.started_at = datetime.now().isoformat()
                state.add_log("Session started")
                self.backend.save(session_id, state)
        
        self._broadcast_event("session_started", {"session_id": session_id})
    
    def start_iteration(
        self, 
        session_id: Optional[str] = None,
        iteration: Optional[int] = None
    ) -> int:
        """
        Start a new iteration.
        
        Returns:
            Iteration number
        """
        session_id = session_id or self._current_session_id
        if not session_id:
            return 0
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state:
                return 0
            
            iteration = iteration or (state.current_iteration + 1)
            state.current_iteration = iteration
            
            record = IterationRecord(
                iteration=iteration,
                started_at=datetime.now().isoformat()
            )
            state.iterations.append(record)
            state.add_log(f"Iteration {iteration} started")
            
            self.backend.save(session_id, state)
        
        self._broadcast_event("iteration_started", {
            "session_id": session_id,
            "iteration": iteration
        })
        
        return iteration
    
    def record_output(
        self,
        session_id: Optional[str] = None,
        output_type: str = "code",
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an output for the current iteration.
        
        Args:
            session_id: Session ID
            output_type: Type of output ("code", "evaluation", "artifact")
            output: Output content
            metadata: Additional metadata
        """
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state or not state.iterations:
                return
            
            record = state.iterations[-1]
            
            if output_type == "code":
                record.code = str(output)
            elif output_type == "output":
                record.output = str(output)
            elif output_type == "artifact":
                artifact_name = metadata.get("name", "artifact") if metadata else "artifact"
                record.artifacts[artifact_name] = output
            else:
                record.artifacts[output_type] = output
            
            self.backend.save(session_id, state)
    
    def record_agent_trace(
        self,
        session_id: Optional[str] = None,
        agent_name: str = "",
        phase: str = "",
        input_text: str = "",
        output_text: str = "",
        duration_seconds: Optional[float] = None,
        llm_metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Record an agent execution trace."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state or not state.iterations:
                return
            
            record = state.iterations[-1]
            record.agent_traces.append({
                "agent_name": agent_name,
                "phase": phase,
                "timestamp": datetime.now().isoformat(),
                "input_preview": input_text[:500] if input_text else "",
                "output_preview": output_text[:500] if output_text else "",
                "duration_seconds": duration_seconds,
                "llm_metrics": llm_metrics or {},
                "error": error
            })
            
            state.current_agent = agent_name
            self.backend.save(session_id, state)
        
        self._broadcast_event("agent_executed", {
            "session_id": session_id,
            "agent_name": agent_name,
            "phase": phase,
            "duration_seconds": duration_seconds
        })
    
    def complete_iteration(
        self,
        session_id: Optional[str] = None,
        quality_score: Optional[float] = None,
        passed: bool = False,
        issues: Optional[List[Dict[str, str]]] = None,
        recommendations: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete the current iteration with evaluation results."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state or not state.iterations:
                return
            
            record = state.iterations[-1]
            record.completed_at = datetime.now().isoformat()
            record.quality_score = quality_score
            record.passed = passed
            record.issues = issues or []
            record.recommendations = recommendations or []
            record.metrics = metrics or {}
            
            state.add_log(
                f"Iteration {record.iteration} completed: "
                f"score={quality_score}, passed={passed}"
            )
            
            self.backend.save(session_id, state)
        
        self._broadcast_event("iteration_completed", {
            "session_id": session_id,
            "iteration": record.iteration,
            "quality_score": quality_score,
            "passed": passed
        })
    
    def start_phase(
        self,
        session_id: Optional[str] = None,
        phase_name: str = "",
        phase_type: str = ""
    ) -> None:
        """Start a phase (for mission mode)."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state:
                return
            
            phase = PhaseRecord(
                name=phase_name,
                phase_type=phase_type,
                status=PhaseStatus.RUNNING.value,
                started_at=datetime.now().isoformat()
            )
            state.phases.append(phase)
            state.current_phase_index = len(state.phases) - 1
            state.add_log(f"Phase '{phase_name}' started")
            
            self.backend.save(session_id, state)
        
        self._broadcast_event("phase_started", {
            "session_id": session_id,
            "phase_name": phase_name,
            "phase_type": phase_type
        })
    
    def complete_phase(
        self,
        session_id: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        output_summary: Optional[str] = None
    ) -> None:
        """Complete the current phase."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state or not state.phases:
                return
            
            phase = state.phases[state.current_phase_index]
            phase.status = PhaseStatus.COMPLETED.value
            phase.completed_at = datetime.now().isoformat()
            phase.artifacts = artifacts or {}
            phase.output_summary = output_summary
            
            # Calculate duration
            if phase.started_at:
                start = datetime.fromisoformat(phase.started_at)
                end = datetime.fromisoformat(phase.completed_at)
                phase.duration_seconds = (end - start).total_seconds()
            
            state.add_log(f"Phase '{phase.name}' completed")
            self.backend.save(session_id, state)
        
        self._broadcast_event("phase_completed", {
            "session_id": session_id,
            "phase_name": phase.name,
            "duration_seconds": phase.duration_seconds
        })
    
    def complete_session(
        self,
        session_id: Optional[str] = None,
        final_output: Optional[str] = None,
        final_artifacts: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Complete the session."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if not state:
                return
            
            state.completed_at = datetime.now().isoformat()
            state.status = (
                SessionStatus.FAILED.value if error 
                else SessionStatus.COMPLETED.value
            )
            state.final_output = final_output
            state.final_artifacts = final_artifacts or {}
            state.error = error
            state.current_agent = None
            
            state.add_log(f"Session completed: {state.status}")
            self.backend.save(session_id, state)
            
            # Notify callbacks
            self._notify_state_change(session_id, state)
        
        self._broadcast_event("session_completed", {
            "session_id": session_id,
            "status": state.status,
            "error": error
        })
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[SessionState]:
        """Get session state."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return None
        return self.backend.load(session_id)
    
    def get_current_session(self) -> Optional[SessionState]:
        """Get current session state."""
        return self.get_session(self._current_session_id)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return self.backend.list_sessions()
    
    def set_context(
        self,
        session_id: Optional[str] = None,
        key: str = "",
        value: Any = None
    ) -> None:
        """Set a context value."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return
        
        with self._lock:
            state = self.backend.load(session_id)
            if state:
                state.context[key] = value
                self.backend.save(session_id, state)
    
    def get_context(
        self,
        session_id: Optional[str] = None,
        key: Optional[str] = None
    ) -> Any:
        """Get context value or entire context."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return None
        
        state = self.backend.load(session_id)
        if not state:
            return None
        
        if key:
            return state.context.get(key)
        return state.context
    
    # Event System
    
    def subscribe_events(self) -> queue.Queue:
        """Subscribe to state events."""
        with self._event_lock:
            q = queue.Queue(maxsize=100)
            self._event_queues.append(q)
            return q
    
    def unsubscribe_events(self, q: queue.Queue) -> None:
        """Unsubscribe from events."""
        with self._event_lock:
            if q in self._event_queues:
                self._event_queues.remove(q)
    
    def _broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast event to subscribers."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with self._event_lock:
            dead = []
            for q in self._event_queues:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            
            for q in dead:
                self._event_queues.remove(q)
    
    def add_state_callback(
        self, 
        callback: Callable[[str, SessionState], None]
    ) -> None:
        """Add callback for state changes."""
        self._state_callbacks.append(callback)
    
    def _notify_state_change(self, session_id: str, state: SessionState) -> None:
        """Notify callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                callback(session_id, state)
            except Exception as e:
                logger.error(f"State callback failed: {e}")


# =============================================================================
# Global Instance
# =============================================================================

state_layer = StateLayer()


def initialize_state_layer(
    backend: str = "transient",
    state_dir: str = ".deepthinker_state"
) -> StateLayer:
    """Initialize the global state layer."""
    state_layer.initialize(backend=backend, state_dir=state_dir)
    return state_layer


def get_state_layer() -> StateLayer:
    """Get the global state layer."""
    return state_layer


