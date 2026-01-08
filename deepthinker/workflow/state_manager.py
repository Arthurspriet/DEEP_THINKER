"""
Council State Manager for DeepThinker 2.0.

Thread-safe state management for council-based workflow execution,
tracking council decisions, consensus results, and workflow progress.
"""

import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import queue

# Verbose logging integration
try:
    from ..cli import verbose_logger
    VERBOSE_LOGGER_AVAILABLE = True
except ImportError:
    VERBOSE_LOGGER_AVAILABLE = False
    verbose_logger = None


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CouncilPhase(Enum):
    """Council execution phases."""
    META_PLANNING = "meta_planning"
    PLANNER_COUNCIL = "planner_council"
    RESEARCHER_COUNCIL = "researcher_council"
    CODER_COUNCIL = "coder_council"
    EVALUATOR_COUNCIL = "evaluator_council"
    SIMULATION_COUNCIL = "simulation_council"
    OPTIMIST_COUNCIL = "optimist_council"
    SKEPTIC_COUNCIL = "skeptic_council"
    PHASE_DEEPENING = "phase_deepening"
    ARBITRATION = "arbitration"
    EXECUTION = "execution"


@dataclass
class CouncilExecutionTrace:
    """Trace of a single council execution."""
    
    council_name: str
    phase: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    models_used: List[str] = field(default_factory=list)
    consensus_method: str = ""
    consensus_confidence: float = 0.0
    input_preview: str = ""
    output_preview: str = ""
    full_output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_full: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if not include_full:
            data.pop("full_output", None)
        return data


@dataclass
class CouncilIterationState:
    """State of a single workflow iteration."""
    
    iteration: int
    start_time: str
    end_time: Optional[str] = None
    code: str = ""
    quality_score: Optional[float] = None
    passed: bool = False
    council_traces: List[CouncilExecutionTrace] = field(default_factory=list)
    arbiter_decision: Optional[Dict[str, Any]] = None
    
    def to_dict(self, include_full: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["council_traces"] = [
            trace.to_dict(include_full=include_full)
            for trace in self.council_traces
        ]
        return data


@dataclass
class CouncilWorkflowState:
    """Complete council workflow execution state."""
    
    workflow_id: str
    status: str
    objective: str
    start_time: str
    end_time: Optional[str] = None
    meta_plan: Optional[Dict[str, Any]] = None
    current_iteration: int = 0
    max_iterations: int = 3
    current_phase: Optional[str] = None
    current_council: Optional[str] = None
    iterations: List[CouncilIterationState] = field(default_factory=list)
    final_output: Any = None
    error: Optional[str] = None
    
    def to_dict(self, include_full: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["iterations"] = [
            iteration.to_dict(include_full=include_full)
            for iteration in self.iterations
        ]
        return data


class CouncilStateManager:
    """
    Thread-safe manager for council workflow state.
    
    Singleton pattern for global access with event broadcasting for SSE.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize state manager."""
        if self._initialized:
            return
        
        self._workflows: Dict[str, CouncilWorkflowState] = {}
        self._current_workflow_id: Optional[str] = None
        self._state_lock = threading.RLock()
        self._event_queues: List[queue.Queue] = []
        self._event_queue_lock = threading.Lock()
        self._initialized = True
    
    def start_workflow(
        self,
        objective: str,
        max_iterations: int = 3
    ) -> str:
        """Start a new council workflow."""
        workflow_id = str(uuid.uuid4())
        
        with self._state_lock:
            workflow = CouncilWorkflowState(
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING.value,
                objective=objective,
                start_time=datetime.now().isoformat(),
                max_iterations=max_iterations
            )
            self._workflows[workflow_id] = workflow
            self._current_workflow_id = workflow_id
        
        self._broadcast_event('workflow_started', {
            'workflow_id': workflow_id,
            'objective': objective,
            'max_iterations': max_iterations
        })
        
        return workflow_id
    
    def set_meta_plan(self, meta_plan: Dict[str, Any]) -> None:
        """Set the meta-plan for current workflow."""
        with self._state_lock:
            if self._current_workflow_id:
                self._workflows[self._current_workflow_id].meta_plan = meta_plan
        
        self._broadcast_event('meta_plan_set', {
            'workflow_id': self._current_workflow_id,
            'meta_plan': meta_plan
        })
    
    def start_iteration(self, iteration: int) -> None:
        """Start a new iteration."""
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            workflow.current_iteration = iteration
            
            iteration_state = CouncilIterationState(
                iteration=iteration,
                start_time=datetime.now().isoformat()
            )
            workflow.iterations.append(iteration_state)
        
        self._broadcast_event('iteration_started', {
            'workflow_id': self._current_workflow_id,
            'iteration': iteration
        })
    
    def start_council_execution(
        self,
        council_name: str,
        phase: CouncilPhase,
        models: List[str],
        consensus_method: str
    ) -> None:
        """Record the start of a council execution."""
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            workflow.current_council = council_name
            workflow.current_phase = phase.value
            
            if not workflow.iterations:
                self.start_iteration(1)
            
            current_iteration = workflow.iterations[-1]
            
            trace = CouncilExecutionTrace(
                council_name=council_name,
                phase=phase.value,
                start_time=datetime.now().isoformat(),
                models_used=models,
                consensus_method=consensus_method
            )
            current_iteration.council_traces.append(trace)
        
        self._broadcast_event('council_started', {
            'workflow_id': self._current_workflow_id,
            'council_name': council_name,
            'phase': phase.value,
            'models': models
        })
    
    def complete_council_execution(
        self,
        council_name: str,
        output: Any,
        consensus_confidence: float = 1.0,
        error: Optional[str] = None
    ) -> None:
        """Record the completion of a council execution."""
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            
            if not workflow.iterations:
                return
            
            current_iteration = workflow.iterations[-1]
            
            for trace in reversed(current_iteration.council_traces):
                if trace.council_name == council_name and trace.end_time is None:
                    end_time = datetime.now()
                    trace.end_time = end_time.isoformat()
                    
                    start_time = datetime.fromisoformat(trace.start_time)
                    trace.duration_seconds = (end_time - start_time).total_seconds()
                    
                    output_str = str(output)
                    trace.output_preview = output_str[:500] + "..." if len(output_str) > 500 else output_str
                    trace.full_output = output
                    trace.consensus_confidence = consensus_confidence
                    trace.error = error
                    break
            
            workflow.current_council = None
            workflow.current_phase = None
        
        self._broadcast_event('council_completed', {
            'workflow_id': self._current_workflow_id,
            'council_name': council_name,
            'confidence': consensus_confidence,
            'error': error
        })
    
    def update_iteration_results(
        self,
        code: str,
        quality_score: float,
        passed: bool,
        arbiter_decision: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update iteration results."""
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            
            if not workflow.iterations:
                return
            
            current_iteration = workflow.iterations[-1]
            current_iteration.code = code
            current_iteration.quality_score = quality_score
            current_iteration.passed = passed
            current_iteration.arbiter_decision = arbiter_decision
            current_iteration.end_time = datetime.now().isoformat()
        
        self._broadcast_event('iteration_completed', {
            'workflow_id': self._current_workflow_id,
            'iteration': current_iteration.iteration,
            'quality_score': quality_score,
            'passed': passed
        })
    
    def complete_workflow(
        self,
        final_output: Any = None,
        error: Optional[str] = None
    ) -> None:
        """Mark workflow as completed or failed."""
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            workflow.end_time = datetime.now().isoformat()
            workflow.status = WorkflowStatus.FAILED.value if error else WorkflowStatus.COMPLETED.value
            workflow.final_output = final_output
            workflow.error = error
            workflow.current_council = None
            workflow.current_phase = None
        
        self._broadcast_event('workflow_completed', {
            'workflow_id': self._current_workflow_id,
            'status': workflow.status,
            'error': error
        })
    
    def get_current_workflow(self, include_full: bool = False) -> Optional[Dict[str, Any]]:
        """Get the current workflow state."""
        with self._state_lock:
            if not self._current_workflow_id:
                return None
            
            workflow = self._workflows.get(self._current_workflow_id)
            if not workflow:
                return None
            
            return workflow.to_dict(include_full=include_full)
    
    def get_workflow(self, workflow_id: str, include_full: bool = False) -> Optional[Dict[str, Any]]:
        """Get a specific workflow by ID."""
        with self._state_lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return None
            
            return workflow.to_dict(include_full=include_full)
    
    def subscribe_to_events(self) -> queue.Queue:
        """Subscribe to state change events."""
        with self._event_queue_lock:
            event_queue = queue.Queue(maxsize=100)
            self._event_queues.append(event_queue)
            return event_queue
    
    def unsubscribe_from_events(self, event_queue: queue.Queue) -> None:
        """Unsubscribe from state change events."""
        with self._event_queue_lock:
            if event_queue in self._event_queues:
                self._event_queues.remove(event_queue)
    
    def _broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to all subscribed clients."""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with self._event_queue_lock:
            dead_queues = []
            for event_queue in self._event_queues:
                try:
                    event_queue.put_nowait(event)
                except queue.Full:
                    dead_queues.append(event_queue)
            
            for dead_queue in dead_queues:
                self._event_queues.remove(dead_queue)
    
    def log_snapshot(self, label: str = "Council State") -> None:
        """
        Log a snapshot of the current state via verbose logger.
        
        Args:
            label: Label for the snapshot
        """
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_state_manager_snapshot(self, label)


# Global instance
council_state_manager = CouncilStateManager()

