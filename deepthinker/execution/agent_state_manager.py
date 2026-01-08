"""
Agent State Manager - Centralized tracking for multi-agent workflow execution.

Provides thread-safe state management for monitoring agent execution traces,
workflow progress, and real-time status updates with event broadcasting for SSE.
"""

import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
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


class AgentPhase(Enum):
    """Agent execution phases."""
    PLANNING = "planning"
    RESEARCH = "research"
    CODE_GENERATION = "code_generation"
    CODE_REVISION = "code_revision"
    EVALUATION = "evaluation"
    SIMULATION = "simulation"
    METRIC_EXECUTION = "metric_execution"


@dataclass
class AgentExecutionTrace:
    """Trace of a single agent execution."""
    agent_name: str  # "coder", "evaluator", "simulator"
    phase: str  # AgentPhase value
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    input_preview: str = ""  # Truncated input
    output_preview: str = ""  # Truncated output
    full_input: str = ""  # Complete input
    full_output: str = ""  # Complete output
    llm_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_full: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally including full inputs/outputs."""
        data = asdict(self)
        if not include_full:
            data.pop("full_input", None)
            data.pop("full_output", None)
        return data


@dataclass
class IterationState:
    """State of a single workflow iteration."""
    iteration: int
    start_time: str
    end_time: Optional[str] = None
    code: str = ""
    quality_score: Optional[float] = None
    passed: bool = False
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    agent_traces: List[AgentExecutionTrace] = field(default_factory=list)
    
    def to_dict(self, include_full_traces: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["agent_traces"] = [
            trace.to_dict(include_full=include_full_traces) 
            for trace in self.agent_traces
        ]
        return data


@dataclass
class WorkflowState:
    """Complete workflow execution state."""
    workflow_id: str
    status: str  # WorkflowStatus value
    objective: str
    model_name: str
    start_time: str
    end_time: Optional[str] = None
    current_iteration: int = 0
    max_iterations: int = 3
    current_phase: Optional[str] = None
    current_agent: Optional[str] = None
    iterations: List[IterationState] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self, include_full_traces: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["iterations"] = [
            iteration.to_dict(include_full_traces=include_full_traces)
            for iteration in self.iterations
        ]
        return data


class AgentStateManager:
    """
    Thread-safe manager for agent execution state.
    
    Singleton pattern for global access across the application.
    Supports event broadcasting for real-time SSE updates.
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
        
        self._workflows: Dict[str, WorkflowState] = {}
        self._current_workflow_id: Optional[str] = None
        self._state_lock = threading.RLock()
        
        # Event broadcasting for SSE
        self._event_queues: List[queue.Queue] = []
        self._event_queue_lock = threading.Lock()
        
        self._initialized = True
    
    def start_workflow(
        self,
        objective: str,
        model_name: str,
        max_iterations: int
    ) -> str:
        """
        Start a new workflow and return its ID.
        
        Args:
            objective: Workflow objective
            model_name: LLM model name
            max_iterations: Maximum iterations
            
        Returns:
            Workflow ID
        """
        workflow_id = str(uuid.uuid4())
        
        with self._state_lock:
            workflow = WorkflowState(
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING.value,
                objective=objective,
                model_name=model_name,
                start_time=datetime.now().isoformat(),
                max_iterations=max_iterations
            )
            self._workflows[workflow_id] = workflow
            self._current_workflow_id = workflow_id
        
        # Broadcast event
        self._broadcast_event('workflow_started', {
            'workflow_id': workflow_id,
            'objective': objective,
            'model_name': model_name,
            'max_iterations': max_iterations
        })
        
        return workflow_id
    
    def start_iteration(self, iteration: int) -> None:
        """Start a new iteration."""
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            workflow.current_iteration = iteration
            
            iteration_state = IterationState(
                iteration=iteration,
                start_time=datetime.now().isoformat()
            )
            workflow.iterations.append(iteration_state)
        
        # Broadcast event
        self._broadcast_event('iteration_started', {
            'workflow_id': self._current_workflow_id,
            'iteration': iteration
        })
    
    def start_agent_execution(
        self,
        agent_name: str,
        phase: AgentPhase,
        input_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record the start of an agent execution.
        
        Args:
            agent_name: Name of the agent (coder, evaluator, simulator)
            phase: Execution phase
            input_text: Input prompt/text
            metadata: Additional metadata
        """
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            workflow.current_agent = agent_name
            workflow.current_phase = phase.value
            
            if not workflow.iterations:
                self.start_iteration(1)
            
            current_iteration = workflow.iterations[-1]
            
            # Create execution trace
            trace = AgentExecutionTrace(
                agent_name=agent_name,
                phase=phase.value,
                start_time=datetime.now().isoformat(),
                input_preview=self._truncate(input_text, 500),
                full_input=input_text,
                metadata=metadata or {}
            )
            current_iteration.agent_traces.append(trace)
        
        # Broadcast event
        self._broadcast_event('agent_started', {
            'workflow_id': self._current_workflow_id,
            'agent_name': agent_name,
            'phase': phase.value,
            'iteration': current_iteration.iteration
        })
    
    def complete_agent_execution(
        self,
        agent_name: str,
        output_text: str,
        llm_metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record the completion of an agent execution.
        
        Args:
            agent_name: Name of the agent
            output_text: Output from the agent
            llm_metrics: LLM usage metrics (tokens, latency, cost)
            error: Error message if failed
        """
        duration_seconds = None
        iteration_num = None
        
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            
            if not workflow.iterations:
                return
            
            current_iteration = workflow.iterations[-1]
            iteration_num = current_iteration.iteration
            
            # Find the most recent trace for this agent
            for trace in reversed(current_iteration.agent_traces):
                if trace.agent_name == agent_name and trace.end_time is None:
                    end_time = datetime.now()
                    trace.end_time = end_time.isoformat()
                    
                    # Calculate duration
                    start_time = datetime.fromisoformat(trace.start_time)
                    trace.duration_seconds = (end_time - start_time).total_seconds()
                    duration_seconds = trace.duration_seconds
                    
                    trace.output_preview = self._truncate(output_text, 500)
                    trace.full_output = output_text
                    trace.llm_metrics = llm_metrics or {}
                    trace.error = error
                    break
            
            workflow.current_agent = None
            workflow.current_phase = None
        
        # Broadcast event
        self._broadcast_event('agent_completed', {
            'workflow_id': self._current_workflow_id,
            'agent_name': agent_name,
            'iteration': iteration_num,
            'duration_seconds': duration_seconds,
            'error': error
        })
    
    def update_iteration_results(
        self,
        code: str,
        quality_score: float,
        passed: bool,
        issues: List[Dict[str, Any]],
        recommendations: List[str],
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update iteration results with evaluation data.
        
        Args:
            code: Generated code
            quality_score: Quality score (0-10)
            passed: Whether evaluation passed
            issues: List of issues found
            recommendations: List of recommendations
            metrics: Performance metrics if available
        """
        iteration_num = None
        
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            
            if not workflow.iterations:
                return
            
            current_iteration = workflow.iterations[-1]
            iteration_num = current_iteration.iteration
            current_iteration.code = code
            current_iteration.quality_score = quality_score
            current_iteration.passed = passed
            current_iteration.issues = issues
            current_iteration.recommendations = recommendations
            current_iteration.metrics = metrics
            current_iteration.end_time = datetime.now().isoformat()
        
        # Broadcast event
        self._broadcast_event('iteration_completed', {
            'workflow_id': self._current_workflow_id,
            'iteration': iteration_num,
            'quality_score': quality_score,
            'passed': passed,
            'issues_count': len(issues)
        })
    
    def complete_workflow(self, error: Optional[str] = None) -> None:
        """
        Mark workflow as completed or failed.
        
        Args:
            error: Error message if workflow failed
        """
        status = None
        
        with self._state_lock:
            if not self._current_workflow_id:
                return
            
            workflow = self._workflows[self._current_workflow_id]
            workflow.end_time = datetime.now().isoformat()
            workflow.status = WorkflowStatus.FAILED.value if error else WorkflowStatus.COMPLETED.value
            status = workflow.status
            workflow.error = error
            workflow.current_agent = None
            workflow.current_phase = None
        
        # Broadcast event
        self._broadcast_event('workflow_completed', {
            'workflow_id': self._current_workflow_id,
            'status': status,
            'error': error
        })
    
    def get_current_workflow(self, include_full_traces: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the current workflow state.
        
        Args:
            include_full_traces: Include full input/output in traces
            
        Returns:
            Workflow state dictionary or None
        """
        with self._state_lock:
            if not self._current_workflow_id:
                return None
            
            workflow = self._workflows.get(self._current_workflow_id)
            if not workflow:
                return None
            
            return workflow.to_dict(include_full_traces=include_full_traces)
    
    def get_workflow(self, workflow_id: str, include_full_traces: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a specific workflow by ID.
        
        Args:
            workflow_id: Workflow identifier
            include_full_traces: Include full input/output in traces
            
        Returns:
            Workflow state dictionary or None
        """
        with self._state_lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return None
            
            return workflow.to_dict(include_full_traces=include_full_traces)
    
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflow summaries (without full traces)."""
        with self._state_lock:
            return [
                workflow.to_dict(include_full_traces=False)
                for workflow in self._workflows.values()
            ]
    
    def get_agent_metrics_summary(self) -> Dict[str, Any]:
        """
        Get aggregated metrics per agent across all workflows.
        
        Returns:
            Dictionary with per-agent statistics
        """
        with self._state_lock:
            agent_stats = {
                "coder": {"calls": 0, "total_tokens": 0, "total_latency": 0, "total_cost": 0},
                "evaluator": {"calls": 0, "total_tokens": 0, "total_latency": 0, "total_cost": 0},
                "simulator": {"calls": 0, "total_tokens": 0, "total_latency": 0, "total_cost": 0}
            }
            
            for workflow in self._workflows.values():
                for iteration in workflow.iterations:
                    for trace in iteration.agent_traces:
                        agent = trace.agent_name
                        if agent not in agent_stats:
                            continue
                        
                        metrics = trace.llm_metrics
                        agent_stats[agent]["calls"] += 1
                        agent_stats[agent]["total_tokens"] += metrics.get("total_tokens", 0)
                        agent_stats[agent]["total_latency"] += metrics.get("latency_seconds", 0)
                        agent_stats[agent]["total_cost"] += metrics.get("cost_usd", 0)
            
            # Calculate averages
            for agent, stats in agent_stats.items():
                if stats["calls"] > 0:
                    stats["avg_tokens"] = stats["total_tokens"] / stats["calls"]
                    stats["avg_latency"] = stats["total_latency"] / stats["calls"]
                else:
                    stats["avg_tokens"] = 0
                    stats["avg_latency"] = 0
            
            return agent_stats
    
    def clear_old_workflows(self, keep_recent: int = 10) -> None:
        """
        Clear old workflows, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent workflows to keep
        """
        with self._state_lock:
            if len(self._workflows) <= keep_recent:
                return
            
            # Sort by start time
            sorted_workflows = sorted(
                self._workflows.items(),
                key=lambda x: x[1].start_time,
                reverse=True
            )
            
            # Keep only recent ones
            self._workflows = dict(sorted_workflows[:keep_recent])
    
    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    # Event Broadcasting Methods for SSE
    
    def subscribe_to_events(self) -> queue.Queue:
        """
        Subscribe to state change events.
        
        Returns:
            Queue that will receive event dictionaries
        """
        with self._event_queue_lock:
            event_queue = queue.Queue(maxsize=100)
            self._event_queues.append(event_queue)
            return event_queue
    
    def unsubscribe_from_events(self, event_queue: queue.Queue) -> None:
        """
        Unsubscribe from state change events.
        
        Args:
            event_queue: Queue to remove from subscriptions
        """
        with self._event_queue_lock:
            if event_queue in self._event_queues:
                self._event_queues.remove(event_queue)
    
    def _broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast an event to all subscribed clients.
        
        Args:
            event_type: Type of event (e.g., 'workflow_started', 'agent_started')
            data: Event data dictionary
        """
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with self._event_queue_lock:
            # Remove dead queues and broadcast to live ones
            dead_queues = []
            for event_queue in self._event_queues:
                try:
                    event_queue.put_nowait(event)
                except queue.Full:
                    # Queue is full, skip this client
                    dead_queues.append(event_queue)
            
            # Remove dead queues
            for dead_queue in dead_queues:
                self._event_queues.remove(dead_queue)
    
    def log_snapshot(self, label: str = "Agent State") -> None:
        """
        Log a snapshot of the current state via verbose logger.
        
        Args:
            label: Label for the snapshot
        """
        if VERBOSE_LOGGER_AVAILABLE and verbose_logger and verbose_logger.enabled:
            verbose_logger.log_state_manager_snapshot(self, label)


# Global instance
agent_state_manager = AgentStateManager()

