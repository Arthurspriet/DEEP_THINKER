"""
Dynamic Resource Queue for DeepThinker 2.0.

Provides intelligent request queuing with:
- Dynamic priority reprioritization based on time remaining
- Request timeouts with automatic cancellation
- Phase importance-aware scheduling
- Preemption support for critical phases
- Adaptive scheduling based on mission state

This queue replaces the static PriorityQueue approach with
a more intelligent system that considers the mission context.
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a queued task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


@dataclass
class QueuedTask:
    """
    A task in the dynamic resource queue.
    
    Priority is calculated dynamically based on:
    - Base priority (phase importance)
    - Time urgency (deadline proximity)
    - Age (how long it's been waiting)
    """
    
    request_id: str
    models: List[str]
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    
    # Priority factors
    phase_importance: float = 0.5  # 0-1, higher = more important
    base_priority: int = 0  # User-specified priority
    
    # Timing
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # Absolute deadline timestamp
    max_execution_time: float = 120.0  # Max seconds for execution
    estimated_duration: float = 30.0  # Estimated execution time
    
    # State
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Preemption
    preemptible: bool = True  # Can this task be preempted?
    preemption_checkpoint: Optional[Any] = None  # Partial result if preempted
    
    def calculate_priority(self, current_time: Optional[float] = None) -> float:
        """
        Calculate dynamic priority score.
        
        Higher score = higher priority (processed first).
        
        Components:
        - Phase importance (0-1) * 100 = 0-100 points
        - Time urgency (0-1) * 50 = 0-50 points (increases as deadline approaches)
        - Age bonus (0-1) * 20 = 0-20 points (increases over time)
        - Base priority * 10 = user boost
        
        Returns:
            Priority score (higher = more urgent)
        """
        if current_time is None:
            current_time = time.time()
        
        # Base from phase importance
        score = self.phase_importance * 100
        
        # Time urgency - increases as deadline approaches
        if self.deadline is not None:
            time_until_deadline = self.deadline - current_time
            total_time = self.deadline - self.created_at
            
            if total_time > 0:
                urgency = 1.0 - max(0, min(1, time_until_deadline / total_time))
                score += urgency * 50
        
        # Age bonus - older tasks get slight priority boost (prevent starvation)
        age_seconds = current_time - self.created_at
        age_factor = min(1.0, age_seconds / 60.0)  # Max bonus at 60s
        score += age_factor * 20
        
        # User-specified base priority
        score += self.base_priority * 10
        
        return score
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if task has expired (past deadline)."""
        if self.deadline is None:
            return False
        if current_time is None:
            current_time = time.time()
        return current_time > self.deadline
    
    def is_execution_timeout(self, current_time: Optional[float] = None) -> bool:
        """Check if task has exceeded max execution time."""
        if self.started_at is None:
            return False
        if current_time is None:
            current_time = time.time()
        return (current_time - self.started_at) > self.max_execution_time
    
    def age_seconds(self, current_time: Optional[float] = None) -> float:
        """Get age of task in seconds."""
        if current_time is None:
            current_time = time.time()
        return current_time - self.created_at


class DynamicResourceQueue:
    """
    Dynamic resource queue with intelligent scheduling.
    
    Features:
    - Priority recalculation on each access
    - Automatic expiration handling
    - Preemption support for critical tasks
    - Integration with mission time budget
    - Visible logging of queue decisions
    
    Usage:
        queue = DynamicResourceQueue()
        
        # Add a task
        task_id = queue.enqueue(
            models=["cogito:14b"],
            prompt="...",
            phase_importance=0.9,
            deadline=time.time() + 300  # 5 min deadline
        )
        
        # Get next task to execute
        task = queue.dequeue()
        
        # Mark completed
        queue.complete(task.request_id, result=output)
    """
    
    # Default timeouts by phase type
    DEFAULT_TIMEOUTS = {
        "research": 120.0,
        "design": 90.0,
        "implementation": 90.0,
        "testing": 60.0,
        "synthesis": 120.0,
        "default": 90.0,
    }
    
    def __init__(
        self,
        max_queue_size: int = 100,
        enable_preemption: bool = True,
        log_decisions: bool = True
    ):
        """
        Initialize the dynamic queue.
        
        Args:
            max_queue_size: Maximum tasks in queue
            enable_preemption: Allow task preemption
            log_decisions: Log queue decisions visibly
        """
        self.max_queue_size = max_queue_size
        self.enable_preemption = enable_preemption
        self.log_decisions = log_decisions
        
        # Task storage
        self._tasks: Dict[str, QueuedTask] = {}
        self._heap: List[Tuple[float, str]] = []  # (-priority, task_id)
        self._lock = threading.RLock()
        
        # Tracking
        self._task_counter = 0
        self._running_tasks: Dict[str, QueuedTask] = {}
        
        # Mission context
        self._mission_deadline: Optional[float] = None
        self._mission_start: Optional[float] = None
    
    def set_mission_context(
        self,
        deadline: Optional[float] = None,
        start_time: Optional[float] = None
    ) -> None:
        """
        Set mission context for deadline-aware scheduling.
        
        Args:
            deadline: Mission deadline timestamp
            start_time: Mission start timestamp
        """
        with self._lock:
            self._mission_deadline = deadline
            self._mission_start = start_time or time.time()
            
            if self.log_decisions:
                remaining = (deadline - time.time()) / 60 if deadline else "∞"
                logger.info(f"[QUEUE] Mission context set: {remaining:.1f}min remaining")
    
    def get_mission_time_ratio(self) -> float:
        """
        Get ratio of time remaining vs total mission time.
        
        Returns:
            0.0 = mission ended, 1.0 = just started
        """
        if self._mission_deadline is None or self._mission_start is None:
            return 1.0
        
        now = time.time()
        total = self._mission_deadline - self._mission_start
        remaining = self._mission_deadline - now
        
        if total <= 0:
            return 0.0
        
        return max(0.0, min(1.0, remaining / total))
    
    def enqueue(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        phase_importance: float = 0.5,
        base_priority: int = 0,
        deadline: Optional[float] = None,
        max_execution_time: Optional[float] = None,
        estimated_duration: float = 30.0,
        phase_type: str = "default",
        preemptible: bool = True,
        request_id: Optional[str] = None
    ) -> str:
        """
        Add a task to the queue.
        
        Args:
            models: List of model names to use
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            phase_importance: Importance of this phase (0-1)
            base_priority: User-specified priority boost
            deadline: Absolute deadline timestamp
            max_execution_time: Max execution time in seconds
            estimated_duration: Estimated execution time
            phase_type: Type of phase for default timeout
            preemptible: Whether this task can be preempted
            request_id: Optional custom request ID
            
        Returns:
            Request ID for tracking
        """
        with self._lock:
            # Generate request ID
            if request_id is None:
                self._task_counter += 1
                request_id = f"task_{self._task_counter}_{int(time.time())}"
            
            # Use mission deadline if no specific deadline
            if deadline is None and self._mission_deadline is not None:
                deadline = self._mission_deadline
            
            # Get default timeout if not specified
            if max_execution_time is None:
                max_execution_time = self.DEFAULT_TIMEOUTS.get(
                    phase_type, 
                    self.DEFAULT_TIMEOUTS["default"]
                )
            
            # Adjust timeout based on time remaining
            time_ratio = self.get_mission_time_ratio()
            if time_ratio < 0.3:
                # Less than 30% time remaining - reduce timeouts
                max_execution_time = min(max_execution_time, 60.0)
            if time_ratio < 0.1:
                # Less than 10% time remaining - aggressive reduction
                max_execution_time = min(max_execution_time, 30.0)
            
            # Create task
            task = QueuedTask(
                request_id=request_id,
                models=models,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                phase_importance=phase_importance,
                base_priority=base_priority,
                deadline=deadline,
                max_execution_time=max_execution_time,
                estimated_duration=estimated_duration,
                preemptible=preemptible,
            )
            
            # Check queue capacity
            if len(self._tasks) >= self.max_queue_size:
                # Try to expire old tasks
                self._expire_old_tasks()
                if len(self._tasks) >= self.max_queue_size:
                    raise RuntimeError("Queue is full")
            
            # Add to storage
            self._tasks[request_id] = task
            
            # Add to heap with initial priority
            priority = task.calculate_priority()
            heapq.heappush(self._heap, (-priority, request_id))
            
            if self.log_decisions:
                logger.info(
                    f"[QUEUE] Enqueued {request_id}: "
                    f"importance={phase_importance:.2f}, "
                    f"priority={priority:.1f}, "
                    f"timeout={max_execution_time:.0f}s, "
                    f"queue_size={len(self._tasks)}"
                )
            
            return request_id
    
    def dequeue(self) -> Optional[QueuedTask]:
        """
        Get the highest priority task to execute.
        
        Recalculates priorities before selection.
        Skips expired tasks automatically.
        
        Returns:
            QueuedTask or None if queue is empty
        """
        with self._lock:
            self._recalculate_priorities()
            self._expire_old_tasks()
            
            while self._heap:
                _, request_id = heapq.heappop(self._heap)
                
                task = self._tasks.get(request_id)
                if task is None:
                    continue  # Task was removed
                
                if task.status != TaskStatus.PENDING:
                    continue  # Already processed
                
                if task.is_expired():
                    task.status = TaskStatus.TIMEOUT
                    task.error = "Task expired before execution"
                    if self.log_decisions:
                        logger.warning(f"[QUEUE] Task {request_id} expired before execution")
                    continue
                
                # Mark as running
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                self._running_tasks[request_id] = task
                
                if self.log_decisions:
                    logger.info(
                        f"[QUEUE] Dequeued {request_id}: "
                        f"age={task.age_seconds():.1f}s, "
                        f"running={len(self._running_tasks)}"
                    )
                
                return task
            
            return None
    
    def complete(
        self,
        request_id: str,
        result: Any = None,
        error: Optional[str] = None
    ) -> None:
        """
        Mark a task as completed.
        
        Args:
            request_id: Task to mark
            result: Task result
            error: Error message if failed
        """
        with self._lock:
            task = self._tasks.get(request_id)
            if task is None:
                return
            
            task.completed_at = time.time()
            task.result = result
            task.error = error
            
            if error:
                task.status = TaskStatus.CANCELLED
            else:
                task.status = TaskStatus.COMPLETED
            
            self._running_tasks.pop(request_id, None)
            
            if self.log_decisions:
                duration = (task.completed_at - task.started_at) if task.started_at else 0
                status = "completed" if not error else f"failed: {error}"
                logger.info(
                    f"[QUEUE] Task {request_id} {status} "
                    f"in {duration:.1f}s"
                )
    
    def timeout(self, request_id: str) -> None:
        """Mark a task as timed out."""
        with self._lock:
            task = self._tasks.get(request_id)
            if task is None:
                return
            
            task.status = TaskStatus.TIMEOUT
            task.completed_at = time.time()
            task.error = "Execution timeout"
            
            self._running_tasks.pop(request_id, None)
            
            if self.log_decisions:
                duration = (task.completed_at - task.started_at) if task.started_at else 0
                logger.warning(
                    f"[QUEUE] Task {request_id} timed out after {duration:.1f}s "
                    f"(limit: {task.max_execution_time:.0f}s)"
                )
    
    def preempt(self, request_id: str, checkpoint: Any = None) -> Optional[QueuedTask]:
        """
        Preempt a running task to make room for higher priority work.
        
        Args:
            request_id: Task to preempt
            checkpoint: Partial result to save
            
        Returns:
            The preempted task, or None if not found/not preemptible
        """
        if not self.enable_preemption:
            return None
        
        with self._lock:
            task = self._running_tasks.get(request_id)
            if task is None:
                return None
            
            if not task.preemptible:
                return None
            
            task.status = TaskStatus.PREEMPTED
            task.preemption_checkpoint = checkpoint
            task.completed_at = time.time()
            
            self._running_tasks.pop(request_id, None)
            
            if self.log_decisions:
                duration = (task.completed_at - task.started_at) if task.started_at else 0
                logger.warning(
                    f"[QUEUE] Task {request_id} preempted after {duration:.1f}s"
                )
            
            return task
    
    def should_preempt_for(self, new_importance: float) -> Optional[str]:
        """
        Check if any running task should be preempted for a higher priority task.
        
        Args:
            new_importance: Importance of the new task
            
        Returns:
            Request ID to preempt, or None
        """
        if not self.enable_preemption:
            return None
        
        with self._lock:
            for task_id, task in self._running_tasks.items():
                if not task.preemptible:
                    continue
                
                # Preempt if new task has significantly higher importance
                if new_importance > task.phase_importance + 0.3:
                    return task_id
                
                # Preempt long-running low-importance tasks
                if task.age_seconds() > 60 and task.phase_importance < 0.5:
                    if new_importance > 0.7:
                        return task_id
            
            return None
    
    def get_running_timeouts(self) -> List[str]:
        """
        Get list of running tasks that have exceeded their timeout.
        
        Returns:
            List of request IDs that have timed out
        """
        with self._lock:
            now = time.time()
            return [
                task_id
                for task_id, task in self._running_tasks.items()
                if task.is_execution_timeout(now)
            ]
    
    def _recalculate_priorities(self) -> None:
        """Rebuild heap with recalculated priorities."""
        current_time = time.time()
        new_heap = []
        
        for task_id, task in self._tasks.items():
            if task.status == TaskStatus.PENDING:
                priority = task.calculate_priority(current_time)
                heapq.heappush(new_heap, (-priority, task_id))
        
        self._heap = new_heap
    
    def _expire_old_tasks(self) -> None:
        """Mark expired tasks as timed out."""
        current_time = time.time()
        
        for task in list(self._tasks.values()):
            if task.status == TaskStatus.PENDING and task.is_expired(current_time):
                task.status = TaskStatus.TIMEOUT
                task.error = "Task expired"
                
                if self.log_decisions:
                    logger.warning(f"[QUEUE] Task {task.request_id} expired (deadline passed)")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
            running = len(self._running_tasks)
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.status in (
                TaskStatus.TIMEOUT, TaskStatus.CANCELLED
            ))
            
            return {
                "total_tasks": len(self._tasks),
                "pending": pending,
                "running": running,
                "completed": completed,
                "failed": failed,
                "time_ratio": self.get_mission_time_ratio(),
                "mission_deadline": self._mission_deadline,
            }
    
    def clear(self) -> int:
        """
        Clear all pending tasks.
        
        Returns:
            Number of tasks cleared
        """
        with self._lock:
            count = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
            
            for task in self._tasks.values():
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    task.error = "Queue cleared"
            
            self._heap.clear()
            
            if self.log_decisions and count > 0:
                logger.info(f"[QUEUE] Cleared {count} pending tasks")
            
            return count
    
    def get_task(self, request_id: str) -> Optional[QueuedTask]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(request_id)
    
    def cancel(self, request_id: str) -> bool:
        """
        Cancel a pending task.
        
        Returns:
            True if cancelled, False if not found or already running
        """
        with self._lock:
            task = self._tasks.get(request_id)
            if task is None:
                return False
            
            if task.status != TaskStatus.PENDING:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.error = "Cancelled by user"
            
            if self.log_decisions:
                logger.info(f"[QUEUE] Task {request_id} cancelled")
            
            return True



