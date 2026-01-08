"""
CPU/GPU Capability Router - Routes tasks to appropriate compute resources.
"""

import logging
from typing import Literal, Optional

from ..schemas import RoutingDecision

logger = logging.getLogger(__name__)


class CPUGPUCapabilityRouter:
    """
    Routes tasks to appropriate compute:
    - Summarization → CPU (lightweight models)
    - Embeddings → GPU (if available)
    - Parsing → CPU
    """
    
    # Task routing rules
    TASK_ROUTING = {
        "summarization": "cpu",
        "summary": "cpu",
        "parsing": "cpu",
        "parse": "cpu",
        "extraction": "cpu",
        "embedding": "gpu",
        "embeddings": "gpu",
        "inference": "gpu",
        "generation": "gpu",
        "training": "gpu",
        "fine-tuning": "gpu",
    }
    
    # CPU-only tasks (never use GPU)
    CPU_ONLY_TASKS = {
        "parsing", "parse", "extraction", "summarization", "summary",
        "tokenization", "preprocessing", "postprocessing"
    }
    
    # GPU-preferable tasks (use GPU if available)
    GPU_PREFERRED_TASKS = {
        "embedding", "embeddings", "inference", "generation",
        "training", "fine-tuning", "model_forward", "model_backward"
    }
    
    def __init__(self, gpu_available: bool = False):
        """
        Initialize capability router.
        
        Args:
            gpu_available: Whether GPU is available
        """
        self.gpu_available = gpu_available
    
    def route_task(
        self,
        task: str,
        task_type: Optional[str] = None,
        estimated_cost: Optional[float] = None
    ) -> RoutingDecision:
        """
        Route a task to CPU or GPU.
        
        Args:
            task: Task name or description
            task_type: Optional explicit task type
            estimated_cost: Optional estimated resource cost
            
        Returns:
            RoutingDecision with target and reason
        """
        # Normalize task name
        task_lower = task.lower()
        
        # Check explicit task type first
        if task_type:
            task_lower = task_type.lower()
        
        # Check CPU-only tasks
        if any(cpu_task in task_lower for cpu_task in self.CPU_ONLY_TASKS):
            return RoutingDecision(
                task=task,
                target="cpu",
                reason="Task is CPU-only (parsing, summarization, etc.)",
                estimated_cost=estimated_cost
            )
        
        # Check GPU-preferred tasks
        if any(gpu_task in task_lower for gpu_task in self.GPU_PREFERRED_TASKS):
            if self.gpu_available:
                return RoutingDecision(
                    task=task,
                    target="gpu",
                    reason="Task benefits from GPU acceleration and GPU is available",
                    estimated_cost=estimated_cost
                )
            else:
                return RoutingDecision(
                    task=task,
                    target="cpu",
                    reason="Task prefers GPU but GPU not available, using CPU",
                    estimated_cost=estimated_cost
                )
        
        # Check routing rules
        for pattern, target in self.TASK_ROUTING.items():
            if pattern in task_lower:
                # Override if GPU not available and target is GPU
                if target == "gpu" and not self.gpu_available:
                    return RoutingDecision(
                        task=task,
                        target="cpu",
                        reason=f"Task type '{pattern}' prefers GPU but GPU not available",
                        estimated_cost=estimated_cost
                    )
                return RoutingDecision(
                    task=task,
                    target=target,
                    reason=f"Task matches routing pattern '{pattern}'",
                    estimated_cost=estimated_cost
                )
        
        # Default: CPU (safer, more available)
        return RoutingDecision(
            task=task,
            target="cpu",
            reason="Default routing to CPU (task type not recognized)",
            estimated_cost=estimated_cost
        )
    
    def update_gpu_availability(self, available: bool) -> None:
        """
        Update GPU availability status.
        
        Args:
            available: Whether GPU is now available
        """
        self.gpu_available = available
        logger.debug(f"Updated GPU availability: {available}")

