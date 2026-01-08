"""
GPU Resource Manager for DeepThinker 2.0.

Provides GPU capacity checking, safety margins, and queuing support
for intelligent model execution.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ..monitoring.gpu_monitor import GPUMonitor, GPUStats


@dataclass
class GPUResourceStats:
    """
    Simplified GPU statistics for resource management decisions.
    
    Attributes:
        total_mem: Total GPU memory in MB
        used_mem: Used GPU memory in MB
        free_mem: Free GPU memory in MB
        utilization: GPU compute utilization percentage (0-100)
        gpu_count: Number of available GPUs
    """
    total_mem: int
    used_mem: int
    free_mem: int
    utilization: int
    gpu_count: int = 1
    
    @classmethod
    def from_gpu_stats_list(cls, stats_list: List[GPUStats]) -> "GPUResourceStats":
        """
        Create aggregate stats from multiple GPU stats.
        
        Args:
            stats_list: List of GPUStats from GPUMonitor
            
        Returns:
            Aggregated GPUResourceStats
        """
        if not stats_list:
            return cls(
                total_mem=0,
                used_mem=0,
                free_mem=0,
                utilization=0,
                gpu_count=0
            )
        
        total_mem = sum(int(s.memory_total_mb) for s in stats_list)
        used_mem = sum(int(s.memory_used_mb) for s in stats_list)
        free_mem = sum(int(s.memory_free_mb) for s in stats_list)
        avg_util = sum(s.utilization_gpu for s in stats_list) / len(stats_list)
        
        return cls(
            total_mem=total_mem,
            used_mem=used_mem,
            free_mem=free_mem,
            utilization=int(avg_util),
            gpu_count=len(stats_list)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_mem": self.total_mem,
            "used_mem": self.used_mem,
            "free_mem": self.free_mem,
            "utilization": self.utilization,
            "gpu_count": self.gpu_count
        }


class GPUResourceManager:
    """
    Manages GPU resources for model execution.
    
    Wraps the existing GPUMonitor singleton and adds:
    - Safety margins for memory allocation
    - Capacity checking before model execution
    - Wait-for-capacity queuing mechanism
    - Model VRAM cost lookups
    """
    
    _instance: Optional["GPUResourceManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, safety_margin_mb: int = 2000, ram_monitor: Optional[Any] = None):
        """
        Initialize the GPU resource manager.
        
        Args:
            safety_margin_mb: Memory safety margin in MB to keep free
            ram_monitor: Optional RAMMonitor instance for RAM-aware decisions
        """
        if self._initialized:
            return
        
        self._monitor = GPUMonitor()
        self.safety_margin = safety_margin_mb
        self._model_costs: Optional[Dict[str, Dict[str, Any]]] = None
        self.ram_monitor = ram_monitor
        self._initialized = True
    
    def get_reserved_headroom_mb(self) -> int:
        """
        Get the reserved VRAM headroom in MB.
        
        For high-end GPUs, reserves 30GB minimum or 30% of total VRAM,
        whichever is higher.
        
        Returns:
            Reserved headroom in MB
        """
        stats = self.get_stats()
        if stats.total_mem == 0:
            return 30000  # Default for unknown GPUs
        return max(30000, int(stats.total_mem * 0.3))
    
    def get_stats(self) -> GPUResourceStats:
        """
        Get current GPU resource statistics.
        
        Returns:
            GPUResourceStats with aggregated GPU info
        """
        raw_stats = self._monitor.get_stats()
        return GPUResourceStats.from_gpu_stats_list(raw_stats)
    
    def get_detailed_stats(self) -> List[GPUStats]:
        """
        Get detailed per-GPU statistics.
        
        Returns:
            List of GPUStats objects from the monitor
        """
        return self._monitor.get_stats()
    
    def is_available(self) -> bool:
        """
        Check if GPU monitoring is available.
        
        Returns:
            True if nvidia-smi is available and GPUs detected
        """
        return self._monitor.is_available()
    
    def get_available_vram(self) -> int:
        """
        Get available VRAM accounting for safety margin.
        
        Returns:
            Available VRAM in MB (free - safety_margin)
        """
        stats = self.get_stats()
        available = stats.free_mem - self.safety_margin
        return max(0, available)
    
    def can_run_model(self, estimated_vram_mb: int) -> bool:
        """
        Check if a model can be loaded given current GPU memory.
        
        Args:
            estimated_vram_mb: Estimated VRAM requirement in MB
            
        Returns:
            True if enough VRAM is available
        """
        available = self.get_available_vram()
        return available >= estimated_vram_mb
    
    def can_run_models(self, model_names: List[str]) -> bool:
        """
        Check if a set of models can be loaded.
        
        Args:
            model_names: List of model names to check
            
        Returns:
            True if enough VRAM is available for all models
        """
        total_vram = sum(self.get_loading_cost(m).get("vram_mb", 0) for m in model_names)
        return self.can_run_model(total_vram)
    
    def wait_for_capacity(
        self,
        estimated_vram_mb: int,
        interval: float = 1.0,
        timeout: float = 300.0
    ) -> bool:
        """
        Wait until enough VRAM is available.
        
        Implements a simple polling mechanism to wait for GPU capacity.
        
        Args:
            estimated_vram_mb: Required VRAM in MB
            interval: Polling interval in seconds
            timeout: Maximum wait time in seconds
            
        Returns:
            True if capacity became available, False on timeout
        """
        # Check if capacity is already available - return immediately
        if self.can_run_model(estimated_vram_mb):
            return True
        
        # Capacity not immediately available, poll until it becomes available
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.can_run_model(estimated_vram_mb):
                return True
            time.sleep(interval)
        
        return False
    
    def get_loading_cost(self, model_name: str) -> Dict[str, Any]:
        """
        Get approximate resource cost for loading a model.
        
        Args:
            model_name: Name of the model (e.g., "llama3.2:70b")
            
        Returns:
            Dictionary with:
            - vram_mb: Estimated VRAM usage in MB
            - load_time_s: Estimated load time in seconds
            - tokens_per_sec: Estimated inference speed
            - tier: Model tier (small/medium/large/xlarge)
        """
        if self._model_costs is None:
            self._load_model_costs()
        
        # Try exact match first
        if model_name in self._model_costs:
            return self._model_costs[model_name]
        
        # Try base name match (without tag)
        base_name = model_name.split(":")[0] if ":" in model_name else model_name
        for key, cost in self._model_costs.items():
            if key.split(":")[0] == base_name:
                return cost
        
        # Estimate based on model name patterns
        return self._estimate_cost(model_name)
    
    def _load_model_costs(self) -> None:
        """Load model cost database."""
        try:
            from .model_costs import MODEL_COSTS
            self._model_costs = MODEL_COSTS
        except ImportError:
            self._model_costs = {}
    
    def _estimate_cost(self, model_name: str) -> Dict[str, Any]:
        """
        Estimate model cost based on naming patterns.
        
        Args:
            model_name: Model name to estimate
            
        Returns:
            Estimated cost dictionary
        """
        name_lower = model_name.lower()
        
        # Check for size indicators in name
        if any(x in name_lower for x in ["70b", "72b", "65b"]):
            return {
                "vram_mb": 42000,
                "load_time_s": 60,
                "tokens_per_sec": 15,
                "tier": "xlarge"
            }
        elif any(x in name_lower for x in ["32b", "33b", "34b", "27b"]):
            return {
                "vram_mb": 20000,
                "load_time_s": 30,
                "tokens_per_sec": 25,
                "tier": "large"
            }
        elif any(x in name_lower for x in ["13b", "14b", "12b"]):
            return {
                "vram_mb": 10000,
                "load_time_s": 15,
                "tokens_per_sec": 40,
                "tier": "medium"
            }
        elif any(x in name_lower for x in ["7b", "8b", "9b"]):
            return {
                "vram_mb": 6000,
                "load_time_s": 10,
                "tokens_per_sec": 60,
                "tier": "medium"
            }
        elif any(x in name_lower for x in ["3b", "4b", "1b", "2b"]):
            return {
                "vram_mb": 3000,
                "load_time_s": 5,
                "tokens_per_sec": 100,
                "tier": "small"
            }
        else:
            # Default to medium estimate
            return {
                "vram_mb": 8000,
                "load_time_s": 12,
                "tokens_per_sec": 50,
                "tier": "medium"
            }
    
    def get_recommended_parallelism(self, models: Optional[List[str]] = None) -> int:
        """
        Get recommended number of parallel model executions (model-aware and tier-aware).
        
        Phase 2.3: Removes 8GB assumption, makes recommendations based on actual models.
        Enforces tier-based caps: REASONING/LARGE=1, MEDIUM=2, SMALL=4.
        
        Args:
            models: Optional list of model names to base recommendation on.
                   If None, returns conservative default (1).
        
        Returns:
            Recommended parallelism count (1-4) based on model tiers and VRAM
        """
        stats = self.get_stats()
        
        if stats.gpu_count == 0:
            return 1
        
        available_vram = self.get_available_vram()
        
        # If no models provided, return conservative default
        if not models:
            return 1
        
        # Check model tiers to determine parallelism cap
        try:
            from ..models.model_registry import ModelRegistry, ModelTier
            registry = ModelRegistry()
            
            # Phase 2.1/2.2: REASONING/LARGE models must serialize
            if registry.requires_serialization(models):
                return 1
            
            # Determine tier distribution
            tiers = set()
            total_vram_needed = 0
            for name in models:
                info = registry._models.get(name)
                if info:
                    tiers.add(info.tier)
                    total_vram_needed += info.vram_mb
                else:
                    # Unknown model - estimate from loading cost
                    cost = self.get_loading_cost(name)
                    total_vram_needed += cost.get("vram_mb", 8000)
            
            # All MEDIUM tier
            if len(tiers) == 1 and ModelTier.MEDIUM in tiers:
                # Cap at 2, or VRAM-based (each MEDIUM model needs ~12GB)
                vram_based = available_vram // 12000
                return min(2, max(1, vram_based))
            
            # All SMALL tier
            if len(tiers) == 1 and ModelTier.SMALL in tiers:
                # Cap at 4, or VRAM-based (each SMALL model needs ~8GB)
                vram_based = available_vram // 8000
                return min(4, max(1, vram_based))
            
            # Mixed tiers or unknown: serialize for safety
            return 1
            
        except Exception:
            # If registry unavailable, fall back to conservative estimate
            # Estimate based on total VRAM needed
            if models:
                total_vram = sum(
                    self.get_loading_cost(m).get("vram_mb", 8000) for m in models
                )
                if total_vram > available_vram:
                    return 1
                # Conservative: cap at 2 for unknown models
                return min(2, available_vram // total_vram) if total_vram > 0 else 1
            return 1
    
    def get_resource_pressure(self) -> str:
        """
        Get current resource pressure level using reserved headroom approach.
        
        For high-end GPUs (RTX 5090 with ~32GB VRAM), we reserve 30GB headroom
        (or 30% of total VRAM, whichever is higher) before considering pressure.
        This prevents irrational downgrades when abundant VRAM remains.
        
        If RAM is abundant (>64GB free), pressure is reduced by one level to account
        for potential CPU/RAM offloading by Ollama.
        
        Returns:
            "low", "medium", "high", or "critical"
        """
        stats = self.get_stats()
        
        if stats.gpu_count == 0:
            return "critical"
        
        # Calculate reserved headroom: 30GB minimum or 30% of total, whichever is higher
        reserved_mb = max(30000, int(stats.total_mem * 0.3))
        free_mem = stats.free_mem
        
        # Determine base pressure level
        base_pressure = "low"
        if free_mem < reserved_mb or stats.utilization > 95:
            base_pressure = "critical"
        elif free_mem < reserved_mb * 1.5 or stats.utilization > 85:
            base_pressure = "high"
        elif free_mem < reserved_mb * 2 or stats.utilization > 70:
            base_pressure = "medium"
        
        # Reduce pressure by one level if RAM is abundant (Ollama can offload)
        if self.ram_monitor is not None:
            try:
                ram_stats = self.ram_monitor.get_stats()
                if ram_stats.available_ram_mb > 64000:  # 64GB free
                    # Reduce pressure: critical->high, high->medium, medium->low, low stays low
                    if base_pressure == "critical":
                        return "high"
                    elif base_pressure == "high":
                        return "medium"
                    elif base_pressure == "medium":
                        return "low"
            except Exception:
                # If RAM monitor fails, use base pressure
                pass
        
        return base_pressure
    
    def suggest_model_tier(self) -> str:
        """
        Suggest appropriate model tier based on current resources.
        
        Returns:
            Suggested tier: "small", "medium", "large", or "xlarge"
        """
        pressure = self.get_resource_pressure()
        available = self.get_available_vram()
        
        if pressure == "critical" or available < 4000:
            return "small"
        elif pressure == "high" or available < 12000:
            return "medium"
        elif pressure == "medium" or available < 25000:
            return "large"
        else:
            return "xlarge"

