"""
GPU & System Resource API Routes.

Provides endpoints for GPU statistics and system resource monitoring.
"""

import subprocess
import re
from typing import Optional, List, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["resources"])


class GPUInfo(BaseModel):
    """Information about a single GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    temperature_c: Optional[int]


class GPUStats(BaseModel):
    """Overall GPU statistics."""
    available: bool
    gpu_count: int
    gpus: List[GPUInfo]
    total_memory_mb: int
    total_used_mb: int
    total_free_mb: int
    overall_utilization: float


class ResourceStatus(BaseModel):
    """Overall system resource status."""
    gpu: GPUStats
    pressure: str  # low, medium, high, critical
    can_run_large_models: bool
    recommended_parallelism: int


def _parse_nvidia_smi() -> Optional[List[Dict[str, Any]]]:
    """Parse nvidia-smi output for GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "utilization_percent": int(parts[5]) if parts[5].isdigit() else 0,
                    "temperature_c": int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None
                })
        
        return gpus
        
    except FileNotFoundError:
        # nvidia-smi not installed
        return None
    except subprocess.TimeoutExpired:
        import logging
        logging.getLogger(__name__).warning("nvidia-smi command timed out")
        return None
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Error running nvidia-smi: {e}")
        return None


def _get_gpu_stats() -> GPUStats:
    """Get GPU statistics."""
    gpus_data = _parse_nvidia_smi()
    
    if not gpus_data:
        return GPUStats(
            available=False,
            gpu_count=0,
            gpus=[],
            total_memory_mb=0,
            total_used_mb=0,
            total_free_mb=0,
            overall_utilization=0.0
        )
    
    gpus = [GPUInfo(**g) for g in gpus_data]
    total_mem = sum(g.memory_total_mb for g in gpus)
    total_used = sum(g.memory_used_mb for g in gpus)
    total_free = sum(g.memory_free_mb for g in gpus)
    avg_util = sum(g.utilization_percent for g in gpus) / len(gpus) if gpus else 0
    
    return GPUStats(
        available=True,
        gpu_count=len(gpus),
        gpus=gpus,
        total_memory_mb=total_mem,
        total_used_mb=total_used,
        total_free_mb=total_free,
        overall_utilization=round(avg_util, 1)
    )


def _calculate_pressure(gpu_stats: GPUStats) -> str:
    """Calculate resource pressure level."""
    if not gpu_stats.available:
        return "critical"
    
    memory_usage = gpu_stats.total_used_mb / gpu_stats.total_memory_mb if gpu_stats.total_memory_mb > 0 else 1.0
    
    if memory_usage < 0.5:
        return "low"
    elif memory_usage < 0.7:
        return "medium"
    elif memory_usage < 0.9:
        return "high"
    else:
        return "critical"


@router.get("/gpu/stats", response_model=GPUStats)
async def get_gpu_stats():
    """Get detailed GPU statistics."""
    return _get_gpu_stats()


@router.get("/resources/status", response_model=ResourceStatus)
async def get_resource_status():
    """Get overall system resource status."""
    gpu_stats = _get_gpu_stats()
    pressure = _calculate_pressure(gpu_stats)
    
    # Determine if we can run large models (need at least 8GB free)
    can_run_large = gpu_stats.total_free_mb >= 8000
    
    # Recommend parallelism based on available memory
    if gpu_stats.total_free_mb >= 32000:
        parallelism = 4
    elif gpu_stats.total_free_mb >= 16000:
        parallelism = 3
    elif gpu_stats.total_free_mb >= 8000:
        parallelism = 2
    else:
        parallelism = 1
    
    return ResourceStatus(
        gpu=gpu_stats,
        pressure=pressure,
        can_run_large_models=can_run_large,
        recommended_parallelism=parallelism
    )

