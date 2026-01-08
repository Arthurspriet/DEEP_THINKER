"""
GPU Monitoring Module for NVIDIA GPUs.

Uses nvidia-smi to track GPU utilization, memory usage, temperature,
power draw, and per-process memory allocation for LLM inference monitoring.
"""

import subprocess
import xml.etree.ElementTree as ET
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class GPUProcess:
    """Information about a process using GPU memory."""
    pid: int
    process_name: str
    used_memory_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPUStats:
    """Comprehensive GPU statistics."""
    gpu_id: int
    name: str
    utilization_gpu: float  # 0-100
    utilization_memory: float  # 0-100
    memory_used_mb: float
    memory_free_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    clock_graphics_mhz: Optional[int] = None
    clock_memory_mhz: Optional[int] = None
    clock_sm_mhz: Optional[int] = None
    processes: List[GPUProcess] = None
    
    def __post_init__(self):
        if self.processes is None:
            self.processes = []
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['processes'] = [p.to_dict() for p in self.processes]
        return data


class GPUMonitor:
    """
    Thread-safe GPU monitor using nvidia-smi.
    
    Singleton pattern for global access across the application.
    Caches results to avoid excessive subprocess calls.
    """
    
    _instance: Optional['GPUMonitor'] = None
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
        """Initialize GPU monitor."""
        if self._initialized:
            return
        
        self._cache_lock = threading.RLock()
        self._cached_stats: Optional[List[GPUStats]] = None
        self._cache_timestamp: float = 0
        self._cache_duration: float = 1.0  # Cache for 1 second
        self._available: Optional[bool] = None
        self._initialized = True
    
    def is_available(self) -> bool:
        """
        Check if nvidia-smi is available on the system.
        
        Returns:
            True if nvidia-smi is available and can query GPUs
        """
        if self._available is not None:
            return self._available
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self._available = result.returncode == 0 and result.stdout.strip().isdigit()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            self._available = False
        
        return self._available
    
    def get_gpu_count(self) -> int:
        """
        Get the number of available GPUs.
        
        Returns:
            Number of GPUs, or 0 if nvidia-smi is not available
        """
        if not self.is_available():
            return 0
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        
        return 0
    
    def get_stats(self, force_refresh: bool = False) -> List[GPUStats]:
        """
        Get current GPU statistics for all GPUs.
        
        Args:
            force_refresh: Force refresh cache even if not expired
            
        Returns:
            List of GPUStats objects, one per GPU
        """
        if not self.is_available():
            return []
        
        with self._cache_lock:
            # Check cache
            current_time = time.time()
            if (not force_refresh and 
                self._cached_stats is not None and 
                current_time - self._cache_timestamp < self._cache_duration):
                return self._cached_stats
            
            # Query nvidia-smi with XML output for comprehensive data
            try:
                result = subprocess.run(
                    ['nvidia-smi', '-q', '-x'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode != 0:
                    return []
                
                # Parse XML
                root = ET.fromstring(result.stdout)
                gpu_stats = []
                
                for gpu_idx, gpu in enumerate(root.findall('gpu')):
                    stats = self._parse_gpu_xml(gpu, gpu_idx)
                    if stats:
                        gpu_stats.append(stats)
                
                # Update cache
                self._cached_stats = gpu_stats
                self._cache_timestamp = current_time
                
                return gpu_stats
                
            except Exception as e:
                # Return empty list on error
                return []
    
    def _parse_gpu_xml(self, gpu_elem: ET.Element, gpu_id: int) -> Optional[GPUStats]:
        """Parse GPU statistics from XML element."""
        try:
            # Helper function to safely get text
            def get_text(elem, path: str, default: str = "0") -> str:
                node = elem.find(path)
                if node is not None and node.text:
                    return node.text.strip()
                return default
            
            # Helper to extract numeric value
            def get_numeric(text: str) -> float:
                # Remove units and convert to float
                text = text.split()[0]  # Take first token (number)
                text = text.replace('%', '').replace('MiB', '').replace('W', '').replace('C', '')
                try:
                    return float(text)
                except ValueError:
                    return 0.0
            
            # Basic info
            name = get_text(gpu_elem, 'product_name', 'Unknown GPU')
            
            # Utilization
            util_gpu = get_numeric(get_text(gpu_elem, 'utilization/gpu_util'))
            util_mem = get_numeric(get_text(gpu_elem, 'utilization/memory_util'))
            
            # Memory
            mem_used = get_numeric(get_text(gpu_elem, 'fb_memory_usage/used'))
            mem_free = get_numeric(get_text(gpu_elem, 'fb_memory_usage/free'))
            mem_total = get_numeric(get_text(gpu_elem, 'fb_memory_usage/total'))
            
            # Temperature
            temp = get_numeric(get_text(gpu_elem, 'temperature/gpu_temp'))
            
            # Power (optional)
            power_draw_text = get_text(gpu_elem, 'power_readings/power_draw', None)
            power_limit_text = get_text(gpu_elem, 'power_readings/power_limit', None)
            power_draw = get_numeric(power_draw_text) if power_draw_text else None
            power_limit = get_numeric(power_limit_text) if power_limit_text else None
            
            # Fan speed (optional)
            fan_text = get_text(gpu_elem, 'fan_speed', None)
            fan_speed = get_numeric(fan_text) if fan_text else None
            
            # Clock speeds (optional)
            clock_graphics_text = get_text(gpu_elem, 'clocks/graphics_clock', None)
            clock_memory_text = get_text(gpu_elem, 'clocks/mem_clock', None)
            clock_sm_text = get_text(gpu_elem, 'clocks/sm_clock', None)
            
            clock_graphics = int(get_numeric(clock_graphics_text)) if clock_graphics_text else None
            clock_memory = int(get_numeric(clock_memory_text)) if clock_memory_text else None
            clock_sm = int(get_numeric(clock_sm_text)) if clock_sm_text else None
            
            # Processes
            processes = []
            processes_elem = gpu_elem.find('processes')
            if processes_elem is not None:
                for proc in processes_elem.findall('process_info'):
                    try:
                        pid = int(get_text(proc, 'pid', '0'))
                        proc_name = get_text(proc, 'process_name', 'unknown')  # Fixed: was overwriting 'name'
                        mem = get_numeric(get_text(proc, 'used_memory', '0'))
                        
                        if pid > 0:
                            processes.append(GPUProcess(
                                pid=pid,
                                process_name=proc_name,  # Fixed: use proc_name
                                used_memory_mb=mem
                            ))
                    except Exception:
                        continue
            
            return GPUStats(
                gpu_id=gpu_id,
                name=name,
                utilization_gpu=util_gpu,
                utilization_memory=util_mem,
                memory_used_mb=mem_used,
                memory_free_mb=mem_free,
                memory_total_mb=mem_total,
                temperature_c=temp,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                fan_speed_percent=fan_speed,
                clock_graphics_mhz=clock_graphics,
                clock_memory_mhz=clock_memory,
                clock_sm_mhz=clock_sm,
                processes=processes
            )
            
        except Exception as e:
            return None
    
    def get_processes(self) -> List[Dict[str, Any]]:
        """
        Get all processes using GPU memory across all GPUs.
        
        Returns:
            List of process dictionaries with GPU info
        """
        all_processes = []
        stats = self.get_stats()
        
        for gpu_stats in stats:
            for proc in gpu_stats.processes:
                all_processes.append({
                    'gpu_id': gpu_stats.gpu_id,
                    'gpu_name': gpu_stats.name,
                    'pid': proc.pid,
                    'process_name': proc.process_name,
                    'used_memory_mb': proc.used_memory_mb
                })
        
        return all_processes
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all GPU statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        stats = self.get_stats()
        
        if not stats:
            return {
                'available': False,
                'gpu_count': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        total_memory_used = sum(s.memory_used_mb for s in stats)
        total_memory_total = sum(s.memory_total_mb for s in stats)
        avg_utilization = sum(s.utilization_gpu for s in stats) / len(stats)
        avg_memory_util = sum(s.utilization_memory for s in stats) / len(stats)
        avg_temperature = sum(s.temperature_c for s in stats) / len(stats)
        total_processes = sum(len(s.processes) for s in stats)
        
        return {
            'available': True,
            'gpu_count': len(stats),
            'total_memory_used_mb': total_memory_used,
            'total_memory_total_mb': total_memory_total,
            'memory_usage_percent': (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else 0,
            'avg_gpu_utilization': avg_utilization,
            'avg_memory_utilization': avg_memory_util,
            'avg_temperature_c': avg_temperature,
            'total_processes': total_processes,
            'gpus': [s.to_dict() for s in stats],
            'timestamp': datetime.now().isoformat()
        }


# Global instance and convenience functions
_gpu_monitor = GPUMonitor()


def is_gpu_available() -> bool:
    """Check if GPU monitoring is available."""
    return _gpu_monitor.is_available()


def get_gpu_stats(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Get current GPU statistics."""
    stats = _gpu_monitor.get_stats(force_refresh=force_refresh)
    return [s.to_dict() for s in stats]


def get_gpu_processes() -> List[Dict[str, Any]]:
    """Get all processes using GPU memory."""
    return _gpu_monitor.get_processes()


def get_gpu_summary() -> Dict[str, Any]:
    """Get GPU summary statistics."""
    return _gpu_monitor.get_summary()

