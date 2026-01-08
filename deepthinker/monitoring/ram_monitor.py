"""
RAM Monitor for DeepThinker 2.0.

Provides system RAM monitoring using psutil, with caching to minimize
overhead. Singleton pattern similar to GPUMonitor.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class RAMStats:
    """
    System RAM statistics.
    
    Attributes:
        total_ram_mb: Total system RAM in MB
        available_ram_mb: Available RAM in MB
        used_ram_mb: Used RAM in MB
        percent_used: Percentage of RAM used (0-100)
    """
    total_ram_mb: int
    available_ram_mb: int
    used_ram_mb: int
    percent_used: float


class RAMMonitor:
    """
    System RAM monitor singleton.
    
    Provides cached RAM statistics with 1-second cache duration to minimize
    overhead from frequent psutil calls.
    """
    
    _instance: Optional["RAMMonitor"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_duration: float = 1.0):
        """
        Initialize the RAM monitor.
        
        Args:
            cache_duration: Cache duration in seconds (default: 1.0)
        """
        if self._initialized:
            return
        
        self.cache_duration = cache_duration
        self._cached_stats: Optional[RAMStats] = None
        self._cache_timestamp: float = 0.0
        self._cache_lock = threading.Lock()
        self._initialized = True
    
    def is_available(self) -> bool:
        """
        Check if RAM monitoring is available (psutil installed).
        
        Returns:
            True if psutil is available
        """
        return PSUTIL_AVAILABLE
    
    def get_stats(self, force_refresh: bool = False) -> RAMStats:
        """
        Get current RAM statistics.
        
        Results are cached for cache_duration seconds to minimize overhead.
        
        Args:
            force_refresh: Force refresh cache even if not expired
            
        Returns:
            RAMStats object with current RAM information
        """
        if not self.is_available():
            # Return empty stats if psutil unavailable
            return RAMStats(
                total_ram_mb=0,
                available_ram_mb=0,
                used_ram_mb=0,
                percent_used=100.0
            )
        
        with self._cache_lock:
            current_time = time.time()
            
            # Check cache
            if (not force_refresh and 
                self._cached_stats is not None and 
                current_time - self._cache_timestamp < self.cache_duration):
                return self._cached_stats
            
            # Query psutil
            try:
                mem = psutil.virtual_memory()
                
                stats = RAMStats(
                    total_ram_mb=int(mem.total / (1024 * 1024)),  # Convert bytes to MB
                    available_ram_mb=int(mem.available / (1024 * 1024)),
                    used_ram_mb=int(mem.used / (1024 * 1024)),
                    percent_used=mem.percent
                )
                
                # Update cache
                self._cached_stats = stats
                self._cache_timestamp = current_time
                
                return stats
                
            except Exception:
                # If psutil call fails, return last cached value or empty stats
                if self._cached_stats is not None:
                    return self._cached_stats
                return RAMStats(
                    total_ram_mb=0,
                    available_ram_mb=0,
                    used_ram_mb=0,
                    percent_used=100.0
                )

