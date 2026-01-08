"""
Execution metrics for observability and feedback loops.

Tracks resource usage, network activity, and security violations
during code execution.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExecutionMetrics:
    """
    Metrics collected during code execution.
    
    Attributes:
        ram_peak_mb: Peak RAM usage in megabytes
        cpu_time_seconds: Total CPU time consumed
        gpu_memory_mb: Peak GPU memory usage (if GPU enabled)
        gpu_seconds: GPU compute time (if GPU enabled)
        network_calls: Number of network requests made
        network_bytes_sent: Bytes sent over network
        network_bytes_received: Bytes received over network
        file_io_operations: Number of file I/O operations
        security_violations: List of security violations detected
    """
    
    ram_peak_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    gpu_memory_mb: Optional[float] = None
    gpu_seconds: Optional[float] = None
    network_calls: int = 0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    file_io_operations: int = 0
    security_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ram_peak_mb": self.ram_peak_mb,
            "cpu_time_seconds": self.cpu_time_seconds,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_seconds": self.gpu_seconds,
            "network_calls": self.network_calls,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_received": self.network_bytes_received,
            "file_io_operations": self.file_io_operations,
            "security_violations": self.security_violations,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionMetrics":
        """Create from dictionary."""
        return cls(**data)

