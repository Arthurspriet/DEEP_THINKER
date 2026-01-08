"""
Sandbox Executor for DeepThinker 2.0.

Secure code execution in isolated environments.
Wraps the Docker executor with council-aware integration.
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from .scanner import SecurityScanner, RiskLevel

# Try to import Docker executor
try:
    from ..execution.docker_executor import DockerExecutor, DOCKER_AVAILABLE
except ImportError:
    DOCKER_AVAILABLE = False
    DockerExecutor = None

from ..execution.code_executor import CodeExecutor
from ..execution.data_config import DataConfig


@dataclass
class SandboxResult:
    """Result from sandbox execution."""
    
    success: bool
    output: Any
    error: Optional[str] = None
    security_report: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    sandbox_type: str = "subprocess"


class SandboxExecutor:
    """
    Secure sandbox executor for DeepThinker 2.0.
    
    Provides multiple execution backends:
    - Docker (preferred, most secure)
    - Subprocess (fallback)
    
    Integrates security scanning before execution.
    """
    
    def __init__(
        self,
        prefer_docker: bool = True,
        enable_security_scanning: bool = True,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        timeout: int = 30
    ):
        """
        Initialize sandbox executor.
        
        Args:
            prefer_docker: Use Docker if available
            enable_security_scanning: Scan code before execution
            memory_limit: Memory limit for Docker
            cpu_limit: CPU limit for Docker
            timeout: Execution timeout in seconds
        """
        self.prefer_docker = prefer_docker
        self.enable_security_scanning = enable_security_scanning
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        
        self._docker_executor: Optional[Any] = None
        self._security_scanner = SecurityScanner(strict_mode=True)
        
        # Initialize Docker if available and preferred
        if self.prefer_docker and DOCKER_AVAILABLE:
            try:
                self._docker_executor = DockerExecutor(
                    memory_limit=memory_limit,
                    cpu_limit=cpu_limit,
                    timeout=timeout,
                    enable_security_scanning=enable_security_scanning
                )
            except Exception:
                self._docker_executor = None
    
    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker is available."""
        return DOCKER_AVAILABLE
    
    def get_backend(self) -> str:
        """Get the active execution backend."""
        if self._docker_executor:
            return "docker"
        return "subprocess"
    
    def scan_code(self, code: str) -> Dict[str, Any]:
        """
        Scan code for security issues.
        
        Args:
            code: Python code to scan
            
        Returns:
            Security scan report
        """
        self._security_scanner.scan_code(code)
        return self._security_scanner.get_report()
    
    def execute(
        self,
        code: str,
        data_config: Optional[DataConfig] = None,
        allow_unsafe: bool = False
    ) -> SandboxResult:
        """
        Execute code in sandbox.
        
        Args:
            code: Python code to execute
            data_config: Optional data configuration
            allow_unsafe: Allow execution even if security scan fails
            
        Returns:
            SandboxResult with execution output
        """
        import time
        start_time = time.time()
        
        # Security scan
        security_report = None
        if self.enable_security_scanning:
            security_report = self.scan_code(code)
            
            if not allow_unsafe and not self._security_scanner.is_safe():
                return SandboxResult(
                    success=False,
                    output=None,
                    error=f"Security scan failed: {security_report['max_risk_level']}",
                    security_report=security_report,
                    execution_time=time.time() - start_time,
                    sandbox_type="none"
                )
        
        # Execute with appropriate backend
        try:
            if self._docker_executor and data_config:
                exec_result, y_test, y_pred = self._docker_executor.execute_model_on_data(
                    code, data_config
                )
                
                return SandboxResult(
                    success=exec_result.success,
                    output={
                        "exec_result": exec_result,
                        "y_test": y_test.tolist() if y_test is not None else None,
                        "y_pred": y_pred.tolist() if y_pred is not None else None
                    },
                    error=exec_result.error if not exec_result.success else None,
                    security_report=security_report,
                    execution_time=time.time() - start_time,
                    sandbox_type="docker"
                )
            
            elif data_config:
                # Fallback to subprocess
                exec_result, y_test, y_pred = CodeExecutor.execute_model_on_data(
                    code, data_config
                )
                
                return SandboxResult(
                    success=exec_result.success,
                    output={
                        "exec_result": exec_result,
                        "y_test": y_test.tolist() if y_test is not None else None,
                        "y_pred": y_pred.tolist() if y_pred is not None else None
                    },
                    error=exec_result.error if not exec_result.success else None,
                    security_report=security_report,
                    execution_time=time.time() - start_time,
                    sandbox_type="subprocess"
                )
            
            else:
                # Execute without data config
                exec_result = CodeExecutor.execute_code(code, timeout=self.timeout)
                
                return SandboxResult(
                    success=exec_result.success,
                    output=exec_result.output if exec_result.success else None,
                    error=exec_result.error if not exec_result.success else None,
                    security_report=security_report,
                    execution_time=time.time() - start_time,
                    sandbox_type="subprocess"
                )
                
        except Exception as e:
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
                security_report=security_report,
                execution_time=time.time() - start_time,
                sandbox_type="error"
            )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._docker_executor:
            try:
                self._docker_executor.cleanup()
            except Exception:
                pass

