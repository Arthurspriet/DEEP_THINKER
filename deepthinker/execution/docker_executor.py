"""
Docker-based secure code execution sandbox.

Provides isolated container execution with comprehensive security measures
to protect against malicious code, resource exhaustion, and container breakout.
"""

import ast
import json
import tempfile
import time
import os
from pathlib import Path
from typing import Any, Tuple, Optional, Dict
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import docker
    from docker.types import DeviceRequest, Mount
    from docker.errors import DockerException, APIError, ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

from ..evaluation.metric_result import ExecutionResult
from .data_config import DataConfig
from .security_scanner import SecurityScanner, RiskLevel
from .execution_profile import ExecutionProfile
from .profile_registry import get_default_registry
from .execution_metrics import ExecutionMetrics


class DockerExecutor:
    """
    Secure code executor using Docker containers.
    
    Provides isolated execution environment with:
    - No network access
    - Read-only root filesystem
    - Resource limits (CPU, memory, PIDs)
    - Execution timeout
    - Security scanning before execution
    """
    
    DEFAULT_IMAGE = "deepthinker-sandbox:latest"
    
    def __init__(
        self,
        profile: Optional[ExecutionProfile] = None,
        image_name: Optional[str] = None,
        memory_limit: Optional[str] = None,
        cpu_limit: Optional[float] = None,
        timeout: Optional[int] = None,
        enable_security_scanning: Optional[bool] = None,
        auto_build_image: bool = True
    ):
        """
        Initialize Docker executor.
        
        Args:
            profile: ExecutionProfile to use (preferred). If None, uses legacy parameters or defaults to SAFE_ML.
            image_name: Docker image to use (legacy, ignored if profile provided)
            memory_limit: Memory limit (legacy, ignored if profile provided)
            cpu_limit: CPU limit (legacy, ignored if profile provided)
            timeout: Execution timeout (legacy, ignored if profile provided)
            enable_security_scanning: Whether to scan code (legacy, ignored if profile provided)
            auto_build_image: Automatically build image if not found
        """
        if not DOCKER_AVAILABLE:
            raise ImportError(
                "Docker SDK not available. Install with: pip install docker>=7.0.0"
            )
        
        # Use profile if provided, otherwise construct from legacy params or default
        if profile is not None:
            self.profile = profile
        else:
            # Backward compatibility: construct SAFE_ML profile from legacy params
            registry = get_default_registry()
            default_profile = registry.get_default_profile()
            
            # Override with provided legacy parameters
            self.profile = ExecutionProfile(
                name="SAFE_ML",
                ram_limit=memory_limit or default_profile.ram_limit,
                cpu_limit=cpu_limit if cpu_limit is not None else default_profile.cpu_limit,
                gpu_enabled=False,
                network_policy="none",
                network_allowlist=[],
                docker_image=image_name or default_profile.docker_image,
                security_scan_level="strict" if (enable_security_scanning is None or enable_security_scanning) else "disabled",
                allowed_languages=default_profile.allowed_languages,
                max_execution_time=timeout if timeout is not None else default_profile.max_execution_time
            )
        
        self.auto_build_image = auto_build_image
        
        # Initialize Docker client
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
        except DockerException as e:
            raise RuntimeError(
                f"Failed to connect to Docker daemon. "
                f"Ensure Docker is running: {str(e)}"
            )
        
        # Ensure image exists
        self._ensure_image()
    
    def _ensure_image(self):
        """Ensure Docker image exists, build if necessary."""
        try:
            self.client.images.get(self.profile.docker_image)
        except ImageNotFound:
            if self.auto_build_image:
                self._build_image()
            else:
                raise RuntimeError(
                    f"Docker image '{self.profile.docker_image}' not found. "
                    f"Build it manually or set auto_build_image=True"
                )
    
    def _build_image(self):
        """Build Docker sandbox image."""
        # Map image names to dockerfile names
        image_to_dockerfile = {
            "deepthinker-sandbox:latest": "Dockerfile.sandbox",
            "deepthinker-sandbox-gpu:latest": "Dockerfile.sandbox-gpu",
            "deepthinker-sandbox-browser:latest": "Dockerfile.sandbox-browser",
            "deepthinker-sandbox-node:latest": "Dockerfile.sandbox-node",
            "deepthinker-sandbox-trusted:latest": "Dockerfile.sandbox-trusted",
        }
        
        dockerfile_name = image_to_dockerfile.get(
            self.profile.docker_image,
            "Dockerfile.sandbox"  # Default fallback
        )
        
        # Find dockerfile in project root
        dockerfile_path = Path(__file__).parent.parent.parent / dockerfile_name
        
        if not dockerfile_path.exists():
            raise FileNotFoundError(
                f"Dockerfile '{dockerfile_name}' not found at {dockerfile_path}"
            )
        
        print(f"Building Docker image '{self.profile.docker_image}' from {dockerfile_name}...")
        
        try:
            # Build image
            image, build_logs = self.client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=dockerfile_name,
                tag=self.profile.docker_image,
                rm=True,
                forcerm=True
            )
            
            print(f"Successfully built image: {self.profile.docker_image}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to build Docker image: {str(e)}")
    
    def execute_model_on_data(
        self,
        code: str,
        config: DataConfig,
        profile: Optional[ExecutionProfile] = None
    ) -> Tuple[ExecutionResult, np.ndarray, np.ndarray]:
        """
        Execute generated model code on a dataset in Docker container.
        
        Args:
            code: Python code containing model class
            config: DataConfig with dataset path and execution parameters
            profile: Optional ExecutionProfile override (uses instance profile if None)
            
        Returns:
            Tuple of (ExecutionResult, y_test, y_pred)
        """
        start_time = time.time()
        
        # Use provided profile or instance profile
        exec_profile = profile or self.profile
        
        # Security scanning based on profile
        security_issues_list = []
        execution_allowed = True
        if exec_profile.security_scan_level != "disabled":
            scanner = SecurityScanner(scan_level=exec_profile.security_scan_level)
            issues = scanner.scan_code(code)
            security_issues_list = [issue.to_dict() for issue in issues]
            
            if not scanner.is_safe(max_allowed_risk=RiskLevel.MEDIUM):
                execution_allowed = False
                # Build error message with security issues
                issue_details = "\n".join([
                    f"  - [{issue.risk_level.value.upper()}] {issue.description} (line {issue.line_number})"
                    for issue in issues
                    if issue.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                ])
                
                # Log security panel
                try:
                    from ...cli import verbose_logger
                    if verbose_logger and verbose_logger.enabled:
                        verbose_logger.log_security_execution_panel(
                            security_issues=security_issues_list,
                            execution_environment="docker",
                            safety_checks=["code_scanning", "resource_limits", "network_isolation"],
                            code_scanned=True,
                            execution_allowed=False
                        )
                except Exception:
                    pass  # Don't fail if logging fails
                
                return (
                    ExecutionResult(
                        success=False,
                        error_type="SecurityError",
                        error_message=(
                            f"Code failed security scan with {len(issues)} issue(s). "
                            f"High/Critical issues:\n{issue_details}"
                        ),
                        traceback="",
                        execution_time=time.time() - start_time
                    ),
                    np.array([]),
                    np.array([])
                )
            else:
                # Log security panel even when safe
                try:
                    from ...cli import verbose_logger
                    if verbose_logger and verbose_logger.enabled:
                        verbose_logger.log_security_execution_panel(
                            security_issues=security_issues_list,
                            execution_environment="docker",
                            safety_checks=["code_scanning", "resource_limits", "network_isolation"],
                            code_scanned=True,
                            execution_allowed=True
                        )
                except Exception:
                    pass  # Don't fail if logging fails
        
        try:
            # Load dataset
            X, y = self._load_dataset(config.data_path, config.target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_split_ratio, random_state=config.random_seed
            )
            
            # Execute in Docker container
            exec_result, metrics = self._execute_in_container(
                code, X_train, y_train, X_test, exec_profile
            )
            
            if not exec_result["success"]:
                result = ExecutionResult(
                    success=False,
                    error_type=exec_result["error_type"],
                    error_message=exec_result["error_message"],
                    traceback=exec_result.get("traceback", ""),
                    execution_time=time.time() - start_time
                )
                # Attach metrics even on failure
                if hasattr(result, 'metrics'):
                    result.metrics = metrics
                return (
                    result,
                    np.array([]),
                    np.array([])
                )
            
            predictions = np.array(exec_result["predictions"])
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=True,
                predictions=predictions.tolist(),
                execution_time=execution_time
            )
            # Attach metrics
            if hasattr(result, 'metrics'):
                result.metrics = metrics
            
            return (
                result,
                y_test,
                predictions
            )
            
        except Exception as e:
            return (
                ExecutionResult(
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback="",
                    execution_time=time.time() - start_time
                ),
                np.array([]),
                np.array([])
            )
    
    def _execute_in_container(
        self,
        code: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        profile: ExecutionProfile
    ) -> Tuple[Dict[str, Any], ExecutionMetrics]:
        """
        Execute model code in isolated Docker container.
        
        Args:
            code: Python code containing model class
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            profile: Execution profile to use
            
        Returns:
            Tuple of (execution result dict, execution metrics)
        """
        # Create temporary directory for data exchange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Save data to temporary files
            np.save(tmpdir_path / "X_train.npy", X_train)
            np.save(tmpdir_path / "y_train.npy", y_train)
            np.save(tmpdir_path / "X_test.npy", X_test)
            
            # Save code to file
            code_file = tmpdir_path / "model_code.py"
            code_file.write_text(code)
            
            # Create execution script
            exec_script = tmpdir_path / "exec_model.py"
            exec_script.write_text('''
import sys
import json
import traceback
import numpy as np
from pathlib import Path

tmpdir = Path("/workspace")

try:
    # Load data
    X_train = np.load(tmpdir / "X_train.npy", allow_pickle=True)
    y_train = np.load(tmpdir / "y_train.npy", allow_pickle=True)
    X_test = np.load(tmpdir / "X_test.npy", allow_pickle=True)
    
    # Load and execute model code
    with open(tmpdir / "model_code.py", "r") as f:
        code = f.read()
    
    # Parse to find class name
    import ast
    tree = ast.parse(code)
    class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    if not class_defs:
        result = {
            "success": False,
            "error_type": "InterfaceError",
            "error_message": "No class definition found in generated code",
            "traceback": ""
        }
        print(json.dumps(result))
        sys.exit(0)
    
    class_name = class_defs[0].name
    
    # Execute code in namespace
    namespace = {"np": np, "numpy": np}
    exec(code, namespace)
    
    if class_name not in namespace:
        result = {
            "success": False,
            "error_type": "InterfaceError",
            "error_message": f"Class '{class_name}' not found after execution",
            "traceback": ""
        }
        print(json.dumps(result))
        sys.exit(0)
    
    # Instantiate model
    ModelClass = namespace[class_name]
    model = ModelClass()
    
    # Check for required methods
    if not hasattr(model, "fit"):
        result = {
            "success": False,
            "error_type": "InterfaceError",
            "error_message": "Model class must have a 'fit' method",
            "traceback": ""
        }
        print(json.dumps(result))
        sys.exit(0)
    
    if not hasattr(model, "predict"):
        result = {
            "success": False,
            "error_type": "InterfaceError",
            "error_message": "Model class must have a 'predict' method",
            "traceback": ""
        }
        print(json.dumps(result))
        sys.exit(0)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Convert to list
    if hasattr(predictions, "tolist"):
        pred_list = predictions.tolist()
    else:
        pred_list = list(predictions)
    
    result = {
        "success": True,
        "predictions": pred_list
    }
    print(json.dumps(result))
    
except SyntaxError as e:
    result = {
        "success": False,
        "error_type": "SyntaxError",
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }
    print(json.dumps(result))
    
except Exception as e:
    result = {
        "success": False,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }
    print(json.dumps(result))
''')
            
            try:
                # Configure GPU access if enabled
                device_requests = []
                if profile.gpu_enabled:
                    device_requests.append(
                        DeviceRequest(count=1, capabilities=[['gpu']])
                    )
                
                # Configure network based on profile
                network_mode = "none"
                if profile.network_policy == "full":
                    network_mode = "bridge"  # Default bridge network
                elif profile.network_policy == "allowlist":
                    network_mode = "none"  # TODO: Implement allowlist via iptables
                elif profile.network_policy == "proxy":
                    network_mode = "none"  # TODO: Implement proxy container
                # else: network_mode = "none" (default)
                
                # Run container with profile-based configuration
                container = self.client.containers.run(
                    profile.docker_image,
                    command=["python3", "/workspace/exec_model.py"],
                    # Network configuration
                    network_mode=network_mode,
                    # Security: Read-only root filesystem
                    read_only=True,
                    # Mount workspace as writable (needed for numpy operations)
                    mounts=[
                        Mount(
                            target="/workspace",
                            source=str(tmpdir_path),
                            type="bind",
                            read_only=False
                        )
                    ],
                    # Resource limits from profile
                    mem_limit=profile.ram_limit,
                    nano_cpus=int(profile.cpu_limit * 1e9),  # Convert to nanocpus
                    pids_limit=100,  # Limit number of processes
                    # GPU access
                    device_requests=device_requests,
                    # Security: Drop all capabilities
                    cap_drop=["ALL"],
                    # Security: No privileged mode
                    privileged=False,
                    # Automatic cleanup
                    remove=True,
                    # Capture output
                    detach=False,
                    stdout=True,
                    stderr=True,
                    # Timeout from profile
                    timeout=profile.max_execution_time
                )
                
                # Collect metrics
                metrics = self._collect_metrics(container, profile)
                
                # Parse output
                output = container.decode('utf-8').strip()
                
                try:
                    result = json.loads(output)
                    return result, metrics
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error_type": "RuntimeError",
                        "error_message": "Failed to parse container output",
                        "traceback": f"Output: {output}"
                    }, metrics
                    
            except ContainerError as e:
                return {
                    "success": False,
                    "error_type": "ContainerError",
                    "error_message": f"Container execution failed: {str(e)}",
                    "traceback": str(e)
                }, ExecutionMetrics()
            
            except APIError as e:
                if "timeout" in str(e).lower():
                    return {
                        "success": False,
                        "error_type": "TimeoutError",
                        "error_message": f"Execution exceeded timeout of {profile.max_execution_time} seconds",
                        "traceback": ""
                    }, ExecutionMetrics()
                return {
                    "success": False,
                    "error_type": "DockerAPIError",
                    "error_message": f"Docker API error: {str(e)}",
                    "traceback": str(e)
                }, ExecutionMetrics()
            
            except Exception as e:
                return {
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": f"Unexpected error: {str(e)}",
                    "traceback": ""
                }, ExecutionMetrics()
    
    def _collect_metrics(
        self,
        container: Any,
        profile: ExecutionProfile
    ) -> ExecutionMetrics:
        """
        Collect execution metrics from container.
        
        Args:
            container: Docker container object (may be bytes output if container removed)
            profile: Execution profile used
            
        Returns:
            ExecutionMetrics with collected data
        """
        metrics = ExecutionMetrics()
        
        try:
            # Container may have been removed (remove=True), so stats may not be available
            # This is best-effort metrics collection
            if hasattr(container, 'stats'):
                # Get container stats (if available)
                stats = container.stats(stream=False)
                
                # Extract memory usage
                if "memory_stats" in stats and "max_usage" in stats["memory_stats"]:
                    metrics.ram_peak_mb = stats["memory_stats"]["max_usage"] / (1024 * 1024)
                
                # Extract CPU time
                if "cpu_stats" in stats and "cpu_usage" in stats["cpu_stats"]:
                    cpu_usage = stats["cpu_stats"]["cpu_usage"]
                    if "total_usage" in cpu_usage:
                        # Convert nanoseconds to seconds
                        metrics.cpu_time_seconds = cpu_usage["total_usage"] / 1e9
                
                # GPU metrics (if enabled)
                if profile.gpu_enabled:
                    # TODO: Parse GPU stats from container logs or nvidia-smi
                    # For now, set to None (will be populated if GPU monitoring is available)
                    pass
        except Exception:
            # Metrics collection is best-effort, don't fail on errors
            pass
        
        return metrics
    
    @staticmethod
    def _load_dataset(
        data_path: str,
        target_column: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from file.
        
        Args:
            data_path: Path to CSV or JSON file
            target_column: Name of target column (None = last column)
            
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        import pandas as pd
        
        # Load based on file extension
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".json"):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Determine target column
        if target_column is None:
            target_column = df.columns[-1]
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Split features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        return X, y
    
    @staticmethod
    def is_available() -> bool:
        """Check if Docker is available."""
        if not DOCKER_AVAILABLE:
            return False
        
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up Docker client connection."""
        if hasattr(self, 'client'):
            self.client.close()

