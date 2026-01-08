"""
Safe execution of generated code on datasets.
"""

import ast
import sys
import traceback
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Any, Optional, Tuple
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ..evaluation.metric_result import ExecutionResult
from .data_config import DataConfig


class CodeExecutor:
    """
    Executes generated code safely on datasets.
    
    Handles model instantiation, training, and prediction with comprehensive
    error handling and timeout protection.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize code executor.
        
        Args:
            timeout: Maximum execution time in seconds (deprecated, use config.execution_timeout)
        """
        self.timeout = timeout
    
    @staticmethod
    def execute_model_on_data(
        code: str,
        config: DataConfig
    ) -> Tuple[ExecutionResult, np.ndarray, np.ndarray]:
        """
        Execute generated model code on a dataset.
        
        Args:
            code: Python code containing model class
            config: DataConfig with dataset path, task type, and execution parameters
            
        Returns:
            Tuple of (ExecutionResult, y_test, y_pred)
            - ExecutionResult: execution status and predictions or error info
            - y_test: True labels for test set (empty array if execution failed)
            - y_pred: Model predictions (empty array if execution failed)
        """
        start_time = time.time()
        
        try:
            # Load dataset
            X, y = CodeExecutor._load_dataset(config.data_path, config.target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_split_ratio, random_state=config.random_seed
            )
            
            # Execute in subprocess for isolation and timeout
            exec_result = CodeExecutor._execute_in_subprocess(
                code, X_train, y_train, X_test, config.execution_timeout
            )
            
            if not exec_result["success"]:
                return (
                    ExecutionResult(
                        success=False,
                        error_type=exec_result["error_type"],
                        error_message=exec_result["error_message"],
                        traceback=exec_result.get("traceback", ""),
                        execution_time=time.time() - start_time
                    ),
                    np.array([]),
                    np.array([])
                )
            
            predictions = np.array(exec_result["predictions"])
            execution_time = time.time() - start_time
            
            return (
                ExecutionResult(
                    success=True,
                    predictions=predictions.tolist(),
                    execution_time=execution_time
                ),
                y_test,
                predictions
            )
            
        except subprocess.TimeoutExpired:
            return (
                ExecutionResult(
                    success=False,
                    error_type="TimeoutError",
                    error_message=f"Execution exceeded timeout of {config.execution_timeout} seconds",
                    traceback="",
                    execution_time=config.execution_timeout
                ),
                np.array([]),
                np.array([])
            )
        
        except Exception as e:
            return (
                ExecutionResult(
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    execution_time=time.time() - start_time
                ),
                np.array([]),
                np.array([])
            )
    
    @staticmethod
    def _execute_in_subprocess(
        code: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        timeout: int
    ) -> dict:
        """
        Execute model code in an isolated subprocess with timeout and resource limits.
        
        Args:
            code: Python code containing model class
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results:
            - success: bool
            - predictions: list (if successful)
            - error_type: str (if failed)
            - error_message: str (if failed)
            - traceback: str (if failed)
        """
        # Create temporary directory for inter-process communication
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
            exec_script.write_text(f'''
import sys
import json
import traceback
import numpy as np
from pathlib import Path

tmpdir = Path("{tmpdir}")

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
        result = {{
            "success": False,
            "error_type": "InterfaceError",
            "error_message": "No class definition found in generated code",
            "traceback": ""
        }}
        print(json.dumps(result))
        sys.exit(0)
    
    class_name = class_defs[0].name
    
    # Execute code in namespace
    namespace = {{"np": np, "numpy": np}}
    exec(code, namespace)
    
    if class_name not in namespace:
        result = {{
            "success": False,
            "error_type": "InterfaceError",
            "error_message": f"Class '{{class_name}}' not found after execution",
            "traceback": ""
        }}
        print(json.dumps(result))
        sys.exit(0)
    
    # Instantiate model
    ModelClass = namespace[class_name]
    model = ModelClass()
    
    # Check for required methods
    if not hasattr(model, "fit"):
        result = {{
            "success": False,
            "error_type": "InterfaceError",
            "error_message": "Model class must have a 'fit' method",
            "traceback": ""
        }}
        print(json.dumps(result))
        sys.exit(0)
    
    if not hasattr(model, "predict"):
        result = {{
            "success": False,
            "error_type": "InterfaceError",
            "error_message": "Model class must have a 'predict' method",
            "traceback": ""
        }}
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
    
    result = {{
        "success": True,
        "predictions": pred_list
    }}
    print(json.dumps(result))
    
except SyntaxError as e:
    result = {{
        "success": False,
        "error_type": "SyntaxError",
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(result))
    
except Exception as e:
    result = {{
        "success": False,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(result))
''')
            
            try:
                # Set resource limits if on Unix-like system
                preexec_fn = None
                if sys.platform != "win32":
                    try:
                        import resource
                        def limit_resources():
                            # Limit memory to 1GB
                            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
                            # Limit CPU time
                            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
                        preexec_fn = limit_resources
                    except ImportError:
                        pass  # resource module not available
                
                # Execute in subprocess
                result = subprocess.run(
                    [sys.executable, str(exec_script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    preexec_fn=preexec_fn
                )
                
                # Parse result
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error_type": "RuntimeError",
                        "error_message": f"Subprocess exited with code {result.returncode}",
                        "traceback": result.stderr
                    }
                
                # Parse JSON output
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error_type": "RuntimeError",
                        "error_message": "Failed to parse subprocess output",
                        "traceback": f"stdout: {result.stdout}\nstderr: {result.stderr}"
                    }
                    
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error_type": "TimeoutError",
                    "error_message": f"Execution exceeded timeout of {timeout} seconds",
                    "traceback": ""
                }
    
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

