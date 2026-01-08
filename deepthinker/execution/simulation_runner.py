"""
Core simulation execution engine for running scenarios.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .code_executor import CodeExecutor
from .metric_computer import MetricComputer
from .data_config import DataConfig
from .simulation_config import SimulationConfig, ScenarioConfig, NoiseConfig
from ..evaluation.simulation_result import (
    SimulationSummary,
    ScenarioResult,
    SamplePrediction
)
from ..evaluation.metric_result import ExecutionResult


class SimulationRunner:
    """
    Executes model code across multiple simulation scenarios.
    
    Manages scenario execution, data handling, and result collection.
    """
    
    def __init__(
        self,
        base_data_config: Optional[DataConfig] = None,
        execution_timeout: int = 30,
        verbose: bool = False
    ):
        """
        Initialize simulation runner.
        
        Args:
            base_data_config: Base data configuration (used as defaults)
            execution_timeout: Timeout for code execution in seconds
            verbose: Whether to print progress messages
        """
        self.base_data_config = base_data_config
        self.execution_timeout = execution_timeout
        self.verbose = verbose
        self.executor = CodeExecutor(timeout=execution_timeout)
    
    def run_scenarios(
        self,
        code: str,
        simulation_config: SimulationConfig
    ) -> SimulationSummary:
        """
        Execute code across all configured scenarios.
        
        Args:
            code: Generated model code to test
            simulation_config: Simulation configuration
            
        Returns:
            SimulationSummary with results from all scenarios
        """
        if not simulation_config.is_enabled():
            return SimulationSummary()
        
        if self.verbose:
            print(f"\n🔬 Running simulation with {len(simulation_config.scenarios)} scenario(s)...")
        
        results = []
        
        for i, scenario_config in enumerate(simulation_config.scenarios):
            if self.verbose:
                print(f"\n  Scenario {i+1}/{len(simulation_config.scenarios)}: {scenario_config.name}")
            
            result = self._execute_scenario(
                code,
                scenario_config,
                simulation_config.random_seed
            )
            results.append(result)
            
            if self.verbose:
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.scenario_name}: {result.error_message if not result.success else 'Success'}")
        
        summary = SimulationSummary(scenario_results=results)
        
        if self.verbose:
            print(f"\n✨ Simulation complete: {summary.successful_scenarios}/{summary.total_scenarios} scenarios succeeded")
        
        return summary
    
    def _execute_scenario(
        self,
        code: str,
        scenario_config: ScenarioConfig,
        random_seed: int
    ) -> ScenarioResult:
        """
        Execute code on a single scenario.
        
        Args:
            code: Model code to execute
            scenario_config: Scenario configuration
            random_seed: Random seed for reproducibility
            
        Returns:
            ScenarioResult with execution details
        """
        try:
            # Load and prepare data
            X_train, X_test, y_train, y_test = self._prepare_scenario_data(
                scenario_config,
                random_seed
            )
            
            # Check for empty datasets
            if len(X_train) == 0 or len(X_test) == 0:
                return ScenarioResult(
                    scenario_name=scenario_config.name,
                    scenario_description=scenario_config.description,
                    success=False,
                    error_message="Insufficient data after filtering (empty train or test set)"
                )
            
            # Inject noise if configured
            if scenario_config.noise_config:
                y_train = self._inject_noise(
                    y_train,
                    scenario_config.noise_config,
                    random_seed
                )
            
            # Execute model using CodeExecutor
            exec_result = self._execute_model(code, X_train, X_test, y_train, y_test)
            
            if not exec_result.success:
                return ScenarioResult(
                    scenario_name=scenario_config.name,
                    scenario_description=scenario_config.description,
                    execution_result=exec_result,
                    success=False,
                    error_message=exec_result.error_message or "Execution failed"
                )
            
            # Compute metrics
            task_type = self._determine_task_type(y_train)
            metrics = MetricComputer.compute_metrics(
                y_test,
                np.array(exec_result.predictions),
                task_type
            )
            
            # Capture sample predictions and errors
            sample_predictions, sample_errors = self._capture_samples(
                y_test,
                np.array(exec_result.predictions)
            )
            
            return ScenarioResult(
                scenario_name=scenario_config.name,
                scenario_description=scenario_config.description,
                metrics=metrics,
                execution_result=exec_result,
                sample_predictions=sample_predictions,
                sample_errors=sample_errors,
                num_samples=len(y_test),
                success=True
            )
            
        except Exception as e:
            return ScenarioResult(
                scenario_name=scenario_config.name,
                scenario_description=scenario_config.description,
                success=False,
                error_message=f"{type(e).__name__}: {str(e)}"
            )
    
    def _prepare_scenario_data(
        self,
        scenario_config: ScenarioConfig,
        random_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data for a scenario.
        
        Args:
            scenario_config: Scenario configuration
            random_seed: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Determine data path
        if scenario_config.data_path:
            data_path = scenario_config.data_path
        elif self.base_data_config and self.base_data_config.data_path:
            data_path = self.base_data_config.data_path
        else:
            raise ValueError("No data path specified in scenario or base config")
        
        # Load dataset
        X, y = self._load_dataset(
            data_path,
            self.base_data_config.target_column if self.base_data_config else None
        )
        
        # Apply data filter if specified
        if scenario_config.data_filter:
            X, y = self._apply_data_filter(X, y, scenario_config.data_filter, data_path)
        
        # Determine test split ratio
        test_split = scenario_config.test_split_ratio
        if test_split is None and self.base_data_config:
            test_split = self.base_data_config.test_split_ratio
        if test_split is None:
            test_split = 0.2
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed
        )
        
        return X_train, X_test, y_train, y_test
    
    def _load_dataset(
        self,
        data_path: str,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from file.
        
        Args:
            data_path: Path to dataset file
            target_column: Target column name (None = last column)
            
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
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
    
    def _apply_data_filter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        filter_expr: str,
        data_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply pandas query filter to dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            filter_expr: Pandas query expression
            data_path: Original data path (for column names)
            
        Returns:
            Filtered (X, y)
        """
        # Reload as DataFrame for filtering
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".json"):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format for filtering: {data_path}")
        
        # Apply filter
        try:
            filtered_df = df.query(filter_expr)
        except Exception as e:
            raise ValueError(f"Invalid filter expression '{filter_expr}': {e}")
        
        if len(filtered_df) == 0:
            raise ValueError(f"Filter '{filter_expr}' resulted in empty dataset")
        
        # Get target column
        target_column = self.base_data_config.target_column if self.base_data_config else df.columns[-1]
        
        # Re-extract X and y
        X_filtered = filtered_df.drop(columns=[target_column]).values
        y_filtered = filtered_df[target_column].values
        
        return X_filtered, y_filtered
    
    def _inject_noise(
        self,
        y: np.ndarray,
        noise_config: NoiseConfig,
        random_seed: int
    ) -> np.ndarray:
        """
        Inject noise into labels.
        
        Args:
            y: Original labels
            noise_config: Noise configuration
            random_seed: Random seed
            
        Returns:
            Noisy labels
        """
        if noise_config.type == "label_flip":
            rng = np.random.RandomState(random_seed)
            y_noisy = y.copy()
            
            # Randomly flip labels based on probability
            flip_mask = rng.random(len(y)) < noise_config.probability
            
            # For classification: flip to random other class
            # For regression: add random noise
            unique_values = np.unique(y)
            
            if len(unique_values) <= 20:  # Likely classification
                for i in np.where(flip_mask)[0]:
                    other_classes = unique_values[unique_values != y[i]]
                    if len(other_classes) > 0:
                        y_noisy[i] = rng.choice(other_classes)
            else:  # Likely regression
                noise = rng.normal(0, np.std(y) * 0.1, size=np.sum(flip_mask))
                y_noisy[flip_mask] += noise
            
            return y_noisy
        
        return y
    
    def _execute_model(
        self,
        code: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> ExecutionResult:
        """
        Execute model code with prepared data.
        
        Args:
            code: Model code
            X_train, X_test, y_train, y_test: Train/test splits
            
        Returns:
            ExecutionResult
        """
        import ast
        import time
        import traceback
        
        start_time = time.time()
        
        try:
            # Parse and extract model class
            tree = ast.parse(code)
            class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if not class_defs:
                return ExecutionResult(
                    success=False,
                    error_type="ValueError",
                    error_message="No class definition found in code",
                    execution_time=time.time() - start_time
                )
            
            # Execute code in namespace
            namespace = {"np": np, "numpy": np}
            exec(code, namespace)
            
            model_class = namespace[class_defs[0].name]
            model = model_class()
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            return ExecutionResult(
                success=True,
                predictions=predictions.tolist() if hasattr(predictions, "tolist") else list(predictions),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
    
    def _determine_task_type(self, y: np.ndarray) -> str:
        """
        Determine if task is classification or regression.
        
        Args:
            y: Target vector
            
        Returns:
            "classification" or "regression"
        """
        if self.base_data_config and self.base_data_config.task_type:
            return self.base_data_config.task_type
        
        # Heuristic: if few unique values, likely classification
        unique_values = len(np.unique(y))
        if unique_values <= 20:
            return "classification"
        return "regression"
    
    def _capture_samples(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 5
    ) -> Tuple[List[SamplePrediction], List[SamplePrediction]]:
        """
        Capture sample predictions and worst errors.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            n_samples: Number of samples to capture
            
        Returns:
            Tuple of (sample_predictions, sample_errors)
        """
        # Compute errors
        errors = np.abs(y_true - y_pred)
        
        # First n predictions
        sample_predictions = []
        for i in range(min(n_samples, len(y_true))):
            sample_predictions.append(SamplePrediction(
                index=i,
                true_value=float(y_true[i]),
                predicted_value=float(y_pred[i]),
                error=float(errors[i])
            ))
        
        # Worst n errors
        worst_indices = np.argsort(errors)[-n_samples:][::-1]
        sample_errors = []
        for idx in worst_indices:
            sample_errors.append(SamplePrediction(
                index=int(idx),
                true_value=float(y_true[idx]),
                predicted_value=float(y_pred[idx]),
                error=float(errors[idx])
            ))
        
        return sample_predictions, sample_errors

