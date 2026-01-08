"""
Evaluation Logger for Phase Risk Predictor Shadow Mode.

Logs prediction vs actual outcomes to enable model improvement.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .schemas import PhaseRiskPrediction, PhaseRiskEvaluation
from .config import EVAL_LOG_PATH

logger = logging.getLogger(__name__)


class PhaseRiskEvaluationLogger:
    """
    Logger for phase risk prediction evaluations.
    
    Stores prediction vs actual comparisons in JSONL format
    for analysis and model improvement.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize the evaluation logger.
        
        Args:
            log_path: Path to evaluation log file.
                     Defaults to kb/orchestration/phase_risk_eval.jsonl
        """
        if log_path is None:
            log_path = Path(EVAL_LOG_PATH)
        
        self.log_path = Path(log_path)
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure the log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_evaluation(
        self,
        mission_id: str,
        phase_name: str,
        phase_type: str,
        prediction: PhaseRiskPrediction,
        actual_retry_count: int,
        actual_failure_mode: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a prediction vs actual evaluation.
        
        Args:
            mission_id: Mission identifier
            phase_name: Name of the phase
            phase_type: Type of the phase
            prediction: The prediction made before execution
            actual_retry_count: Actual number of retries that occurred
            actual_failure_mode: Actual failure mode observed
            extra_data: Optional additional data to include
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Create evaluation record
        evaluation = PhaseRiskEvaluation.from_prediction_and_actual(
            timestamp=timestamp,
            mission_id=mission_id,
            phase_name=phase_name,
            phase_type=phase_type,
            prediction=prediction,
            actual_retry_count=actual_retry_count,
            actual_failure_mode=actual_failure_mode,
        )
        
        # Convert to dict and add extra data
        record = evaluation.to_dict()
        if extra_data:
            record["extra"] = extra_data
        
        # Append to log file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Log summary
            logger.info(
                f"[SHADOW] Risk prediction eval: {phase_name} ({phase_type}) - "
                f"predicted retries: {prediction.expected_retries:.2f}, "
                f"actual: {actual_retry_count}, "
                f"error: {evaluation.retry_error_abs:.2f}, "
                f"used_fallback: {prediction.used_fallback}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to log risk evaluation: {e}")
    
    def read_evaluations(
        self,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Iterator[PhaseRiskEvaluation]:
        """
        Read evaluation records from the log.
        
        Args:
            filter_dict: Optional filter (e.g., {"phase_type": "synthesis"})
            
        Yields:
            PhaseRiskEvaluation records matching filter
        """
        if not self.log_path.exists():
            return
        
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Apply filter
                        if filter_dict:
                            match = True
                            for key, value in filter_dict.items():
                                if data.get(key) != value:
                                    match = False
                                    break
                            if not match:
                                continue
                        
                        # Remove extra field before creating dataclass
                        data.pop("extra", None)
                        
                        yield PhaseRiskEvaluation(**data)
                        
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Failed to parse risk evaluation record: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to read risk evaluations: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over evaluation records.
        
        Returns:
            Dictionary with evaluation statistics
        """
        total = 0
        fallback_count = 0
        ml_count = 0
        
        retry_errors = []
        failure_mode_matches = 0
        
        by_phase_type: Dict[str, Dict[str, Any]] = {}
        
        for eval_record in self.read_evaluations():
            total += 1
            
            if eval_record.prediction_used_fallback:
                fallback_count += 1
            else:
                ml_count += 1
            
            retry_errors.append(eval_record.retry_error_abs)
            
            # Check failure mode match
            if eval_record.predicted_failure_mode == eval_record.actual_failure_mode:
                failure_mode_matches += 1
            
            # Per-phase-type stats
            pt = eval_record.phase_type
            if pt not in by_phase_type:
                by_phase_type[pt] = {
                    "count": 0,
                    "retry_errors": [],
                }
            by_phase_type[pt]["count"] += 1
            by_phase_type[pt]["retry_errors"].append(eval_record.retry_error_abs)
        
        if total == 0:
            return {
                "total_evaluations": 0,
                "log_path": str(self.log_path),
            }
        
        import numpy as np
        
        # Compute per-phase-type MAE
        for pt, data in by_phase_type.items():
            errors = data["retry_errors"]
            data["retry_mae"] = float(np.mean(errors)) if errors else 0.0
            del data["retry_errors"]  # Remove raw data
        
        return {
            "total_evaluations": total,
            "ml_predictions": ml_count,
            "fallback_predictions": fallback_count,
            "fallback_ratio": fallback_count / total,
            "retry_mae": float(np.mean(retry_errors)),
            "failure_mode_accuracy": failure_mode_matches / total,
            "by_phase_type": by_phase_type,
            "log_path": str(self.log_path),
        }
    
    def clear_log(self) -> bool:
        """
        Clear the evaluation log.
        
        Returns:
            True if successful
        """
        try:
            if self.log_path.exists():
                self.log_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to clear risk eval log: {e}")
            return False

