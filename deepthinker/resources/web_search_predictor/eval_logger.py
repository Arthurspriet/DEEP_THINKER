"""
Evaluation Logger for Web Search Predictor Shadow Mode.

Logs prediction vs actual outcomes to enable model improvement.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .schemas import WebSearchPrediction, WebSearchEvaluation
from .config import EVAL_LOG_PATH

logger = logging.getLogger(__name__)


class WebSearchEvaluationLogger:
    """
    Logger for web search prediction evaluations.
    
    Stores prediction vs actual comparisons in JSONL format
    for analysis and model improvement.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize the evaluation logger.
        
        Args:
            log_path: Path to evaluation log file.
                     Defaults to kb/orchestration/web_search_eval.jsonl
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
        prediction: WebSearchPrediction,
        actual_search_used: bool,
        actual_num_queries: int,
        actual_hallucination_detected: bool,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a prediction vs actual evaluation.
        
        Args:
            mission_id: Mission identifier
            phase_name: Name of the phase
            phase_type: Type of the phase
            prediction: The prediction made before execution
            actual_search_used: Whether web search was actually used
            actual_num_queries: Actual number of search queries executed
            actual_hallucination_detected: Whether hallucination was detected
            extra_data: Optional additional data to include
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Create evaluation record
        evaluation = WebSearchEvaluation.from_prediction_and_actual(
            timestamp=timestamp,
            mission_id=mission_id,
            phase_name=phase_name,
            phase_type=phase_type,
            prediction=prediction,
            actual_search_used=actual_search_used,
            actual_num_queries=actual_num_queries,
            actual_hallucination_detected=actual_hallucination_detected,
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
                f"[SHADOW] Web search prediction eval: {phase_name} ({phase_type}) - "
                f"predicted search: {prediction.search_required}, "
                f"actual: {actual_search_used}, "
                f"correct: {evaluation.search_prediction_correct}, "
                f"used_fallback: {prediction.used_fallback}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to log web search evaluation: {e}")
    
    def read_evaluations(
        self,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Iterator[WebSearchEvaluation]:
        """
        Read evaluation records from the log.
        
        Args:
            filter_dict: Optional filter (e.g., {"phase_type": "research"})
            
        Yields:
            WebSearchEvaluation records matching filter
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
                        
                        yield WebSearchEvaluation(**data)
                        
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Failed to parse web search evaluation record: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to read web search evaluations: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over evaluation records.
        
        Returns:
            Dictionary with evaluation statistics
        """
        total = 0
        fallback_count = 0
        ml_count = 0
        
        search_correct_count = 0
        query_errors = []
        hallucination_when_no_search_predicted = 0
        hallucination_when_search_predicted = 0
        
        by_phase_type: Dict[str, Dict[str, Any]] = {}
        
        for eval_record in self.read_evaluations():
            total += 1
            
            if eval_record.prediction_used_fallback:
                fallback_count += 1
            else:
                ml_count += 1
            
            if eval_record.search_prediction_correct:
                search_correct_count += 1
            
            query_errors.append(eval_record.query_count_error_abs)
            
            # Track hallucinations vs predictions
            if eval_record.actual_hallucination_detected:
                if eval_record.predicted_search_required:
                    hallucination_when_search_predicted += 1
                else:
                    hallucination_when_no_search_predicted += 1
            
            # Per-phase-type stats
            pt = eval_record.phase_type
            if pt not in by_phase_type:
                by_phase_type[pt] = {
                    "count": 0,
                    "search_correct": 0,
                    "query_errors": [],
                    "hallucinations": 0,
                }
            by_phase_type[pt]["count"] += 1
            if eval_record.search_prediction_correct:
                by_phase_type[pt]["search_correct"] += 1
            by_phase_type[pt]["query_errors"].append(eval_record.query_count_error_abs)
            if eval_record.actual_hallucination_detected:
                by_phase_type[pt]["hallucinations"] += 1
        
        if total == 0:
            return {
                "total_evaluations": 0,
                "log_path": str(self.log_path),
            }
        
        import numpy as np
        
        # Compute per-phase-type MAE and accuracy
        for pt, data in by_phase_type.items():
            errors = data["query_errors"]
            data["query_mae"] = float(np.mean(errors)) if errors else 0.0
            data["search_accuracy"] = data["search_correct"] / data["count"] if data["count"] > 0 else 0.0
            del data["query_errors"]  # Remove raw data
            del data["search_correct"]  # Already computed accuracy
        
        return {
            "total_evaluations": total,
            "ml_predictions": ml_count,
            "fallback_predictions": fallback_count,
            "fallback_ratio": fallback_count / total,
            "search_prediction_accuracy": search_correct_count / total,
            "query_count_mae": float(np.mean(query_errors)),
            "hallucination_when_search_predicted": hallucination_when_search_predicted,
            "hallucination_when_no_search_predicted": hallucination_when_no_search_predicted,
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
            logger.error(f"Failed to clear web search eval log: {e}")
            return False

