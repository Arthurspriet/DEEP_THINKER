"""
Shadow mode for latent memory retrieval.

Runs retrieval silently in the background without injecting results into
model generation. Logs results to disk for later evaluation.

This is a read-only observability feature that has zero impact on mission behavior.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import LATENT_MEMORY_ENABLED
from .retriever import LatentMissionRetriever

logger = logging.getLogger(__name__)


def run_shadow_latent_retrieval(mission_id: str, objective: str) -> Optional[Dict[str, Any]]:
    """
    Run latent memory retrieval in shadow mode.
    
    Retrieves similar missions but does NOT inject them into model generation.
    Results are returned as a structured dictionary for logging purposes only.
    
    Args:
        mission_id: Current mission ID
        objective: Mission objective text to use as query
        
    Returns:
        Dictionary with retrieval results, or None if:
        - LATENT_MEMORY_ENABLED is False
        - Retrieval fails or is unavailable
        - Objective is empty
        
    Structure:
        {
            "mission_id": str,
            "objective": str,
            "timestamp": str (ISO format),
            "retrieved_missions": [
                {
                    "mission_id": str,
                    "objective": str,
                    "similarity_score": float
                }
            ]
        }
    """
    if not LATENT_MEMORY_ENABLED:
        return None
    
    if not objective or not objective.strip():
        return None
    
    try:
        # Instantiate retriever
        retriever = LatentMissionRetriever()
        
        # Retrieve similar missions
        retrieved = retriever.retrieve(objective)
        
        # Build result structure (exclude memory_tokens - we don't need them)
        retrieved_missions = [
            {
                "mission_id": mission.mission_id,
                "objective": mission.objective,
                "similarity_score": float(mission.similarity_score),
            }
            for mission in retrieved
        ]
        
        return {
            "mission_id": mission_id,
            "objective": objective,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "retrieved_missions": retrieved_missions,
        }
        
    except Exception as e:
        # Silent fail - don't log to avoid noise if latent memory is unavailable
        logger.debug(f"Shadow retrieval failed for mission {mission_id}: {e}")
        return None


def persist_shadow_log(result: Dict[str, Any]) -> None:
    """
    Persist shadow retrieval results to disk.
    
    Creates the shadow logs directory if it doesn't exist and writes
    a JSON file for the mission.
    
    Args:
        result: Dictionary returned by run_shadow_latent_retrieval()
        
    Raises:
        None - all errors are silently handled
    """
    if not result:
        return
    
    mission_id = result.get("mission_id")
    if not mission_id:
        return
    
    try:
        # Get base directory from environment or default
        base_dir = Path(os.getenv("DEEPTHINKER_KB_DIR", "kb"))
        shadow_logs_dir = base_dir / "latent_memory" / "shadow_logs"
        
        # Create directory if it doesn't exist
        shadow_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file (overwrite if exists)
        log_file = shadow_logs_dir / f"{mission_id}.json"
        with open(log_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.debug(f"Shadow log persisted to {log_file}")
        
    except Exception as e:
        # Silent fail - don't log to avoid noise
        logger.debug(f"Failed to persist shadow log for mission {mission_id}: {e}")


def list_shadow_logs() -> List[Path]:
    """
    List all shadow log files.
    
    Returns:
        List of Path objects for shadow log JSON files
    """
    try:
        base_dir = Path(os.getenv("DEEPTHINKER_KB_DIR", "kb"))
        shadow_logs_dir = base_dir / "latent_memory" / "shadow_logs"
        
        if not shadow_logs_dir.exists():
            return []
        
        # Return all JSON files in the directory
        return sorted(shadow_logs_dir.glob("*.json"))
        
    except Exception as e:
        logger.debug(f"Failed to list shadow logs: {e}")
        return []


def summarize_shadow_logs() -> Dict[str, Any]:
    """
    Summarize shadow log statistics.
    
    Returns:
        Dictionary with summary statistics:
        {
            "total_logs": int,
            "total_retrievals": int,
            "average_similarity": float,
            "max_similarity": float,
            "min_similarity": float
        }
    """
    try:
        log_files = list_shadow_logs()
        
        if not log_files:
            return {
                "total_logs": 0,
                "total_retrievals": 0,
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
            }
        
        all_scores = []
        total_retrievals = 0
        
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
                    retrieved = data.get("retrieved_missions", [])
                    total_retrievals += len(retrieved)
                    for mission in retrieved:
                        score = mission.get("similarity_score", 0.0)
                        if score > 0:
                            all_scores.append(score)
            except Exception:
                # Skip invalid files
                continue
        
        if not all_scores:
            return {
                "total_logs": len(log_files),
                "total_retrievals": total_retrievals,
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
            }
        
        return {
            "total_logs": len(log_files),
            "total_retrievals": total_retrievals,
            "average_similarity": sum(all_scores) / len(all_scores),
            "max_similarity": max(all_scores),
            "min_similarity": min(all_scores),
        }
        
    except Exception as e:
        logger.debug(f"Failed to summarize shadow logs: {e}")
        return {
            "total_logs": 0,
            "total_retrievals": 0,
            "average_similarity": 0.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
        }

