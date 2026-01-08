"""
Mission Persistence Store for DeepThinker 2.0.

Provides JSON-based persistence for mission state with proper
datetime serialization and dataclass reconstruction.
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from dataclasses import asdict

from .mission_types import MissionState, MissionConstraints, MissionPhase
from ..outputs.output_types import OutputArtifact


class MissionStore:
    """
    JSON-based persistence layer for mission state.
    
    Stores each mission as a separate JSON file in the base directory.
    Handles datetime serialization/deserialization transparently.
    """
    
    def __init__(self, base_dir: str = ".deepthinker_missions"):
        """
        Initialize the mission store.
        
        Args:
            base_dir: Directory to store mission JSON files
        """
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _path_for(self, mission_id: str) -> Path:
        """Get the file path for a mission ID."""
        return self.base_path / f"{mission_id}.json"
    
    def _serialize_datetime(self, obj):
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _deserialize_datetime(self, obj):
        """Custom JSON deserializer for datetime objects."""
        if isinstance(obj, dict) and "__datetime__" in obj:
            return datetime.fromisoformat(obj["__datetime__"])
        return obj
    
    def _dict_to_mission_state(self, data: dict) -> MissionState:
        """
        Reconstruct a MissionState from a dictionary.
        
        Handles nested dataclass reconstruction and datetime parsing.
        """
        # Reconstruct MissionConstraints
        constraints_data = data["constraints"]
        constraints = MissionConstraints(
            time_budget_minutes=constraints_data["time_budget_minutes"],
            max_iterations=constraints_data.get("max_iterations", 100),
            allow_internet=constraints_data.get("allow_internet", True),
            allow_code_execution=constraints_data.get("allow_code_execution", True),
            notes=constraints_data.get("notes")
        )
        
        # Reconstruct MissionPhase list
        phases = []
        for phase_data in data.get("phases", []):
            phase = MissionPhase(
                name=phase_data["name"],
                description=phase_data["description"],
                status=phase_data.get("status", "pending"),
                started_at=self._parse_datetime(phase_data.get("started_at")),
                ended_at=self._parse_datetime(phase_data.get("ended_at")),
                iterations=phase_data.get("iterations", 0),
                artifacts=phase_data.get("artifacts", {})
            )
            phases.append(phase)
        
        # Reconstruct output_deliverables
        import logging
        logger = logging.getLogger(__name__)
        output_deliverables = []
        for od_data in data.get("output_deliverables", []):
            try:
                output_deliverables.append(OutputArtifact.from_dict(od_data))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed output deliverable entry: {e}")
                continue
        
        # Reconstruct MissionState
        return MissionState(
            mission_id=data["mission_id"],
            objective=data["objective"],
            constraints=constraints,
            created_at=self._parse_datetime(data["created_at"]),
            deadline_at=self._parse_datetime(data["deadline_at"]),
            current_phase_index=data.get("current_phase_index", 0),
            phases=phases,
            status=data.get("status", "pending"),
            # Failure tracking fields (new for stability)
            failure_reason=data.get("failure_reason"),
            failure_details=data.get("failure_details", {}),
            logs=data.get("logs", []),
            final_artifacts=data.get("final_artifacts", {}),
            output_deliverables=output_deliverables
        )
    
    def _parse_datetime(self, value) -> Optional[datetime]:
        """Parse a datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, dict) and "__datetime__" in value:
            return datetime.fromisoformat(value["__datetime__"])
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
    
    def _state_to_dict(self, state: MissionState) -> dict:
        """Convert MissionState to a serializable dictionary."""
        data = asdict(state)
        
        # Convert datetime fields to ISO format strings
        def convert_datetimes(obj):
            if isinstance(obj, dict):
                return {k: convert_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetimes(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        return convert_datetimes(data)
    
    def save(self, state: MissionState) -> None:
        """
        Save a mission state to disk.
        
        Args:
            state: MissionState to persist
        """
        path = self._path_for(state.mission_id)
        data = self._state_to_dict(state)
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def load(self, mission_id: str) -> MissionState:
        """
        Load a mission state from disk.
        
        Args:
            mission_id: ID of the mission to load
            
        Returns:
            Reconstructed MissionState
            
        Raises:
            FileNotFoundError: If the mission does not exist
        """
        path = self._path_for(mission_id)
        if not path.exists():
            raise FileNotFoundError(f"Mission {mission_id} not found")
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        return self._dict_to_mission_state(data)
    
    def exists(self, mission_id: str) -> bool:
        """Check if a mission exists in the store."""
        return self._path_for(mission_id).exists()
    
    def delete(self, mission_id: str) -> bool:
        """
        Delete a mission from the store.
        
        Args:
            mission_id: ID of the mission to delete
            
        Returns:
            True if deleted, False if not found
        """
        path = self._path_for(mission_id)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_missions(self) -> List[str]:
        """
        List all mission IDs in the store.
        
        Returns:
            List of mission IDs
        """
        missions = []
        for path in self.base_path.glob("*.json"):
            missions.append(path.stem)
        return sorted(missions)
    
    def list_missions_with_status(self) -> List[dict]:
        """
        List all missions with their current status.
        
        Returns:
            List of dicts with mission_id, status, objective, created_at
        """
        import logging
        logger = logging.getLogger(__name__)
        results = []
        for mission_id in self.list_missions():
            try:
                state = self.load(mission_id)
                results.append({
                    "mission_id": state.mission_id,
                    "status": state.status,
                    "objective": state.objective[:100] + "..." if len(state.objective) > 100 else state.objective,
                    "created_at": state.created_at.isoformat(),
                    "remaining_minutes": state.remaining_minutes()
                })
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logger.debug(f"Skipping corrupted mission file {mission_id}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error loading mission {mission_id}: {e}")
                continue
        return results

