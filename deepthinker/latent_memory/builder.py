"""
Builder for creating latent memory index from past mission outputs.

Scans mission state files and builds the FAISS index.

Enhanced with Decision Fingerprint builder for decision accountability.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deepthinker.memory.structured_state import StructuredMissionState

from .config import INDEX_PATH, STORE_PATH
from .compressor import LatentCompressor
from .index import LatentIndex

logger = logging.getLogger(__name__)

# Decision fingerprint storage paths
DECISION_FINGERPRINT_PATH = "kb/latent_memory/decision_fingerprints.json"


def build_latent_memory_from_missions(base_dir: Optional[Path] = None) -> None:
    """
    Build latent memory index from past mission outputs.
    
    Scans kb/missions/*/state.json and:
    - Extracts final outputs from phase_outputs
    - Compresses them to latent vectors
    - Adds them to FAISS index
    - Saves index and store to disk
    
    Args:
        base_dir: Base directory for missions (default: Path("kb/missions"))
    """
    if base_dir is None:
        base_dir = Path("kb/missions")
    
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        logger.error(f"Mission directory not found: {base_dir}")
        return
    
    logger.info(f"Building latent memory index from {base_dir}")
    
    try:
        # Initialize compressor
        compressor = LatentCompressor()
        hidden_size = compressor.hidden_size
        
        # Initialize index
        index = LatentIndex(dimension=hidden_size)
        
        # Scan mission directories
        mission_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(mission_dirs)} mission directories")
        
        processed = 0
        skipped = 0
        failed = 0
        
        for mission_dir in mission_dirs:
            state_file = mission_dir / "state.json"
            
            if not state_file.exists():
                skipped += 1
                continue
            
            try:
                # Load mission state
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                
                # StructuredMissionState.from_dict expects base_dir to be parent of "missions"
                # If base_dir is "kb/missions", then base_dir.parent is "kb" (correct)
                kb_dir = base_dir.parent if base_dir.name == "missions" else base_dir
                state = StructuredMissionState.from_dict(state_data, base_dir=kb_dir)
                
                # Check if mission is completed
                # Note: StructuredMissionState doesn't have status field directly,
                # but we can check if it has phase_outputs
                if not state.phase_outputs:
                    skipped += 1
                    logger.debug(f"Skipping {mission_dir.name}: no phase outputs")
                    continue
                
                # Extract final output
                # Prefer "Synthesis & Report" phase, fallback to last phase with final_output
                final_output = None
                
                # Try Synthesis & Report first
                if "Synthesis & Report" in state.phase_outputs:
                    phase_output = state.phase_outputs["Synthesis & Report"]
                    final_output = phase_output.final_output
                
                # Fallback: find last phase with final_output
                if not final_output:
                    for phase_name in reversed(list(state.phase_outputs.keys())):
                        phase_output = state.phase_outputs[phase_name]
                        if phase_output.final_output:
                            final_output = phase_output.final_output
                            break
                
                # Skip if no final output
                if not final_output or not final_output.strip():
                    skipped += 1
                    logger.debug(f"Skipping {mission_dir.name}: no final output")
                    continue
                
                # Compress document
                memory_tokens = compressor.compress_document(final_output)
                
                # Prepare metadata
                metadata = {
                    "mission_id": state.mission_id,
                    "objective": state.objective,
                    "created_at": state.created_at.isoformat() if state.created_at else None,
                    "mission_type": state.mission_type,
                    "total_phases_completed": state.total_phases_completed,
                }
                
                # Add to index
                index.add(
                    doc_id=state.mission_id,
                    mem_tokens=memory_tokens,
                    metadata=metadata,
                )
                
                processed += 1
                logger.debug(f"Processed mission {state.mission_id}")
                
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to process {mission_dir.name}: {e}")
                continue
        
        # Build index
        index.build()
        
        # Save to disk
        index.save(
            index_path=Path(INDEX_PATH),
            store_path=Path(STORE_PATH),
        )
        
        logger.info(
            f"Index build complete: {processed} processed, {skipped} skipped, {failed} failed. "
            f"Index size: {len(index.store)} documents"
        )
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        raise


# =============================================================================
# Decision Fingerprint Builder (Decision Accountability Layer)
# =============================================================================

@dataclass
class DecisionFingerprint:
    """
    Compressed decision pattern for latent memory.
    
    Captures aggregated decision patterns (not content) for learning.
    """
    mission_id: str
    objective: str
    
    # Aggregated decision patterns
    total_decisions: int = 0
    escalation_count: int = 0
    governance_block_count: int = 0
    retry_count: int = 0
    
    # Model tier sequence
    model_tier_sequence: List[str] = None
    
    # Outcome cause sequence
    outcome_cause_sequence: List[str] = None
    
    # Derived features
    escalation_ratio: float = 0.0
    success_rate: float = 0.0
    avg_governance_severity: float = 0.0
    
    def __post_init__(self):
        if self.model_tier_sequence is None:
            self.model_tier_sequence = []
        if self.outcome_cause_sequence is None:
            self.outcome_cause_sequence = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionFingerprint":
        """Create from dictionary."""
        return cls(**data)
    
    def to_vector(self) -> np.ndarray:
        """
        Convert to fixed-size vector for similarity matching.
        
        Returns:
            Numpy array of shape (16,) with normalized features
        """
        # Normalize counts
        max_decisions = max(self.total_decisions, 1)
        
        features = [
            # Raw counts (normalized)
            min(self.total_decisions / 50, 1.0),  # Cap at 50
            min(self.escalation_count / 10, 1.0),  # Cap at 10
            min(self.governance_block_count / 10, 1.0),
            min(self.retry_count / 10, 1.0),
            
            # Derived ratios
            self.escalation_ratio,
            self.success_rate,
            min(self.avg_governance_severity, 1.0),
            
            # Tier sequence features (one-hot style)
            1.0 if "small" in self.model_tier_sequence else 0.0,
            1.0 if "medium" in self.model_tier_sequence else 0.0,
            1.0 if "large" in self.model_tier_sequence else 0.0,
            1.0 if "reasoning" in self.model_tier_sequence else 0.0,
            
            # Outcome features
            1.0 if "convergence" in self.outcome_cause_sequence else 0.0,
            1.0 if "governance_veto" in self.outcome_cause_sequence else 0.0,
            1.0 if "retry_exhaustion" in self.outcome_cause_sequence else 0.0,
            1.0 if "time_exhaustion" in self.outcome_cause_sequence else 0.0,
            1.0 if "model_underpowered" in self.outcome_cause_sequence else 0.0,
        ]
        
        return np.array(features, dtype=np.float32)


def build_decision_fingerprints(base_dir: Optional[Path] = None) -> List[DecisionFingerprint]:
    """
    Build decision fingerprint index from mission decision logs.
    
    Scans kb/missions/*/decisions.jsonl and:
    - Extracts decision patterns
    - Computes aggregated features
    - Saves fingerprints to disk
    
    Args:
        base_dir: Base directory for missions (default: Path("kb/missions"))
        
    Returns:
        List of DecisionFingerprint objects
    """
    if base_dir is None:
        base_dir = Path("kb/missions")
    
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        logger.error(f"Mission directory not found: {base_dir}")
        return []
    
    logger.info(f"Building decision fingerprints from {base_dir}")
    
    fingerprints = []
    processed = 0
    skipped = 0
    
    # Import DecisionType for pattern matching
    try:
        from ..decisions.decision_record import DecisionType
        DECISION_TYPES_AVAILABLE = True
    except ImportError:
        DECISION_TYPES_AVAILABLE = False
        logger.warning("DecisionType not available, using string matching")
    
    # Scan mission directories
    mission_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for mission_dir in mission_dirs:
        decisions_file = mission_dir / "decisions.jsonl"
        state_file = mission_dir / "state.json"
        
        if not decisions_file.exists():
            skipped += 1
            continue
        
        try:
            # Load mission objective from state if available
            objective = ""
            if state_file.exists():
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                    objective = state_data.get("objective", "")
            
            # Load decisions
            decisions = []
            with open(decisions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            decisions.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            if not decisions:
                skipped += 1
                continue
            
            # Build fingerprint
            fingerprint = _build_fingerprint_from_decisions(
                mission_id=mission_dir.name,
                objective=objective,
                decisions=decisions,
            )
            
            fingerprints.append(fingerprint)
            processed += 1
            
        except Exception as e:
            logger.warning(f"Failed to process {mission_dir.name}: {e}")
            continue
    
    # Save fingerprints to disk
    if fingerprints:
        fingerprint_path = Path(DECISION_FINGERPRINT_PATH)
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(
                [fp.to_dict() for fp in fingerprints],
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        logger.info(
            f"Decision fingerprints built: {processed} missions, "
            f"{skipped} skipped. Saved to {fingerprint_path}"
        )
    
    return fingerprints


def _build_fingerprint_from_decisions(
    mission_id: str,
    objective: str,
    decisions: List[Dict[str, Any]],
) -> DecisionFingerprint:
    """
    Build a DecisionFingerprint from a list of decision records.
    
    Args:
        mission_id: Mission identifier
        objective: Mission objective
        decisions: List of decision dictionaries
        
    Returns:
        DecisionFingerprint with computed features
    """
    total = len(decisions)
    escalation_count = 0
    governance_block_count = 0
    retry_count = 0
    
    model_tiers = []
    outcome_causes = []
    governance_severities = []
    successful_phases = 0
    total_phases = 0
    
    for d in decisions:
        dtype = d.get("decision_type", "")
        constraints = d.get("constraints_snapshot", {})
        
        # Count by type
        if dtype == "retry_escalation":
            escalation_count += 1
            retry_count = max(retry_count, constraints.get("retry_count", 0))
        
        elif dtype == "governance_veto":
            governance_block_count += 1
            severity = constraints.get("aggregate_severity", 0)
            governance_severities.append(severity)
        
        elif dtype == "model_selection":
            # Extract tier info if available
            if constraints.get("downgraded"):
                model_tiers.append("small")
            else:
                importance = constraints.get("importance", 0.5)
                if importance > 0.8:
                    model_tiers.append("reasoning")
                elif importance > 0.6:
                    model_tiers.append("large")
                else:
                    model_tiers.append("medium")
        
        elif dtype == "phase_termination":
            total_phases += 1
            status = d.get("selected_option", "")
            outcome_cause = constraints.get("outcome_cause", "")
            
            if outcome_cause:
                outcome_causes.append(outcome_cause)
            
            if status == "completed":
                successful_phases += 1
    
    # Compute derived features
    escalation_ratio = escalation_count / max(total, 1)
    success_rate = successful_phases / max(total_phases, 1)
    avg_severity = sum(governance_severities) / max(len(governance_severities), 1)
    
    return DecisionFingerprint(
        mission_id=mission_id,
        objective=objective,
        total_decisions=total,
        escalation_count=escalation_count,
        governance_block_count=governance_block_count,
        retry_count=retry_count,
        model_tier_sequence=list(set(model_tiers)),  # Unique tiers used
        outcome_cause_sequence=list(set(outcome_causes)),  # Unique causes
        escalation_ratio=escalation_ratio,
        success_rate=success_rate,
        avg_governance_severity=avg_severity,
    )


def load_decision_fingerprints() -> List[DecisionFingerprint]:
    """
    Load decision fingerprints from disk.
    
    Returns:
        List of DecisionFingerprint objects
    """
    fingerprint_path = Path(DECISION_FINGERPRINT_PATH)
    
    if not fingerprint_path.exists():
        return []
    
    try:
        with open(fingerprint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [DecisionFingerprint.from_dict(d) for d in data]
        
    except Exception as e:
        logger.warning(f"Failed to load decision fingerprints: {e}")
        return []

