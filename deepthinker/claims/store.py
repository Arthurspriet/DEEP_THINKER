"""
Claim Store for persisting extracted claims.

Stores claims to kb/claims/{mission_id}.jsonl in append-only format.

Format:
- First line: Run metadata (extractor_mode, extraction_time_ms, claim_count)
- Subsequent lines: Individual claims

Constraint 3: Persist extractor_mode, extraction_time_ms, claim_count per run.
"""

import json
import logging
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class ClaimStore:
    """
    JSONL-based store for extracted claims.
    
    Stores claims in kb/claims/{mission_id}.jsonl format.
    Append-only for durability and easy analysis.
    
    Usage:
        store = ClaimStore()
        
        # Write claims
        store.write_extraction_result(result, mission_id)
        
        # Read claims
        claims = store.read_claims(mission_id)
        
        # Read run metadata
        runs = store.read_run_metadata(mission_id)
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the claim store.
        
        Args:
            base_dir: Base directory for claims (default: kb/claims)
        """
        self.base_dir = base_dir or Path("kb/claims")
        self._lock = threading.Lock()
    
    def _get_claims_path(self, mission_id: str) -> Path:
        """Get path to claims file for a mission."""
        # Sanitize mission_id
        safe_id = "".join(c for c in mission_id if c.isalnum() or c in "-_")
        if not safe_id:
            safe_id = hashlib.md5(mission_id.encode()).hexdigest()[:16]
        return self.base_dir / f"{safe_id}.jsonl"
    
    def write_extraction_result(
        self,
        result: "ClaimExtractionResult",
        mission_id: str,
        model_used: Optional[str] = None,
    ) -> bool:
        """
        Write extraction result to the claims store.
        
        Writes run metadata first, then individual claims.
        
        Args:
            result: ClaimExtractionResult from claim extractor
            mission_id: Mission identifier
            model_used: Model used for extraction (optional)
            
        Returns:
            True if successful
        """
        from deepthinker.hf_instruments.claim_extractor import ClaimExtractionResult
        
        try:
            with self._lock:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                claims_path = self._get_claims_path(mission_id)
                
                with open(claims_path, "a", encoding="utf-8") as f:
                    # Write run metadata
                    run_meta = {
                        "_run_metadata": True,
                        "mission_id": mission_id,
                        "extractor_mode": result.extractor_mode,
                        "extraction_time_ms": result.extraction_time_ms,
                        "claim_count": result.claim_count,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "model_used": model_used,
                        "source": result.source,
                        "error": result.error,
                    }
                    f.write(json.dumps(run_meta, ensure_ascii=False) + "\n")
                    
                    # Write individual claims
                    for claim in result.claims:
                        claim_dict = claim.to_dict()
                        claim_dict["mission_id"] = mission_id  # Ensure mission_id is set
                        f.write(json.dumps(claim_dict, ensure_ascii=False) + "\n")
                
                logger.debug(
                    f"Wrote {result.claim_count} claims for mission {mission_id} "
                    f"(mode={result.extractor_mode}, time={result.extraction_time_ms:.1f}ms)"
                )
                return True
                
        except Exception as e:
            logger.warning(f"Failed to write claims for mission {mission_id}: {e}")
            return False
    
    def read_claims(self, mission_id: str) -> List[Dict[str, Any]]:
        """
        Read all claims for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            List of claim dictionaries
        """
        claims = []
        claims_path = self._get_claims_path(mission_id)
        
        if not claims_path.exists():
            return claims
        
        try:
            with open(claims_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        # Skip run metadata
                        if not data.get("_run_metadata"):
                            claims.append(data)
                    except json.JSONDecodeError:
                        continue
            
            return claims
            
        except Exception as e:
            logger.warning(f"Failed to read claims for mission {mission_id}: {e}")
            return []
    
    def read_run_metadata(self, mission_id: str) -> List[Dict[str, Any]]:
        """
        Read run metadata for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            List of run metadata dictionaries
        """
        runs = []
        claims_path = self._get_claims_path(mission_id)
        
        if not claims_path.exists():
            return runs
        
        try:
            with open(claims_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if data.get("_run_metadata"):
                            runs.append(data)
                    except json.JSONDecodeError:
                        continue
            
            return runs
            
        except Exception as e:
            logger.warning(f"Failed to read run metadata for mission {mission_id}: {e}")
            return []
    
    def iter_claims(self, mission_id: str) -> Iterator[Dict[str, Any]]:
        """
        Iterate over claims for a mission.
        
        Useful for large claim files.
        
        Args:
            mission_id: Mission identifier
            
        Yields:
            Claim dictionaries
        """
        claims_path = self._get_claims_path(mission_id)
        
        if not claims_path.exists():
            return
        
        try:
            with open(claims_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if not data.get("_run_metadata"):
                            yield data
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to iterate claims for mission {mission_id}: {e}")
    
    def get_claim_count(self, mission_id: str) -> int:
        """
        Get total claim count for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            Number of claims
        """
        return sum(1 for _ in self.iter_claims(mission_id))
    
    def list_missions(self) -> List[str]:
        """
        List all missions with claims.
        
        Returns:
            List of mission IDs
        """
        if not self.base_dir.exists():
            return []
        
        missions = []
        for path in self.base_dir.glob("*.jsonl"):
            mission_id = path.stem
            if mission_id:
                missions.append(mission_id)
        
        return sorted(missions)
    
    def delete_claims(self, mission_id: str) -> bool:
        """
        Delete claims for a mission.
        
        Args:
            mission_id: Mission identifier
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                claims_path = self._get_claims_path(mission_id)
                if claims_path.exists():
                    claims_path.unlink()
                    logger.info(f"Deleted claims for mission {mission_id}")
                    return True
                return False
        except Exception as e:
            logger.warning(f"Failed to delete claims for mission {mission_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the claim store.
        
        Returns:
            Dictionary with stats
        """
        missions = self.list_missions()
        total_claims = 0
        total_runs = 0
        
        for mission_id in missions:
            total_claims += self.get_claim_count(mission_id)
            total_runs += len(self.read_run_metadata(mission_id))
        
        return {
            "mission_count": len(missions),
            "total_claims": total_claims,
            "total_runs": total_runs,
            "base_dir": str(self.base_dir),
        }


# Module-level singleton
_store: Optional[ClaimStore] = None
_store_lock = threading.Lock()


def get_claim_store(base_dir: Optional[Path] = None) -> ClaimStore:
    """
    Get the global claim store instance.
    
    Args:
        base_dir: Optional base directory override
        
    Returns:
        ClaimStore instance
    """
    global _store
    
    if _store is None or base_dir is not None:
        with _store_lock:
            if _store is None or base_dir is not None:
                _store = ClaimStore(base_dir)
    
    return _store


def extract_and_store_claims(
    text: str,
    mission_id: str,
    source_type: str = "unknown",
    source_ref: str = "",
    phase: str = "",
    mode: Optional[str] = None,
    model_used: Optional[str] = None,
) -> "ClaimExtractionResult":
    """
    Extract claims and store them in kb/claims/.
    
    Convenience function that combines extraction and storage.
    
    Args:
        text: Text to extract claims from
        mission_id: Mission identifier
        source_type: Type of source (final_answer, phase_output, etc.)
        source_ref: Reference to source
        phase: Phase name
        mode: Extraction mode (regex, hf, llm-json) or None for config default
        model_used: Model used for extraction
        
    Returns:
        ClaimExtractionResult with extraction metadata
    """
    from deepthinker.hf_instruments.claim_extractor import extract_claims
    
    source = {
        "source_type": source_type,
        "source_ref": source_ref,
        "mission_id": mission_id,
        "phase": phase,
    }
    
    # Extract claims
    result = extract_claims(text, source, mode)
    
    # Store claims
    store = get_claim_store()
    store.write_extraction_result(result, mission_id, model_used)
    
    # Emit observability event
    _emit_claims_extracted_event(result, mission_id)
    
    return result


def _emit_claims_extracted_event(
    result: "ClaimExtractionResult",
    mission_id: str,
) -> None:
    """Emit observability event for claim extraction."""
    try:
        from deepthinker.observability.ml_influence import (
            get_influence_tracker,
            MLInfluenceEvent,
        )
        
        tracker = get_influence_tracker()
        
        event = MLInfluenceEvent(
            mission_id=mission_id,
            phase_name=result.source.get("phase", "unknown"),
            predictor_name="hf_claim_extractor",
            predictor_mode="active" if result.extractor_mode != "regex" else "baseline",
            prediction_summary={
                "event_type": "claims_extracted",
                "mission_id": mission_id,
                "extractor_mode": result.extractor_mode,
                "claim_count": result.claim_count,
                "extraction_time_ms": result.extraction_time_ms,
                "source_type": result.source.get("source_type", "unknown"),
                "error": result.error,
            },
        )
        tracker.record_event(event)
        
    except Exception as e:
        logger.debug(f"Failed to emit claims_extracted event: {e}")


__all__ = [
    "ClaimStore",
    "get_claim_store",
    "extract_and_store_claims",
]

