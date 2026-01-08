"""
Model Registry for Phase Risk Predictor.

Local storage and versioning for trained ML models.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import MODEL_STORAGE_DIR, FEATURE_VECTOR_VERSION

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata for a trained prediction model.
    
    Attributes:
        version: Model version number
        trained_at: ISO timestamp of training
        dataset_size: Number of samples used for training
        feature_version: Version of feature encoding used
        validation_metrics: Validation metrics (retry_mae, retry_auc, etc.)
        model_type: Type of model (e.g., "RandomForest", "XGBoost")
        training_config: Configuration used during training
    """
    version: int
    trained_at: str
    dataset_size: int
    feature_version: int
    validation_metrics: Dict[str, float]
    model_type: str = "RandomForest"
    training_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.training_config is None:
            self.training_config = {}
        if self.validation_metrics is None:
            self.validation_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


class PhaseRiskModelRegistry:
    """
    Local registry for trained phase risk prediction models.
    
    Features:
    - Versioned model storage
    - Metadata tracking (training time, dataset size, validation metrics)
    - Automatic version numbering
    - Feature version compatibility checking
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the model registry.
        
        Args:
            base_dir: Base directory for model storage. 
                     Defaults to kb/models/phase_risk_predictor
        """
        if base_dir is None:
            base_dir = Path(MODEL_STORAGE_DIR)
        
        self.base_dir = Path(base_dir)
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure the storage directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _model_path(self, version: int) -> Path:
        """Get path for model file."""
        return self.base_dir / f"model_v{version}.joblib"
    
    def _metadata_path(self, version: int) -> Path:
        """Get path for metadata file."""
        return self.base_dir / f"metadata_v{version}.json"
    
    def list_versions(self) -> List[int]:
        """
        List all available model versions.
        
        Returns:
            List of version numbers, sorted ascending
        """
        versions = []
        
        for path in self.base_dir.glob("model_v*.joblib"):
            try:
                # Extract version from filename
                version_str = path.stem.replace("model_v", "")
                version = int(version_str)
                versions.append(version)
            except (ValueError, IndexError):
                logger.warning(f"Invalid model filename: {path}")
        
        versions.sort()
        return versions
    
    def get_next_version(self) -> int:
        """
        Get the next available version number.
        
        Returns:
            Next version number (1 if no models exist)
        """
        versions = self.list_versions()
        if not versions:
            return 1
        return max(versions) + 1
    
    def get_latest_version(self) -> Optional[int]:
        """
        Get the latest model version.
        
        Returns:
            Latest version number, or None if no models exist
        """
        versions = self.list_versions()
        if not versions:
            return None
        return max(versions)
    
    def load_metadata(self, version: int) -> Optional[ModelMetadata]:
        """
        Load metadata for a specific model version.
        
        Args:
            version: Model version to load metadata for
            
        Returns:
            ModelMetadata or None if not found
        """
        metadata_path = self._metadata_path(version)
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found for version {version}")
            return None
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load metadata for version {version}: {e}")
            return None
    
    def load_model(self, version: int) -> Optional[Any]:
        """
        Load a model by version number.
        
        Args:
            version: Model version to load
            
        Returns:
            Loaded model object, or None if not found
        """
        model_path = self._model_path(version)
        
        if not model_path.exists():
            logger.warning(f"Model not found for version {version}")
            return None
        
        try:
            import joblib
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model version {version}: {e}")
            return None
    
    def load_latest(self) -> Optional[Tuple[Any, ModelMetadata]]:
        """
        Load the latest compatible model with its metadata.
        
        Compatibility is checked based on feature version.
        
        Returns:
            Tuple of (model, metadata) or None if no compatible model
        """
        versions = self.list_versions()
        
        # Try versions from newest to oldest
        for version in reversed(versions):
            metadata = self.load_metadata(version)
            
            if metadata is None:
                continue
            
            # Check feature version compatibility
            if metadata.feature_version != FEATURE_VECTOR_VERSION:
                logger.warning(
                    f"Model v{version} has incompatible feature version "
                    f"({metadata.feature_version} != {FEATURE_VECTOR_VERSION})"
                )
                continue
            
            # Load the model
            model = self.load_model(version)
            if model is not None:
                logger.info(f"Loaded model v{version} (trained on {metadata.dataset_size} samples)")
                return model, metadata
        
        logger.info("No compatible trained model found")
        return None
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> bool:
        """
        Save a trained model with its metadata.
        
        Args:
            model: Trained model object (must be joblib-serializable)
            metadata: Model metadata
            
        Returns:
            True if save was successful
        """
        self._ensure_dir()
        
        version = metadata.version
        model_path = self._model_path(version)
        metadata_path = self._metadata_path(version)
        
        try:
            # Save model
            import joblib
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")
            
            # Save metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model version {version}: {e}")
            # Clean up partial saves
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            return False
    
    def delete_version(self, version: int) -> bool:
        """
        Delete a specific model version.
        
        Args:
            version: Version to delete
            
        Returns:
            True if deletion was successful
        """
        model_path = self._model_path(version)
        metadata_path = self._metadata_path(version)
        
        deleted = False
        
        if model_path.exists():
            model_path.unlink()
            deleted = True
        
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        
        if deleted:
            logger.info(f"Deleted model version {version}")
        
        return deleted
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored models.
        
        Returns:
            Dictionary with registry statistics
        """
        versions = self.list_versions()
        
        stats = {
            "total_versions": len(versions),
            "versions": versions,
            "storage_dir": str(self.base_dir),
            "current_feature_version": FEATURE_VECTOR_VERSION,
        }
        
        if versions:
            latest = max(versions)
            metadata = self.load_metadata(latest)
            if metadata:
                stats["latest_version"] = latest
                stats["latest_trained_at"] = metadata.trained_at
                stats["latest_dataset_size"] = metadata.dataset_size
                stats["latest_validation_metrics"] = metadata.validation_metrics
        
        return stats

