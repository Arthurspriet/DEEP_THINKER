"""
Disk cache for arXiv Connector.

Provides caching for downloaded PDFs and source files with SHA256 deduplication.
Each cached file has a sidecar JSON file with metadata.

Cache key format: {arxiv_id}__{kind}__{version or 'latest'}
"""

import hashlib
import json
import logging
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import ArxivConfig, get_arxiv_config

logger = logging.getLogger(__name__)


class ArxivCache:
    """
    Disk cache for arXiv downloads.
    
    Storage structure:
        {cache_dir}/
            2501.01234__pdf__v1.pdf
            2501.01234__pdf__v1.json  (sidecar metadata)
            2501.01234__source__latest.tar.gz
            2501.01234__source__latest.json
    
    Sidecar JSON contains:
        - sha256: Content hash
        - retrieved_at: When downloaded
        - request_url: URL used
        - arxiv_id: Paper ID
        - version: Version string
        - kind: "pdf" or "source"
        - size_bytes: File size
    """
    
    def __init__(self, config: Optional[ArxivConfig] = None):
        """
        Initialize the cache.
        
        Args:
            config: ArxivConfig instance (uses global if None)
        """
        self.config = config or get_arxiv_config()
        self._cache_dir = Path(self.config.cache_dir)
        self._lock = threading.Lock()
        self._initialized = False
    
    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self._cache_dir
    
    def _ensure_dir(self) -> bool:
        """Ensure cache directory exists."""
        if self._initialized:
            return True
        
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.debug(f"[ARXIV_CACHE] Initialized cache dir: {self._cache_dir}")
            return True
        except Exception as e:
            logger.warning(f"[ARXIV_CACHE] Failed to create cache dir: {e}")
            return False
    
    def _make_key(self, arxiv_id: str, kind: str, version: Optional[str]) -> str:
        """
        Generate cache key from components.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            version: Version string or None for latest
            
        Returns:
            Cache key string
        """
        # Sanitize arxiv_id (remove any path separators)
        safe_id = arxiv_id.replace("/", "_").replace("\\", "_")
        version_str = version if version else "latest"
        return f"{safe_id}__{kind}__{version_str}"
    
    def _get_extension(self, kind: str) -> str:
        """Get file extension for kind."""
        if kind == "pdf":
            return ".pdf"
        elif kind == "source":
            return ".tar.gz"
        return ""
    
    def get_cache_path(
        self,
        arxiv_id: str,
        kind: str,
        version: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Get paths for cached file and sidecar.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            version: Version string or None
            
        Returns:
            Tuple of (file_path, sidecar_path)
        """
        key = self._make_key(arxiv_id, kind, version)
        ext = self._get_extension(kind)
        
        file_path = self._cache_dir / f"{key}{ext}"
        sidecar_path = self._cache_dir / f"{key}.json"
        
        return file_path, sidecar_path
    
    def get(
        self,
        arxiv_id: str,
        kind: str,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached item metadata if it exists.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            version: Version string or None
            
        Returns:
            Sidecar metadata dict if cached, None otherwise
        """
        file_path, sidecar_path = self.get_cache_path(arxiv_id, kind, version)
        
        if not file_path.exists() or not sidecar_path.exists():
            return None
        
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Verify file still exists and sha256 matches
            if file_path.exists():
                actual_sha256 = self._compute_sha256(file_path)
                if actual_sha256 == metadata.get("sha256"):
                    metadata["local_path"] = str(file_path)
                    return metadata
                else:
                    logger.warning(
                        f"[ARXIV_CACHE] SHA256 mismatch for {arxiv_id}, "
                        f"cache invalidated"
                    )
                    return None
            
            return None
            
        except Exception as e:
            logger.debug(f"[ARXIV_CACHE] Failed to read sidecar: {e}")
            return None
    
    def put(
        self,
        arxiv_id: str,
        kind: str,
        version: Optional[str],
        content: bytes,
        request_url: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Store content in cache.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            version: Version string or None
            content: File content bytes
            request_url: URL used to download
            
        Returns:
            Sidecar metadata dict if successful, None otherwise
        """
        if not self._ensure_dir():
            return None
        
        file_path, sidecar_path = self.get_cache_path(arxiv_id, kind, version)
        sha256 = hashlib.sha256(content).hexdigest()
        
        # Check for duplicate content (same sha256 in cache)
        existing = self._find_by_sha256(sha256, kind)
        if existing:
            logger.info(
                f"[ARXIV_CACHE] Deduplicating: {arxiv_id} matches "
                f"existing {existing.get('arxiv_id')}"
            )
        
        metadata = {
            "arxiv_id": arxiv_id,
            "version": version,
            "kind": kind,
            "sha256": sha256,
            "retrieved_at": datetime.utcnow().isoformat(),
            "request_url": request_url,
            "size_bytes": len(content),
            "local_path": str(file_path),
        }
        
        try:
            with self._lock:
                # Write content file
                with open(file_path, "wb") as f:
                    f.write(content)
                
                # Write sidecar metadata
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(
                f"[ARXIV_CACHE] Cached {arxiv_id} {kind} "
                f"({len(content)} bytes, sha256={sha256[:16]}...)"
            )
            return metadata
            
        except Exception as e:
            logger.warning(f"[ARXIV_CACHE] Failed to write cache: {e}")
            # Clean up partial writes
            try:
                if file_path.exists():
                    file_path.unlink()
                if sidecar_path.exists():
                    sidecar_path.unlink()
            except Exception:
                pass
            return None
    
    def has(
        self,
        arxiv_id: str,
        kind: str,
        version: Optional[str] = None,
    ) -> bool:
        """
        Check if item is cached.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            version: Version string or None
            
        Returns:
            True if cached
        """
        return self.get(arxiv_id, kind, version) is not None
    
    def delete(
        self,
        arxiv_id: str,
        kind: str,
        version: Optional[str] = None,
    ) -> bool:
        """
        Delete cached item.
        
        Args:
            arxiv_id: arXiv paper ID
            kind: "pdf" or "source"
            version: Version string or None
            
        Returns:
            True if deleted
        """
        file_path, sidecar_path = self.get_cache_path(arxiv_id, kind, version)
        
        deleted = False
        try:
            with self._lock:
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
                if sidecar_path.exists():
                    sidecar_path.unlink()
                    deleted = True
        except Exception as e:
            logger.warning(f"[ARXIV_CACHE] Failed to delete: {e}")
        
        return deleted
    
    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _find_by_sha256(self, sha256: str, kind: str) -> Optional[Dict[str, Any]]:
        """
        Find cached item by SHA256 hash.
        
        Args:
            sha256: Content hash
            kind: "pdf" or "source"
            
        Returns:
            Sidecar metadata if found, None otherwise
        """
        if not self._cache_dir.exists():
            return None
        
        try:
            for sidecar_path in self._cache_dir.glob(f"*__{kind}__*.json"):
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    if metadata.get("sha256") == sha256:
                        return metadata
                except Exception:
                    continue
        except Exception:
            pass
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self._cache_dir.exists():
            return {
                "cache_dir": str(self._cache_dir),
                "initialized": False,
                "pdf_count": 0,
                "source_count": 0,
                "total_size_bytes": 0,
            }
        
        pdf_count = 0
        source_count = 0
        total_size = 0
        
        try:
            for path in self._cache_dir.iterdir():
                if path.suffix == ".json":
                    continue
                if "__pdf__" in path.name:
                    pdf_count += 1
                elif "__source__" in path.name:
                    source_count += 1
                total_size += path.stat().st_size
        except Exception as e:
            logger.debug(f"[ARXIV_CACHE] Error getting stats: {e}")
        
        return {
            "cache_dir": str(self._cache_dir),
            "initialized": True,
            "pdf_count": pdf_count,
            "source_count": source_count,
            "total_size_bytes": total_size,
        }
    
    def clear(self) -> int:
        """
        Clear all cached items.
        
        Returns:
            Number of items deleted
        """
        if not self._cache_dir.exists():
            return 0
        
        count = 0
        try:
            with self._lock:
                for path in self._cache_dir.iterdir():
                    try:
                        path.unlink()
                        count += 1
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"[ARXIV_CACHE] Error clearing cache: {e}")
        
        logger.info(f"[ARXIV_CACHE] Cleared {count} cached files")
        return count


# Global cache instance
_cache: Optional[ArxivCache] = None


def get_arxiv_cache(config: Optional[ArxivConfig] = None) -> ArxivCache:
    """
    Get global arXiv cache instance.
    
    Args:
        config: Optional config override
        
    Returns:
        ArxivCache instance
    """
    global _cache
    if _cache is None or config is not None:
        _cache = ArxivCache(config)
    return _cache


def reset_arxiv_cache() -> None:
    """Reset global cache instance (for testing)."""
    global _cache
    _cache = None


