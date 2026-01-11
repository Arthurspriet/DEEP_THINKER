"""
Configuration for arXiv Connector.

Provides feature flags and settings for arXiv API access.
All settings are configurable via environment variables.
All feature flags default to OFF for backward compatibility.

Environment Variables:
    DEEPTHINKER_ARXIV_ENABLED: Master enable flag (default: false)
    DEEPTHINKER_ARXIV_INGEST_ENABLED: RAG ingestion flag (default: false)
    DEEPTHINKER_ARXIV_API_INTERVAL_SEC: Min seconds between API calls (default: 3)
    DEEPTHINKER_ARXIV_DL_INTERVAL_SEC: Min seconds between downloads (default: 10)
    DEEPTHINKER_ARXIV_CACHE_DIR: Cache directory path (default: kb/arxiv/cache)
    DEEPTHINKER_ARXIV_USER_AGENT: User-Agent header (default: DeepThinker/1.0)
    DEEPTHINKER_ARXIV_MAX_RESULTS: Max results per search (default: 50)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ArxivConfig:
    """
    Configuration for the arXiv connector.
    
    All settings are configurable via environment variables.
    All feature flags default to OFF for backward compatibility.
    
    Attributes:
        enabled: Master enable flag for arXiv connector
        ingest_enabled: Whether to ingest PDFs into RAG store
        api_interval_sec: Minimum seconds between API calls (rate limiting)
        dl_interval_sec: Minimum seconds between downloads (rate limiting)
        cache_dir: Directory for caching downloaded files
        user_agent: User-Agent header for API requests
        max_results: Maximum results per search query
        api_base_url: arXiv API base URL
        pdf_base_url: arXiv PDF download base URL
        source_base_url: arXiv source (e-print) download base URL
    """
    
    # Feature flags (default OFF)
    enabled: bool = False
    ingest_enabled: bool = False
    
    # Rate limiting
    api_interval_sec: float = 3.0
    dl_interval_sec: float = 10.0
    
    # Storage
    cache_dir: str = "kb/arxiv/cache"
    
    # API settings
    user_agent: str = "DeepThinker/1.0 (https://github.com/deepthinker)"
    max_results: int = 50
    
    # API endpoints
    api_base_url: str = "https://export.arxiv.org/api/query"
    pdf_base_url: str = "https://arxiv.org/pdf"
    source_base_url: str = "https://arxiv.org/e-print"
    
    @property
    def is_enabled(self) -> bool:
        """Check if arXiv connector is enabled."""
        return self.enabled
    
    @property
    def is_ingest_enabled(self) -> bool:
        """Check if RAG ingestion is enabled."""
        return self.enabled and self.ingest_enabled
    
    @classmethod
    def from_env(cls) -> "ArxivConfig":
        """
        Create configuration from environment variables.
        
        All feature flags default to OFF for backward compatibility.
        
        Returns:
            ArxivConfig instance
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes", "on"):
                return True
            elif val in ("false", "0", "no", "off"):
                return False
            return default
        
        def get_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        
        config = cls(
            enabled=get_bool("DEEPTHINKER_ARXIV_ENABLED", False),
            ingest_enabled=get_bool("DEEPTHINKER_ARXIV_INGEST_ENABLED", False),
            api_interval_sec=get_float("DEEPTHINKER_ARXIV_API_INTERVAL_SEC", 3.0),
            dl_interval_sec=get_float("DEEPTHINKER_ARXIV_DL_INTERVAL_SEC", 10.0),
            cache_dir=os.environ.get("DEEPTHINKER_ARXIV_CACHE_DIR", "kb/arxiv/cache"),
            user_agent=os.environ.get(
                "DEEPTHINKER_ARXIV_USER_AGENT",
                "DeepThinker/1.0 (https://github.com/deepthinker)"
            ),
            max_results=get_int("DEEPTHINKER_ARXIV_MAX_RESULTS", 50),
        )
        
        if config.enabled:
            logger.info(
                f"[ARXIV] Config loaded: enabled={config.enabled}, "
                f"ingest={config.ingest_enabled}, "
                f"api_interval={config.api_interval_sec}s, "
                f"cache_dir={config.cache_dir}"
            )
        
        return config
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "ingest_enabled": self.ingest_enabled,
            "api_interval_sec": self.api_interval_sec,
            "dl_interval_sec": self.dl_interval_sec,
            "cache_dir": self.cache_dir,
            "user_agent": self.user_agent,
            "max_results": self.max_results,
            "api_base_url": self.api_base_url,
            "pdf_base_url": self.pdf_base_url,
            "source_base_url": self.source_base_url,
        }


# Global config instance
_config: Optional[ArxivConfig] = None


def get_arxiv_config() -> ArxivConfig:
    """
    Get global arXiv config (lazy-loaded from env).
    
    Returns:
        ArxivConfig instance
    """
    global _config
    if _config is None:
        _config = ArxivConfig.from_env()
    return _config


def reset_arxiv_config() -> None:
    """Reset global config (for testing)."""
    global _config
    _config = None


def is_arxiv_enabled() -> bool:
    """
    Check if arXiv connector is enabled.
    
    Convenience function for quick checks.
    
    Returns:
        True if enabled
    """
    return get_arxiv_config().is_enabled


