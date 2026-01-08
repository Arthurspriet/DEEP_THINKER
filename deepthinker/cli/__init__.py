"""
DeepThinker CLI utilities for verbose logging and context inspection.

This package provides comprehensive tools for understanding and debugging
the DeepThinker workflow execution, including:

- Council auto-discovery and metadata extraction
- Context flow tracing between components
- Rich terminal output for readable logging
- State manager inspection
"""

from .verbose_logger import (
    VerboseLogger,
    verbose_logger,
    configure_verbose_logging,
    RICH_AVAILABLE,
)

__all__ = [
    "VerboseLogger",
    "verbose_logger",
    "configure_verbose_logging",
    "RICH_AVAILABLE",
]

