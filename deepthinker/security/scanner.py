"""
Security Scanner for DeepThinker 2.0.

Static code analysis for detecting dangerous patterns and security risks.
Re-exports from execution module for backwards compatibility.
"""

# Re-export from existing implementation
from ..execution.security_scanner import (
    SecurityScanner,
    SecurityIssue,
    RiskLevel,
)

__all__ = [
    "SecurityScanner",
    "SecurityIssue",
    "RiskLevel",
]

