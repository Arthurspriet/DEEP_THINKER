"""
Supervisor package for DeepThinker 2.0.

Provides intelligent model supervision using a lightweight CPU-based LLM
for dynamic model selection and resource optimization.

Key features:
- GPU-aware model selection based on available VRAM
- Phase importance weighting for critical tasks
- Wait-for-capacity decisions vs immediate downgrade
- Fallback model support for timeout scenarios
"""

from .model_supervisor import (
    ModelSupervisor,
    SupervisorDecision,
    PHASE_IMPORTANCE,
    MODEL_TIER_QUALITY,
)

__all__ = [
    "ModelSupervisor",
    "SupervisorDecision",
    "PHASE_IMPORTANCE",
    "MODEL_TIER_QUALITY",
]

