"""
API Routes for DeepThinker.
"""

from .missions import router as missions_router
from .workflows import router as workflows_router
from .agents import router as agents_router
from .gpu import router as gpu_router
from .config import router as config_router

__all__ = [
    "missions_router",
    "workflows_router", 
    "agents_router",
    "gpu_router",
    "config_router"
]

