"""
Hybrid Memory System for DeepThinker 2.0.

Provides persistent storage for mission cognition including:
- Structured mission state with hypotheses, reflections, debates
- Per-mission and global RAG stores for evidence retrieval
- Long-term mission summaries for cross-mission intelligence

DeepThinker 2.0 Memory Discipline:
- MemoryPolicy: Defines memory caps and importance thresholds
- MemoryGuard: Enforces memory discipline with importance scoring

All memory operations are fail-safe - errors are logged but don't crash missions.
"""

from typing import TYPE_CHECKING

# Schemas - always available
from .schemas import (
    HypothesisSchema,
    EvidenceSchema,
    PhaseOutputSchema,
    SupervisorSignalsSchema,
    ReflectionSchema,
    DebateSchema,
    PlanRevisionSchema,
    MissionSummarySchema,
)

# Memory discipline (new in 2.0)
from .memory_policy import (
    MemoryPolicy,
    ContentType,
    IMPORTANCE_WEIGHTS,
    DEFAULT_MEMORY_POLICY,
    STRICT_MEMORY_POLICY,
    PERMISSIVE_MEMORY_POLICY,
)

from .memory_guard import (
    MemoryGuard,
    MemoryWriteRequest,
    MemoryWriteResult,
    get_memory_guard,
)

# Core components with graceful import handling
try:
    from .structured_state import StructuredMissionState
    STRUCTURED_STATE_AVAILABLE = True
except ImportError as e:
    STRUCTURED_STATE_AVAILABLE = False
    StructuredMissionState = None

try:
    from .rag_store import MissionRAGStore, GlobalRAGStore
    RAG_STORE_AVAILABLE = True
except ImportError as e:
    RAG_STORE_AVAILABLE = False
    MissionRAGStore = None
    GlobalRAGStore = None

try:
    from .general_knowledge_store import GeneralKnowledgeStore
    GENERAL_KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    GENERAL_KNOWLEDGE_AVAILABLE = False
    GeneralKnowledgeStore = None

try:
    from .summary_memory import SummaryMemory
    SUMMARY_MEMORY_AVAILABLE = True
except ImportError as e:
    SUMMARY_MEMORY_AVAILABLE = False
    SummaryMemory = None

try:
    from .memory_manager import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    MemoryManager = None

try:
    from .knowledge_router import (
        KnowledgeRouter,
        KnowledgeItem,
        RoutedKnowledge,
        get_knowledge_router,
        route_knowledge_for_persona,
        route_knowledge_for_council,
        enrich_context,
        PERSONA_DOMAIN_MAPPING,
        COUNCIL_DOMAIN_MAPPING,
    )
    KNOWLEDGE_ROUTER_AVAILABLE = True
except ImportError as e:
    KNOWLEDGE_ROUTER_AVAILABLE = False
    KnowledgeRouter = None
    KnowledgeItem = None
    RoutedKnowledge = None
    get_knowledge_router = None
    route_knowledge_for_persona = None
    route_knowledge_for_council = None
    enrich_context = None
    PERSONA_DOMAIN_MAPPING = {}
    COUNCIL_DOMAIN_MAPPING = {}

# Check if full memory system is available
MEMORY_SYSTEM_AVAILABLE = all([
    STRUCTURED_STATE_AVAILABLE,
    RAG_STORE_AVAILABLE,
    SUMMARY_MEMORY_AVAILABLE,
    MEMORY_MANAGER_AVAILABLE,
])

__all__ = [
    # Schemas
    "HypothesisSchema",
    "EvidenceSchema",
    "PhaseOutputSchema",
    "SupervisorSignalsSchema",
    "ReflectionSchema",
    "DebateSchema",
    "PlanRevisionSchema",
    "MissionSummarySchema",
    # Memory discipline (new in 2.0)
    "MemoryPolicy",
    "ContentType",
    "IMPORTANCE_WEIGHTS",
    "DEFAULT_MEMORY_POLICY",
    "STRICT_MEMORY_POLICY",
    "PERMISSIVE_MEMORY_POLICY",
    "MemoryGuard",
    "MemoryWriteRequest",
    "MemoryWriteResult",
    "get_memory_guard",
    # Core components
    "StructuredMissionState",
    "MissionRAGStore",
    "GlobalRAGStore",
    "GeneralKnowledgeStore",
    "SummaryMemory",
    "MemoryManager",
    # Knowledge Router
    "KnowledgeRouter",
    "KnowledgeItem",
    "RoutedKnowledge",
    "get_knowledge_router",
    "route_knowledge_for_persona",
    "route_knowledge_for_council",
    "enrich_context",
    "PERSONA_DOMAIN_MAPPING",
    "COUNCIL_DOMAIN_MAPPING",
    # Availability flags
    "MEMORY_SYSTEM_AVAILABLE",
    "STRUCTURED_STATE_AVAILABLE",
    "RAG_STORE_AVAILABLE",
    "GENERAL_KNOWLEDGE_AVAILABLE",
    "SUMMARY_MEMORY_AVAILABLE",
    "MEMORY_MANAGER_AVAILABLE",
    "KNOWLEDGE_ROUTER_AVAILABLE",
]

