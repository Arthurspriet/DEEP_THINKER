"""
Decision Accountability Layer for DeepThinker.

Provides first-class decision records that capture system choices,
enabling cost attribution, mission review, and learning signals.

Key components:
- DecisionRecord: First-class artifact for system choices
- DecisionType: Enum of decision categories
- OutcomeCause: Enum of phase outcome causes
- DecisionEmitter: Stateless emitter interface
- DecisionStore: JSONL persistence layer
"""

from .decision_record import (
    DecisionRecord,
    DecisionType,
    OutcomeCause,
)
from .decision_emitter import DecisionEmitter
from .decision_store import DecisionStore

__all__ = [
    # Core data types
    "DecisionRecord",
    "DecisionType",
    "OutcomeCause",
    # Emitter
    "DecisionEmitter",
    # Storage
    "DecisionStore",
]

