# Memory Module

The memory module provides **persistent and contextual memory systems** for maintaining knowledge across missions and phases.

## Core Concepts

### Memory Types

| Type | Scope | Purpose |
|------|-------|---------|
| **Short-Term** | Current phase | Immediate context, recent outputs |
| **Long-Term** | Across missions | Learned patterns, knowledge base |
| **Episodic** | Per mission | Decision history, hypotheses |

## Components

### 1. RAG (Retrieval-Augmented Generation)
Vector-based retrieval for finding relevant prior knowledge.

```
Query → Embed → Vector Search → Top-K Results → Augment Prompt
```

**Files**: `rag_store.py`, `embeddings.py`

### 2. Knowledge Base
Structured storage for facts, documents, and learned information.

**Files**: `knowledge_base.py`, `document_store.py`

### 3. Context Manager
Manages context flow between phases, compressing and summarizing as needed.

**Files**: `context_manager.py`, `compression.py`

### 4. Session Memory
Tracks the current execution session's state and history.

**Files**: `session_memory.py`

## Key Files

| File | Purpose |
|------|---------|
| `rag_store.py` | Vector store for retrieval |
| `embeddings.py` | Text embedding generation |
| `knowledge_base.py` | Structured knowledge storage |
| `context_manager.py` | Phase-to-phase context flow |
| `compression.py` | Context compression for token limits |
| `session_memory.py` | Current session state |

## Memory Flow

```
Phase N Execution
       ↓
   Artifacts Generated
       ↓
   ┌───────────────────────────────────┐
   │  Short-Term: Pass to Phase N+1   │
   │  Long-Term: Index for future RAG │
   │  Episodic: Log decisions made    │
   └───────────────────────────────────┘
       ↓
Phase N+1 Execution (with context)
```

## Usage

```python
from deepthinker.memory import RAGStore, ContextManager

# RAG retrieval
rag = RAGStore(store_path="kb/rag")
relevant_docs = rag.search("How to implement authentication?", top_k=5)

# Context management
ctx_manager = ContextManager()
ctx_manager.add_phase_context("research", {"findings": "..."})
compressed = ctx_manager.get_compressed_context(max_tokens=2000)
```

## Storage Locations

```
kb/
├── rag/              # Vector embeddings
├── general_knowledge/ # Shared knowledge base
├── missions/         # Per-mission memory
│   └── {mission_id}/
│       ├── state.json
│       └── rag/
└── long_memory/      # Cross-mission summaries
```


