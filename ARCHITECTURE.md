# DeepThinker Architecture Guide

This document provides a visual guide to the DeepThinker multi-agent AI system architecture.

## Table of Contents

- [System Overview](#system-overview)
- [Mission Lifecycle](#mission-lifecycle)
- [Council Architecture](#council-architecture)
- [Phase Execution](#phase-execution)
- [Consensus Mechanisms](#consensus-mechanisms)
- [Memory Systems](#memory-systems)
- [Key Concepts](#key-concepts)

---

## System Overview

DeepThinker is a **time-bounded, multi-agent autonomous AI system** that orchestrates specialized councils to accomplish complex objectives.

```mermaid
flowchart TB
    subgraph UserInterface [User Interface Layer]
        CLI[CLI - main.py]
        WebUI[React Frontend]
        API[FastAPI Server]
    end

    subgraph Orchestration [Orchestration Layer]
        MO[Mission Orchestrator]
        TM[Time Manager]
        CS[Cognitive Spine]
    end

    subgraph Councils [Council Layer]
        PC[Planner Council]
        RC[Researcher Council]
        CC[Coder Council]
        EC[Evaluator Council]
        SC[Simulation Council]
    end

    subgraph Support [Support Systems]
        ARB[Arbiter]
        MEM[Memory Systems]
        LLM[Ollama LLMs]
    end

    CLI --> MO
    WebUI --> API
    API --> MO
    
    MO --> TM
    MO --> CS
    MO --> PC
    MO --> RC
    MO --> CC
    MO --> EC
    MO --> SC
    
    PC --> ARB
    RC --> ARB
    CC --> ARB
    EC --> ARB
    SC --> ARB
    
    Councils --> MEM
    Councils --> LLM
```

---

## Mission Lifecycle

A mission progresses through distinct phases, each with allocated time budgets.

```mermaid
stateDiagram-v2
    [*] --> Pending: Mission Created
    Pending --> Running: Start Execution
    
    Running --> Reconnaissance: Phase 1
    Reconnaissance --> AnalysisPlanning: Phase 2
    AnalysisPlanning --> DeepAnalysis: Phase 3
    DeepAnalysis --> Synthesis: Phase 4
    
    Synthesis --> Completed: All Phases Done
    Running --> Expired: Time Budget Exhausted
    Running --> Aborted: User Abort
    Running --> Failed: Unrecoverable Error
    
    Completed --> [*]
    Expired --> [*]
    Aborted --> [*]
    Failed --> [*]
```

### Phase Time Allocation

Time budgets are automatically distributed across phases based on mission complexity:

```mermaid
pie title Time Budget Distribution (30-min Mission)
    "Reconnaissance" : 15
    "Analysis & Planning" : 25
    "Deep Analysis" : 40
    "Synthesis & Report" : 20
```

---

## Council Architecture

Each council is a specialized team of AI agents working toward a common goal.

### Council Responsibilities

```mermaid
flowchart LR
    subgraph PlannerCouncil [Planner Council]
        direction TB
        P1[Strategy Agent]
        P2[Workflow Agent]
        P3[Requirements Agent]
    end

    subgraph ResearcherCouncil [Researcher Council]
        direction TB
        R1[Web Search Agent]
        R2[RAG Agent]
        R3[Source Validator]
    end

    subgraph CoderCouncil [Coder Council]
        direction TB
        C1[Code Generator]
        C2[Code Reviewer]
        C3[Refactor Agent]
    end

    subgraph EvaluatorCouncil [Evaluator Council]
        direction TB
        E1[Quality Assessor]
        E2[Feedback Agent]
        E3[Scoring Agent]
    end

    subgraph SimulationCouncil [Simulation Council]
        direction TB
        S1[Scenario Builder]
        S2[Test Runner]
        S3[Edge Case Finder]
    end

    Objective([Objective]) --> PlannerCouncil
    PlannerCouncil --> ResearcherCouncil
    ResearcherCouncil --> CoderCouncil
    CoderCouncil --> EvaluatorCouncil
    EvaluatorCouncil --> SimulationCouncil
    SimulationCouncil --> Output([Final Output])
```

### Multi-View Reasoning

For balanced analysis, DeepThinker employs optimist and skeptic councils:

```mermaid
flowchart TB
    Input[Analysis Input] --> Split{Multi-View Split}
    
    Split --> Optimist[Optimist Council]
    Split --> Skeptic[Skeptic Council]
    
    Optimist --> OA[Strengths & Opportunities]
    Skeptic --> SA[Risks & Weaknesses]
    
    OA --> Blend[Weighted Blend]
    SA --> Blend
    
    Blend --> Arbiter[Arbiter Reconciliation]
    Arbiter --> Final[Balanced Output]
```

---

## Phase Execution

Each phase contains discrete steps executed by the Step Engine.

```mermaid
sequenceDiagram
    participant MO as Mission Orchestrator
    participant Phase as Current Phase
    participant SE as Step Executor
    participant Council as Active Council
    participant LLM as Ollama Model

    MO->>Phase: Start Phase
    Phase->>SE: Execute Steps
    
    loop For Each Step
        SE->>Council: Delegate Step
        Council->>LLM: Generate Response
        LLM-->>Council: Model Output
        Council->>SE: Step Result
        SE->>SE: Validate Output
    end
    
    SE->>Phase: All Steps Complete
    Phase->>MO: Phase Artifacts
    MO->>MO: Checkpoint State
```

### Iterative Refinement Loop

Councils can iterate to improve quality:

```mermaid
flowchart TB
    Start([Start Phase]) --> Execute[Execute Council]
    Execute --> Evaluate{Evaluate Quality}
    
    Evaluate -->|Score >= Threshold| Complete([Phase Complete])
    Evaluate -->|Score < Threshold| Check{Iterations < Max?}
    
    Check -->|Yes| Feedback[Generate Feedback]
    Feedback --> Revise[Revise Output]
    Revise --> Execute
    
    Check -->|No| Complete
```

---

## Consensus Mechanisms

Multiple agents reach consensus through structured mechanisms.

### Critique Exchange

```mermaid
flowchart LR
    subgraph Round1 [Round 1: Initial Proposals]
        A1[Agent 1 Proposal]
        A2[Agent 2 Proposal]
        A3[Agent 3 Proposal]
    end

    subgraph Round2 [Round 2: Critique]
        C1[Agent 1 Critiques 2,3]
        C2[Agent 2 Critiques 1,3]
        C3[Agent 3 Critiques 1,2]
    end

    subgraph Round3 [Round 3: Synthesis]
        S[Synthesized Output]
    end

    A1 --> C2
    A1 --> C3
    A2 --> C1
    A2 --> C3
    A3 --> C1
    A3 --> C2
    
    C1 --> S
    C2 --> S
    C3 --> S
```

### Voting Mechanism

```mermaid
flowchart TB
    Proposals[Multiple Proposals] --> Vote{Voting Round}
    
    Vote --> V1[Vote: Proposal A - 3]
    Vote --> V2[Vote: Proposal B - 2]
    Vote --> V3[Vote: Proposal C - 1]
    
    V1 --> Winner[Winner: Proposal A]
    
    Winner --> Blend[Weighted Blend with Runner-up]
    Blend --> Final[Final Consensus]
```

---

## Memory Systems

DeepThinker uses multiple memory systems for context retention.

```mermaid
flowchart TB
    subgraph ShortTerm [Short-Term Memory]
        CTX[Phase Context]
        ART[Phase Artifacts]
        LOG[Execution Logs]
    end

    subgraph LongTerm [Long-Term Memory]
        RAG[RAG Vector Store]
        KB[Knowledge Base]
        SUM[Compressed Summaries]
    end

    subgraph Episodic [Episodic Memory]
        DEC[Decision Records]
        CLM[Claim Registry]
        HYP[Hypothesis Tracker]
    end

    Mission[Active Mission] --> ShortTerm
    ShortTerm --> LongTerm
    
    Mission --> Episodic
    Episodic --> LongTerm
    
    LongTerm --> Retrieval[Context Retrieval]
    Retrieval --> Mission
```

### RAG Flow

```mermaid
sequenceDiagram
    participant Q as Query
    participant E as Embeddings
    participant VS as Vector Store
    participant LLM as Language Model

    Q->>E: Generate Query Embedding
    E->>VS: Similarity Search
    VS-->>E: Top-K Documents
    E->>LLM: Query + Retrieved Context
    LLM-->>Q: Augmented Response
```

---

## Key Concepts

### Effort Levels

Missions are categorized by time investment:

| Level | Duration | Use Case |
|-------|----------|----------|
| Quick | 5-15 min | Simple research, quick analysis |
| Standard | 15-60 min | Moderate complexity tasks |
| Deep | 1-4 hours | In-depth analysis, complex coding |
| Marathon | 4+ hours | Comprehensive projects |

### Phase Types

| Phase | Purpose | Primary Council |
|-------|---------|-----------------|
| Reconnaissance | Gather context and resources | Researcher |
| Analysis & Planning | Strategic planning | Planner |
| Deep Analysis | In-depth investigation | Evaluator + Evidence |
| Synthesis | Consolidate findings | Planner |

### Arbiter Role

The Arbiter is the final decision-maker that:

```mermaid
flowchart LR
    C1[Council 1 Output] --> ARB[Arbiter]
    C2[Council 2 Output] --> ARB
    C3[Council 3 Output] --> ARB
    
    ARB --> Analyze[Analyze Conflicts]
    Analyze --> Reconcile[Reconcile Differences]
    Reconcile --> Final[Final Decision]
```

---

## Data Flow Example

Complete flow for a research mission:

```mermaid
flowchart TB
    subgraph Input
        OBJ[User Objective]
        TIME[Time Budget]
        OPTS[Options]
    end

    subgraph Phase1 [Phase 1: Reconnaissance]
        WS[Web Search]
        RAG1[RAG Retrieval]
        SRC[Source Collection]
    end

    subgraph Phase2 [Phase 2: Analysis]
        PLAN[Strategic Plan]
        REQ[Requirements]
        WF[Workflow Design]
    end

    subgraph Phase3 [Phase 3: Deep Analysis]
        EVD[Evidence Gathering]
        EVAL[Quality Evaluation]
        MV[Multi-View Analysis]
    end

    subgraph Phase4 [Phase 4: Synthesis]
        CON[Consolidation]
        RPT[Report Generation]
        DEL[Deliverables]
    end

    subgraph Output
        ARTS[Artifacts]
        FILES[Output Files]
        STATE[Mission State]
    end

    Input --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Output
```

---

## Technology Stack

```mermaid
flowchart TB
    subgraph Frontend
        React[React 18]
        Vite[Vite]
        TW[Tailwind CSS]
    end

    subgraph Backend
        FastAPI[FastAPI]
        Pydantic[Pydantic]
        SSE[Server-Sent Events]
    end

    subgraph Core
        Python[Python 3.10+]
        Click[Click CLI]
        Asyncio[Asyncio]
    end

    subgraph AI
        Ollama[Ollama]
        LangChain[LangChain]
        FAISS[FAISS Vector Store]
    end

    subgraph Infrastructure
        Docker[Docker Sandbox]
        SQLite[SQLite/JSON Store]
    end

    Frontend --> Backend
    Backend --> Core
    Core --> AI
    Core --> Infrastructure
```

---

## Further Reading

- [README.md](README.md) - Quick start and CLI reference
- [deepthinker/missions/](deepthinker/missions/) - Mission orchestration code
- [deepthinker/councils/](deepthinker/councils/) - Council implementations
- [deepthinker/consensus/](deepthinker/consensus/) - Consensus mechanisms

