# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepThinker is a long-running, time-bounded multi-agent autonomous AI system. It uses local LLMs via Ollama, orchestrated through two distinct execution stacks: a CrewAI-based code generation workflow and a council-based mission engine for complex multi-phase analysis.

Python 3.10+. Built with Poetry. Uses Click for CLI, FastAPI for the REST API, React+Vite for the frontend.

## Commands

```bash
# Install
pip install -e .
# or
poetry install

# Run tests
pytest tests/ -v
pytest tests/test_arxiv.py -v                    # single file
pytest tests/ --cov=deepthinker --cov-report=html # with coverage

# CLI
python3 main.py --help
python3 main.py test-connection
python3 main.py list-models

# Run workflow (CrewAI code generation stack)
python3 main.py run "Build a binary search tree" --model deepseek-r1:8b --verbose

# Run mission (council-based multi-phase stack)
python3 main.py mission start -o "Analyze topic X" -t 30 --verbose
python3 main.py mission status --id <id>
python3 main.py mission list
python3 main.py mission resume --id <id>
python3 main.py mission abort --id <id>

# Context inspection
python3 main.py context councils --show-prompt
python3 main.py context mission --id <id> --phases --iterations
python3 main.py context state

# Knowledge packs
python3 main.py knowledge list
python3 main.py knowledge show --pack <id>
python3 main.py knowledge install <spec_path>
python3 main.py knowledge scaffold <topic> --install

# API server
uvicorn api.server:app --reload --port 8000

# Frontend dev server
cd frontend && npm run dev
```

## Two Execution Stacks

### 1. Run Workflow (`python3 main.py run`)
- **Purpose**: Single-pass code generation with iterative refinement
- **Engine**: CrewAI agents (`deepthinker/execution/run_workflow.py`)
- **Agents**: Planner, WebSearch, Coder, Evaluator, Simulator, Executor (from `deepthinker/agents/`)
- **Flow**: Optional planning → optional research → code → evaluate → iterate until quality threshold
- **No time budgets** — bounded by iteration count and quality score

### 2. Mission Engine (`python3 main.py mission start`)
- **Purpose**: Long-running multi-phase research/analysis with time budgets
- **Engine**: Council architecture (`deepthinker/missions/mission_orchestrator.py`)
- **Phases**: Reconnaissance → Analysis → Deep → Synthesis (auto-planned based on objective)
- **Councils**: Multi-LLM deliberation with consensus mechanisms
- **Effort levels**: QUICK (<5min), STANDARD (5-30min), DEEP (30-120min), MARATHON (>120min) — auto-inferred from time budget
- **Supports**: Checkpointing, resumability, SSE events, REST API access

## Architecture Layers

**Entry points**: `main.py` (CLI), `api/server.py` (FastAPI), `frontend/` (React/Vite)

**Mission orchestration** (`deepthinker/missions/`): `MissionOrchestrator` creates missions with time budgets, manages phase execution, coordinates councils, checkpoints state. `MissionTimeManager` tracks budgets. `MissionState`/`MissionPhase`/`MissionConstraints` in `mission_types.py`.

**Councils** (`deepthinker/councils/`): `BaseCouncil` is the abstract base for all councils. Each council runs multiple LLMs via `ModelPool`, applies consensus, validates outputs. Council types: Planner, Researcher, Coder, Evaluator, Simulation, Explorer, Evidence, Multi-View (Optimist/Skeptic), OutputAdapter. `DynamicCouncilFactory` builds councils at runtime based on mission context.

**Step Engine** (`deepthinker/steps/`): `StepExecutor` executes individual steps using single models (sits below councils). Default model and temperature mappings per step type in `step_executor.py`. Councils handle strategy; StepExecutor handles doing.

**Core** (`deepthinker/core/`):
- `CognitiveSpine`: Central schema validation, output contracts, resource budget enforcement, memory compression
- `SafetyCoreRegistry` (`safety_registry.py`): Centralized registry for optional safety modules with graceful degradation
- `PhaseValidator`: Hard phase contract enforcement
- `ConfigManager`: Unified configuration management
- `StateLayer`: Session state management (transient or persistent backends)
- `CouncilBridge`: Bridges between CrewAI agents and council system

**Consensus** (`deepthinker/consensus/`): Critique exchange (3 rounds), voting (semantic similarity clustering), weighted blend, semantic distance (embedding-based).

**Arbiter** (`deepthinker/arbiter/`): Final decision reconciliation across council outputs.

**Models** (`deepthinker/models/`): `OllamaLoader` connects to Ollama. `ModelPool` manages concurrent multi-model execution. `ModelRegistry` categorizes models into tiers (SMALL/MEDIUM/LARGE). `ModelSelector` picks models based on phase/context. `call_model`/`call_model_async` are the unified calling interface. `StepTierPolicy` enforces model tier requirements for truth-critical steps.

**Outputs** (`deepthinker/outputs/`): Generates mission deliverables in PDF, HTML dashboard, slides, and code repo formats. Uses a design system (`DeepThinkerStyle`).

**Meta-cognition** (`deepthinker/meta/`): After each mission phase — reflection, hypothesis updates, internal debate (optimist vs skeptic), plan revision, reasoning supervision, depth evaluation.

**ML learning layer**:
- Bandits (`deepthinker/bandits/`): UCB & Thompson Sampling for model/council/config selection
- Routing (`deepthinker/routing/`): ML-based advisory router for council set and model tier selection
- Rewards (`deepthinker/rewards/`): Unified reward computation clamped to [-1, +1]
- Learning (`deepthinker/learning/`): Online stop/escalate predictor, quality prediction
- Alignment (`deepthinker/alignment/`): Drift detection, escalation ladder, corrective pressure

**Safety & governance**:
- Constitution (`deepthinker/constitution/`): Rule-based enforcement, blinded reasoning, audit ledger
- Governance (`deepthinker/governance/`): Normative layer with rules, phase contracts
- Proofs (`deepthinker/proofs/`): Proof-carrying reasoning, claim extraction, evidence binding, integrity checks

**Memory** (`deepthinker/memory/`): RAG store (FAISS), general knowledge store, memory manager, knowledge router. Runtime data stored in `kb/` directory.

**Knowledge Packs** (`deepthinker/knowledge_packs/`): Installable config bundles defining knowledge sources (arXiv queries), retrieval policy, tool/council/evaluator policies. Managed via CLI or API.

## Key Patterns

**Graceful degradation via try/except imports**: Nearly every module uses optional imports with `*_AVAILABLE` boolean flags. If a component fails to import, the system logs it and continues without it. `SafetyCoreRegistry` centralizes this pattern for safety modules. When adding new optional dependencies, follow the same pattern.

**Effort-based execution**: Mission time budget auto-determines effort level (QUICK/STANDARD/DEEP/MARATHON), which controls rounds, branches, model selection, and quality thresholds via `EFFORT_PRESETS` in `mission_types.py`.

**Model refusal detection**: `BaseCouncil` detects LLM refusals via `MODEL_REFUSAL_PATTERNS` and retries with different prompting.

**Default model mapping**: `StepExecutor` maps step types to default models (research→gemma3:12b, coding→deepseek-r1:8b, synthesis→gemma3:27b, etc.) and temperatures.

**SSE event publishing**: Councils and orchestrator publish real-time events via `api/sse.py` for the web UI. Helper `_publish_sse_event()` safely handles async scheduling from sync code.

## Environment

Ollama must be running locally. Configure via env vars or `.env` files (project root and `deepthinker/.env`):
- `OLLAMA_API_BASE` (default: `http://localhost:11434`)
- `DEEPTHINKER_SEARCH_BACKEND` — `tavily` or `ddgs`
- `TAVILY_API_KEY` — required if using Tavily search backend

## Testing

pytest with `asyncio_mode = "auto"`. Fixtures in `tests/conftest.py` provide: event loop, temp dirs, mock Ollama/LLM responses, sample data, mission constraints, FastAPI test client (via httpx). API tests use `AsyncClient` with `ASGITransport`.

## API Routes

All under `/api/` prefix. Docs at `/api/docs`.
- Missions: CRUD + start/abort/logs/events(SSE)/artifacts
- Workflows: code generation
- Agents, GPU status, Config, Knowledge Packs
- CORS allows localhost:3000 and localhost:5173

## Frontend

React 18 + Vite + Tailwind CSS + React Router v6. Dev server on port 5173. Production build served by FastAPI from `frontend/dist/`.
