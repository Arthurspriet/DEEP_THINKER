# DeepThinker

A long-running, time-bounded multi-agent autonomous AI system for complex reasoning and analysis tasks.

DeepThinker orchestrates multiple specialized AI councils to accomplish complex objectives through structured phases, iterative refinement, and metric-based evaluation—all running locally via Ollama.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Mission Orchestrator          │
                    │    (Time-bounded, multi-phase execution)│
                    └────────────────────┬────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│    Planner    │              │  Researcher   │              │    Coder      │
│    Council    │              │    Council    │              │    Council    │
│               │              │               │              │               │
│ - Strategy    │              │ - Web search  │              │ - Code gen    │
│ - Workflow    │              │ - RAG memory  │              │ - Revision    │
│ - Synthesis   │              │ - Sources     │              │ - Consensus   │
└───────────────┘              └───────────────┘              └───────────────┘
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   Evaluator   │              │  Simulation   │              │    Arbiter    │
│    Council    │              │    Council    │              │               │
│               │              │               │              │ Final decision│
│ - Quality     │              │ - Testing     │              │ reconciliation│
│ - Feedback    │              │ - Scenarios   │              │               │
│ - Scoring     │              │ - Validation  │              │               │
└───────────────┘              └───────────────┘              └───────────────┘
```

### Mission Phases

Missions execute through structured phases, each with time budgets:

1. **Reconnaissance** - Gather context, background information, and relevant resources
2. **Analysis & Planning** - Strategic planning and workflow design
3. **Deep Analysis** - In-depth investigation using evidence and evaluation councils
4. **Synthesis & Report** - Consolidate findings into actionable outputs

## Features

- **Multi-Council Architecture**: Specialized councils for planning, research, coding, evaluation, and simulation
- **Time-Bounded Missions**: Configurable time budgets with automatic phase allocation
- **Iterative Refinement**: Automatic quality improvement through multiple refinement cycles
- **Web Research Integration**: Automated research to gather documentation and best practices
- **Multi-View Reasoning**: Optimist/Skeptic councils for balanced analysis
- **Secure Code Execution**: Docker-based sandbox with security scanning
- **Local-First**: Runs entirely on your machine with Ollama—no external API calls
- **Checkpointing**: Mission state saved for resumability
- **Web UI & API**: React frontend with FastAPI backend for mission management

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama: https://ollama.ai
   ollama serve
   ollama pull deepseek-r1:8b
   ```

### Installation

```bash
cd deep_thinker

# Install with pip
pip install -e .

# Or with Poetry
poetry install
```

### Run a Mission

```bash
# Start a research/analysis mission (no code execution)
python3 main.py mission start \
  -o "Analyze how AI will transform global labor markets" \
  -t 15 \
  --no-code-exec \
  --verbose

# Start a coding mission
python3 main.py mission start \
  -o "Build a FastAPI REST API with authentication" \
  -t 30 \
  --allow-code-exec

# Check mission status
python3 main.py mission status --id <mission_id>

# List all missions
python3 main.py mission list
```

### Run the Web UI

```bash
# Start the API server
cd api && uvicorn server:app --reload

# In another terminal, start the frontend
cd frontend && npm install && npm run dev
```

## CLI Reference

### Mission Commands

```bash
# Start a new mission
python3 main.py mission start \
  --objective "Your objective here" \
  --time 60 \                        # Time budget in minutes
  --allow-internet \                 # Enable web research (default)
  --allow-code-exec \                # Enable code execution
  --verbose                          # Detailed output

# Other mission commands
python3 main.py mission status --id <id>    # Check status
python3 main.py mission list                 # List all missions
python3 main.py mission resume --id <id>    # Resume a mission
python3 main.py mission abort --id <id>     # Abort a mission
```

### Legacy Run Command

For simple code generation tasks:

```bash
python3 main.py run "Create a binary search tree class" \
  --max-iterations 5 \
  --quality-threshold 8.0
```

### Utility Commands

```bash
python3 main.py test-connection    # Test Ollama connection
python3 main.py list-models        # List available models
python3 main.py context councils   # Inspect council configurations
```

## Project Structure

```
deep_thinker/
├── deepthinker/              # Main package
│   ├── agents/               # Individual AI agents
│   ├── councils/             # Council implementations
│   │   ├── planner_council/
│   │   ├── researcher_council/
│   │   ├── coder_council/
│   │   ├── evaluator_council/
│   │   └── simulation_council/
│   ├── missions/             # Mission orchestration
│   ├── arbiter/              # Final decision reconciliation
│   ├── consensus/            # Multi-agent consensus mechanisms
│   ├── memory/               # RAG and long-term memory
│   ├── execution/            # Workflow and code execution
│   ├── models/               # Ollama LLM integration
│   └── ...
├── api/                      # FastAPI backend
├── frontend/                 # React/Vite frontend
├── tests/                    # Test suite
├── main.py                   # CLI entry point
└── requirements.txt          # Dependencies
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Ollama configuration
OLLAMA_API_BASE=http://localhost:11434

# Default models
OLLAMA_MODEL=deepseek-r1:8b
```

### Model Selection

Different models can be assigned to different agents:

```bash
python3 main.py run "..." \
  --planner-model cogito:14b \
  --coder-model deepseek-r1:8b \
  --evaluator-model gemma3:27b
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### API Development

```bash
cd api
uvicorn server:app --reload --port 8000
```

## License

MIT License - See LICENSE file for details.

---

Built with Ollama and Python
