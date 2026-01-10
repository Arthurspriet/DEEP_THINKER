# Models Module

The models module handles **LLM integration and model management** via Ollama.

## Core Concepts

### Ollama Integration
DeepThinker uses Ollama for local LLM inference:
- No external API calls
- Full privacy and control
- Multiple model support
- GPU acceleration when available

### Model Pool
Manages model instances with caching and configuration.

## Components

### 1. Ollama Loader
Connects to Ollama and loads models.

**File**: `ollama_loader.py`

### 2. Model Pool
Caches and manages LLM instances.

**File**: `model_pool.py`

### 3. Model Caller
Unified interface for calling any model.

**File**: `model_caller.py`

### 4. Agent Model Config
Per-agent model configuration.

**File**: `agent_model_config.py`

## Key Files

| File | Purpose |
|------|---------|
| `ollama_loader.py` | Ollama server connection |
| `model_pool.py` | Model instance caching |
| `model_caller.py` | Unified calling interface |
| `agent_model_config.py` | Agent-specific model settings |
| `monitoring.py` | Token usage and cost tracking |

## Supported Models

| Model | Best For | Size |
|-------|----------|------|
| `deepseek-r1:8b` | Code generation, reasoning | 8B |
| `cogito:14b` | Complex planning | 14B |
| `gemma3:12b` | General tasks | 12B |
| `gemma3:27b` | Quality evaluation | 27B |
| `llama3.2:3b` | Fast execution | 3B |
| `mistral:instruct` | Instruction following | 7B |

## Usage

```python
from deepthinker.models import OllamaLoader, ModelPool, AgentModelConfig

# Basic connection
loader = OllamaLoader(base_url="http://localhost:11434")
if loader.validate_connection():
    models = loader.list_available_models()

# Model pool with caching
pool = ModelPool(base_url="http://localhost:11434")
llm = pool.get_model("deepseek-r1:8b", temperature=0.7)
response = llm.invoke("Write a Python function...")

# Agent-specific config
config = AgentModelConfig(
    planner_model="cogito:14b",
    coder_model="deepseek-r1:8b",
    evaluator_model="gemma3:27b"
)
```

## Model Selection Strategy

Different agents use different models based on task requirements:

```
┌─────────────────────────────────────────────────────────────┐
│ Task Complexity        │ Recommended Model               │
├─────────────────────────────────────────────────────────────┤
│ Fast/Simple            │ llama3.2:3b                       │
│ Code Generation        │ deepseek-r1:8b                    │
│ Complex Reasoning      │ cogito:14b                        │
│ Quality Evaluation     │ gemma3:27b                        │
│ General Purpose        │ gemma3:12b, mistral:instruct      │
└─────────────────────────────────────────────────────────────┘
```

## Monitoring

Track token usage and costs:

```python
from deepthinker.models import enable_monitoring, print_monitoring_summary

enable_monitoring(log_dir="logs", verbose=True)

# ... run workflow ...

print_monitoring_summary()
# Output: Tokens used, latency, estimated costs
```

## Environment Variables

```bash
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b  # Default model
```


