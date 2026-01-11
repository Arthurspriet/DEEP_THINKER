# Execution Module

The execution module handles **workflow orchestration and secure code execution**.

## Core Concepts

### Workflow Execution
Manages the overall flow of agent/council execution:
- Task distribution
- Result collection
- Iteration loops
- Error handling

### Code Execution
Safely runs generated code in isolated environments:
- Subprocess execution (fast, local)
- Docker sandbox (secure, isolated)

## Components

### 1. Workflow Runner
Coordinates the execution of the full DeepThinker workflow.

**File**: `run_workflow.py`

### 2. Step Executor
Executes individual steps within phases.

**File**: `step_executor.py`

### 3. Code Executor
Runs Python code with configurable backends.

**Files**: `code_executor.py`, `docker_executor.py`

### 4. Security Scanner
Analyzes code for dangerous patterns before execution.

**File**: `security_scanner.py`

## Key Files

| File | Purpose |
|------|---------|
| `run_workflow.py` | Main workflow orchestration |
| `step_executor.py` | Phase step execution |
| `code_executor.py` | Code execution abstraction |
| `docker_executor.py` | Docker sandbox execution |
| `security_scanner.py` | Pre-execution security checks |
| `data_config.py` | Dataset and execution configuration |
| `metric_computer.py` | Compute metrics on code outputs |

## Execution Backends

### Subprocess (Default)
Fast, runs in local Python process.
```python
executor = CodeExecutor(backend="subprocess")
result = executor.execute(code, timeout=30)
```

### Docker (Secure)
Isolated container with resource limits.
```python
from deepthinker.execution import DockerConfig

config = DockerConfig(
    memory_limit="512m",
    cpu_limit=1.0,
    network_disabled=True
)
executor = CodeExecutor(backend="docker", docker_config=config)
```

## Security Features

The security scanner checks for:
- File system access (`open()`, `os.remove()`)
- Network calls (`socket`, `requests`)
- System commands (`subprocess`, `os.system`)
- Code execution (`eval()`, `exec()`)
- Import of dangerous modules

```python
from deepthinker.execution import SecurityScanner

scanner = SecurityScanner()
issues = scanner.scan(code)
if issues.has_critical():
    raise SecurityError("Dangerous code detected")
```

## Workflow Flow

```
Objective
    ↓
┌─────────────────────────────────────┐
│  Planning Phase                     │
│  → Planner Council                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Research Phase                     │
│  → Researcher Council               │
│  → Web Search, RAG                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Code Generation (Iterative)        │
│  → Coder Council                    │
│  → Evaluator Council                │
│  → Loop until quality threshold     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Simulation Phase                   │
│  → Simulation Council               │
│  → Code Execution                   │
└─────────────────────────────────────┘
    ↓
Final Output
```

## Configuration

```python
from deepthinker.execution import IterationConfig, DataConfig

iteration_config = IterationConfig(
    max_iterations=5,
    quality_threshold=8.0,
    enabled=True
)

data_config = DataConfig(
    data_path="data/dataset.csv",
    task_type="classification",
    target_column="label"
)
```




