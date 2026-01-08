# DeepThinker 🧠

A long-running multi-agent autonomous AI system using CrewAI and local LLMs via Ollama.

DeepThinker performs complex tasks over extended periods through structured, sequential phases with iterative refinement and metric-based evaluation.

## 🏗️ Architecture

```
deepthinker/
├── agents/           # AI agent definitions (Coder, Evaluator, Simulator)
├── tasks/            # Task definitions for each phase
├── execution/        # Workflow orchestration and code execution
├── evaluation/       # Result parsing and quality assessment
└── models/           # Ollama LLM integration
```

### Multi-Agent System

- **PlannerAgent**: Analyzes objectives and creates detailed execution plans with requirements for each agent
- **ResearchAgent**: Searches web for documentation, examples, and best practices
- **CoderAgent**: Generates and revises Python code from specifications
- **EvaluatorAgent**: Assesses code quality with structured feedback
- **SimulatorAgent**: Runs scenario-based testing and simulation
- **ExecutorAgent**: Manages secure code execution with security monitoring

### Workflow Phases

```
1. Planning → 2. Research → 3. Code Generation → 4. Evaluation → 5. Revision (if needed) → 6. Simulation
                                      ↓
                                [Iterative Loop]
                                Continues until:
                                - Quality threshold met
                                - Max iterations reached
```

## ✨ Key Features

- **🤖 Multi-Agent System**: Specialized agents for planning, research, coding, evaluation, simulation, and secure execution
- **📋 Strategic Planning**: Planner agent analyzes objectives and creates detailed execution plans with specific requirements for each agent
- **🔍 Web Research**: Automated research phase to gather documentation and best practices before coding
- **🔄 Iterative Refinement**: Automatic code improvement based on quality metrics
- **📊 Metric-Based Evaluation**: Real dataset testing with accuracy/performance metrics
- **🧪 Scenario Simulation**: Advanced testing across multiple data scenarios
- **🔒 Secure Code Execution**: Docker-based sandbox with comprehensive security measures
  - Container isolation with no network access
  - Static code analysis and security scanning
  - Resource limits (CPU, memory, timeout)
  - Protection against malicious code
- **📈 LiteLLM Monitoring**: Comprehensive observability for all LLM interactions
  - Token usage tracking
  - Latency monitoring
  - Cost estimation
  - Request/response logging
  - Error tracking
- **🔌 Local-First**: Runs entirely on your machine with Ollama

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running locally
   - Install: https://ollama.ai
   - Start: `ollama serve`
   - Pull a model: `ollama pull deepseek-r1:8b`

### Installation

```bash
# Navigate to project directory
cd /home/arthurspriet/deep_thinker

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Simple code generation (planning enabled by default)
python3 main.py run "Create a binary search tree class"

# With iteration configuration
python3 main.py run "Implement a LRU cache" \
  --max-iterations 5 \
  --quality-threshold 8.0

# With dataset-based evaluation
python3 main.py run "Build a decision tree classifier" \
  --data-path examples/iris_classification.csv \
  --task-type classification \
  --target-column species

# Save workflow plan to file
python3 main.py run "Build a neural network" \
  --save-plan plans/neural_net_plan.json

# Disable planning for simple tasks
python3 main.py run "Create a calculator class" \
  --no-planning

# Test Ollama connection
python3 main.py test-connection

# List available models
python3 main.py list-models
```

## 📋 Features

### ✅ Iterative Refinement Loop
- Automatic quality improvement through multiple refinement cycles
- Configurable quality threshold and max iterations
- Structured feedback with categorized issues (critical/major/minor)
- Anti-loop mechanism prevents unproductive iterations

### ✅ Metric-Based Evaluation
- Execute generated code on real datasets
- Compute task-specific metrics (classification/regression)
- Combine LLM quality assessment with quantitative performance
- Configurable metric weighting

### ✅ Scenario Simulation
- Test code against diverse scenarios
- Identify edge cases and failure modes
- Comprehensive simulation reports

### ✅ Local LLM Integration
- Run entirely locally via Ollama
- Support for multiple models (deepseek-r1:8b, codellama, mistral, etc.)
- Model-specific temperature tuning
- No external API calls or costs

## 🔧 CLI Options

### `run` Command

```bash
python3 main.py run OBJECTIVE [OPTIONS]
```

**Core Options:**
- `--model TEXT`: Ollama model to use (default: deepseek-r1:8b)
- `--verbose`: Enable detailed progress output

**Iteration Options:**
- `--max-iterations INT`: Maximum refinement cycles (default: 3)
- `--quality-threshold FLOAT`: Min quality score 0-10 (default: 7.0)
- `--no-iteration`: Disable iteration (single pass only)

**Dataset Options:**
- `--data-path PATH`: Dataset file for evaluation (CSV/JSON)
- `--task-type [classification|regression]`: ML task type
- `--target-column TEXT`: Target column name
- `--test-split FLOAT`: Test data fraction (default: 0.2)
- `--metric-weight FLOAT`: Metric weight in score (default: 0.5)

**Context Options:**
- `--context-file PATH`: JSON file with additional context
- `--scenarios-file PATH`: JSON file with simulation scenarios

**Output Options:**
- `--output PATH`: Save results to JSON file

## 📊 Example Workflows

### Basic Code Generation

```bash
python3 main.py run "Create a function to calculate fibonacci numbers"
```

### High-Quality Code with Iterations

```bash
python3 main.py run "Implement a thread-safe singleton pattern" \
  --max-iterations 5 \
  --quality-threshold 9.0 \
  --verbose
```

### ML Model with Dataset Evaluation

```bash
python3 main.py run "Create a k-nearest neighbors classifier" \
  --data-path examples/iris_classification.csv \
  --task-type classification \
  --target-column species \
  --max-iterations 4 \
  --metric-weight 0.7 \
  --output results.json
```

### With Custom Context

```json
// context.json
{
  "language": "Python 3.10",
  "constraints": [
    "Use only standard library",
    "Must be thread-safe",
    "Include comprehensive docstrings"
  ],
  "style": "Google Python Style Guide"
}
```

```bash
python3 main.py run "Create a connection pool" \
  --context-file context.json
```

### With Simulation Scenarios

```json
// scenarios.json
{
  "scenarios": [
    "Empty input",
    "Single element",
    "Large dataset (10000 items)",
    "Concurrent access from 10 threads",
    "Invalid/malformed input"
  ]
}
```

```bash
python3 main.py run "Build a rate limiter" \
  --scenarios-file scenarios.json
```

## 🔍 Project Structure

```
deep_thinker/
├── deepthinker/              # Main package
│   ├── agents/               # Agent definitions
│   │   ├── coder_agent.py
│   │   ├── evaluator_agent.py
│   │   └── simulator_agent.py
│   ├── tasks/                # Task definitions
│   │   ├── code_task.py
│   │   ├── evaluate_task.py
│   │   ├── revise_task.py
│   │   └── simulate_task.py
│   ├── execution/            # Workflow orchestration
│   │   ├── run_workflow.py
│   │   ├── data_config.py
│   │   ├── code_executor.py
│   │   └── metric_computer.py
│   ├── evaluation/           # Result parsing
│   │   ├── evaluation_result.py
│   │   ├── metric_result.py
│   │   └── result_parser.py
│   └── models/               # LLM integration
│       └── ollama_loader.py
├── examples/                 # Example datasets
├── main.py                   # CLI entry point
├── pyproject.toml            # Poetry config
└── README.md                 # This file
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b

# Default iteration settings
DEFAULT_MAX_ITERATIONS=3
DEFAULT_QUALITY_THRESHOLD=7.0

# Execution settings
CODE_EXECUTION_TIMEOUT=30
```

### Model Selection

Different models for different tasks:

```bash
# General purpose
python3 main.py run "..." --model deepseek-r1:8b

# Code generation (if available)
python3 main.py run "..." --model codellama

# Advanced reasoning
python3 main.py run "..." --model mixtral
```

## 🐛 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test connection
python3 main.py test-connection

# Start Ollama if not running
ollama serve
```

### Import Errors

```bash
# Reinstall dependencies
poetry install
# or
pip install -e .
```

### Memory Issues with Large Models

```bash
# Use smaller model
python3 main.py run "..." --model deepseek-r1:8b:7b

# Or increase Ollama memory limit
# Edit Ollama service configuration
```

## 🚧 Development Status

- ✅ Core architecture implemented
- ✅ Planner agent with dynamic workflow coordination
- ✅ Web research integration
- ✅ Iterative refinement working
- ✅ Metric-based evaluation functional
- ✅ Secure Docker execution
- ✅ CLI complete
- 🚧 Advanced agent logic (ongoing refinement)
- 🚧 Extended model support
- 🚧 Distributed execution

## 📝 License

[Add license information]

## 🤝 Contributing

[Add contribution guidelines]

## 📧 Contact

[Add contact information]

---

Built with ❤️ using CrewAI, Ollama, and Python
