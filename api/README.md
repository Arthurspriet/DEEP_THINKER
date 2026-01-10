# API Module

The API module provides a **FastAPI backend** for the DeepThinker web interface.

## Core Concepts

### REST API
HTTP endpoints for managing missions and workflows:
- Create and start missions
- Query mission status
- Stream real-time updates
- Manage configuration

### Server-Sent Events (SSE)
Real-time streaming of mission progress to the frontend.

## Components

### 1. Server
Main FastAPI application.

**File**: `server.py`

### 2. SSE Manager
Manages real-time event streaming.

**File**: `sse.py`

### 3. Routes
Endpoint implementations organized by feature.

**Directory**: `routes/`

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app, middleware, startup |
| `sse.py` | Server-Sent Events for real-time updates |
| `routes/missions.py` | Mission CRUD and execution |
| `routes/workflows.py` | Workflow management |
| `routes/agents.py` | Agent configuration |
| `routes/config.py` | System configuration |
| `routes/gpu.py` | GPU resource management |

## API Endpoints

### Missions
```
POST   /api/missions           Create new mission
GET    /api/missions           List all missions
GET    /api/missions/{id}      Get mission status
POST   /api/missions/{id}/start   Start mission
POST   /api/missions/{id}/abort   Abort mission
DELETE /api/missions/{id}      Delete mission
```

### Workflows
```
POST   /api/workflows/run      Run legacy workflow
GET    /api/workflows/status   Get workflow status
```

### Configuration
```
GET    /api/config             Get current config
PUT    /api/config             Update config
GET    /api/models             List available models
```

### Real-time Updates
```
GET    /api/events/{mission_id}   SSE stream for mission
```

## Running the Server

```bash
# Development
cd api
uvicorn server:app --reload --port 8000

# Production
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## SSE Events

Real-time events streamed to clients:

```javascript
// Frontend example
const eventSource = new EventSource('/api/events/mission-123');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.type: 'phase_started' | 'phase_completed' | 'step_output' | ...
    // data.payload: event-specific data
};
```

Event Types:
- `mission_started` - Mission execution began
- `phase_started` - New phase starting
- `phase_completed` - Phase finished
- `step_output` - Incremental step output
- `mission_completed` - Mission finished
- `error` - Error occurred

## CORS Configuration

The API allows cross-origin requests for local development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Integration with Frontend

```
┌─────────────────┐         ┌─────────────────┐
│  React Frontend │ ◄─────► │   FastAPI       │
│  (port 5173)    │  HTTP   │   (port 8000)   │
│                 │  + SSE  │                 │
└─────────────────┘         └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │  DeepThinker    │
                            │  Core Engine    │
                            └─────────────────┘
```


