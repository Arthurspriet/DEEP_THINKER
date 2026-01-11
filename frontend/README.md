# Frontend Module

The frontend provides a **React-based web interface** for managing DeepThinker missions.

## Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Getting Started

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The dev server runs on `http://localhost:5173`

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Page components
│   ├── hooks/          # Custom React hooks
│   ├── services/       # API client services
│   ├── utils/          # Utility functions
│   ├── App.jsx         # Main app component
│   ├── main.jsx        # Entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML template
├── package.json        # Dependencies
├── tailwind.config.js  # Tailwind configuration
└── vite.config.js      # Vite configuration
```

## Key Features

### Mission Dashboard
- View all missions with status indicators
- Real-time progress updates via SSE
- Mission details and artifacts

### Mission Creation
- Set objectives and time budgets
- Configure constraints (code execution, internet)
- Select effort levels

### Real-time Updates
- Server-Sent Events for live progress
- Phase transitions
- Step-by-step output streaming

### Artifact Viewer
- View generated reports
- Code output display
- Download deliverables

## Configuration

### API Connection
The frontend connects to the API server:

```javascript
// In services/api.js
const API_BASE = 'http://localhost:8000';
```

### Environment Variables
Create `.env` for custom configuration:

```bash
VITE_API_URL=http://localhost:8000
```

## Development

### Adding a New Page

1. Create component in `src/pages/`
2. Add route in `App.jsx`
3. Add navigation link if needed

### Adding a Component

1. Create in `src/components/`
2. Export from `index.js` if shared
3. Use Tailwind for styling

### API Integration

```javascript
import { api } from '../services/api';

// Fetch missions
const missions = await api.getMissions();

// Create mission
const mission = await api.createMission({
    objective: "...",
    time_budget: 30
});

// Subscribe to updates
api.subscribeToMission(missionId, (event) => {
    console.log('Update:', event);
});
```

## Build & Deploy

```bash
# Build production bundle
npm run build

# Output in dist/ directory
# Serve with any static file server
```

## Integration

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                   React Frontend                       │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │ │
│  │  │Dashboard│ │ Create  │ │ Details │ │Settings │     │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │ │
│  │       │           │           │           │           │ │
│  │  ┌────▼───────────▼───────────▼───────────▼────┐     │ │
│  │  │              API Service Layer              │     │ │
│  │  └─────────────────────┬───────────────────────┘     │ │
│  └────────────────────────│───────────────────────────────┘ │
│                           │ HTTP + SSE                      │
└───────────────────────────┼─────────────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │  FastAPI Server │
                   │  (port 8000)    │
                   └─────────────────┘
```




