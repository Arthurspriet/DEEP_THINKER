"""
DeepThinker API Server.

FastAPI application with all routes mounted.
Run with: uvicorn api.server:app --reload --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import (
    missions_router,
    workflows_router,
    agents_router,
    gpu_router,
    config_router
)

# Configure Ollama base URL
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

app = FastAPI(
    title="DeepThinker API",
    description="REST API for the DeepThinker multi-agent autonomous AI system",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(missions_router)
app.include_router(workflows_router)
app.include_router(agents_router)
app.include_router(gpu_router)
app.include_router(config_router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "DeepThinker API",
        "version": "2.0.0"
    }


# Serve frontend static files in production
FRONTEND_BUILD_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_BUILD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_BUILD_DIR / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend for all non-API routes."""
        # Don't serve frontend for API routes
        if full_path.startswith("api/"):
            return {"detail": "Not found"}
        
        # Try to serve the specific file
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Fallback to index.html for SPA routing
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

