"""
FastAPI application for CALISOTA AI Engine.
Provides REST API for code generation, execution, and management.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.calisota.core.config import get_settings
from src.calisota.api.routes import tasks, health, rag
from src.calisota.rag.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)
settings = get_settings()


# Global instances
faiss_manager: FAISSManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global faiss_manager

    logger.info("Starting CALISOTA AI Engine...")

    # Initialize FAISS manager
    faiss_manager = FAISSManager(settings)
    faiss_manager.initialize_index()

    logger.info("CALISOTA AI Engine started successfully")

    yield

    # Cleanup
    logger.info("Shutting down CALISOTA AI Engine...")
    if faiss_manager:
        faiss_manager.save_index()


# Create FastAPI app
app = FastAPI(
    title="CALISOTA AI Engine",
    description=(
        "Autonomous AI-powered software generation system with RAG, "
        "Actor-Critic ensembles, and multi-language sandboxes. "
        "Optimized for NVIDIA RTX 4080 16GB."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])

# Prometheus metrics
if settings.enable_prometheus:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "CALISOTA AI Engine",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/api/docs",
        "health": "/api/health"
    }


def get_faiss_manager() -> FAISSManager:
    """Get FAISS manager dependency."""
    if faiss_manager is None:
        raise HTTPException(status_code=503, detail="FAISS manager not initialized")
    return faiss_manager
