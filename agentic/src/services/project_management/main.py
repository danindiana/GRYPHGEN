"""Project Management Service API."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

from ...common.config import get_settings
from ...common.logger import setup_logging
from .models import OptimizationRequest, OptimizationResponse
from .optimizer import ProjectOptimizer

# Setup
setup_logging()
settings = get_settings()

# Global optimizer instance
project_optimizer: ProjectOptimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global project_optimizer

    logger.info("Starting Project Management Service")
    logger.info(f"GPU enabled: {settings.use_gpu}")

    # Initialize optimizer
    project_optimizer = ProjectOptimizer()

    yield

    logger.info("Shutting down Project Management Service")


# Create FastAPI app
app = FastAPI(
    title="Project Management Service",
    description="RL-based project management and task optimization",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Project Management Service",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_project(request: OptimizationRequest):
    """
    Optimize task assignments and project timeline.

    Args:
        request: Optimization request with tasks and team members

    Returns:
        Optimized assignments and timeline
    """
    try:
        logger.info(f"Optimizing project {request.project_id}")
        result = await project_optimizer.optimize(request)
        return result
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
