"""Main API Gateway application."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from ..common.config import get_settings
from ..common.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    logger.info("Starting GRYPHGEN Agentic API Gateway")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Startup logic here
    yield

    # Shutdown logic here
    logger.info("Shutting down GRYPHGEN Agentic API Gateway")


# Create FastAPI application
app = FastAPI(
    title="GRYPHGEN Agentic API Gateway",
    description="AI-powered development assistant API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "service": "GRYPHGEN Agentic API Gateway",
        "version": "0.1.0",
        "status": "operational",
        "environment": settings.environment,
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": settings.service_name}


@app.get("/ready")
async def readiness_check() -> dict:
    """Readiness check endpoint."""
    # Add checks for database, kafka, etc.
    return {"status": "ready", "service": settings.service_name}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# Import and include routers
from ..services.code_generation import router as code_generation_router
from ..services.automated_testing import router as testing_router
from ..services.project_management import router as project_mgmt_router
from ..services.documentation import router as docs_router
from ..services.collaboration import router as collab_router
from ..services.self_improvement import router as improvement_router

app.include_router(
    code_generation_router,
    prefix="/api/v1/code",
    tags=["Code Generation"],
)
app.include_router(
    testing_router,
    prefix="/api/v1/test",
    tags=["Automated Testing"],
)
app.include_router(
    project_mgmt_router,
    prefix="/api/v1/project",
    tags=["Project Management"],
)
app.include_router(
    docs_router,
    prefix="/api/v1/docs",
    tags=["Documentation"],
)
app.include_router(
    collab_router,
    prefix="/api/v1/collaboration",
    tags=["Collaboration"],
)
app.include_router(
    improvement_router,
    prefix="/api/v1/improve",
    tags=["Self-Improvement"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
