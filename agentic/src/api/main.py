"""Main API Gateway application."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from ..auth.api_keys import extract_key_from_headers, key_fingerprint
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with key fingerprint, method, path, status, and duration."""
    start = time.monotonic()
    raw_key = extract_key_from_headers(dict(request.headers))
    key_id = key_fingerprint(raw_key) if raw_key else "anonymous"

    response = await call_next(request)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        f"key={key_id} {request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)"
    )
    return response


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


# Core routers — always loaded
from ..services.code_generation import router as code_generation_router
from ..services.agent import router as agent_service_module
from ..auth.api_keys import require_api_key

app.include_router(code_generation_router, prefix="/api/v1/code", tags=["Code Generation"])


@app.get("/api/v1/tools/search", tags=["Tools"])
async def web_search(
    q: str,
    max_results: int = 5,
    api_key: str = Security(require_api_key),
) -> dict:
    """Search DuckDuckGo and return results. No extra API key required."""
    from ..tools.web_search import search
    results = await search(q, max_results=max_results)
    return {"query": q, "results": results, "source": "ddg"}

try:
    from ..services.agent.router import router as agent_router
    app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
except Exception as _e:
    logger.warning(f"Skipping agent router: {_e}")

# Optional routers — skip gracefully if their deps aren't installed
_optional_routers = [
    ("..services.automated_testing", "router", "/api/v1/test", "Automated Testing"),
    ("..services.project_management", "router", "/api/v1/project", "Project Management"),
    ("..services.documentation", "router", "/api/v1/docs", "Documentation"),
    ("..services.collaboration", "router", "/api/v1/collaboration", "Collaboration"),
    ("..services.self_improvement", "router", "/api/v1/improve", "Self-Improvement"),
]

import importlib

for _mod, _attr, _prefix, _tag in _optional_routers:
    try:
        _m = importlib.import_module(_mod, package=__package__)
        app.include_router(getattr(_m, _attr), prefix=_prefix, tags=[_tag])
    except Exception as _e:
        logger.warning(f"Skipping router {_mod}: {_e}")

try:
    from ..websockets import websocket_router
    app.include_router(websocket_router, tags=["WebSocket"])
except Exception as _e:
    logger.warning(f"Skipping WebSocket router: {_e}")

# MCP server — mounted at /mcp (SSE transport)
# Clients connect via: GET /mcp/sse
# Messages sent via:   POST /mcp/messages/
try:
    from ..gryphgen_mcp.gryphgen_server import create_sse_app as _create_mcp_sse
    app.mount("/mcp", _create_mcp_sse())
    logger.info("MCP server mounted at /mcp/sse")
except Exception as _e:
    logger.warning(f"Skipping MCP server: {_e}")


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
