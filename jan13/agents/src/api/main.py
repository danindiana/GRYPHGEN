"""FastAPI application for GRYPHGEN Infrastructure Agents."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.agents.infrastructure import InfrastructureAgent
from src.models.config import DeploymentConfig, DeploymentStatus, OllamaConfig, NginxConfig, GPUConfig
from src.utils.logging import setup_logging
from src.utils.monitoring import PrometheusMetrics

# Global state
infrastructure_agent: Optional[InfrastructureAgent] = None
prometheus_metrics: Optional[PrometheusMetrics] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global infrastructure_agent, prometheus_metrics

    # Startup
    setup_logging(level="INFO", structured=True)

    prometheus_metrics = PrometheusMetrics()

    # Create infrastructure agent (don't deploy yet)
    infrastructure_agent = InfrastructureAgent()

    yield

    # Shutdown
    if infrastructure_agent and infrastructure_agent.status.value == "running":
        await infrastructure_agent.stop()


# Create FastAPI app
app = FastAPI(
    title="GRYPHGEN Infrastructure Agents API",
    description="REST API for managing infrastructure deployment and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


# Request/Response Models
class DeployRequest(BaseModel):
    """Request model for deployment."""

    ollama_port: int = 11435
    nginx_port: int = 11434
    gpu_enabled: bool = True
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.9
    auto_pull_models: bool = True
    models: list[str] = ["llama2", "mistral"]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    services: dict


class StatusResponse(BaseModel):
    """Status response."""

    overall_status: str
    ollama_status: str
    nginx_status: str
    timestamp: str
    metrics: Optional[dict] = None


# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "name": "GRYPHGEN Infrastructure Agents API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not infrastructure_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Infrastructure agent not initialized",
        )

    health = await infrastructure_agent.health_check()

    return HealthResponse(
        status=health.status.value,
        timestamp=health.timestamp.isoformat(),
        services=health.details or {},
    )


@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status():
    """Get infrastructure status."""
    if not infrastructure_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Infrastructure agent not initialized",
        )

    deployment_status = await infrastructure_agent.get_status()

    return StatusResponse(
        overall_status=deployment_status.overall_status.value,
        ollama_status=deployment_status.ollama.status.value,
        nginx_status=deployment_status.nginx.status.value,
        timestamp=deployment_status.timestamp.isoformat(),
        metrics=deployment_status.metrics.dict() if deployment_status.metrics else None,
    )


@app.post("/api/v1/deploy", status_code=status.HTTP_202_ACCEPTED)
async def deploy_infrastructure(request: DeployRequest):
    """Deploy infrastructure stack."""
    global infrastructure_agent

    try:
        # Create configuration
        config = DeploymentConfig()
        config.ollama.port = request.ollama_port
        config.ollama.auto_pull_models = request.auto_pull_models
        config.ollama.models = request.models
        config.nginx.port = request.nginx_port
        config.nginx.upstream_port = request.ollama_port
        config.gpu.enabled = request.gpu_enabled
        config.gpu.gpu_id = request.gpu_id
        config.gpu.memory_fraction = request.gpu_memory_fraction

        # Validate configuration
        errors = config.validate_ports()
        if errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Configuration errors: {errors}",
            )

        # Create and deploy
        infrastructure_agent = InfrastructureAgent(config)
        success = await infrastructure_agent.deploy()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Deployment failed",
            )

        # Update Prometheus metrics
        if prometheus_metrics:
            prometheus_metrics.update_service_status("ollama", True)
            prometheus_metrics.update_service_status("nginx", True)

        return {
            "message": "Deployment successful",
            "status": "running",
            "ollama_url": f"http://{config.ollama.host}:{config.ollama.port}",
            "nginx_url": f"http://{config.nginx.host}:{config.nginx.port}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment error: {str(e)}",
        )


@app.post("/api/v1/stop")
async def stop_infrastructure():
    """Stop infrastructure services."""
    if not infrastructure_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No infrastructure deployed",
        )

    success = await infrastructure_agent.stop()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop services",
        )

    # Update Prometheus metrics
    if prometheus_metrics:
        prometheus_metrics.update_service_status("ollama", False)
        prometheus_metrics.update_service_status("nginx", False)

    return {"message": "Infrastructure stopped successfully"}


@app.get("/api/v1/models")
async def list_models():
    """List available Ollama models."""
    if not infrastructure_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No infrastructure deployed",
        )

    models = await infrastructure_agent.ollama_agent.list_models()

    return {"models": models, "count": len(models)}


@app.get("/api/v1/logs/nginx/access")
async def get_nginx_access_logs(lines: int = 50):
    """Get Nginx access logs."""
    if not infrastructure_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No infrastructure deployed",
        )

    logs = await infrastructure_agent.nginx_agent.get_access_logs(lines=lines)

    return PlainTextResponse(content=logs)


@app.get("/api/v1/logs/nginx/error")
async def get_nginx_error_logs(lines: int = 50):
    """Get Nginx error logs."""
    if not infrastructure_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No infrastructure deployed",
        )

    logs = await infrastructure_agent.nginx_agent.get_error_logs(lines=lines)

    return PlainTextResponse(content=logs)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not prometheus_metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics not available",
        )

    # Update metrics before serving
    if infrastructure_agent:
        deployment_status = await infrastructure_agent.get_status()
        if deployment_status.metrics:
            prometheus_metrics.update_system_metrics(deployment_status.metrics)

    return PlainTextResponse(
        content=generate_latest(prometheus_metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
    )


def get_application() -> FastAPI:
    """Get FastAPI application instance.

    Returns:
        Configured FastAPI app
    """
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
