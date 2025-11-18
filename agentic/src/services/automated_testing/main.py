"""Automated Testing Service API."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

from ...common.config import get_settings
from ...common.logger import setup_logging
from .models import TestGenerationRequest, TestGenerationResponse
from .generator import TestGenerator

# Setup
setup_logging()
settings = get_settings()

# Global generator instance
test_generator: TestGenerator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global test_generator

    logger.info("Starting Automated Testing Service")

    # Initialize test generator
    test_generator = TestGenerator()

    yield

    logger.info("Shutting down Automated Testing Service")


# Create FastAPI app
app = FastAPI(
    title="Automated Testing Service",
    description="ML-based automated test generation",
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
        "service": "Automated Testing Service",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate", response_model=TestGenerationResponse)
async def generate_tests(request: TestGenerationRequest):
    """
    Generate comprehensive test cases for the provided code.

    Args:
        request: Test generation request

    Returns:
        Generated test suites and cases
    """
    try:
        logger.info(f"Received test generation request: {request.framework}")
        result = await test_generator.generate_tests(request)
        return result
    except Exception as e:
        logger.error(f"Error in test generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test generation failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
