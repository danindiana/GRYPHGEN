"""Code Generation Service API."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

from ...common.config import get_settings
from ...common.logger import setup_logging
from .models import CodeGenerationRequest, CodeGenerationResponse
from .generator import CodeGenerator

# Setup
setup_logging()
settings = get_settings()

# Global generator instance
code_generator: CodeGenerator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global code_generator

    logger.info("Starting Code Generation Service")
    logger.info(f"GPU enabled: {settings.use_gpu}")

    # Initialize code generator
    code_generator = CodeGenerator()

    yield

    logger.info("Shutting down Code Generation Service")


# Create FastAPI app
app = FastAPI(
    title="Code Generation Service",
    description="AI-powered code generation using transformer models",
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
        "service": "Code Generation Service",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """
    Generate code based on the provided prompt.

    Args:
        request: Code generation request

    Returns:
        Generated code with tests and documentation
    """
    try:
        logger.info(f"Received code generation request: {request.language}")
        result = await code_generator.generate(request)
        return result
    except Exception as e:
        logger.error(f"Error in code generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {str(e)}",
        )


@app.post("/analyze")
async def analyze_code(code: str, language: str):
    """
    Analyze code for quality and issues.

    Args:
        code: Code to analyze
        language: Programming language

    Returns:
        Analysis results
    """
    # Placeholder for code analysis
    return {
        "is_valid": True,
        "syntax_errors": [],
        "warnings": [],
        "suggestions": ["Consider adding type hints", "Add docstrings"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
