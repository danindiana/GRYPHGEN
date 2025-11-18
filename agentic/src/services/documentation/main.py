"""Documentation Service API - NLP-based documentation generation."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from loguru import logger
from openai import AsyncOpenAI

from ...common.config import get_settings
from ...common.logger import setup_logging

# Setup
setup_logging()
settings = get_settings()

# Global client
openai_client: Optional[AsyncOpenAI] = None


class DocGenerationRequest(BaseModel):
    """Documentation generation request."""

    code: str = Field(..., description="Code to document")
    language: str = Field(..., description="Programming language")
    doc_type: str = Field(default="comprehensive", description="Type of documentation")
    include_examples: bool = Field(default=True)
    include_api_spec: bool = Field(default=False)


class DocGenerationResponse(BaseModel):
    """Documentation generation response."""

    request_id: str
    markdown: str
    docstrings: Optional[str] = None
    api_spec: Optional[dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global openai_client

    logger.info("Starting Documentation Service")

    if settings.openai_api_key:
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    yield

    logger.info("Shutting down Documentation Service")


# Create FastAPI app
app = FastAPI(
    title="Documentation Service",
    description="NLP-based documentation generation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS and metrics
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "Documentation Service", "version": "0.1.0", "status": "operational"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate", response_model=DocGenerationResponse)
async def generate_documentation(request: DocGenerationRequest):
    """Generate comprehensive documentation for code."""
    import uuid

    try:
        request_id = str(uuid.uuid4())

        if not openai_client:
            raise HTTPException(
                status_code=503, detail="OpenAI API key not configured"
            )

        prompt = f"""Generate comprehensive documentation for this {request.language} code:

{request.code}

Include:
- Overview and purpose
- Detailed function/class descriptions
- Parameter documentation with types
- Return value descriptions
- Usage examples
- Error handling notes

Format as Markdown.
"""

        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a technical documentation expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2048,
        )

        markdown = response.choices[0].message.content

        return DocGenerationResponse(request_id=request_id, markdown=markdown)

    except Exception as e:
        logger.error(f"Documentation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)
