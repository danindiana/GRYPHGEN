"""Code Generation Service Router."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field

from ...auth.api_keys import require_api_key
from ...auth.rate_limit import generate_limiter
from ...llm.generator import get_generator

router = APIRouter()


class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description of the code to generate")
    language: str = Field(default="python", description="Programming language")
    framework: Optional[str] = Field(None, description="Optional framework to use")
    include_tests: bool = Field(default=False, description="Generate tests alongside code")
    include_docs: bool = Field(default=True, description="Generate documentation")
    style_guide: Optional[str] = Field(None, description="Code style guide")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    model: Optional[str] = Field(None, description="Override model (backend-specific name)")
    context_files: Optional[list[str]] = Field(
        None, description="Filenames involved — signals multi-file complexity to router"
    )
    reasoning_effort: Optional[str] = Field(
        None, description="Override routing: 'none' for fast, 'high' for deep reasoning"
    )


class CodeGenerationResponse(BaseModel):
    request_id: str
    code: str
    language: str
    framework: Optional[str] = None
    model_used: str
    tokens_used: int
    generation_time: float
    tier: str = Field(default="", description="Complexity tier chosen by router")
    reasoning_effort_used: str = Field(default="", description="reasoning_effort applied")
    backend_used: str = Field(default="", description="Backend that produced the response")


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    api_key: str = Security(require_api_key),
) -> CodeGenerationResponse:
    """Generate code from a natural language prompt using the configured LLM backend."""
    generate_limiter.check(api_key)
    generator = get_generator()

    try:
        result = await generator.generate(
            prompt=request.prompt,
            language=request.language,
            framework=request.framework,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_tests=request.include_tests,
            include_docs=request.include_docs,
            style_guide=request.style_guide,
            context_files=request.context_files,
            reasoning_effort=request.reasoning_effort,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM backend error: {exc}") from exc

    return CodeGenerationResponse(
        request_id=str(uuid.uuid4()),
        code=result.text,
        language=request.language,
        framework=request.framework,
        model_used=result.model,
        tokens_used=result.tokens_used,
        generation_time=result.generation_time,
        tier=result.tier,
        reasoning_effort_used=result.reasoning_effort,
        backend_used=result.backend_used,
    )


@router.get("/models")
async def list_available_models() -> list[str]:
    """List models available from the current backend."""
    generator = get_generator()
    try:
        return await generator.list_models()
    except Exception:
        return []


@router.get("/languages")
async def list_supported_languages() -> list[str]:
    return [
        "python", "javascript", "typescript", "go", "rust",
        "java", "c", "c++", "c#", "bash", "ruby", "php",
    ]
