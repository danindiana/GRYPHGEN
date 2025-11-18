"""Data models for Code Generation Service."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator


class CodeLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"


class CodeStyle(str, Enum):
    """Code generation styles."""

    FUNCTIONAL = "functional"
    OOP = "oop"
    PROCEDURAL = "procedural"
    REACTIVE = "reactive"


class ModelProvider(str, Enum):
    """AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""

    prompt: str = Field(..., description="Description of code to generate", min_length=10)
    language: CodeLanguage = Field(default=CodeLanguage.PYTHON, description="Target language")
    framework: Optional[str] = Field(None, description="Optional framework (e.g., fastapi, react)")
    style: CodeStyle = Field(default=CodeStyle.FUNCTIONAL, description="Code style preference")
    include_tests: bool = Field(default=True, description="Generate test cases")
    include_docs: bool = Field(default=True, description="Generate documentation")
    max_tokens: int = Field(default=4096, ge=100, le=8192, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    @validator("prompt")
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty and meaningful."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class GeneratedCode(BaseModel):
    """Generated code with metadata."""

    code: str = Field(..., description="Generated code")
    language: CodeLanguage
    framework: Optional[str] = None
    explanation: Optional[str] = Field(None, description="Code explanation")
    complexity: Optional[str] = Field(None, description="Time/space complexity")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")


class GeneratedTests(BaseModel):
    """Generated test cases."""

    test_code: str = Field(..., description="Test code")
    test_framework: str = Field(..., description="Testing framework used")
    test_count: int = Field(default=0, description="Number of test cases")
    coverage_estimate: Optional[float] = Field(None, ge=0.0, le=100.0)


class GeneratedDocumentation(BaseModel):
    """Generated documentation."""

    markdown: str = Field(..., description="Documentation in markdown")
    docstrings: Optional[str] = Field(None, description="Inline docstrings")
    api_spec: Optional[Dict[str, Any]] = Field(None, description="API specification")


class CodeGenerationResponse(BaseModel):
    """Response model for code generation."""

    request_id: str = Field(..., description="Unique request identifier")
    code: GeneratedCode
    tests: Optional[GeneratedTests] = None
    documentation: Optional[GeneratedDocumentation] = None
    model_used: str = Field(..., description="Model used for generation")
    provider: ModelProvider
    tokens_used: int = Field(..., description="Total tokens consumed")
    generation_time: float = Field(..., description="Time taken in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeQualityMetrics(BaseModel):
    """Code quality metrics."""

    cyclomatic_complexity: Optional[int] = None
    maintainability_index: Optional[float] = None
    lines_of_code: int
    comment_ratio: Optional[float] = None
    duplicate_code: Optional[float] = None


class CodeAnalysisResult(BaseModel):
    """Code analysis result."""

    is_valid: bool
    syntax_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    quality_metrics: Optional[CodeQualityMetrics] = None
    security_issues: List[str] = Field(default_factory=list)
