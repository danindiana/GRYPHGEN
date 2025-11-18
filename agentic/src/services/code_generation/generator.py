"""Code generation engine using transformer models."""

import asyncio
import time
import uuid
from typing import Optional, Dict, Any

import torch
from loguru import logger
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ...common.config import get_settings
from .models import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    GeneratedCode,
    GeneratedTests,
    GeneratedDocumentation,
    ModelProvider,
)


class CodeGenerator:
    """Code generation using transformer models."""

    def __init__(self):
        """Initialize code generator."""
        self.settings = get_settings()
        self.openai_client = None
        self.anthropic_client = None

        # Initialize clients if API keys are available
        if self.settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

        if self.settings.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() and self.settings.use_gpu else "cpu"
        logger.info(f"Code Generator initialized with device: {self.device}")

        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    async def generate(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """
        Generate code based on request.

        Args:
            request: Code generation request

        Returns:
            Code generation response with code, tests, and documentation
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"Generating code for request {request_id}")
        logger.info(f"Language: {request.language}, Framework: {request.framework}")

        try:
            # Generate code
            code_result = await self._generate_code(request)

            # Generate tests if requested
            tests_result = None
            if request.include_tests:
                tests_result = await self._generate_tests(code_result.code, request.language)

            # Generate documentation if requested
            docs_result = None
            if request.include_docs:
                docs_result = await self._generate_documentation(code_result.code, request.language)

            generation_time = time.time() - start_time

            response = CodeGenerationResponse(
                request_id=request_id,
                code=code_result,
                tests=tests_result,
                documentation=docs_result,
                model_used=self.settings.code_gen_model,
                provider=ModelProvider.OPENAI if self.openai_client else ModelProvider.ANTHROPIC,
                tokens_used=0,  # Updated by actual API response
                generation_time=generation_time,
                metadata={
                    "prompt": request.prompt,
                    "framework": request.framework,
                    "style": request.style.value,
                },
            )

            logger.info(f"Code generation completed in {generation_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    async def _generate_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code using AI model."""
        # Build prompt
        prompt = self._build_code_prompt(request)

        # Use OpenAI if available
        if self.openai_client:
            code = await self._generate_with_openai(prompt, request)
        elif self.anthropic_client:
            code = await self._generate_with_anthropic(prompt, request)
        else:
            # Fallback to local generation (placeholder)
            code = self._generate_local(prompt, request)

        # Extract dependencies
        dependencies = self._extract_dependencies(code, request.language)

        return GeneratedCode(
            code=code,
            language=request.language,
            framework=request.framework,
            explanation=f"Generated {request.language.value} code for: {request.prompt}",
            dependencies=dependencies,
        )

    async def _generate_with_openai(self, prompt: str, request: CodeGenerationRequest) -> str:
        """Generate code using OpenAI API."""
        logger.info("Generating code with OpenAI")

        response = await self.openai_client.chat.completions.create(
            model=self.settings.code_gen_model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {request.language.value} developer. "
                    "Generate clean, efficient, well-documented code.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return response.choices[0].message.content

    async def _generate_with_anthropic(self, prompt: str, request: CodeGenerationRequest) -> str:
        """Generate code using Anthropic Claude API."""
        logger.info("Generating code with Anthropic Claude")

        message = await self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _generate_local(self, prompt: str, request: CodeGenerationRequest) -> str:
        """Generate code using local model (placeholder)."""
        logger.warning("No API keys configured, using placeholder generation")

        # This is a placeholder - in production, would use local transformer model
        return f"""# Generated {request.language.value} code
# Prompt: {request.prompt}

def placeholder_function():
    '''
    This is a placeholder implementation.
    Configure OpenAI or Anthropic API keys for actual code generation.
    '''
    pass
"""

    async def _generate_tests(self, code: str, language: str) -> GeneratedTests:
        """Generate test cases for the code."""
        test_frameworks = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "java": "junit",
            "go": "testing",
        }

        framework = test_frameworks.get(language, "pytest")

        if self.openai_client:
            prompt = f"""Generate comprehensive test cases for this {language} code:

{code}

Use {framework}. Include:
- Unit tests for all functions
- Edge cases
- Error handling tests
"""

            response = await self.openai_client.chat.completions.create(
                model=self.settings.code_gen_model,
                messages=[
                    {"role": "system", "content": "You are a testing expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=2048,
            )

            test_code = response.choices[0].message.content

            return GeneratedTests(
                test_code=test_code,
                test_framework=framework,
                test_count=test_code.count("def test_") if language == "python" else 0,
                coverage_estimate=85.0,
            )

        # Placeholder
        return GeneratedTests(
            test_code=f"# {framework} tests\n# Configure API key for test generation",
            test_framework=framework,
            test_count=0,
        )

    async def _generate_documentation(
        self, code: str, language: str
    ) -> GeneratedDocumentation:
        """Generate documentation for the code."""
        if self.openai_client:
            prompt = f"""Generate comprehensive documentation for this {language} code:

{code}

Include:
- Overview and purpose
- Function/class descriptions
- Parameter documentation
- Usage examples
- Return value descriptions
"""

            response = await self.openai_client.chat.completions.create(
                model=self.settings.code_gen_model,
                messages=[
                    {"role": "system", "content": "You are a technical writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=2048,
            )

            markdown = response.choices[0].message.content

            return GeneratedDocumentation(markdown=markdown)

        # Placeholder
        return GeneratedDocumentation(
            markdown="# Documentation\n\nConfigure API key for documentation generation"
        )

    def _build_code_prompt(self, request: CodeGenerationRequest) -> str:
        """Build the prompt for code generation."""
        prompt_parts = [f"Generate {request.language.value} code for the following requirement:"]

        prompt_parts.append(f"\nRequirement: {request.prompt}")

        if request.framework:
            prompt_parts.append(f"\nFramework: {request.framework}")

        prompt_parts.append(f"\nCode Style: {request.style.value}")

        if request.context:
            prompt_parts.append(f"\nAdditional Context: {request.context}")

        prompt_parts.append("\nRequirements:")
        prompt_parts.append("- Clean, readable code")
        prompt_parts.append("- Proper error handling")
        prompt_parts.append("- Type hints (if applicable)")
        prompt_parts.append("- Docstrings/comments")
        prompt_parts.append("- Follow best practices")

        return "\n".join(prompt_parts)

    def _extract_dependencies(self, code: str, language: str) -> list:
        """Extract dependencies from generated code."""
        dependencies = []

        if language == "python":
            # Simple import extraction
            lines = code.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    # Extract package name
                    parts = line.split()
                    if len(parts) >= 2:
                        pkg = parts[1].split(".")[0]
                        if pkg not in ["os", "sys", "re", "json", "typing"]:  # Skip stdlib
                            dependencies.append(pkg)

        return list(set(dependencies))
