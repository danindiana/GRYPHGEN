"""Tests for Code Generation Service."""

import pytest
from src.services.code_generation.models import (
    CodeGenerationRequest,
    CodeLanguage,
    CodeStyle,
)


@pytest.mark.unit
def test_code_generation_request_validation():
    """Test request model validation."""
    request = CodeGenerationRequest(
        prompt="Create a hello world function",
        language=CodeLanguage.PYTHON,
        style=CodeStyle.FUNCTIONAL,
    )

    assert request.prompt == "Create a hello world function"
    assert request.language == CodeLanguage.PYTHON
    assert request.include_tests is True


@pytest.mark.unit
def test_code_generation_request_invalid_prompt():
    """Test that empty prompt raises validation error."""
    with pytest.raises(ValueError):
        CodeGenerationRequest(
            prompt="   ",
            language=CodeLanguage.PYTHON,
        )


@pytest.mark.unit
def test_supported_languages():
    """Test all supported languages are valid."""
    languages = [
        CodeLanguage.PYTHON,
        CodeLanguage.JAVASCRIPT,
        CodeLanguage.TYPESCRIPT,
        CodeLanguage.JAVA,
        CodeLanguage.GO,
    ]

    for lang in languages:
        request = CodeGenerationRequest(
            prompt="Test function",
            language=lang,
        )
        assert request.language == lang
