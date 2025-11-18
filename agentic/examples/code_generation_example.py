#!/usr/bin/env python3
"""
Example: Using the Code Generation Service

This script demonstrates how to use the GRYPHGEN Agentic code generation
service to generate code, tests, and documentation.
"""

import asyncio
from typing import Dict, Any

import httpx


async def generate_code(
    prompt: str,
    language: str = "python",
    framework: str = None,
    api_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Generate code using the agentic API.

    Args:
        prompt: Description of the code to generate
        language: Programming language (default: python)
        framework: Optional framework to use
        api_url: Base URL of the API gateway

    Returns:
        Dictionary containing generated code, tests, and documentation
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_url}/api/v1/code/generate",
            json={
                "prompt": prompt,
                "language": language,
                "framework": framework,
                "include_tests": True,
                "include_docs": True,
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Main example function."""

    print("=" * 80)
    print("GRYPHGEN Agentic - Code Generation Example")
    print("=" * 80)
    print()

    # Example 1: Simple function
    print("Example 1: Generating a Fibonacci function")
    print("-" * 80)

    result = await generate_code(
        prompt="Create a Python function to calculate the nth Fibonacci number using memoization",
        language="python"
    )

    print("Generated Code:")
    print(result.get("code", "No code generated"))
    print()
    print("Generated Tests:")
    print(result.get("tests", "No tests generated"))
    print()
    print("Generated Documentation:")
    print(result.get("documentation", "No documentation generated"))
    print()

    # Example 2: REST API
    print("Example 2: Generating a REST API")
    print("-" * 80)

    result = await generate_code(
        prompt="Create a FastAPI REST API with CRUD endpoints for a todo list",
        language="python",
        framework="fastapi"
    )

    print("Generated Code:")
    print(result.get("code", "No code generated"))
    print()

    # Example 3: Data processing
    print("Example 3: Generating a data processing pipeline")
    print("-" * 80)

    result = await generate_code(
        prompt="Create a pandas pipeline to clean and analyze CSV data with missing values",
        language="python",
        framework="pandas"
    )

    print("Generated Code:")
    print(result.get("code", "No code generated"))
    print()

    print("=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
