#!/usr/bin/env python3
"""Basic usage examples for ShellGenie."""

import asyncio

from shellgenie import ShellGenieCore
from shellgenie.models import (
    AppConfig,
    CommandRequest,
    ModelBackend,
    ModelConfig,
    SecurityLevel,
)


async def example_basic_generation():
    """Example: Generate a simple command."""
    print("=== Example 1: Basic Command Generation ===\n")

    # Initialize ShellGenie
    core = ShellGenieCore()

    # Create a request
    request = CommandRequest(
        prompt="list all PDF files in the current directory"
    )

    # Generate command
    response = await core.generate_command(request)

    print(f"Generated Command: {response.command}")
    print(f"Explanation: {response.explanation}")
    print(f"Risk Level: {response.risk_level}")
    if response.warnings:
        print(f"Warnings: {', '.join(response.warnings)}")
    print()


async def example_with_custom_config():
    """Example: Use custom configuration."""
    print("=== Example 2: Custom Configuration ===\n")

    # Custom model configuration
    model_config = ModelConfig(
        backend=ModelBackend.OLLAMA,
        model_name="llama3.2",
        temperature=0.5,  # Lower temperature for more deterministic output
        max_tokens=256,
    )

    # Custom app configuration
    app_config = AppConfig(
        security_level=SecurityLevel.STRICT,
        auto_execute=False,
    )

    # Initialize with custom configs
    core = ShellGenieCore(
        model_config=model_config,
        app_config=app_config,
    )

    # Process request
    response, _ = await core.process_request(
        "find all files larger than 100MB"
    )

    print(f"Command: {response.command}")
    print(f"Risk: {response.risk_level}")
    print()


async def example_with_execution():
    """Example: Generate and execute a command."""
    print("=== Example 3: Command Execution ===\n")

    core = ShellGenieCore()

    # Generate command
    response, _ = await core.process_request(
        "show current directory"
    )

    print(f"Generated: {response.command}")

    # Execute the command
    if response.risk_level == "low":
        result = core.execute(response.command)

        print(f"Exit Code: {result.return_code}")
        print(f"Output:\n{result.stdout}")
        print(f"Execution Time: {result.execution_time:.2f}s")
    else:
        print(f"Skipping execution due to risk level: {response.risk_level}")
    print()


async def example_with_context():
    """Example: Provide context for better results."""
    print("=== Example 4: With Context ===\n")

    core = ShellGenieCore()

    # Add context
    context = {
        "project_type": "Python",
        "task": "cleanup",
    }

    request = CommandRequest(
        prompt="remove all Python cache files",
        context=context,
    )

    response = await core.generate_command(request)

    print(f"Command: {response.command}")
    print(f"Explanation: {response.explanation}")
    print()


async def example_with_history():
    """Example: Use command history for context."""
    print("=== Example 5: With History ===\n")

    core = ShellGenieCore()

    # Simulate some history
    history = [
        "cd /tmp",
        "mkdir test_dir",
        "cd test_dir",
    ]

    request = CommandRequest(
        prompt="create a file called test.txt",
        history=history,
    )

    response = await core.generate_command(request)

    print(f"Command: {response.command}")
    print(f"Given history, the command understands the current context")
    print()


async def example_security_levels():
    """Example: Different security levels."""
    print("=== Example 6: Security Levels ===\n")

    dangerous_command_prompt = "delete all files in current directory"

    for level in [SecurityLevel.STRICT, SecurityLevel.MODERATE, SecurityLevel.PERMISSIVE]:
        print(f"Security Level: {level.value}")

        app_config = AppConfig(security_level=level)
        core = ShellGenieCore(app_config=app_config)

        request = CommandRequest(prompt=dangerous_command_prompt)
        response = await core.generate_command(request)

        print(f"  Command: {response.command or 'BLOCKED'}")
        print(f"  Risk: {response.risk_level}")
        if response.warnings:
            print(f"  Warnings: {len(response.warnings)}")
        print()


async def main():
    """Run all examples."""
    print("ShellGenie Usage Examples\n")
    print("=" * 60)
    print()

    await example_basic_generation()
    await example_with_custom_config()
    await example_with_execution()
    await example_with_context()
    await example_with_history()
    await example_security_levels()

    print("=" * 60)
    print("\nAll examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
