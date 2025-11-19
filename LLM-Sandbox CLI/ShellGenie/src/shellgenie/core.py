"""Core functionality for ShellGenie."""

import asyncio
import json
from typing import Dict, List, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from shellgenie.models import (
    AppConfig,
    CommandRequest,
    CommandResponse,
    ExecutionResult,
    ModelBackend,
    ModelConfig,
    SecurityLevel,
)
from shellgenie.security import SecurityValidator
from shellgenie.utils import (
    create_prompt_template,
    execute_command,
    get_system_info,
    load_command_history,
    save_to_history,
)


class ShellGenieCore:
    """Core ShellGenie functionality."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        app_config: Optional[AppConfig] = None,
    ):
        """Initialize ShellGenie.

        Args:
            model_config: Model configuration
            app_config: Application configuration
        """
        self.model_config = model_config or ModelConfig()
        self.app_config = app_config or AppConfig()
        self.security = SecurityValidator(self.app_config.security_level)
        self.system_info = get_system_info()
        self.history: List[str] = []

        # Load history
        if self.app_config.history_file:
            self.history = load_command_history(
                self.app_config.history_file,
                self.app_config.max_history
            )

        logger.info("ShellGenie initialized")
        logger.debug(f"Model backend: {self.model_config.backend}")
        logger.debug(f"Security level: {self.app_config.security_level}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_command(self, request: CommandRequest) -> CommandResponse:
        """Generate a bash command from natural language.

        Args:
            request: Command generation request

        Returns:
            CommandResponse with generated command

        Raises:
            Exception: If generation fails
        """
        # Create prompt
        prompt = create_prompt_template(
            task=request.prompt,
            context=request.context,
            system_info=self.system_info,
        )

        # Add history context if available
        if request.history or self.history:
            history_context = request.history or self.history[-5:]
            if history_context:
                prompt += "\n\nRecent commands:\n" + "\n".join(history_context)

        # Generate command based on backend
        if self.model_config.backend == ModelBackend.OLLAMA:
            command = await self._generate_with_ollama(prompt)
        elif self.model_config.backend == ModelBackend.LLAMA_CPP:
            command = await self._generate_with_llama_cpp(prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.model_config.backend}")

        # Clean up the generated command
        command = self._clean_command(command)

        # Validate security
        response = self.security.validate_command(command)

        # If command was blocked, return empty response
        if not response.command:
            logger.warning("Command blocked by security validator")
            return response

        return response

    async def _generate_with_ollama(self, prompt: str) -> str:
        """Generate command using Ollama.

        Args:
            prompt: The prompt to send

        Returns:
            Generated command string
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.model_config.api_base}/api/generate",
                    json={
                        "model": self.model_config.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.model_config.temperature,
                            "num_predict": self.model_config.max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")

        except httpx.HTTPError as e:
            logger.error(f"Ollama request failed: {e}")
            raise Exception(f"Failed to generate command with Ollama: {e}")

    async def _generate_with_llama_cpp(self, prompt: str) -> str:
        """Generate command using llama.cpp.

        Args:
            prompt: The prompt to send

        Returns:
            Generated command string
        """
        try:
            from llama_cpp import Llama

            # Initialize model if not already done
            if not hasattr(self, "_llama_model"):
                logger.info("Loading llama.cpp model...")
                self._llama_model = Llama(
                    model_path=self.model_config.model_path,
                    n_gpu_layers=self.model_config.gpu_layers,
                    n_threads=self.model_config.threads,
                    n_batch=self.model_config.batch_size,
                    n_ctx=self.model_config.context_length,
                )

            # Generate
            result = self._llama_model(
                prompt,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                stop=["User request:", "\n\n"],
            )

            return result["choices"][0]["text"]

        except ImportError:
            raise Exception("llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"llama.cpp generation failed: {e}")
            raise Exception(f"Failed to generate command with llama.cpp: {e}")

    def _clean_command(self, command: str) -> str:
        """Clean up generated command.

        Args:
            command: Raw generated command

        Returns:
            Cleaned command string
        """
        # Remove common prefixes
        prefixes = ["$ ", "bash\n", "```bash\n", "```\n", "> "]
        for prefix in prefixes:
            if command.startswith(prefix):
                command = command[len(prefix):]

        # Remove common suffixes
        suffixes = ["\n```", "```"]
        for suffix in suffixes:
            if command.endswith(suffix):
                command = command[:-len(suffix)]

        # Strip whitespace
        command = command.strip()

        # If multi-line, take first non-empty line
        lines = [line.strip() for line in command.split("\n") if line.strip()]
        if lines:
            command = lines[0]

        return command

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a command.

        Args:
            command: Command to execute
            timeout: Execution timeout
            env: Environment variables

        Returns:
            ExecutionResult with execution details
        """
        timeout = timeout or self.app_config.timeout

        logger.info(f"Executing command: {command}")

        # Execute
        result = execute_command(command, timeout=timeout, env=env)

        # Save to history if successful
        if result.success and self.app_config.history_file:
            save_to_history(self.app_config.history_file, command, self.app_config.max_history)
            self.history.append(command)
            if len(self.history) > self.app_config.max_history:
                self.history = self.history[-self.app_config.max_history:]

        logger.info(f"Command execution {'succeeded' if result.success else 'failed'} in {result.execution_time:.2f}s")

        return result

    async def process_request(
        self,
        prompt: str,
        context: Optional[Dict[str, any]] = None,
        auto_execute: Optional[bool] = None,
    ) -> tuple[CommandResponse, Optional[ExecutionResult]]:
        """Process a complete request from prompt to execution.

        Args:
            prompt: Natural language prompt
            context: Additional context
            auto_execute: Whether to auto-execute (overrides config)

        Returns:
            Tuple of (CommandResponse, ExecutionResult or None)
        """
        # Create request
        request = CommandRequest(
            prompt=prompt,
            context=context or {},
            history=self.history[-5:] if self.history else None,
        )

        # Generate command
        response = await self.generate_command(request)

        # Check if command was blocked
        if not response.command:
            return response, None

        # Execute if auto_execute is enabled
        execution_result = None
        should_execute = auto_execute if auto_execute is not None else self.app_config.auto_execute

        if should_execute and response.risk_level not in ["high", "critical"]:
            execution_result = self.execute(response.command)

        return response, execution_result

    def get_stats(self) -> Dict[str, any]:
        """Get usage statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "commands_in_history": len(self.history),
            "security_level": self.app_config.security_level.value,
            "model_backend": self.model_config.backend.value,
            "model_name": self.model_config.model_name,
            "gpu_available": self.system_info.gpu_available,
            "gpu_name": self.system_info.gpu_name,
        }

    def set_security_level(self, level: SecurityLevel) -> None:
        """Set security level.

        Args:
            level: New security level
        """
        self.app_config.security_level = level
        self.security.set_security_level(level)
        logger.info(f"Security level changed to: {level.value}")

    def clear_history(self) -> None:
        """Clear command history."""
        self.history = []
        if self.app_config.history_file:
            try:
                from pathlib import Path
                history_path = Path(self.app_config.history_file).expanduser()
                if history_path.exists():
                    history_path.unlink()
                logger.info("History cleared")
            except Exception as e:
                logger.error(f"Failed to clear history: {e}")
