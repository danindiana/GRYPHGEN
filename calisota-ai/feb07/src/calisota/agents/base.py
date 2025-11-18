"""Base agent classes for CALISOTA ensemble system."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from src.calisota.core.config import Settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the ensemble."""

    def __init__(self, settings: Settings, model_name: str) -> None:
        """Initialize base agent."""
        self.settings = settings
        self.model_name = model_name

        # Initialize API clients
        self.openai_client: Optional[AsyncOpenAI] = None
        self.anthropic_client: Optional[AsyncAnthropic] = None

        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        if settings.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    @abstractmethod
    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process input and return output."""
        pass

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Call LLM API based on model configuration."""
        if "gpt" in self.model_name.lower() and self.openai_client:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""

        elif "claude" in self.model_name.lower() and self.anthropic_client:
            # Convert messages format for Anthropic
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]

            response = await self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=user_messages
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")


class SlowThinkerAgent(BaseAgent):
    """
    Large 'slow-thinker' agent for high-level planning and guidance.

    Responsibilities:
    - Task decomposition
    - High-level planning
    - Context retrieval from RAG
    - Clarification requests to frontier models
    """

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process task and create high-level plan.

        Args:
            input_data: Contains 'task', 'context', etc.

        Returns:
            Plan with subtasks, approach, and guidance
        """
        task = input_data.get("task", "")
        context = input_data.get("context", "")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software architect. Break down complex tasks "
                    "into actionable subtasks. Provide high-level guidance and approach."
                )
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nContext: {context}\n\n"
                          f"Create a detailed plan with subtasks."
            }
        ]

        plan = await self.call_llm(messages, temperature=0.3)

        logger.info(f"Slow thinker created plan for task: {task[:50]}...")

        return {
            "plan": plan,
            "task": task,
            "agent": "slow_thinker",
            "model": self.model_name
        }


class FastThinkerCodeGenerator(BaseAgent):
    """
    Smaller 'fast-thinker' agent for rapid code generation.

    Responsibilities:
    - Generate code based on guidance
    - Retrieve code samples from RAG
    - Quick iteration on code improvements
    """

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate code based on plan and guidance.

        Args:
            input_data: Contains 'plan', 'language', 'guidance', etc.

        Returns:
            Generated code with explanation
        """
        plan = input_data.get("plan", "")
        language = input_data.get("language", "python")
        guidance = input_data.get("guidance", "")
        code_samples = input_data.get("code_samples", "")

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert {language} developer. Generate clean, "
                    f"efficient, well-documented code following best practices."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Plan: {plan}\n\n"
                    f"Guidance: {guidance}\n\n"
                    f"Reference code samples: {code_samples}\n\n"
                    f"Generate {language} code to implement this plan."
                )
            }
        ]

        code = await self.call_llm(messages, temperature=0.7)

        logger.info(f"Fast thinker generated {language} code")

        return {
            "code": code,
            "language": language,
            "agent": "fast_thinker",
            "model": self.model_name
        }


class ActorCriticAgent(BaseAgent):
    """
    Actor-Critic agent for evaluating and improving code.

    Responsibilities:
    - Evaluate generated code quality
    - Identify bugs and improvements
    - Provide feedback for refinement
    - Score code against criteria
    """

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate code and provide feedback.

        Args:
            input_data: Contains 'code', 'execution_result', 'plan', etc.

        Returns:
            Evaluation with score, feedback, and improvement suggestions
        """
        code = input_data.get("code", "")
        execution_result = input_data.get("execution_result", "")
        plan = input_data.get("plan", "")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code reviewer and quality analyst. Evaluate code for:\n"
                    "1. Correctness and functionality\n"
                    "2. Code quality and best practices\n"
                    "3. Performance and efficiency\n"
                    "4. Security and safety\n"
                    "5. Documentation and readability\n\n"
                    "Provide a score (0-100) and detailed feedback."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Plan: {plan}\n\n"
                    f"Generated Code:\n```\n{code}\n```\n\n"
                    f"Execution Result: {execution_result}\n\n"
                    f"Evaluate this code and provide:\n"
                    f"1. Overall score (0-100)\n"
                    f"2. Strengths\n"
                    f"3. Weaknesses\n"
                    f"4. Specific improvements needed\n"
                    f"5. Refinement suggestions"
                )
            }
        ]

        evaluation = await self.call_llm(messages, temperature=0.3)

        # Parse score from evaluation (simple regex)
        import re
        score_match = re.search(r'score[:\s]+(\d+)', evaluation, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 50

        logger.info(f"Actor-critic evaluated code: Score {score}/100")

        return {
            "evaluation": evaluation,
            "score": score,
            "needs_refinement": score < 80,
            "agent": "actor_critic",
            "model": self.model_name
        }
