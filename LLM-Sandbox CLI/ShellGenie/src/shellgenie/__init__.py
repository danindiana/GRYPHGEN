"""
ShellGenie - AI-powered bash shell assistant using LLMs with GPU acceleration.

A modern, secure, and efficient CLI tool that bridges natural language
and bash commands using local language models.
"""

__version__ = "2.0.0"
__author__ = "GRYPHGEN Team"
__email__ = "dev@gryphgen.ai"

from shellgenie.core import ShellGenieCore
from shellgenie.models import CommandRequest, CommandResponse, ModelConfig

__all__ = [
    "ShellGenieCore",
    "CommandRequest",
    "CommandResponse",
    "ModelConfig",
    "__version__",
]
