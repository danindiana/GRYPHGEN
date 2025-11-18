"""Task execution endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.calisota.core.config import Settings, get_settings
from src.calisota.agents.base import SlowThinkerAgent, FastThinkerCodeGenerator, ActorCriticAgent
from src.calisota.sandbox.executor import SandboxExecutor
from src.calisota.api.main import get_faiss_manager
from src.calisota.rag.faiss_manager import FAISSManager

router = APIRouter()
logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    """Task execution request."""
    task: str = Field(..., description="Task description")
    language: str = Field(default="python", description="Target programming language")
    context: Optional[str] = Field(default="", description="Additional context")
    use_rag: bool = Field(default=True, description="Use RAG for context retrieval")
    auto_execute: bool = Field(default=False, description="Automatically execute generated code")


class TaskResponse(BaseModel):
    """Task execution response."""
    task_id: str
    status: str
    plan: Optional[str] = None
    code: Optional[str] = None
    evaluation: Optional[dict] = None
    execution_result: Optional[dict] = None


@router.post("/execute", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    settings: Settings = Depends(get_settings),
    faiss: FAISSManager = Depends(get_faiss_manager)
) -> TaskResponse:
    """
    Execute a complete task through the ensemble system.

    Flow:
    1. Slow thinker creates high-level plan
    2. RAG retrieves relevant code samples (if enabled)
    3. Fast thinker generates code
    4. Sandbox executes code (if auto_execute)
    5. Actor-critic evaluates results
    """
    import uuid
    task_id = str(uuid.uuid4())

    try:
        # Step 1: Slow thinker creates plan
        slow_thinker = SlowThinkerAgent(settings, settings.slow_thinker_model)

        # Retrieve context from RAG if enabled
        rag_context = ""
        if request.use_rag:
            rag_results = faiss.search(request.task, top_k=3)
            rag_context = "\n\n".join([r.get("text", "") for r in rag_results])

        plan_result = await slow_thinker.process({
            "task": request.task,
            "context": request.context + "\n\nRAG Context:\n" + rag_context
        })

        # Step 2: Fast thinker generates code
        fast_thinker = FastThinkerCodeGenerator(settings, settings.fast_thinker_model)

        # Get code samples from RAG
        code_samples = ""
        if request.use_rag:
            code_results = faiss.search(
                f"{request.language} code example {request.task}",
                top_k=2
            )
            code_samples = "\n\n".join([r.get("text", "") for r in code_results])

        code_result = await fast_thinker.process({
            "plan": plan_result["plan"],
            "language": request.language,
            "guidance": plan_result["plan"],
            "code_samples": code_samples
        })

        # Step 3: Execute code if requested
        execution_result = None
        if request.auto_execute:
            sandbox = SandboxExecutor(settings)
            execution_result = await sandbox.execute(
                code=code_result["code"],
                language=request.language
            )

        # Step 4: Actor-critic evaluation
        actor_critic = ActorCriticAgent(settings, settings.actor_critic_model)
        evaluation = await actor_critic.process({
            "code": code_result["code"],
            "plan": plan_result["plan"],
            "execution_result": str(execution_result) if execution_result else "Not executed"
        })

        return TaskResponse(
            task_id=task_id,
            status="completed" if not evaluation.get("needs_refinement") else "needs_refinement",
            plan=plan_result["plan"],
            code=code_result["code"],
            evaluation=evaluation,
            execution_result=execution_result
        )

    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages")
async def get_supported_languages(settings: Settings = Depends(get_settings)) -> dict:
    """Get list of supported programming languages."""
    sandbox = SandboxExecutor(settings)
    return {
        "languages": sandbox.get_supported_languages(),
        "default": "python"
    }
