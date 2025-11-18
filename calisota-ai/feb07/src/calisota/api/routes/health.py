"""Health check endpoints."""

import torch
from fastapi import APIRouter, Depends

from src.calisota.core.config import Settings, get_settings
from src.calisota.api.main import get_faiss_manager
from src.calisota.rag.faiss_manager import FAISSManager

router = APIRouter()


@router.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings),
    faiss: FAISSManager = Depends(get_faiss_manager)
) -> dict:
    """Comprehensive health check."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
            "memory_reserved_mb": torch.cuda.memory_reserved(0) / 1024 / 1024,
            "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        }
    else:
        gpu_info = {"available": False}

    return {
        "status": "healthy",
        "version": "1.0.0",
        "gpu": gpu_info,
        "faiss": faiss.get_stats(),
        "config": {
            "debug": settings.debug,
            "gpu_enabled": settings.use_gpu_index,
            "supported_languages": settings.supported_languages,
        }
    }


@router.get("/ready")
async def readiness_check(faiss: FAISSManager = Depends(get_faiss_manager)) -> dict:
    """Readiness check for Kubernetes."""
    stats = faiss.get_stats()
    return {
        "ready": stats["status"] == "ready",
        "faiss_initialized": stats["status"] == "ready",
    }


@router.get("/live")
async def liveness_check() -> dict:
    """Liveness check for Kubernetes."""
    return {"alive": True}
