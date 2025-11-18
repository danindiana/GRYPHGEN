"""Self-Improvement Service API - Meta-learning for continuous improvement."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from loguru import logger

from ...common.config import get_settings
from ...common.logger import setup_logging

# Setup
setup_logging()
settings = get_settings()


class PerformanceMetric(BaseModel):
    """Performance metric."""

    service: str
    metric_name: str
    value: float
    timestamp: datetime


class FeedbackEntry(BaseModel):
    """User feedback entry."""

    service: str
    request_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ImprovementRequest(BaseModel):
    """Request for system improvement analysis."""

    service: str
    metrics: List[PerformanceMetric]
    feedback: List[FeedbackEntry]
    time_period_days: int = Field(default=7, ge=1, le=90)


class Recommendation(BaseModel):
    """Improvement recommendation."""

    category: str
    priority: str  # high, medium, low
    description: str
    expected_impact: str
    implementation_effort: str


class ImprovementResponse(BaseModel):
    """Improvement analysis response."""

    request_id: str
    service: str
    current_performance_score: float
    trend: str  # improving, stable, declining
    recommendations: List[Recommendation]
    model_updates_suggested: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MetaLearner(nn.Module):
    """Simple meta-learner for performance prediction."""

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Global meta-learner
meta_learner: MetaLearner = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global meta_learner

    logger.info("Starting Self-Improvement Service")
    logger.info(f"Device: {device}")

    # Initialize meta-learner
    meta_learner = MetaLearner().to(device)

    yield

    logger.info("Shutting down Self-Improvement Service")


# Create FastAPI app
app = FastAPI(
    title="Self-Improvement Service",
    description="Meta-learning for continuous system improvement",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS and metrics
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Self-Improvement Service",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/analyze", response_model=ImprovementResponse)
async def analyze_improvement(request: ImprovementRequest):
    """Analyze system performance and suggest improvements."""
    import uuid
    import numpy as np

    try:
        request_id = str(uuid.uuid4())

        # Calculate current performance score
        if request.metrics:
            metric_values = [m.value for m in request.metrics]
            current_score = np.mean(metric_values)
        else:
            current_score = 0.0

        # Analyze trend
        if len(request.metrics) > 1:
            recent_avg = np.mean([m.value for m in request.metrics[-3:]])
            older_avg = np.mean([m.value for m in request.metrics[:3]])
            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Generate recommendations based on feedback
        recommendations = []

        # Analyze negative feedback
        negative_feedback = [f for f in request.feedback if f.rating <= 2]
        if len(negative_feedback) > len(request.feedback) * 0.2:
            recommendations.append(
                Recommendation(
                    category="quality",
                    priority="high",
                    description="High volume of negative feedback detected",
                    expected_impact="Improve user satisfaction by 30%",
                    implementation_effort="medium",
                )
            )

        # Performance recommendations
        if current_score < 0.7:
            recommendations.append(
                Recommendation(
                    category="performance",
                    priority="high",
                    description="Performance metrics below threshold",
                    expected_impact="Reduce latency by 40%",
                    implementation_effort="high",
                )
            )

        # Model update suggestions
        model_updates = {
            "retrain_frequency": "weekly" if trend == "declining" else "monthly",
            "learning_rate_adjustment": 0.001 if trend == "improving" else 0.0005,
            "batch_size_recommendation": 32,
        }

        # Add general recommendations
        if not recommendations:
            recommendations.append(
                Recommendation(
                    category="optimization",
                    priority="medium",
                    description="Continue current optimization strategy",
                    expected_impact="Maintain stable performance",
                    implementation_effort="low",
                )
            )

        return ImprovementResponse(
            request_id=request_id,
            service=request.service,
            current_performance_score=current_score,
            trend=trend,
            recommendations=recommendations,
            model_updates_suggested=model_updates,
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackEntry):
    """Submit user feedback for meta-learning."""
    logger.info(f"Received feedback for {feedback.service}: {feedback.rating}/5")
    return {"status": "feedback_received", "feedback_id": str(uuid.uuid4())}


if __name__ == "__main__":
    import uuid
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)
