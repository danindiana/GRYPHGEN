"""Collaboration Service API - GNN-based developer matching."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Developer(BaseModel):
    """Developer profile."""

    id: str
    name: str
    skills: List[str]
    interests: List[str]
    experience_years: int
    projects_completed: int
    availability: float = Field(ge=0.0, le=1.0)


class Project(BaseModel):
    """Project requirements."""

    id: str
    name: str
    required_skills: List[str]
    domain: str
    complexity: int = Field(ge=1, le=10)
    duration_weeks: int


class MatchRequest(BaseModel):
    """Matching request."""

    developers: List[Developer]
    project: Project
    team_size: int = Field(ge=1, le=20)


class DeveloperMatch(BaseModel):
    """Developer match result."""

    developer_id: str
    developer_name: str
    match_score: float = Field(ge=0.0, le=1.0)
    skill_match: float
    interest_match: float
    availability_score: float
    reasoning: str


class MatchResponse(BaseModel):
    """Matching response."""

    request_id: str
    project_id: str
    matches: List[DeveloperMatch]
    team_score: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for developer matching."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Global model
gnn_model: SimpleGNN = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global gnn_model

    logger.info("Starting Collaboration Service")
    logger.info(f"Device: {device}")

    # Initialize GNN
    gnn_model = SimpleGNN(input_dim=10).to(device)

    yield

    logger.info("Shutting down Collaboration Service")


# Create FastAPI app
app = FastAPI(
    title="Collaboration Service",
    description="GNN-based developer-project matching",
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
    return {"service": "Collaboration Service", "version": "0.1.0", "status": "operational"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/match", response_model=MatchResponse)
async def match_developers(request: MatchRequest):
    """Match developers to project using GNN."""
    import uuid
    import numpy as np

    try:
        request_id = str(uuid.uuid4())
        matches = []

        for dev in request.developers:
            # Calculate skill match
            skill_match = len(
                set(dev.skills) & set(request.project.required_skills)
            ) / len(request.project.required_skills)

            # Calculate interest alignment (simple heuristic)
            interest_match = 0.5  # Placeholder

            # Extract features
            features = [
                skill_match,
                dev.experience_years / 20.0,
                dev.projects_completed / 100.0,
                dev.availability,
                request.project.complexity / 10.0,
                request.project.duration_weeks / 52.0,
                interest_match,
                len(dev.skills) / 20.0,
                0.0,  # Placeholder
                0.0,  # Placeholder
            ]

            # Get GNN score
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                match_score = gnn_model(features_tensor).item()

            matches.append(
                DeveloperMatch(
                    developer_id=dev.id,
                    developer_name=dev.name,
                    match_score=match_score,
                    skill_match=skill_match,
                    interest_match=interest_match,
                    availability_score=dev.availability,
                    reasoning=f"Skills: {skill_match:.1%}, Experience: {dev.experience_years}y, Available: {dev.availability:.1%}",
                )
            )

        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        # Select top matches for team
        team_matches = matches[: request.team_size]
        team_score = np.mean([m.match_score for m in team_matches]) if team_matches else 0.0

        return MatchResponse(
            request_id=request_id,
            project_id=request.project.id,
            matches=team_matches,
            team_score=team_score,
        )

    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)
