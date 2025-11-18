"""Data models for Project Management Service."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task status."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Task model."""

    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed description")
    estimated_hours: float = Field(..., ge=0.1, le=1000, description="Estimated hours")
    priority: Priority = Field(default=Priority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.TODO)
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    deadline: Optional[datetime] = None
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamMember(BaseModel):
    """Team member model."""

    id: str
    name: str
    skills: List[str] = Field(..., description="Technical skills")
    skill_levels: Dict[str, int] = Field(
        default_factory=dict, description="Skill proficiency (1-10)"
    )
    availability: float = Field(default=40.0, ge=0, le=168, description="Hours per week")
    current_workload: float = Field(default=0.0, ge=0, description="Current assigned hours")
    preferences: List[str] = Field(default_factory=list, description="Task preferences")
    performance_score: float = Field(default=5.0, ge=0, le=10)


class OptimizationRequest(BaseModel):
    """Request for task optimization."""

    project_id: str
    tasks: List[Task]
    team_members: List[TeamMember]
    optimization_goal: str = Field(
        default="balanced",
        description="balanced, speed, quality, or cost",
    )
    constraints: Dict[str, Any] = Field(default_factory=dict)
    time_horizon_days: int = Field(default=30, ge=1, le=365)


class TaskAssignment(BaseModel):
    """Task assignment result."""

    task_id: str
    assignee_id: str
    assignee_name: str
    estimated_start: datetime
    estimated_end: datetime
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assignment confidence")
    reasoning: str = Field(..., description="Why this assignment was chosen")


class ProjectPhase(BaseModel):
    """Project phase in timeline."""

    phase_number: int
    name: str
    description: str
    task_ids: List[str]
    start_date: datetime
    end_date: datetime
    duration_days: float
    parallel_tasks: int = Field(..., description="Number of parallel tasks")


class OptimizationMetrics(BaseModel):
    """Optimization quality metrics."""

    efficiency_score: float = Field(..., ge=0.0, le=100.0)
    load_balance_score: float = Field(..., ge=0.0, le=100.0)
    skill_match_score: float = Field(..., ge=0.0, le=100.0)
    deadline_compliance: float = Field(..., ge=0.0, le=100.0)
    utilization_rate: float = Field(..., ge=0.0, le=100.0)


class OptimizationResponse(BaseModel):
    """Response from task optimization."""

    request_id: str
    project_id: str
    assignments: List[TaskAssignment]
    timeline: List[ProjectPhase]
    metrics: OptimizationMetrics
    total_duration_days: float
    resource_utilization: Dict[str, float]
    bottlenecks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    optimization_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectStatus(BaseModel):
    """Current project status."""

    project_id: str
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    blocked_tasks: int
    completion_percentage: float
    on_track: bool
    estimated_completion: datetime
    risks: List[str] = Field(default_factory=list)
