"""Tests for Project Management Service."""

import pytest
from datetime import datetime, timedelta
from src.services.project_management.models import (
    Task,
    TeamMember,
    Priority,
    TaskStatus,
)


@pytest.mark.unit
def test_task_creation():
    """Test task model creation."""
    task = Task(
        id="task_1",
        title="Implement feature",
        description="Add new feature to the system",
        estimated_hours=8.0,
        priority=Priority.HIGH,
        required_skills=["python", "fastapi"],
    )

    assert task.id == "task_1"
    assert task.priority == Priority.HIGH
    assert len(task.required_skills) == 2
    assert task.status == TaskStatus.TODO


@pytest.mark.unit
def test_team_member_creation():
    """Test team member model."""
    member = TeamMember(
        id="dev_1",
        name="Alice",
        skills=["python", "docker", "kubernetes"],
        skill_levels={"python": 9, "docker": 7, "kubernetes": 6},
        availability=40.0,
    )

    assert member.id == "dev_1"
    assert len(member.skills) == 3
    assert member.skill_levels["python"] == 9
    assert member.performance_score == 5.0  # Default


@pytest.mark.unit
def test_task_with_dependencies():
    """Test task with dependencies."""
    task = Task(
        id="task_2",
        title="Deploy feature",
        description="Deploy the new feature",
        estimated_hours=4.0,
        dependencies=["task_1"],
    )

    assert "task_1" in task.dependencies
