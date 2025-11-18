#!/usr/bin/env python3
"""
Example: Using the Project Management Service

This script demonstrates how to use the GRYPHGEN Agentic project management
service to optimize task assignments and project planning.
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

import httpx


async def optimize_tasks(
    project_id: str,
    team_members: List[str],
    tasks: List[Dict[str, Any]],
    api_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Optimize task assignments using reinforcement learning.

    Args:
        project_id: Unique identifier for the project
        team_members: List of team member IDs
        tasks: List of task dictionaries
        api_url: Base URL of the API gateway

    Returns:
        Dictionary containing optimized task assignments
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_url}/api/v1/project/optimize",
            json={
                "project_id": project_id,
                "team_members": team_members,
                "tasks": tasks,
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Main example function."""

    print("=" * 80)
    print("GRYPHGEN Agentic - Project Management Example")
    print("=" * 80)
    print()

    # Define team members
    team_members = ["alice", "bob", "charlie", "diana"]

    # Define tasks
    tasks = [
        {
            "id": "task_1",
            "title": "Implement authentication system",
            "description": "Build JWT-based authentication",
            "estimated_hours": 16,
            "priority": "high",
            "required_skills": ["python", "security", "jwt"],
            "deadline": (datetime.now() + timedelta(days=5)).isoformat(),
        },
        {
            "id": "task_2",
            "title": "Create database schema",
            "description": "Design and implement PostgreSQL schema",
            "estimated_hours": 8,
            "priority": "high",
            "required_skills": ["sql", "postgresql", "database_design"],
            "deadline": (datetime.now() + timedelta(days=3)).isoformat(),
        },
        {
            "id": "task_3",
            "title": "Build REST API endpoints",
            "description": "Implement CRUD endpoints for users",
            "estimated_hours": 12,
            "priority": "medium",
            "required_skills": ["python", "fastapi", "rest_api"],
            "deadline": (datetime.now() + timedelta(days=7)).isoformat(),
        },
        {
            "id": "task_4",
            "title": "Write unit tests",
            "description": "Create comprehensive test suite",
            "estimated_hours": 10,
            "priority": "medium",
            "required_skills": ["python", "pytest", "testing"],
            "deadline": (datetime.now() + timedelta(days=10)).isoformat(),
        },
        {
            "id": "task_5",
            "title": "Setup CI/CD pipeline",
            "description": "Configure GitHub Actions for testing and deployment",
            "estimated_hours": 6,
            "priority": "low",
            "required_skills": ["devops", "github_actions", "docker"],
            "deadline": (datetime.now() + timedelta(days=14)).isoformat(),
        },
    ]

    print("Optimizing task assignments...")
    print("-" * 80)
    print(f"Team members: {', '.join(team_members)}")
    print(f"Number of tasks: {len(tasks)}")
    print()

    # Optimize task assignments
    result = await optimize_tasks(
        project_id="proj_example_001",
        team_members=team_members,
        tasks=tasks
    )

    print("Optimized Assignments:")
    print("-" * 80)

    assignments = result.get("assignments", {})
    for task_id, assignee in assignments.items():
        task = next((t for t in tasks if t["id"] == task_id), None)
        if task:
            print(f"Task: {task['title']}")
            print(f"  Assigned to: {assignee}")
            print(f"  Priority: {task['priority']}")
            print(f"  Estimated hours: {task['estimated_hours']}")
            print(f"  Required skills: {', '.join(task['required_skills'])}")
            print()

    # Display optimization metrics
    print("Optimization Metrics:")
    print("-" * 80)
    metrics = result.get("metrics", {})
    print(f"Total efficiency score: {metrics.get('efficiency_score', 'N/A')}")
    print(f"Load balance score: {metrics.get('load_balance', 'N/A')}")
    print(f"Skill match score: {metrics.get('skill_match', 'N/A')}")
    print()

    # Display timeline
    print("Projected Timeline:")
    print("-" * 80)
    timeline = result.get("timeline", [])
    for phase in timeline:
        print(f"Phase {phase.get('phase_number')}: {phase.get('description')}")
        print(f"  Duration: {phase.get('duration_days')} days")
        print(f"  Tasks: {', '.join(phase.get('task_ids', []))}")
        print()

    print("=" * 80)
    print("Project optimization completed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
