"""RL-based project management optimizer."""

import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ...common.config import get_settings
from .models import (
    OptimizationRequest,
    OptimizationResponse,
    TaskAssignment,
    ProjectPhase,
    OptimizationMetrics,
    Task,
    TeamMember,
)


class TaskAssignmentNetwork(nn.Module):
    """Neural network for task-assignee matching."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """Initialize network."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class ProjectOptimizer:
    """RL-based project management optimizer."""

    def __init__(self):
        """Initialize optimizer."""
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() and self.settings.use_gpu else "cpu"

        # Initialize assignment network
        self.assignment_network = TaskAssignmentNetwork(input_dim=20).to(self.device)

        logger.info(f"Project Optimizer initialized with device: {self.device}")

    async def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        """
        Optimize task assignments using RL.

        Args:
            request: Optimization request

        Returns:
            Optimized task assignments and timeline
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"Optimizing project {request.project_id} ({len(request.tasks)} tasks)")

        try:
            # Sort tasks by priority and dependencies
            sorted_tasks = self._sort_tasks(request.tasks)

            # Generate task assignments
            assignments = self._assign_tasks(sorted_tasks, request.team_members)

            # Build project timeline
            timeline = self._build_timeline(assignments, sorted_tasks)

            # Calculate metrics
            metrics = self._calculate_metrics(
                assignments, request.tasks, request.team_members, timeline
            )

            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(assignments, request.team_members)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                assignments, request.tasks, request.team_members, metrics
            )

            # Calculate resource utilization
            utilization = self._calculate_utilization(assignments, request.team_members)

            optimization_time = time.time() - start_time

            response = OptimizationResponse(
                request_id=request_id,
                project_id=request.project_id,
                assignments=assignments,
                timeline=timeline,
                metrics=metrics,
                total_duration_days=timeline[-1].end_date.timestamp()
                - timeline[0].start_date.timestamp()
                if timeline
                else 0,
                resource_utilization=utilization,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                optimization_time=optimization_time,
            )

            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            raise

    def _sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by priority and dependencies."""
        # Simple topological sort considering dependencies
        sorted_tasks = []
        remaining = tasks.copy()
        completed_ids = set()

        priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        while remaining:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = [
                t for t in remaining if all(dep in completed_ids for dep in t.dependencies)
            ]

            if not ready_tasks:
                # If circular dependency, just take highest priority
                ready_tasks = [max(remaining, key=lambda t: priority_weights[t.priority.value])]

            # Sort ready tasks by priority
            ready_tasks.sort(
                key=lambda t: (priority_weights[t.priority.value], -t.estimated_hours),
                reverse=True,
            )

            # Add first ready task
            task = ready_tasks[0]
            sorted_tasks.append(task)
            completed_ids.add(task.id)
            remaining.remove(task)

        return sorted_tasks

    def _assign_tasks(
        self, tasks: List[Task], team_members: List[TeamMember]
    ) -> List[TaskAssignment]:
        """Assign tasks to team members using RL."""
        assignments = []
        current_time = datetime.utcnow()

        # Track member workloads
        member_schedules = {m.id: [] for m in team_members}

        for task in tasks:
            # Find best assignee
            best_assignee, confidence, reasoning = self._find_best_assignee(
                task, team_members, member_schedules
            )

            if best_assignee:
                # Calculate start and end times
                assignee = next(m for m in team_members if m.id == best_assignee)

                # Find earliest start time based on dependencies and availability
                earliest_start = self._calculate_earliest_start(
                    task, assignments, member_schedules[best_assignee], current_time
                )

                # Calculate duration based on skill match and workload
                duration_hours = task.estimated_hours
                skill_match = self._calculate_skill_match(task, assignee)
                if skill_match < 0.8:
                    duration_hours *= 1.2  # Add buffer for skill mismatch

                duration_days = duration_hours / 8  # Assuming 8-hour workdays
                estimated_end = earliest_start + timedelta(days=duration_days)

                assignment = TaskAssignment(
                    task_id=task.id,
                    assignee_id=best_assignee,
                    assignee_name=assignee.name,
                    estimated_start=earliest_start,
                    estimated_end=estimated_end,
                    confidence=confidence,
                    reasoning=reasoning,
                )

                assignments.append(assignment)

                # Update schedule
                member_schedules[best_assignee].append(
                    {"task_id": task.id, "start": earliest_start, "end": estimated_end}
                )

        return assignments

    def _find_best_assignee(
        self,
        task: Task,
        team_members: List[TeamMember],
        schedules: Dict,
    ) -> Tuple[str, float, str]:
        """Find best team member for a task using RL network."""
        scores = []

        for member in team_members:
            # Calculate features for network
            features = self._extract_features(task, member, schedules.get(member.id, []))

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Get score from network
            with torch.no_grad():
                score = self.assignment_network(features_tensor).item()

            scores.append((member.id, member.name, score))

        # Sort by score
        scores.sort(key=lambda x: x[2], reverse=True)

        if scores:
            best_id, best_name, best_score = scores[0]

            # Build reasoning
            member = next(m for m in team_members if m.id == best_id)
            skill_match = self._calculate_skill_match(task, member)
            workload = member.current_workload / member.availability

            reasoning = (
                f"Selected {best_name} based on: "
                f"skill match ({skill_match:.1%}), "
                f"current workload ({workload:.1%}), "
                f"and overall optimization score ({best_score:.1%})"
            )

            return best_id, best_score, reasoning

        return None, 0.0, "No suitable assignee found"

    def _extract_features(
        self, task: Task, member: TeamMember, schedule: List
    ) -> List[float]:
        """Extract features for RL network."""
        features = [0.0] * 20  # Fixed size feature vector

        # Skill match features (0-4)
        skill_match = self._calculate_skill_match(task, member)
        features[0] = skill_match

        # Calculate avg skill level for required skills
        skill_levels = [
            member.skill_levels.get(skill, 0) / 10.0 for skill in task.required_skills
        ]
        features[1] = np.mean(skill_levels) if skill_levels else 0.0

        # Workload features (5-7)
        features[5] = member.current_workload / max(member.availability, 1)
        features[6] = len(schedule) / 10.0  # Normalize scheduled tasks

        # Task features (8-11)
        features[8] = task.estimated_hours / 100.0  # Normalize hours
        features[9] = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}[
            task.priority.value
        ]

        # Member features (12-14)
        features[12] = member.performance_score / 10.0
        features[13] = member.availability / 40.0  # Normalize to full-time

        # Preference match (15)
        features[15] = 1.0 if any(pref in task.title for pref in member.preferences) else 0.0

        return features

    def _calculate_skill_match(self, task: Task, member: TeamMember) -> float:
        """Calculate how well member skills match task requirements."""
        if not task.required_skills:
            return 1.0

        matched_skills = sum(1 for skill in task.required_skills if skill in member.skills)
        return matched_skills / len(task.required_skills)

    def _calculate_earliest_start(
        self,
        task: Task,
        assignments: List[TaskAssignment],
        member_schedule: List,
        current_time: datetime,
    ) -> datetime:
        """Calculate earliest possible start time for a task."""
        earliest = current_time

        # Check dependencies
        for dep_id in task.dependencies:
            dep_assignment = next((a for a in assignments if a.task_id == dep_id), None)
            if dep_assignment and dep_assignment.estimated_end > earliest:
                earliest = dep_assignment.estimated_end

        # Check member availability
        if member_schedule:
            last_task = max(member_schedule, key=lambda x: x["end"])
            if last_task["end"] > earliest:
                earliest = last_task["end"]

        return earliest

    def _build_timeline(
        self, assignments: List[TaskAssignment], tasks: List[Task]
    ) -> List[ProjectPhase]:
        """Build project timeline with phases."""
        if not assignments:
            return []

        # Group tasks into phases based on timing
        phases = []
        sorted_assignments = sorted(assignments, key=lambda a: a.estimated_start)

        current_phase_tasks = []
        phase_start = sorted_assignments[0].estimated_start
        phase_num = 1

        for i, assignment in enumerate(sorted_assignments):
            if not current_phase_tasks or (
                assignment.estimated_start - phase_start
            ).days < 7:
                current_phase_tasks.append(assignment)
            else:
                # Create phase
                phase_end = max(t.estimated_end for t in current_phase_tasks)
                duration = (phase_end - phase_start).days

                task_ids = [t.task_id for t in current_phase_tasks]
                task_titles = [
                    next(task.title for task in tasks if task.id == tid) for tid in task_ids
                ]

                phases.append(
                    ProjectPhase(
                        phase_number=phase_num,
                        name=f"Phase {phase_num}",
                        description=f"Tasks: {', '.join(task_titles[:3])}...",
                        task_ids=task_ids,
                        start_date=phase_start,
                        end_date=phase_end,
                        duration_days=duration,
                        parallel_tasks=len(current_phase_tasks),
                    )
                )

                # Start new phase
                current_phase_tasks = [assignment]
                phase_start = assignment.estimated_start
                phase_num += 1

        # Add final phase
        if current_phase_tasks:
            phase_end = max(t.estimated_end for t in current_phase_tasks)
            duration = (phase_end - phase_start).days

            phases.append(
                ProjectPhase(
                    phase_number=phase_num,
                    name=f"Phase {phase_num}",
                    description=f"{len(current_phase_tasks)} tasks",
                    task_ids=[t.task_id for t in current_phase_tasks],
                    start_date=phase_start,
                    end_date=phase_end,
                    duration_days=duration,
                    parallel_tasks=len(current_phase_tasks),
                )
            )

        return phases

    def _calculate_metrics(
        self,
        assignments: List[TaskAssignment],
        tasks: List[Task],
        team_members: List[TeamMember],
        timeline: List[ProjectPhase],
    ) -> OptimizationMetrics:
        """Calculate optimization quality metrics."""
        # Efficiency: based on parallelization and utilization
        avg_parallel = np.mean([p.parallel_tasks for p in timeline]) if timeline else 1
        efficiency = min(avg_parallel / len(team_members) * 100, 100)

        # Load balance: variance in workload across team members
        workloads = [
            len([a for a in assignments if a.assignee_id == m.id]) for m in team_members
        ]
        load_variance = np.var(workloads) if workloads else 0
        load_balance = max(100 - load_variance * 10, 0)

        # Skill match: average skill match across assignments
        skill_matches = []
        for assignment in assignments:
            task = next(t for t in tasks if t.id == assignment.task_id)
            member = next(m for m in team_members if m.id == assignment.assignee_id)
            skill_matches.append(self._calculate_skill_match(task, member))

        skill_match = np.mean(skill_matches) * 100 if skill_matches else 0

        # Deadline compliance
        tasks_with_deadlines = [t for t in tasks if t.deadline]
        if tasks_with_deadlines:
            on_time = sum(
                1
                for task in tasks_with_deadlines
                if next(a.estimated_end for a in assignments if a.task_id == task.id)
                <= task.deadline
            )
            deadline_compliance = (on_time / len(tasks_with_deadlines)) * 100
        else:
            deadline_compliance = 100.0

        # Utilization
        total_hours = sum(t.estimated_hours for t in tasks)
        total_capacity = sum(m.availability for m in team_members)
        timeline_days = (timeline[-1].end_date - timeline[0].start_date).days if timeline else 1
        available_capacity = total_capacity * (timeline_days / 7)  # Convert to days
        utilization = min((total_hours / available_capacity) * 100, 100) if available_capacity > 0 else 0

        return OptimizationMetrics(
            efficiency_score=efficiency,
            load_balance_score=load_balance,
            skill_match_score=skill_match,
            deadline_compliance=deadline_compliance,
            utilization_rate=utilization,
        )

    def _identify_bottlenecks(
        self, assignments: List[TaskAssignment], team_members: List[TeamMember]
    ) -> List[str]:
        """Identify potential bottlenecks."""
        bottlenecks = []

        # Check for overloaded team members
        for member in team_members:
            member_tasks = [a for a in assignments if a.assignee_id == member.id]
            if len(member_tasks) > len(assignments) * 0.3:  # More than 30% of tasks
                bottlenecks.append(
                    f"{member.name} assigned {len(member_tasks)} tasks - consider redistribution"
                )

        return bottlenecks

    def _generate_recommendations(
        self,
        assignments: List[TaskAssignment],
        tasks: List[Task],
        team_members: List[TeamMember],
        metrics: OptimizationMetrics,
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if metrics.load_balance_score < 70:
            recommendations.append("Consider redistributing tasks for better load balance")

        if metrics.skill_match_score < 75:
            recommendations.append("Some assignments have skill mismatches - consider training or reassignment")

        if metrics.utilization_rate > 90:
            recommendations.append("Team is near full capacity - consider additional resources")

        if metrics.deadline_compliance < 95:
            recommendations.append("Some tasks may miss deadlines - review critical path")

        return recommendations

    def _calculate_utilization(
        self, assignments: List[TaskAssignment], team_members: List[TeamMember]
    ) -> Dict[str, float]:
        """Calculate resource utilization per team member."""
        utilization = {}

        for member in team_members:
            member_assignments = [a for a in assignments if a.assignee_id == member.id]
            total_hours = sum(
                (a.estimated_end - a.estimated_start).total_seconds() / 3600
                for a in member_assignments
            )
            utilization[member.name] = min((total_hours / member.availability) * 100, 100)

        return utilization
