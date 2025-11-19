"""SQLAlchemy database models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
import enum

from .session import Base


class UserRole(str, enum.Enum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class TaskStatus(str, enum.Enum):
    """Task status."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"


class TaskPriority(str, enum.Enum):
    """Task priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Profile information
    skills = Column(JSON, default=list)  # List of skills
    preferences = Column(JSON, default=dict)  # User preferences

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    projects = relationship("Project", back_populates="owner")
    tasks = relationship("Task", back_populates="assigned_to_user")
    code_generations = relationship("CodeGeneration", back_populates="user")
    test_runs = relationship("TestRun", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class Project(Base):
    """Project model."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Project configuration
    language = Column(String(50), default="python")
    framework = Column(String(100))
    repository_url = Column(String(500))

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)

    # Metadata
    metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    owner = relationship("User", back_populates="projects")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    code_generations = relationship("CodeGeneration", back_populates="project")
    test_runs = relationship("TestRun", back_populates="project")
    documentations = relationship("Documentation", back_populates="project")

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}')>"


class Task(Base):
    """Task model."""

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)

    # Assignment
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Status and priority
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.TODO, nullable=False)
    priority = Column(SQLEnum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False)

    # Estimation
    estimated_hours = Column(Float, default=0.0)
    actual_hours = Column(Float, default=0.0)

    # Dependencies
    dependencies = Column(JSON, default=list)  # List of task IDs
    tags = Column(JSON, default=list)  # List of tags

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="tasks")
    assigned_to_user = relationship("User", back_populates="tasks")

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title='{self.title}', status='{self.status}')>"


class CodeGeneration(Base):
    """Code generation request/response model."""

    __tablename__ = "code_generations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Request
    prompt = Column(Text, nullable=False)
    language = Column(String(50), nullable=False)
    framework = Column(String(100), nullable=True)

    # Response
    generated_code = Column(Text)
    generated_tests = Column(Text, nullable=True)
    generated_docs = Column(Text, nullable=True)

    # Metadata
    model_used = Column(String(100))
    tokens_used = Column(Integer, default=0)
    generation_time = Column(Float, default=0.0)  # seconds

    # Status
    status = Column(String(50), default="completed")  # pending, completed, failed
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="code_generations")
    project = relationship("Project", back_populates="code_generations")

    def __repr__(self) -> str:
        return f"<CodeGeneration(id={self.id}, language='{self.language}', status='{self.status}')>"


class TestRun(Base):
    """Test run model."""

    __tablename__ = "test_runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Test configuration
    test_code = Column(Text, nullable=False)
    source_code = Column(Text, nullable=False)
    framework = Column(String(100), default="pytest")

    # Results
    total_tests = Column(Integer, default=0)
    passed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    skipped = Column(Integer, default=0)
    coverage = Column(Float, default=0.0)  # percentage
    duration = Column(Float, default=0.0)  # seconds

    # Detailed results
    results = Column(JSON, default=list)  # List of test results

    # Status
    success = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="test_runs")
    project = relationship("Project", back_populates="test_runs")

    def __repr__(self) -> str:
        return f"<TestRun(id={self.id}, passed={self.passed}/{self.total_tests}, coverage={self.coverage}%)>"


class Documentation(Base):
    """Documentation model."""

    __tablename__ = "documentations"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # Content
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String(50), default="markdown")  # markdown, rst, html

    # Metadata
    version = Column(String(50), default="1.0.0")
    language = Column(String(50), default="python")
    auto_generated = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="documentations")

    def __repr__(self) -> str:
        return f"<Documentation(id={self.id}, title='{self.title}', version='{self.version}')>"


class Feedback(Base):
    """User feedback model for self-improvement."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Feedback details
    service = Column(String(100), nullable=False)  # code_generation, testing, etc.
    request_id = Column(String(255), nullable=False)  # Reference to original request
    feedback_type = Column(String(50), nullable=False)  # positive, negative, neutral
    rating = Column(Float, nullable=False)  # 1.0 to 5.0
    comments = Column(Text, nullable=True)

    # Improvements suggested
    improvements = Column(JSON, default=list)

    # Metadata
    metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="feedback")

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, service='{self.service}', rating={self.rating})>"


class APIKey(Base):
    """API Key model for authentication."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Key details
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Permissions
    scopes = Column(JSON, default=list)  # List of allowed scopes

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"
