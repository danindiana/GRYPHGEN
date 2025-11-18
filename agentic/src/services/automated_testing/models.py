"""Data models for Automated Testing Service."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class TestFramework(str, Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    RSPEC = "rspec"


class TestType(str, Enum):
    """Types of tests."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"


class TestCoverage(str, Enum):
    """Test coverage levels."""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class TestGenerationRequest(BaseModel):
    """Request model for test generation."""

    code: str = Field(..., description="Code to generate tests for", min_length=10)
    language: str = Field(..., description="Programming language")
    framework: TestFramework = Field(default=TestFramework.PYTEST, description="Test framework")
    test_types: List[TestType] = Field(
        default=[TestType.UNIT], description="Types of tests to generate"
    )
    coverage_level: TestCoverage = Field(
        default=TestCoverage.STANDARD, description="Desired test coverage level"
    )
    include_mocks: bool = Field(default=True, description="Include mock objects")
    include_fixtures: bool = Field(default=True, description="Include test fixtures")
    edge_cases: bool = Field(default=True, description="Generate edge case tests")


class TestCase(BaseModel):
    """Individual test case."""

    name: str = Field(..., description="Test case name")
    test_type: TestType
    code: str = Field(..., description="Test code")
    description: str = Field(..., description="What the test validates")
    assertions: int = Field(default=1, description="Number of assertions")
    dependencies: List[str] = Field(default_factory=list)


class TestSuite(BaseModel):
    """Collection of test cases."""

    name: str = Field(..., description="Test suite name")
    test_cases: List[TestCase] = Field(..., description="Test cases in suite")
    setup_code: Optional[str] = Field(None, description="Suite setup code")
    teardown_code: Optional[str] = Field(None, description="Suite teardown code")


class TestGenerationResponse(BaseModel):
    """Response model for test generation."""

    request_id: str
    test_suites: List[TestSuite]
    framework: TestFramework
    total_tests: int = Field(..., description="Total number of tests generated")
    estimated_coverage: float = Field(..., ge=0.0, le=100.0, description="Estimated coverage %")
    fixtures: Optional[str] = Field(None, description="Test fixtures code")
    mocks: Optional[str] = Field(None, description="Mock objects code")
    dependencies: List[str] = Field(default_factory=list)
    generation_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TestExecutionRequest(BaseModel):
    """Request to execute tests."""

    test_code: str = Field(..., description="Test code to execute")
    code: str = Field(..., description="Code being tested")
    framework: TestFramework
    timeout: int = Field(default=30, ge=1, le=300, description="Execution timeout in seconds")


class TestResult(BaseModel):
    """Individual test result."""

    test_name: str
    passed: bool
    duration: float = Field(..., description="Execution time in seconds")
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0


class TestExecutionResponse(BaseModel):
    """Response from test execution."""

    request_id: str
    total_tests: int
    passed: int
    failed: int
    skipped: int = 0
    duration: float
    coverage_percentage: Optional[float] = None
    test_results: List[TestResult]
    exit_code: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
