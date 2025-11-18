"""ML-based test generation engine."""

import ast
import re
import time
import uuid
from typing import List, Optional

from loguru import logger
from openai import AsyncOpenAI

from ...common.config import get_settings
from .models import (
    TestGenerationRequest,
    TestGenerationResponse,
    TestSuite,
    TestCase,
    TestType,
    TestFramework,
)


class TestGenerator:
    """ML-based test generator."""

    def __init__(self):
        """Initialize test generator."""
        self.settings = get_settings()
        self.openai_client = None

        if self.settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

        logger.info("Test Generator initialized")

    async def generate_tests(self, request: TestGenerationRequest) -> TestGenerationResponse:
        """
        Generate comprehensive test cases for given code.

        Args:
            request: Test generation request

        Returns:
            Generated test suites and cases
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"Generating tests for request {request_id}")

        try:
            # Analyze code structure
            code_analysis = self._analyze_code(request.code, request.language)

            # Generate test suites for each test type
            test_suites = []
            for test_type in request.test_types:
                suite = await self._generate_test_suite(
                    request.code,
                    test_type,
                    request.framework,
                    code_analysis,
                    request.coverage_level.value,
                )
                test_suites.append(suite)

            # Generate fixtures if requested
            fixtures = None
            if request.include_fixtures:
                fixtures = await self._generate_fixtures(request.code, request.framework)

            # Generate mocks if requested
            mocks = None
            if request.include_mocks:
                mocks = await self._generate_mocks(code_analysis, request.framework)

            # Calculate total tests and coverage estimate
            total_tests = sum(len(suite.test_cases) for suite in test_suites)
            estimated_coverage = self._estimate_coverage(
                code_analysis, total_tests, request.coverage_level.value
            )

            generation_time = time.time() - start_time

            response = TestGenerationResponse(
                request_id=request_id,
                test_suites=test_suites,
                framework=request.framework,
                total_tests=total_tests,
                estimated_coverage=estimated_coverage,
                fixtures=fixtures,
                mocks=mocks,
                dependencies=self._extract_test_dependencies(request.framework),
                generation_time=generation_time,
            )

            logger.info(
                f"Generated {total_tests} tests in {generation_time:.2f}s "
                f"(estimated coverage: {estimated_coverage:.1f}%)"
            )

            return response

        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            raise

    def _analyze_code(self, code: str, language: str) -> dict:
        """Analyze code structure to identify testable components."""
        analysis = {
            "functions": [],
            "classes": [],
            "methods": [],
            "imports": [],
            "complexity": 0,
        }

        if language == "python":
            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        analysis["functions"].append(
                            {
                                "name": node.name,
                                "args": [arg.arg for arg in node.args.args],
                                "lineno": node.lineno,
                            }
                        )
                    elif isinstance(node, ast.ClassDef):
                        analysis["classes"].append({"name": node.name, "lineno": node.lineno})
                        # Extract methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                analysis["methods"].append(
                                    {
                                        "class": node.name,
                                        "name": item.name,
                                        "args": [arg.arg for arg in item.args.args],
                                    }
                                )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis["imports"].append(node.module)

            except SyntaxError as e:
                logger.warning(f"Code parsing error: {e}")

        return analysis

    async def _generate_test_suite(
        self,
        code: str,
        test_type: TestType,
        framework: TestFramework,
        code_analysis: dict,
        coverage_level: str,
    ) -> TestSuite:
        """Generate a test suite for a specific test type."""
        if self.openai_client:
            prompt = self._build_test_generation_prompt(
                code, test_type, framework, code_analysis, coverage_level
            )

            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert at writing {framework.value} tests. "
                        "Generate comprehensive, well-structured test cases.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=3000,
            )

            test_code = response.choices[0].message.content
            test_cases = self._parse_test_cases(test_code, test_type, framework)

        else:
            # Fallback: Generate basic test structure
            test_cases = self._generate_basic_tests(code_analysis, test_type, framework)

        return TestSuite(
            name=f"{test_type.value}_tests",
            test_cases=test_cases,
            setup_code=self._generate_setup_code(framework),
            teardown_code=self._generate_teardown_code(framework),
        )

    def _build_test_generation_prompt(
        self,
        code: str,
        test_type: TestType,
        framework: TestFramework,
        code_analysis: dict,
        coverage_level: str,
    ) -> str:
        """Build prompt for test generation."""
        components = []

        if code_analysis["functions"]:
            func_list = ", ".join([f["name"] for f in code_analysis["functions"]])
            components.append(f"Functions: {func_list}")

        if code_analysis["classes"]:
            class_list = ", ".join([c["name"] for c in code_analysis["classes"]])
            components.append(f"Classes: {class_list}")

        prompt = f"""Generate {coverage_level} {test_type.value} tests using {framework.value} for this code:

{code}

Code components found:
{chr(10).join(components)}

Requirements:
1. Generate comprehensive {test_type.value} tests
2. Include happy path, edge cases, and error scenarios
3. Use appropriate assertions
4. Add descriptive test names
5. Include setup and teardown if needed
6. Test all public functions/methods
7. Add comments explaining what each test validates

Return only the test code, properly formatted for {framework.value}.
"""
        return prompt

    def _parse_test_cases(
        self, test_code: str, test_type: TestType, framework: TestFramework
    ) -> List[TestCase]:
        """Parse generated test code into individual test cases."""
        test_cases = []

        # Extract test functions based on framework
        if framework == TestFramework.PYTEST:
            pattern = r"def (test_\w+)\(.*?\):(.*?)(?=\ndef |$)"
        else:
            pattern = r"def (test\w+)\(.*?\):(.*?)(?=\ndef |$)"

        matches = re.finditer(pattern, test_code, re.DOTALL)

        for match in matches:
            test_name = match.group(1)
            test_body = match.group(2).strip()

            # Count assertions
            assertions = test_body.count("assert")

            test_cases.append(
                TestCase(
                    name=test_name,
                    test_type=test_type,
                    code=f"def {test_name}():\n{test_body}",
                    description=f"Test case for {test_name.replace('test_', '')}",
                    assertions=assertions,
                )
            )

        return test_cases

    def _generate_basic_tests(
        self, code_analysis: dict, test_type: TestType, framework: TestFramework
    ) -> List[TestCase]:
        """Generate basic test structure when API is not available."""
        test_cases = []

        # Generate a test for each function
        for func in code_analysis["functions"]:
            test_name = f"test_{func['name']}"
            test_code = f"""def {test_name}():
    '''Test {func['name']} function.'''
    # TODO: Implement test
    assert True  # Placeholder
"""
            test_cases.append(
                TestCase(
                    name=test_name,
                    test_type=test_type,
                    code=test_code,
                    description=f"Test for {func['name']}",
                    assertions=1,
                )
            )

        return test_cases

    async def _generate_fixtures(self, code: str, framework: TestFramework) -> str:
        """Generate test fixtures."""
        if self.openai_client and framework == TestFramework.PYTEST:
            prompt = f"""Generate pytest fixtures for this code:

{code}

Include fixtures for:
- Common test data
- Mock objects
- Database connections (if applicable)
- API clients (if applicable)

Return only the fixture code.
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1000,
            )

            return response.choices[0].message.content

        return "# Configure API key for fixture generation"

    async def _generate_mocks(self, code_analysis: dict, framework: TestFramework) -> str:
        """Generate mock objects."""
        if not code_analysis["imports"]:
            return None

        mock_code = "# Mock objects\n"
        mock_code += "from unittest.mock import Mock, patch\n\n"

        return mock_code

    def _generate_setup_code(self, framework: TestFramework) -> str:
        """Generate setup code for test suite."""
        if framework == TestFramework.PYTEST:
            return """@pytest.fixture(autouse=True)
def setup():
    '''Setup for each test.'''
    # Setup code here
    yield
    # Teardown code here
"""
        return ""

    def _generate_teardown_code(self, framework: TestFramework) -> str:
        """Generate teardown code."""
        return ""  # Usually handled in setup fixture

    def _estimate_coverage(
        self, code_analysis: dict, total_tests: int, coverage_level: str
    ) -> float:
        """Estimate test coverage percentage."""
        # Simple heuristic based on number of tests vs code components
        total_components = (
            len(code_analysis["functions"])
            + len(code_analysis["classes"])
            + len(code_analysis["methods"])
        )

        if total_components == 0:
            return 0.0

        base_coverage = min((total_tests / total_components) * 40, 80)

        # Adjust based on coverage level
        multiplier = {"basic": 0.7, "standard": 1.0, "comprehensive": 1.3}.get(
            coverage_level, 1.0
        )

        return min(base_coverage * multiplier, 95.0)

    def _extract_test_dependencies(self, framework: TestFramework) -> List[str]:
        """Extract test framework dependencies."""
        deps = {
            TestFramework.PYTEST: ["pytest", "pytest-cov", "pytest-mock"],
            TestFramework.UNITTEST: [],  # Built-in
            TestFramework.JEST: ["jest", "@types/jest"],
            TestFramework.MOCHA: ["mocha", "chai"],
        }

        return deps.get(framework, [])
