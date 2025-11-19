# Contributing to GRYPHGEN

Thank you for your interest in contributing to GRYPHGEN! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GRYPHGEN.git
   cd GRYPHGEN/gstruct
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/danindiana/GRYPHGEN.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   make install-dev
   ```

3. **Verify installation**:
   ```bash
   make test
   ```

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   make test
   make lint
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples**:
```
feat: add GPU memory optimization for RTX 4080
fix: resolve race condition in scheduler
docs: update installation instructions
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://github.com/psf/black) for code formatting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Add type hints where possible
- Write docstrings for all public functions and classes

**Format your code**:
```bash
make format
```

**Run linters**:
```bash
make lint
```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

**Example Docstring**:
```python
def allocate_resources(task_id: str, requirements: Dict[str, float]) -> Dict[str, Any]:
    """Allocate resources for a task.

    Args:
        task_id: Unique task identifier
        requirements: Dictionary of resource requirements

    Returns:
        Allocation result dictionary with success status and details

    Raises:
        ValueError: If requirements are invalid
        ResourceError: If resources unavailable

    Example:
        >>> result = allocate_resources("task1", {"cpu": 2.0, "memory": 4096})
        >>> print(result["success"])
        True
    """
```

## Testing

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest for testing
- Use async tests for async code

**Test file structure**:
```python
import pytest
from module import function


class TestFeature:
    """Test suite for Feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function()
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        result = await async_function()
        assert result == expected
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_orchestration.py -v

# Run with coverage
make test

# Run fast tests only
make test-fast
```

## Submitting Changes

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes
   - Ensure all CI checks pass

3. **Address review feedback**:
   - Make requested changes
   - Push updates to your branch
   - Respond to comments

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings introduced
- [ ] Backwards compatibility maintained

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- **Clear title** and description
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment details**:
  - OS and version
  - Python version
  - GRYPHGEN version
  - GPU information (if applicable)
- **Error messages** and stack traces
- **Minimal reproducible example**

### Feature Requests

When requesting features, include:

- **Clear use case** description
- **Expected behavior**
- **Alternative solutions** considered
- **Implementation ideas** (optional)

## Development Workflow

### Syncing with Upstream

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes
git checkout main
git merge upstream/main

# Update your feature branch
git checkout feature/your-feature-name
git rebase main
```

### GPU Development

When working on GPU-related features:

- Test on target hardware (RTX 4080) when possible
- Include CPU fallback code
- Add GPU availability checks
- Document GPU requirements
- Test memory management

### Docker Development

```bash
# Build Docker image
make docker-build

# Test in container
docker run -it gryphgen:latest bash

# Run tests in container
make docker-test
```

## Questions?

If you have questions:

- Check existing [issues](https://github.com/danindiana/GRYPHGEN/issues)
- Open a new issue with the `question` label
- Review documentation in `docs/`

## Recognition

Contributors will be recognized in:

- CHANGELOG.md
- README.md (for significant contributions)
- GitHub contributors page

Thank you for contributing to GRYPHGEN! 🚀
