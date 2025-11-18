# Contributing to GRYPHGEN Agentic

Thank you for your interest in contributing to GRYPHGEN Agentic! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment.

## How to Contribute

### Reporting Issues

1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide details**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, GPU, Python version, etc.)
   - Logs and error messages

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the code style guidelines
   - Add tests for new features
   - Update documentation as needed
   - Ensure all tests pass

4. **Commit your changes**:
   ```bash
   git commit -m "feat: add amazing feature"
   ```

   Use conventional commit messages:
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `style:` - Code style changes (formatting, etc.)
   - `refactor:` - Code refactoring
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance tasks

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**:
   - Provide a clear description
   - Reference related issues
   - Include screenshots/examples if applicable

## Development Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (for GPU features)
- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danindiana/GRYPHGEN.git
   cd GRYPHGEN/agentic
   ```

2. **Create virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   make install-dev
   ```

4. **Copy environment file**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests**:
   ```bash
   make test
   ```

6. **Start development environment**:
   ```bash
   make dev
   ```

## Code Style

### Python

We use:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pylint** for additional linting

Format your code before committing:
```bash
make format
```

Check code style:
```bash
make lint
```

### Type Hints

- Use type hints for all function signatures
- Use `typing` module for complex types
- Example:
  ```python
  from typing import List, Optional, Dict, Any

  def process_data(
      data: List[Dict[str, Any]],
      config: Optional[str] = None
  ) -> Dict[str, Any]:
      """Process data with optional config."""
      ...
  ```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Longer description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Examples:
        >>> example_function("test", 42)
        True
    """
    ...
```

## Testing

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest fixtures for common setup
- Separate unit, integration, and GPU tests

Example test:
```python
import pytest
from agentic.services.code_generation import CodeGenerator

@pytest.fixture
def code_generator():
    return CodeGenerator(model="gpt-4-turbo")

def test_code_generation(code_generator):
    """Test basic code generation."""
    result = code_generator.generate(
        prompt="Create a hello world function",
        language="python"
    )
    assert result.code is not None
    assert "def" in result.code
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# GPU tests (requires GPU)
make test-gpu

# With coverage
pytest --cov=src tests/
```

## GPU Development

### Testing GPU Code

- Use `@pytest.mark.gpu` for GPU tests
- Mock GPU operations when possible for CI
- Test both GPU and CPU fallbacks
- Monitor GPU memory usage

Example:
```python
import pytest
import torch

@pytest.mark.gpu
def test_gpu_inference():
    """Test inference on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    model = YourModel().to("cuda")
    result = model.inference(data)
    assert result.device.type == "cuda"
```

### GPU Best Practices

- Use mixed precision training (bf16 on RTX 4080)
- Implement gradient checkpointing for large models
- Monitor GPU memory with `nvidia-smi`
- Profile with PyTorch profiler
- Test memory cleanup after operations

## Documentation

### Code Documentation

- Document all public APIs
- Include examples in docstrings
- Update README for major changes
- Add architecture diagrams when appropriate

### Documentation Site

We use MkDocs with Material theme:

```bash
# Build docs
make docs

# Serve locally
make docs-serve
```

Add new pages in `docs/` and update `mkdocs.yml`.

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests are added/updated and passing
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] All CI checks pass

## Performance Considerations

- Profile code for bottlenecks
- Optimize GPU memory usage
- Use async operations where appropriate
- Implement caching for expensive operations
- Monitor service latency

## Security

- Never commit API keys or secrets
- Use environment variables for configuration
- Sanitize user inputs
- Keep dependencies updated
- Report security issues privately

## Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to GRYPHGEN Agentic! 🚀
