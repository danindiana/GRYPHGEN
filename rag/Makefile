.PHONY: help install install-dev test lint format clean build docker docs

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# Help target
help:
	@echo "GRYPHGEN RAG - Makefile Commands"
	@echo "================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install package and dependencies"
	@echo "  make install-dev    - Install with development dependencies"
	@echo "  make install-gpu    - Install with GPU support (CUDA)"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code with ruff and mypy"
	@echo "  make test           - Run tests with pytest"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo ""
	@echo "Build & Package:"
	@echo "  make build          - Build package distribution"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build documentation"
	@echo "  make docs-serve     - Serve documentation locally"
	@echo ""
	@echo "Examples:"
	@echo "  make run-simgrag    - Run SimGRAG example"
	@echo "  make run-cag        - Run CAG example"

# Installation targets
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✓ Package installed successfully"

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,docs]"
	$(PYTHON) -m pre_commit install
	@echo "✓ Development environment set up successfully"

install-gpu:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[gpu]"
	@echo "✓ GPU support installed"
	@echo "Note: Ensure CUDA 12.x is installed for RTX 4080 support"

# Code quality targets
format:
	@echo "Formatting code with black..."
	$(BLACK) simgrag cag common examples tests
	@echo "✓ Code formatted successfully"

lint:
	@echo "Linting with ruff..."
	$(RUFF) check simgrag cag common examples tests
	@echo "Type checking with mypy..."
	$(MYPY) simgrag cag common
	@echo "✓ Linting complete"

lint-fix:
	@echo "Fixing lint issues..."
	$(RUFF) check --fix simgrag cag common examples tests
	@echo "✓ Lint issues fixed"

# Testing targets
test:
	@echo "Running tests..."
	$(PYTEST) tests/ -v
	@echo "✓ Tests complete"

test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ -v --cov=simgrag --cov=cag --cov=common --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-watch:
	@echo "Running tests in watch mode..."
	$(PYTEST) tests/ -v --watch

# Build targets
build:
	@echo "Building package..."
	$(PYTHON) -m build
	@echo "✓ Package built successfully"
	@ls -lh dist/

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Cleanup complete"

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t gryphgen-rag:latest -f Dockerfile .
	@echo "✓ Docker image built successfully"

docker-run:
	@echo "Running Docker container..."
	docker run --rm -it --gpus all \
		-v $(PWD):/workspace \
		gryphgen-rag:latest \
		bash

docker-test:
	@echo "Running tests in Docker..."
	docker run --rm --gpus all \
		-v $(PWD):/workspace \
		gryphgen-rag:latest \
		make test

# Documentation targets
docs:
	@echo "Building documentation..."
	cd docs && mkdocs build
	@echo "✓ Documentation built in docs/site/"

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	cd docs && mkdocs serve

# Example targets
run-simgrag:
	@echo "Running SimGRAG example..."
	$(PYTHON) examples/simgrag_example.py

run-cag:
	@echo "Running CAG example..."
	$(PYTHON) examples/cag_example.py

# CI/CD targets
ci: lint test
	@echo "✓ CI checks passed"

# GPU check
gpu-check:
	@echo "Checking GPU availability..."
	$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'CUDA version: {torch.version.cuda}'); \
		print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Benchmark targets
benchmark-simgrag:
	@echo "Running SimGRAG benchmark..."
	$(PYTHON) -m pytest tests/benchmark_simgrag.py -v --benchmark-only

benchmark-cag:
	@echo "Running CAG benchmark..."
	$(PYTHON) -m pytest tests/benchmark_cag.py -v --benchmark-only

# Quick setup for RTX 4080
setup-rtx4080: install-gpu
	@echo "Setting up for NVIDIA RTX 4080..."
	@echo "Checking GPU..."
	@make gpu-check
	@echo "✓ RTX 4080 setup complete"

# Development workflow
dev: clean install-dev
	@echo "✓ Development environment ready"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make format' to format code"
	@echo "Run 'make lint' to check code quality"

# All-in-one targets
all: clean install lint test build
	@echo "✓ All tasks complete"

.DEFAULT_GOAL := help
