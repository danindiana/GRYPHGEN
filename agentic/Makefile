.PHONY: help install install-dev clean test lint format build run dev stop logs docker-build docker-up docker-down gpu-check

# Configuration
PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := gryphgen-agentic

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

## help: Display this help message
help:
	@echo "$(BLUE)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## install: Install production dependencies
install:
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

## install-dev: Install development dependencies
install-dev: install
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev,gpu]"
	pre-commit install

## clean: Remove build artifacts and cache
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "$(GREEN)Clean complete!$(NC)"

## test: Run tests
test:
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

## test-unit: Run unit tests only
test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/ -v -m "unit"

## test-integration: Run integration tests only
test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/ -v -m "integration"

## test-gpu: Run GPU tests (requires NVIDIA GPU)
test-gpu: gpu-check
	@echo "$(BLUE)Running GPU tests...$(NC)"
	$(PYTEST) tests/ -v -m "gpu"

## lint: Run linters
lint:
	@echo "$(BLUE)Running linters...$(NC)"
	$(FLAKE8) src/ tests/
	$(MYPY) src/
	$(PYLINT) src/

## format: Format code with black and isort
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

## format-check: Check code formatting
format-check:
	@echo "$(BLUE)Checking code format...$(NC)"
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/

## gpu-check: Verify GPU availability
gpu-check:
	@echo "$(BLUE)Checking GPU availability...$(NC)"
	@$(PYTHON) -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); \
		print(f'CUDA Version: {torch.version.cuda}'); \
		print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); \
		print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else 'N/A')" || \
		echo "$(RED)GPU not available or PyTorch not properly installed$(NC)"

## build: Build Python package
build: clean
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete!$(NC)"

## docker-build: Build Docker images
docker-build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Docker images built!$(NC)"

## docker-build-prod: Build production Docker images
docker-build-prod:
	@echo "$(BLUE)Building production Docker images...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml build
	@echo "$(GREEN)Production Docker images built!$(NC)"

## docker-up: Start Docker containers
docker-up:
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Containers started!$(NC)"
	@echo "API Gateway: http://localhost:8000"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

## docker-down: Stop Docker containers
docker-down:
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Containers stopped!$(NC)"

## docker-logs: View Docker logs
docker-logs:
	$(DOCKER_COMPOSE) logs -f

## docker-ps: Show running containers
docker-ps:
	$(DOCKER_COMPOSE) ps

## docker-clean: Remove Docker containers, volumes, and images
docker-clean:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v --rmi all
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

## dev: Start development environment
dev: docker-up
	@echo "$(BLUE)Development environment ready!$(NC)"

## run: Run services locally (without Docker)
run:
	@echo "$(BLUE)Starting services locally...$(NC)"
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

## docs: Build documentation
docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)Documentation built! Open site/index.html$(NC)"

## docs-serve: Serve documentation locally
docs-serve:
	@echo "$(BLUE)Serving documentation at http://localhost:8001$(NC)"
	mkdocs serve -a localhost:8001

## benchmark: Run performance benchmarks
benchmark: gpu-check
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) scripts/benchmark.py

## profile: Profile GPU memory usage
profile: gpu-check
	@echo "$(BLUE)Profiling GPU memory...$(NC)"
	$(PYTHON) -m torch.utils.bottleneck scripts/profile_gpu.py

## migrate: Run database migrations
migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head

## migrate-create: Create a new migration
migrate-create:
	@echo "$(BLUE)Creating new migration...$(NC)"
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

## seed: Seed database with sample data
seed:
	@echo "$(BLUE)Seeding database...$(NC)"
	$(PYTHON) scripts/seed_db.py

## security-check: Run security vulnerability checks
security-check:
	@echo "$(BLUE)Running security checks...$(NC)"
	$(PIP) install safety bandit
	safety check
	bandit -r src/

## pre-commit: Run pre-commit hooks
pre-commit:
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

## ci: Run CI pipeline locally
ci: format-check lint test
	@echo "$(GREEN)CI pipeline passed!$(NC)"

## version: Display version information
version:
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA: $(shell $(PYTHON) -c 'import torch; print(torch.version.cuda)')"
	@echo "Project: $(PROJECT_NAME)"

## all: Run full build and test pipeline
all: clean install-dev format lint test docker-build
	@echo "$(GREEN)All tasks completed successfully!$(NC)"
