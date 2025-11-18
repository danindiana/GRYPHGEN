.PHONY: help install install-dev clean test lint format run-dev run-prod docker-build docker-up docker-down deploy health

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Python configuration
PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
UVICORN := $(PYTHON) -m uvicorn

# Project configuration
PROJECT_NAME := gryphgen-infrastructure-agents
SRC_DIR := src
TEST_DIR := tests
DOCKER_DIR := docker

# GPU configuration (RTX 4080)
GPU_ID ?= 0
GPU_MEMORY_FRACTION ?= 0.9

# Service ports
OLLAMA_PORT ?= 11435
NGINX_PORT ?= 11434
API_PORT ?= 8080
PROMETHEUS_PORT ?= 9090

help: ## Show this help message
	@echo "$(BLUE)════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  GRYPHGEN Infrastructure Agents - Makefile Commands$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)════════════════════════════════════════════════════════════════$(NC)"

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

install-docker: ## Install Docker-related dependencies
	@echo "$(GREEN)Installing Docker dependencies...$(NC)"
	$(PIP) install -e ".[docker]"
	@echo "$(GREEN)✓ Docker dependencies installed$(NC)"

install-all: ## Install all dependencies (production + dev + docker)
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all]"
	pre-commit install
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

clean: ## Clean build artifacts and cache files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist
	@echo "$(GREEN)✓ Clean complete$(NC)"

format: ## Auto-format code with black and ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run linters (ruff, mypy, bandit)
	@echo "$(GREEN)Running linters...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR)
	$(PYTHON) -m bandit -r $(SRC_DIR) -f screen
	@echo "$(GREEN)✓ Linting complete$(NC)"

test: ## Run tests with coverage
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(NC)"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) -v -m unit
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) -v -m integration
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-gpu: ## Run GPU tests (requires RTX 4080)
	@echo "$(GREEN)Running GPU tests...$(NC)"
	CUDA_VISIBLE_DEVICES=$(GPU_ID) $(PYTEST) -v -m gpu
	@echo "$(GREEN)✓ GPU tests complete$(NC)"

security-scan: ## Run security vulnerability checks
	@echo "$(GREEN)Running security scans...$(NC)"
	$(PYTHON) -m bandit -r $(SRC_DIR) -f json -o bandit-report.json
	$(PYTHON) -m safety check --json
	@echo "$(GREEN)✓ Security scan complete$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(NC)"
	$(MYPY) $(SRC_DIR)
	@echo "$(GREEN)✓ Type checking complete$(NC)"

ci-test: lint type-check test security-scan ## Run all CI checks
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

run-dev: ## Run development server with auto-reload
	@echo "$(GREEN)Starting development server...$(NC)"
	@echo "$(BLUE)API docs: http://localhost:$(API_PORT)/docs$(NC)"
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

run-prod: ## Run production server
	@echo "$(GREEN)Starting production server...$(NC)"
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port $(API_PORT) --workers 4

deploy: ## Deploy infrastructure (Ollama + Nginx)
	@echo "$(GREEN)Deploying infrastructure...$(NC)"
	@echo "$(BLUE)Ollama port: $(OLLAMA_PORT)$(NC)"
	@echo "$(BLUE)Nginx port: $(NGINX_PORT)$(NC)"
	@echo "$(BLUE)GPU: $(GPU_ID) (memory fraction: $(GPU_MEMORY_FRACTION))$(NC)"
	$(PYTHON) -m src.agents.infrastructure deploy \
		--ollama-port $(OLLAMA_PORT) \
		--nginx-port $(NGINX_PORT) \
		--gpu-enabled \
		--gpu-id $(GPU_ID) \
		--gpu-memory-fraction $(GPU_MEMORY_FRACTION)
	@echo "$(GREEN)✓ Deployment complete$(NC)"

health: ## Check infrastructure health
	@echo "$(GREEN)Checking infrastructure health...$(NC)"
	@curl -s http://localhost:$(API_PORT)/health | python3 -m json.tool || echo "$(RED)API not running$(NC)"
	@curl -s http://localhost:$(NGINX_PORT)/health || echo "$(YELLOW)Nginx not responding$(NC)"
	@curl -s http://localhost:$(OLLAMA_PORT)/ || echo "$(YELLOW)Ollama not responding$(NC)"

status: ## Show infrastructure status
	@echo "$(GREEN)Infrastructure Status:$(NC)"
	$(PYTHON) -m src.agents.infrastructure status

stop: ## Stop infrastructure services
	@echo "$(YELLOW)Stopping infrastructure...$(NC)"
	$(PYTHON) -m src.agents.infrastructure stop
	@echo "$(GREEN)✓ Infrastructure stopped$(NC)"

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -f $(DOCKER_DIR)/Dockerfile -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up: ## Start Docker Compose services
	@echo "$(GREEN)Starting Docker services...$(NC)"
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml up -d
	@echo "$(GREEN)✓ Docker services started$(NC)"
	@echo "$(BLUE)View logs: make docker-logs$(NC)"

docker-down: ## Stop Docker Compose services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml down
	@echo "$(GREEN)✓ Docker services stopped$(NC)"

docker-logs: ## Show Docker Compose logs
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml logs -f

docker-shell: ## Open shell in running container
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml exec agent bash

gpu-check: ## Check GPU availability and stats
	@echo "$(GREEN)Checking GPU...$(NC)"
	@nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used --format=csv || echo "$(RED)nvidia-smi not found$(NC)"
	@echo ""
	@echo "$(GREEN)CUDA Version:$(NC)"
	@nvcc --version 2>/dev/null || echo "$(YELLOW)nvcc not found$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTHON) examples/gpu_optimization.py --benchmark
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve

docs-build: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)✓ Documentation built in site/$(NC)"

requirements-update: ## Update requirements.txt from pyproject.toml
	@echo "$(GREEN)Updating requirements.txt...$(NC)"
	$(PIP) install pip-tools
	pip-compile --upgrade --resolver=backtracking -o requirements.txt pyproject.toml
	pip-compile --extra dev --upgrade --resolver=backtracking -o requirements-dev.txt pyproject.toml
	@echo "$(GREEN)✓ Requirements updated$(NC)"

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit complete$(NC)"

git-status: ## Show git status and branch info
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)Git Status$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@git status
	@echo ""
	@echo "$(GREEN)Current Branch:$(NC) $(shell git branch --show-current)"
	@echo "$(GREEN)Recent Commits:$(NC)"
	@git log --oneline -5

version: ## Show versions of key tools
	@echo "$(GREEN)Versions:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "pip: $(shell $(PIP) --version)"
	@echo "CUDA: $(shell nvcc --version 2>/dev/null | grep release || echo 'Not installed')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "nvidia-smi: $(shell nvidia-smi --version 2>/dev/null | head -1 || echo 'Not installed')"

.DEFAULT_GOAL := help
