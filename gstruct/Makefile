.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down run gpu-info

# Variables
PYTHON := python3.11
PIP := $(PYTHON) -m pip
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := gryphgen
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_CYAN := \033[36m

help: ## Show this help message
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)GRYPHGEN - Grid Computing Framework$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)Available targets:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Installing production dependencies...$(COLOR_RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(COLOR_GREEN)✓ Installation complete$(COLOR_RESET)"

install-dev: ## Install development dependencies
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Installing development dependencies...$(COLOR_RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov black ruff mypy sphinx sphinx-rtd-theme
	$(PIP) install -e .
	@echo "$(COLOR_GREEN)✓ Development installation complete$(COLOR_RESET)"

test: ## Run tests
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Running tests...$(COLOR_RESET)"
	pytest -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "$(COLOR_GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(COLOR_RESET)"

test-fast: ## Run tests without coverage
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Running fast tests...$(COLOR_RESET)"
	pytest -v -x
	@echo "$(COLOR_GREEN)✓ Tests complete$(COLOR_RESET)"

lint: ## Run linters
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Running linters...$(COLOR_RESET)"
	ruff check .
	mypy . --ignore-missing-imports
	@echo "$(COLOR_GREEN)✓ Linting complete$(COLOR_RESET)"

format: ## Format code
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Formatting code...$(COLOR_RESET)"
	black .
	ruff check --fix .
	@echo "$(COLOR_GREEN)✓ Code formatted$(COLOR_RESET)"

clean: ## Clean build artifacts
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Cleaning build artifacts...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	@echo "$(COLOR_GREEN)✓ Cleanup complete$(COLOR_RESET)"

docker-build: ## Build Docker images
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Building Docker images...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) build
	@echo "$(COLOR_GREEN)✓ Docker images built$(COLOR_RESET)"

docker-build-gpu: ## Build Docker images with GPU support
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Building Docker images with GPU support...$(COLOR_RESET)"
	docker build --target production -t $(DOCKER_IMAGE) .
	@echo "$(COLOR_GREEN)✓ Docker GPU image built$(COLOR_RESET)"

docker-up: ## Start Docker services
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Starting Docker services...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(COLOR_GREEN)✓ Services started$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Run 'docker-compose logs -f' to view logs$(COLOR_RESET)"

docker-down: ## Stop Docker services
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Stopping Docker services...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(COLOR_GREEN)✓ Services stopped$(COLOR_RESET)"

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

run: ## Run GRYPHGEN locally
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Starting GRYPHGEN...$(COLOR_RESET)"
	$(PYTHON) GRYPHGEN.py start

run-orchestrator: ## Run orchestrator only
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Starting Orchestrator...$(COLOR_RESET)"
	$(PYTHON) -m SYMORQ.orchestration

run-scheduler: ## Run scheduler only
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Starting Scheduler...$(COLOR_RESET)"
	$(PYTHON) -m SYMORG.scheduling

gpu-info: ## Show GPU information
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)GPU Information:$(COLOR_RESET)"
	$(PYTHON) GRYPHGEN.py info

status: ## Show framework status
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Framework Status:$(COLOR_RESET)"
	$(PYTHON) GRYPHGEN.py status

setup-env: ## Setup environment file
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Creating .env file...$(COLOR_RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env 2>/dev/null || echo "# GRYPHGEN Environment Configuration\nGRYPHGEN_LOG_LEVEL=INFO\nGRYPHGEN_GPU_ENABLED=true" > .env; \
		echo "$(COLOR_GREEN)✓ .env file created$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW).env file already exists$(COLOR_RESET)"; \
	fi

init: setup-env install ## Initialize project (first time setup)
	@echo "$(COLOR_BOLD)$(COLOR_GREEN)✓ Project initialized!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Run 'make run' to start GRYPHGEN$(COLOR_RESET)"

dev-init: setup-env install-dev ## Initialize development environment
	@echo "$(COLOR_BOLD)$(COLOR_GREEN)✓ Development environment ready!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Run 'make test' to run tests$(COLOR_RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Running benchmarks...$(COLOR_RESET)"
	$(PYTHON) -m pytest tests/ -v --benchmark-only

docs: ## Generate documentation
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)Generating documentation...$(COLOR_RESET)"
	cd docs && make html
	@echo "$(COLOR_GREEN)✓ Documentation generated$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Open docs/_build/html/index.html$(COLOR_RESET)"

.DEFAULT_GOAL := help
