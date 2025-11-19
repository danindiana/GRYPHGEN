.PHONY: help install install-dev install-gpu test lint format clean docker-build docker-run docker-stop docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

help:  ## Show this help message
	@echo "$(BLUE)ShellGenie Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install:  ## Install ShellGenie (basic)
	@echo "$(BLUE)Installing ShellGenie...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev:  ## Install with development dependencies
	@echo "$(BLUE)Installing ShellGenie with dev dependencies...$(NC)"
	pip install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development installation complete!$(NC)"

install-gpu:  ## Install with GPU support (CUDA, PyTorch, llama-cpp-python)
	@echo "$(BLUE)Installing ShellGenie with GPU support...$(NC)"
	pip install -e ".[gpu,dev]"
	@echo "$(GREEN)✓ GPU installation complete!$(NC)"

install-all:  ## Install everything (dev + GPU)
	@echo "$(BLUE)Installing ShellGenie with all dependencies...$(NC)"
	pip install -e ".[all]"
	pre-commit install
	@echo "$(GREEN)✓ Full installation complete!$(NC)"

test:  ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=shellgenie --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-fast:  ## Run tests without slow tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow" --cov=shellgenie
	@echo "$(GREEN)✓ Fast tests complete!$(NC)"

test-integration:  ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m "integration"
	@echo "$(GREEN)✓ Integration tests complete!$(NC)"

lint:  ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check src/shellgenie tests/
	mypy src/shellgenie
	@echo "$(GREEN)✓ Linting complete!$(NC)"

format:  ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/shellgenie tests/
	ruff check --fix src/shellgenie tests/
	@echo "$(GREEN)✓ Formatting complete!$(NC)"

type-check:  ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/shellgenie
	@echo "$(GREEN)✓ Type checking complete!$(NC)"

clean:  ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

docker-build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -f docker/Dockerfile -t shellgenie:2.0.0 -t shellgenie:latest .
	@echo "$(GREEN)✓ Docker build complete!$(NC)"

docker-run:  ## Run ShellGenie in Docker
	@echo "$(BLUE)Starting ShellGenie container...$(NC)"
	docker-compose -f docker/docker-compose.yml up -d
	@echo "$(GREEN)✓ Container started!$(NC)"
	@echo "$(YELLOW)Run 'docker exec -it shellgenie-dev bash' to access the container$(NC)"

docker-run-standalone:  ## Run with standalone Ollama service
	@echo "$(BLUE)Starting ShellGenie with standalone Ollama...$(NC)"
	docker-compose -f docker/docker-compose.yml --profile standalone up -d
	@echo "$(GREEN)✓ Services started!$(NC)"

docker-stop:  ## Stop Docker containers
	@echo "$(BLUE)Stopping containers...$(NC)"
	docker-compose -f docker/docker-compose.yml down
	@echo "$(GREEN)✓ Containers stopped!$(NC)"

docker-logs:  ## View Docker logs
	docker-compose -f docker/docker-compose.yml logs -f

docker-shell:  ## Open shell in Docker container
	docker exec -it shellgenie-dev /bin/bash

docker-clean:  ## Remove Docker containers and volumes
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose -f docker/docker-compose.yml down -v
	docker rmi shellgenie:2.0.0 shellgenie:latest 2>/dev/null || true
	@echo "$(GREEN)✓ Docker cleanup complete!$(NC)"

check-gpu:  ## Check GPU availability
	@echo "$(BLUE)Checking GPU...$(NC)"
	@nvidia-smi || echo "$(RED)✗ NVIDIA GPU not found$(NC)"
	@python3 -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())" 2>/dev/null || echo "$(YELLOW)PyTorch not installed$(NC)"

info:  ## Show system information
	@echo "$(BLUE)System Information$(NC)"
	@shellgenie info || echo "$(YELLOW)ShellGenie not installed. Run 'make install' first.$(NC)"

run-interactive:  ## Run ShellGenie in interactive mode
	@shellgenie interactive

run-example:  ## Run an example command
	@echo "$(BLUE)Running example: 'list all files in current directory'$(NC)"
	@shellgenie run "list all files in current directory"

setup-ollama:  ## Install and setup Ollama
	@echo "$(BLUE)Setting up Ollama...$(NC)"
	@command -v ollama >/dev/null 2>&1 || (curl -fsSL https://ollama.ai/install.sh | sh)
	@ollama serve > /tmp/ollama.log 2>&1 &
	@sleep 5
	@ollama pull llama3.2
	@echo "$(GREEN)✓ Ollama setup complete!$(NC)"

pull-models:  ## Pull recommended models
	@echo "$(BLUE)Pulling recommended models...$(NC)"
	@ollama pull llama3.2
	@ollama pull llama3.1
	@ollama pull qwen2.5
	@echo "$(GREEN)✓ Models downloaded!$(NC)"

benchmark:  ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@python3 -m pytest tests/ -v -m "benchmark" || echo "$(YELLOW)No benchmarks defined yet$(NC)"

coverage:  ## Generate coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	pytest tests/ --cov=shellgenie --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

docs:  ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@echo "$(YELLOW)Documentation build not yet implemented$(NC)"

build:  ## Build distribution packages
	@echo "$(BLUE)Building packages...$(NC)"
	python3 -m build
	@echo "$(GREEN)✓ Build complete! Packages in dist/$(NC)"

release:  ## Create a new release (requires clean git state)
	@echo "$(BLUE)Creating release...$(NC)"
	@git diff-index --quiet HEAD || (echo "$(RED)✗ Git working directory not clean$(NC)" && exit 1)
	@echo "$(YELLOW)Creating git tag...$(NC)"
	@git tag -a v2.0.0 -m "Release v2.0.0"
	@echo "$(GREEN)✓ Release tag created. Push with: git push origin v2.0.0$(NC)"

dev-setup:  ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(MAKE) install-all
	@$(MAKE) setup-ollama
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

quick-start:  ## Quick start for new users
	@echo "$(BLUE)═══════════════════════════════════════$(NC)"
	@echo "$(BLUE)     ShellGenie Quick Start$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 1:$(NC) Install dependencies"
	@echo "  make install-gpu"
	@echo ""
	@echo "$(YELLOW)Step 2:$(NC) Setup Ollama and pull models"
	@echo "  make setup-ollama"
	@echo ""
	@echo "$(YELLOW)Step 3:$(NC) Run ShellGenie"
	@echo "  shellgenie interactive"
	@echo ""
	@echo "$(GREEN)Or run everything at once:$(NC)"
	@echo "  make dev-setup && shellgenie interactive"
	@echo ""
