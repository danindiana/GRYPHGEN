# CALISOTA AI Engine - Makefile
# Convenient commands for development and deployment

.PHONY: help install test lint format clean run docker-build docker-up docker-down

help:  ## Show this help message
	@echo "CALISOTA AI Engine - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with Poetry
	poetry install

install-dev:  ## Install development dependencies
	poetry install --with dev

test:  ## Run test suite
	poetry run pytest -v --cov=src/calisota --cov-report=html --cov-report=term

test-watch:  ## Run tests in watch mode
	poetry run ptw -- -v

lint:  ## Run linters (ruff, mypy)
	poetry run ruff check src/ tests/
	poetry run mypy src/

format:  ## Format code with Black and Ruff
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

format-check:  ## Check code formatting
	poetry run black --check src/ tests/

clean:  ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml

run:  ## Run development server
	poetry run uvicorn src.calisota.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:  ## Run production server
	poetry run uvicorn src.calisota.api.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-build:  ## Build Docker image
	docker build -t calisota-ai:latest .

docker-up:  ## Start Docker Compose services
	docker-compose up -d

docker-down:  ## Stop Docker Compose services
	docker-compose down

docker-logs:  ## View Docker Compose logs
	docker-compose logs -f

docker-shell:  ## Open shell in API container
	docker-compose exec calisota-api /bin/bash

migrate-docs:  ## Move old documentation to docs/archive
	mkdir -p docs/archive
	mv *.md docs/archive/ 2>/dev/null || true
	mv calisota-feb docs/archive/ 2>/dev/null || true
	mv RAG docs/archive/ 2>/dev/null || true

pre-commit:  ## Install pre-commit hooks
	poetry run pre-commit install

gpu-test:  ## Test GPU availability
	python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

benchmark:  ## Run performance benchmarks
	poetry run python scripts/benchmark.py

init-db:  ## Initialize FAISS database
	poetry run python scripts/init_faiss.py

backup-data:  ## Backup FAISS index and data
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/

.DEFAULT_GOAL := help
