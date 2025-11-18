# Novelty LLM System Makefile

.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down run-dev docs

# Default target
help:
	@echo "Novelty LLM System - Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  setup         - Complete development environment setup"
	@echo ""
	@echo "Development:"
	@echo "  run-dev       - Run development server"
	@echo "  test          - Run tests with coverage"
	@echo "  test-watch    - Run tests in watch mode"
	@echo "  lint          - Run linters (ruff, mypy)"
	@echo "  format        - Format code (black, ruff)"
	@echo "  format-check  - Check code formatting"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  - Build Docker images"
	@echo "  docker-up     - Start all services with docker-compose"
	@echo "  docker-down   - Stop all services"
	@echo "  docker-logs   - View service logs"
	@echo "  docker-shell  - Open shell in main container"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate    - Run database migrations"
	@echo "  db-upgrade    - Upgrade database schema"
	@echo "  db-downgrade  - Downgrade database schema"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         - Clean build artifacts and cache"
	@echo "  clean-all     - Deep clean including dependencies"
	@echo "  docs          - Generate documentation"
	@echo "  security-scan - Run security scanning"

# Python and pip
PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# Directories
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs

# Installation targets
install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	pre-commit install

setup: install-dev
	@echo "Development environment setup complete!"

# Development targets
run-dev:
	uvicorn src.api.gateway:create_app --factory --reload --host 0.0.0.0 --port 8080

# Testing targets
test:
	$(PYTEST) $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-report=xml

test-watch:
	$(PYTEST) $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing -f

test-fast:
	$(PYTEST) $(TESTS_DIR) -v -x --ff

# Code quality targets
lint:
	$(RUFF) check $(SRC_DIR) $(TESTS_DIR)
	$(MYPY) $(SRC_DIR)

format:
	$(BLACK) $(SRC_DIR) $(TESTS_DIR)
	$(RUFF) check --fix $(SRC_DIR) $(TESTS_DIR)

format-check:
	$(BLACK) --check $(SRC_DIR) $(TESTS_DIR)
	$(RUFF) check $(SRC_DIR) $(TESTS_DIR)

# Docker targets
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

docker-shell:
	docker-compose -f docker/docker-compose.yml exec novelty-llm-api /bin/bash

docker-clean:
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f

# Database targets
db-migrate:
	alembic revision --autogenerate -m "$(msg)"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	alembic downgrade base
	alembic upgrade head

# Documentation targets
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Security and auditing
security-scan:
	$(PIP) install safety bandit
	safety check --json
	bandit -r $(SRC_DIR) -f json -o security-report.json

audit:
	$(PIP) audit

# Cleaning targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf site/

clean-all: clean
	rm -rf .venv/
	rm -rf venv/

# Monitoring and debugging
metrics:
	curl -s http://localhost:8080/metrics

health:
	curl -s http://localhost:8080/health | jq .

cache-stats:
	curl -s http://localhost:8080/cache/stats | jq .

# Load testing
load-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8080

# CI targets
ci-test: format-check lint test

ci-build: docker-build

# Release targets
release: clean test lint
	$(PYTHON) -m build
	@echo "Release build complete. Run 'twine upload dist/*' to publish."

# Version management
version:
	@grep "^version" pyproject.toml | cut -d'"' -f2

bump-patch:
	$(PIP) install bump2version
	bump2version patch

bump-minor:
	$(PIP) install bump2version
	bump2version minor

bump-major:
	$(PIP) install bump2version
	bump2version major
