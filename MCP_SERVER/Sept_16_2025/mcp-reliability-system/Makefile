
# MCP Reliability System Makefile

.PHONY: help build test bench clean install docker-build docker-run
.PHONY: format lint docs dev-setup ci-build ci-test release
.PHONY: security-scan profile load-test watch run

# Color output
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RESET  := \033[0m

# Project info
PROJECT_NAME := mcp-reliability-system
VERSION := $(shell grep '^version:' mcp-reliability-system.cabal | awk '{print $$2}')
GHC_VERSION := 9.6.4

# Default target
help:
	@echo "$(BLUE)MCP Reliability System v$(VERSION)$(RESET)"
	@echo ""
	@echo "$(GREEN)Building:$(RESET)"
	@echo "  build         - Build the project"
	@echo "  build-release - Build with optimizations"
	@echo "  install       - Install the binary"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  test          - Run all tests"
	@echo "  test-coverage - Run tests with coverage"
	@echo "  bench         - Run benchmarks"
	@echo "  bench-detailed- Run benchmarks with HTML report"
	@echo "  watch         - Run tests on file changes"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  format        - Format Haskell code"
	@echo "  lint          - Run linter"
	@echo "  docs          - Generate documentation"
	@echo "  ci-build      - Run CI build checks"
	@echo "  ci-test       - Run CI test suite"
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run with Docker Compose"
	@echo "  docker-stop   - Stop Docker containers"
	@echo "  docker-logs   - Show Docker logs"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  dev-setup     - Setup development environment"
	@echo "  run           - Run the server"
	@echo "  profile       - Run with profiling"
	@echo "  security-scan - Run security analysis"
	@echo "  load-test     - Run load tests"
	@echo ""
	@echo "$(GREEN)Release:$(RESET)"
	@echo "  release       - Create release build"
	@echo "  version       - Show version information"

# Build targets
build:
	cabal build

build-release:
	cabal build --enable-optimization=2

# Test targets
test:
	cabal test --test-show-details=streaming

test-coverage:
	cabal test --enable-coverage

# Benchmark targets
bench:
	cabal bench

bench-detailed:
	cabal bench --benchmark-options="--output bench-results.html"

# Development targets
clean:
	cabal clean
	rm -rf dist-newstyle/

install:
	cabal install --install-method=copy --installdir=./bin

# Code quality targets
format:
	find src test bench app -name "*.hs" -exec ormolu --mode inplace {} \;

lint:
	hlint src/ test/ bench/ app/

# Documentation
docs:
	cabal haddock --haddock-all --haddock-hyperlink-source

# Docker targets
docker-build:
	docker build -f docker/Dockerfile -t mcp-reliability-system:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# Development environment
dev-setup:
	cabal update
	cabal build --dependencies-only
	cabal install hlint ormolu

# CI targets
ci-build: build test lint

ci-test: test-coverage bench

# Release targets
release: clean build-release test bench
	@echo "Release build complete"

# Monitoring
metrics:
	curl -s http://localhost:9090/metrics

health:
	curl -s http://localhost:8080/health | jq .

# Database migrations (if needed)
migrate:
	@echo "No migrations needed for this version"

# Security scan
security-scan:
	@echo "$(YELLOW)Running security scan...$(RESET)"
	@command -v hlint >/dev/null 2>&1 && hlint src/ test/ bench/ app/ || echo "$(RED)HLint not found$(RESET)"
	@echo "$(GREEN)Security scan complete$(RESET)"

# Performance profiling
profile:
	@echo "$(YELLOW)Running with profiling...$(RESET)"
	cabal run mcp-server -- +RTS -p -RTS --config config/production.yaml

# Load testing
load-test:
	@echo "$(YELLOW)Running load tests...$(RESET)"
	@echo "$(YELLOW)TODO: Implement load testing with k6 or similar$(RESET)"

# Watch for changes
watch:
	@echo "$(YELLOW)Watching for changes...$(RESET)"
	@command -v ghcid >/dev/null 2>&1 && ghcid --command="cabal repl" --test="main" || \
		echo "$(RED)ghcid not found. Install with: cabal install ghcid$(RESET)"

# Run the server
run:
	@echo "$(YELLOW)Starting MCP server...$(RESET)"
	cabal run mcp-server -- --config config/production.yaml

# Show version information
version:
	@echo "$(BLUE)MCP Reliability System$(RESET)"
	@echo "Version: $(VERSION)"
	@echo "GHC Version: $(GHC_VERSION)"
	@cabal --version
	@ghc --version

# Quick check before commit
pre-commit: format lint test
	@echo "$(GREEN)Pre-commit checks passed!$(RESET)"
