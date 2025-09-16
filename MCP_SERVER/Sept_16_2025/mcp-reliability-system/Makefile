
# MCP Reliability System Makefile

.PHONY: build test bench clean install docker-build docker-run help

# Default target
help:
	@echo "MCP Reliability System - Available targets:"
	@echo "  build        - Build the project"
	@echo "  test         - Run all tests"
	@echo "  bench        - Run benchmarks"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install the binary"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  format       - Format Haskell code"
	@echo "  lint         - Run linter"
	@echo "  docs         - Generate documentation"

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
	@echo "Running security scan..."
	@echo "TODO: Implement security scanning"

# Performance profiling
profile:
	cabal run mcp-server -- +RTS -p -RTS --config config/production.yaml

# Load testing
load-test:
	@echo "Running load tests..."
	@echo "TODO: Implement load testing"
