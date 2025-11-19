# Makefile for GRYPHGEN Perl Pipes Multi-Model IPC
# Optimized for NVIDIA RTX 4080 16GB

.PHONY: help install install-deps test clean setup-pipes cleanup-pipes demo coverage lint format docs docker all

PERL := perl
CPANM := cpanm
PROVE := prove
BIN_DIR := bin
LIB_DIR := lib
TEST_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config
EXAMPLES_DIR := examples

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)GRYPHGEN Perl Pipes - Multi-Model IPC System$(NC)"
	@echo "$(YELLOW)Optimized for NVIDIA RTX 4080 16GB$(NC)"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

all: install test ## Install dependencies and run tests

install-deps: ## Install Perl dependencies via cpanm
	@echo "$(BLUE)Installing Perl dependencies...$(NC)"
	@if ! command -v cpanm &> /dev/null; then \
		echo "$(YELLOW)Installing cpanminus...$(NC)"; \
		curl -L https://cpanmin.us | $(PERL) - App::cpanminus; \
	fi
	@$(CPANM) --quiet --installdeps .
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

install: install-deps ## Install dependencies and make scripts executable
	@echo "$(BLUE)Setting up perl_pipes...$(NC)"
	@chmod +x $(BIN_DIR)/*.pl $(BIN_DIR)/*.sh
	@mkdir -p logs
	@echo "$(GREEN)Installation complete$(NC)"

test: ## Run test suite
	@echo "$(BLUE)Running test suite...$(NC)"
	@$(PROVE) -lr $(TEST_DIR)
	@echo "$(GREEN)Tests completed$(NC)"

coverage: ## Generate test coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	@PERL5OPT=-MDevel::Cover $(PROVE) -lr $(TEST_DIR)
	@cover -report html
	@echo "$(GREEN)Coverage report generated in cover_db/$(NC)"

lint: ## Run Perl::Critic for code quality
	@echo "$(BLUE)Running code quality checks...$(NC)"
	@perlcritic $(BIN_DIR) $(LIB_DIR) || true

format: ## Format code with Perl::Tidy
	@echo "$(BLUE)Formatting Perl code...$(NC)"
	@find $(BIN_DIR) $(LIB_DIR) -name '*.pl' -o -name '*.pm' | xargs perltidy -b -bext='/'

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@mkdir -p $(DOCS_DIR)/man
	@for file in $(BIN_DIR)/*.pl; do \
		pod2man $$file > $(DOCS_DIR)/man/$$(basename $$file .pl).1; \
		pod2html $$file > $(DOCS_DIR)/$$(basename $$file .pl).html; \
	done
	@rm -f pod2htmd.tmp pod2htmi.tmp
	@echo "$(GREEN)Documentation generated in $(DOCS_DIR)/$(NC)"

setup-pipes: ## Create named pipes for IPC
	@echo "$(BLUE)Setting up named pipes...$(NC)"
	@$(BIN_DIR)/setup_pipes.sh
	@echo "$(GREEN)Pipes created successfully$(NC)"

cleanup-pipes: ## Remove named pipes
	@echo "$(BLUE)Cleaning up named pipes...$(NC)"
	@$(BIN_DIR)/cleanup_pipes.sh --force
	@echo "$(GREEN)Pipes removed$(NC)"

demo: setup-pipes ## Run a demo of model communication
	@echo "$(BLUE)Starting demo...$(NC)"
	@echo "$(YELLOW)Starting Model 2 (listener)...$(NC)"
	@$(BIN_DIR)/model2_comm.pl --verbose &
	@sleep 1
	@echo "$(YELLOW)Starting Model 1 (sender)...$(NC)"
	@echo "Demo message from Model 1" | $(BIN_DIR)/model1_comm.pl --verbose
	@sleep 1
	@$(MAKE) cleanup-pipes
	@echo "$(GREEN)Demo completed$(NC)"

find-models: ## Find GGUF models on system (may take a while)
	@echo "$(BLUE)Searching for GGUF models...$(NC)"
	@$(BIN_DIR)/find_gguf_files.pl --dir=$(HOME) --max-size=16GB --verbose

clean: cleanup-pipes ## Clean up generated files and pipes
	@echo "$(BLUE)Cleaning up...$(NC)"
	@rm -rf logs/*.log
	@rm -rf cover_db/
	@rm -rf $(DOCS_DIR)/*.html $(DOCS_DIR)/man/
	@find . -name '*~' -delete
	@find . -name '*.bak' -delete
	@echo "$(GREEN)Cleanup complete$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t gryphgen/perl-pipes:latest .
	@echo "$(GREEN)Docker image built successfully$(NC)"

docker-run: docker-build ## Run in Docker container
	@echo "$(BLUE)Running in Docker...$(NC)"
	@docker run -it --rm --gpus all gryphgen/perl-pipes:latest

check-system: ## Check system compatibility
	@echo "$(BLUE)Checking system compatibility...$(NC)"
	@echo "Perl version: $$($(PERL) -v | grep 'This is perl' | cut -d'v' -f2 | cut -d' ' -f1)"
	@echo -n "NVIDIA GPU: "
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Not found"
	@echo -n "CUDA version: "
	@nvcc --version 2>/dev/null | grep release | cut -d',' -f2 || echo "Not installed"
	@echo "$(GREEN)System check complete$(NC)"

.DEFAULT_GOAL := help
