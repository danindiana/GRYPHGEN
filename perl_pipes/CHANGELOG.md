# Changelog

All notable changes to GRYPHGEN Perl Pipes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-11-19

### Added
- Complete restructuring and modernization of perl_pipes directory
- Modern Perl scripts (v5.36+) with enhanced features:
  - `model1_comm.pl` - Enhanced Model 1 communication with logging
  - `model2_comm.pl` - Enhanced Model 2 communication with logging
  - `find_gguf_files.pl` - GPU-aware GGUF model file scanner
  - `monitor_gpu.pl` - Real-time NVIDIA GPU monitoring
- Shell scripts for pipe management:
  - `setup_pipes.sh` - Automated pipe creation with multi-model support
  - `cleanup_pipes.sh` - Pipe cleanup utility
- Comprehensive test suite using Test::More:
  - `01-basic.t` - Script validation tests
  - `02-pipes.t` - Pipe operations tests
  - `03-find-gguf.t` - GGUF finder tests
- Example scripts:
  - `basic_communication.sh` - Simple communication demo
  - `multi_round_chat.sh` - Multi-round conversation demo
  - `find_gpu_models.sh` - GPU-compatible model finder
- Configuration:
  - `rtx4080_optimized.conf` - NVIDIA RTX 4080 optimized settings
  - `cpanfile` - Perl dependency management
  - `Makefile` - Build automation and common tasks
- Docker support:
  - `Dockerfile` - NVIDIA CUDA-based container
  - `docker-compose.yml` - Multi-container orchestration
- CI/CD:
  - GitHub Actions workflow for automated testing
  - Perl::Critic code quality checks
  - Docker build verification
- Documentation:
  - Comprehensive `README.md` with badges and Mermaid diagrams
  - `ARCHITECTURE.md` - Detailed system architecture
  - `CONTRIBUTING.md` - Contribution guidelines
  - `LICENSE` - MIT License
  - `CHANGELOG.md` - This file
- Features:
  - Timeout protection for all IPC operations
  - Comprehensive logging with configurable levels
  - GPU memory compatibility analysis
  - Multiple output formats (table, JSON, CSV)
  - Environment variable configuration
  - Command-line option parsing with Getopt::Long
  - POD documentation for all scripts

### Changed
- Upgraded to modern Perl (v5.36+)
- Reorganized directory structure:
  - `bin/` - Executable scripts
  - `lib/` - Perl modules (for future extensions)
  - `examples/` - Usage examples
  - `tests/` - Test suite
  - `config/` - Configuration files
  - `docs/` - Documentation
  - `logs/` - Runtime logs
- Enhanced error handling with eval blocks and proper error messages
- Improved code style and formatting
- Updated documentation format with Markdown

### Deprecated
- Old single-file README format (moved to `docs/old_readme.md`)

### Removed
- N/A (initial major refactor)

### Fixed
- N/A (initial major refactor)

### Security
- Added file permission checks for pipes
- Implemented timeout protection to prevent hanging
- Sandboxed pipe operations to /tmp directory

## [1.0.0] - Initial Version

### Added
- Basic Perl pipe communication examples
- Simple GGUF file finder script
- Initial documentation in single README file

---

## Versioning Scheme

- **Major version** (X.0.0): Breaking changes, major rewrites
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, documentation updates

## Links

- [Repository](https://github.com/danindiana/GRYPHGEN)
- [Issues](https://github.com/danindiana/GRYPHGEN/issues)
- [Pull Requests](https://github.com/danindiana/GRYPHGEN/pulls)
