# Contributing to Dynamic Cortex

Thank you for your interest in contributing to the Dynamic Cortex Communication Framework!

## Code of Conduct

This project adheres to a code of professional conduct. By participating, you are expected to uphold respectful, collaborative communication.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear, descriptive title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, hardware, SDK versions)

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/danindiana/GRYPHGEN.git
   cd GRYPHGEN/calisota-ai/July-28-2025/tt-metal
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   make test
   ```

5. **Format your code**
   ```bash
   make format
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: clear description"
   ```

7. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- **C++**: Follow Google C++ Style Guide
- **CUDA**: Use NVIDIA best practices
- **Python**: Follow PEP 8

Format all code with clang-format:
```bash
clang-format -i src/**/*.{cpp,hpp,cu}
```

### Documentation

- Use Doxygen-style comments for all public APIs
- Update README.md for user-facing changes
- Add examples for new features
- Update architecture.md for design changes

### Testing

- Write unit tests for new functions
- Add integration tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

### Commit Messages

Use conventional commit format:

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvements
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(cuda): add Tensor Core support for matmul

Implements mixed-precision matrix multiplication using
CUDA Tensor Cores for 2x throughput improvement.

Fixes #45
```

```
fix(tt-metal): correct NOC routing for sparse transfers

Fixes channel rotation causing dropped packets when
sparse mode is enabled.

Closes #67
```

### Hardware Testing

#### TT-Metal (Greyskull)

Test on actual hardware when possible:
```bash
./scripts/setup_greyskull.sh
make greyskull
./build/greyskull/dynamic_cortex_demo_ttmetal
```

If hardware unavailable, ensure code compiles and document this.

#### CUDA (RTX 4080)

Test on compatible NVIDIA GPU:
```bash
./scripts/setup_rtx4080.sh
make cuda
./build/cuda/dynamic_cortex_demo_cuda
```

Minimum: GTX 1080 (Pascal) for testing

### Performance Benchmarks

For performance-related changes, include benchmarks:

```bash
./scripts/run_benchmarks.sh --hardware greyskull > bench_before.txt
# Apply changes
./scripts/run_benchmarks.sh --hardware greyskull > bench_after.txt
diff bench_before.txt bench_after.txt
```

## Pull Request Process

1. **PR Title**: Use conventional commit format
2. **Description**: Explain what, why, and how
3. **Testing**: Describe tests performed
4. **Checklist**:
   - [ ] Code follows project style
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests pass
   - [ ] No merge conflicts

4. **Review**: Address reviewer feedback promptly
5. **Merge**: Maintainer will merge after approval

## Project Structure

```
tt-metal/
├── include/          # Public headers
├── src/
│   ├── tt-metal/     # TT-Metal implementation
│   └── cuda/         # CUDA implementation
├── docs/             # Documentation
├── tests/            # Test suite
├── scripts/          # Build/setup scripts
└── configs/          # Hardware configs
```

## Development Setup

### Prerequisites

**For TT-Metal:**
- Ubuntu 20.04/22.04
- TT-Metal SDK (install via `scripts/setup_greyskull.sh`)
- CMake >= 3.18
- GCC >= 9.0

**For CUDA:**
- Ubuntu 20.04/22.04
- CUDA Toolkit >= 12.0
- CMake >= 3.18
- GCC >= 9.0

### Build Instructions

See [README.md](README.md#building) for detailed build instructions.

## Areas for Contribution

### High Priority

- [ ] Additional kernel implementations
- [ ] Performance optimizations
- [ ] Extended test coverage
- [ ] Documentation improvements
- [ ] Example programs

### Medium Priority

- [ ] Python bindings
- [ ] Jupyter notebook examples
- [ ] Visualization tools
- [ ] Additional hardware backends
- [ ] Benchmark suite expansion

### Low Priority

- [ ] GUI tools
- [ ] Cloud deployment guides
- [ ] Docker Compose orchestration

## Community

- **Discussions**: https://github.com/danindiana/GRYPHGEN/discussions
- **Issues**: https://github.com/danindiana/GRYPHGEN/issues

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Dynamic Cortex!
