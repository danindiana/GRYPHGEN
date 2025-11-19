# Contributing to GRYPHGEN Perl Pipes

Thank you for your interest in contributing to GRYPHGEN Perl Pipes! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project follows a Code of Conduct that all contributors are expected to adhere to. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/GRYPHGEN.git
   cd GRYPHGEN/perl_pipes
   ```
3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/danindiana/GRYPHGEN.git
   ```

## Development Setup

### Prerequisites

- Perl 5.36 or later
- cpanminus (`cpanm`)
- NVIDIA GPU (optional, for GPU-specific features)
- Docker (optional, for container testing)

### Install Dependencies

```bash
# Install Perl dependencies
make install-deps

# Install development dependencies
cpanm --installdeps . --with-develop

# Verify setup
make check-system
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
prove -v tests/01-basic.t

# Run with coverage
make coverage
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

Example:
```bash
git checkout -b feature/add-websocket-support
```

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Detailed explanation of changes (if needed)

- Bullet points for specifics
- Related issue numbers

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

**Example:**
```
feat(ipc): add WebSocket support for remote model communication

- Implement WebSocket server in lib/WebSocket.pm
- Add --websocket flag to model communication scripts
- Update tests to cover WebSocket functionality

Closes #45
```

## Testing

### Writing Tests

Tests should:
- Be placed in `tests/` directory
- Use `Test::More` framework
- Have `.t` extension
- Cover both success and failure cases

Example test:
```perl
#!/usr/bin/env perl
use v5.36;
use Test::More tests => 3;
use Test::Exception;

use_ok('YourModule');

# Test success case
lives_ok {
    YourModule::some_function();
} "Function executes without errors";

# Test failure case
dies_ok {
    YourModule::some_function(invalid => 'params');
} "Function dies with invalid parameters";

done_testing();
```

### Test Coverage

Aim for:
- 80%+ code coverage
- All public functions tested
- Edge cases covered
- Error handling verified

Check coverage:
```bash
make coverage
open cover_db/coverage.html
```

## Code Style

### Perl Style Guidelines

1. **Use Modern Perl:**
   ```perl
   use v5.36;
   use warnings;
   use autodie;
   ```

2. **Naming Conventions:**
   - `snake_case` for functions and variables
   - `PascalCase` for modules
   - `UPPER_CASE` for constants

3. **Indentation:**
   - 4 spaces (no tabs)
   - Consistent bracing style

4. **Documentation:**
   - POD documentation for all public functions
   - Inline comments for complex logic
   - Function signatures in POD

5. **Error Handling:**
   ```perl
   eval {
       risky_operation();
   };
   if ($@) {
       log_error("Operation failed: $@");
       die $@;
   }
   ```

### Linting

Run Perl::Critic before committing:

```bash
make lint

# Or manually
perlcritic bin/ lib/
```

### Formatting

Format code with Perl::Tidy:

```bash
make format

# Or manually
perltidy -b -bext='/' yourfile.pl
```

## Submitting Changes

### Pull Request Process

1. **Update from upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure tests pass:**
   ```bash
   make test
   ```

3. **Update documentation:**
   - Update README.md if adding features
   - Add POD documentation
   - Update CHANGELOG.md

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature
   ```

5. **Create Pull Request:**
   - Use descriptive title
   - Reference related issues
   - Explain changes in detail
   - Add screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated

## Related Issues
Fixes #(issue number)
```

## Reporting Bugs

### Before Submitting

1. Check existing issues
2. Verify bug on latest version
3. Collect relevant information

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Perl version: [e.g., 5.36]
- GPU: [e.g., RTX 4080]
- Version: [e.g., 2.0.0]

**Logs**
```
Paste relevant log output
```

**Additional context**
Any other relevant information.
```

## Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Screenshots, examples, etc.

**Implementation ideas**
Any thoughts on how to implement this.
```

## Development Guidelines

### Adding New Scripts

1. **Create in `bin/` directory**
2. **Add shebang:**
   ```perl
   #!/usr/bin/env perl
   ```
3. **Use modern Perl features**
4. **Include POD documentation**
5. **Add to Makefile if needed**
6. **Create corresponding tests**
7. **Update README.md**

### Adding New Modules

1. **Create in `lib/` directory**
2. **Follow naming conventions**
3. **Full POD documentation**
4. **Export functions properly**
5. **Add to cpanfile**
6. **Comprehensive tests**

### Adding Examples

1. **Create in `examples/` directory**
2. **Make executable**
3. **Add usage comments**
4. **Test thoroughly**
5. **Document in README**

## Release Process

(For maintainers)

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Create git tag
5. Build Docker image
6. Update documentation
7. Create GitHub release

## Questions?

- Open an issue with your question
- Tag it with `question` label
- Provide context and what you've tried

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to GRYPHGEN Perl Pipes! 🎉
