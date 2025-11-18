# Contributing to MCP Reliability System

First off, thank you for considering contributing to the MCP Reliability System! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include logs and error messages** if applicable.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Explain why this enhancement would be useful** to most users.

### Pull Requests

* Fill in the required template
* Follow the Haskell style guide
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Development Setup

### Prerequisites

* GHC 9.4.8 or higher
* Cabal 3.0 or higher
* Docker (for containerized development)

### Setting Up Your Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/GRYPHGEN.git
   cd GRYPHGEN/MCP_SERVER/Sept_16_2025/mcp-reliability-system
   ```

3. Install dependencies:
   ```bash
   cabal update
   cabal build --dependencies-only
   ```

4. Build the project:
   ```bash
   cabal build
   ```

5. Run tests:
   ```bash
   cabal test
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards below

3. Run the test suite:
   ```bash
   make test
   ```

4. Run the linter:
   ```bash
   make lint
   ```

5. Format your code:
   ```bash
   make format
   ```

6. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "feat: add new circuit breaker configuration option"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request

## Coding Standards

### Haskell Style Guide

* Follow standard Haskell conventions
* Use meaningful variable and function names
* Add type signatures for all top-level functions
* Keep functions small and focused
* Write documentation for exported functions
* Use hlint and ormolu for code formatting

### Naming Conventions

* **Modules**: Use CamelCase (e.g., `MCP.Reliability.CircuitBreaker`)
* **Functions**: Use camelCase (e.g., `executeWithBreaker`)
* **Types**: Use CamelCase (e.g., `CircuitBreakerConfig`)
* **Constants**: Use camelCase (e.g., `defaultTimeout`)

### Documentation

* Add Haddock comments for all exported functions and types
* Include examples in documentation when helpful
* Keep documentation up to date with code changes

Example:
```haskell
-- | Execute an action with circuit breaker protection.
--
-- This function wraps an IO action with circuit breaker logic,
-- preventing cascading failures when a service is unavailable.
--
-- Example:
--
-- > cb <- createCircuitBreaker defaultConfig
-- > result <- executeWithBreaker cb $ callExternalService
executeWithBreaker :: CircuitBreaker -> IO a -> IO (Either CircuitBreakerError a)
```

### Testing

* Write tests for all new functionality
* Use property-based testing where appropriate (QuickCheck)
* Aim for high test coverage (>80%)
* Include both unit tests and integration tests
* Test error cases and edge conditions

Example test structure:
```haskell
spec :: Spec
spec = describe "CircuitBreaker" $ do
  it "opens after threshold failures" $ do
    cb <- createCircuitBreaker $ defaultConfig { failureThreshold = 3 }
    -- Test implementation

  it "allows requests in closed state" $ do
    cb <- createCircuitBreaker defaultConfig
    -- Test implementation
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

* `feat:` A new feature
* `fix:` A bug fix
* `docs:` Documentation only changes
* `style:` Changes that don't affect the meaning of the code
* `refactor:` A code change that neither fixes a bug nor adds a feature
* `perf:` A code change that improves performance
* `test:` Adding missing tests or correcting existing tests
* `chore:` Changes to the build process or auxiliary tools

Examples:
```
feat: add exponential backoff to circuit breaker
fix: resolve race condition in cache invalidation
docs: update README with new configuration options
test: add property tests for parameter validation
```

## Performance Considerations

* Profile your changes if they may impact performance
* Run benchmarks before and after:
  ```bash
  make bench
  ```
* Document any performance implications in your PR

## Security

* Never commit sensitive data (API keys, passwords, etc.)
* Follow security best practices for input validation
* Report security vulnerabilities privately (see [SECURITY.md](SECURITY.md))

## Questions?

Feel free to:
* Open an issue with your question
* Start a discussion in GitHub Discussions
* Contact the maintainers

## Recognition

Contributors will be recognized in:
* The project README
* Release notes for their contributions
* The project's contributor graph

Thank you for contributing to MCP Reliability System! 🎉
