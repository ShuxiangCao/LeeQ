# Contributing to LeeQ

Thank you for your interest in contributing to LeeQ! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/LeeQ.git
   cd LeeQ
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. Write your code following the project style guide
2. Add tests for new functionality
3. Update documentation as needed
4. Run tests locally before committing

### Testing Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run with coverage
pytest --cov=leeq
```

### Code Style

We use the following tools to maintain code quality:

- **flake8** for linting
- **black** for formatting (optional)
- **mypy** for type checking (where applicable)

Run linting checks:
```bash
bash ./ci_scripts/lint.sh
```

## Submitting Changes

### Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `style:` Code style changes

Example:
```bash
git commit -m "feat: add new calibration experiment for XY gate"
```

### Pull Request Process

1. Push your branch to your fork
2. Create a pull request against the main repository
3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
4. Wait for review and address feedback

## Code Guidelines

### Python Style

- Follow PEP 8
- Use descriptive variable names
- Add type hints where beneficial
- Document all public functions with numpy-style docstrings

### Documentation

- Update relevant documentation for any API changes
- Add docstrings to new classes and functions
- Include usage examples where appropriate

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Include integration tests for complex features

## Project Structure

```
leeq/
├── core/           # Core functionality
├── experiments/    # Experiment implementations
├── theory/         # Theoretical simulations
├── compiler/       # Hardware compilation
├── setups/         # Hardware setups
└── utils/          # Utility functions

tests/
├── unit/          # Unit tests
└── integration/   # Integration tests

docs/              # Documentation source
```

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in existing issues
- Reach out to maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.