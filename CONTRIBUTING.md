# Contributing to Infinigen

Thank you for your interest in contributing to Infinigen! This guide will help you get started with contributing to our procedural generation system.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Getting Help](#getting-help)

## Getting Started

Infinigen is a research project for generating infinite photorealistic worlds using procedural generation. Before contributing, please:

1. Read the [README.md](README.md) to understand the project
2. Review our [Installation Guide](docs/Installation.md)
3. Try running the [Hello World](docs/HelloWorld.md) and [Hello Room](docs/HelloRoom.md) examples
4. Explore the [documentation](docs/) to understand the system architecture

## Development Setup

### Prerequisites

- **Python 3.11** (strict requirement)
- **Git** for version control
- **uv** for dependency management
- Sufficient disk space for Blender and dependencies (several GB)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/princeton-vl/infinigen.git
   cd infinigen
   ```

2. **Install dependencies**:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install Infinigen with development dependencies
   uv sync --extra dev
   ```

3. **Compile optional components** (if needed):
   ```bash
   # For terrain generation
   make terrain
   
   # For OpenGL ground truth
   make customgt
   
   # For fluid simulations
   make flip_fluids
   ```

4. **Verify installation**:
   ```bash
   # Run tests (excluding long CI tests)
   uv run pytest tests -k 'not skip_for_ci'
   
   # Try a simple scene generation
   python -m infinigen.datagen.manage_jobs --output_folder outputs/test --num_scenes 1 --configs simple
   ```

## How to Contribute

We welcome various types of contributions:

### Types of Contributions

1. **Code Contributions**
   - Bug fixes
   - New procedural generators
   - Performance improvements
   - New features

2. **Asset Contributions**
   - New procedural assets (objects, plants, creatures)
   - Improved realism for existing assets
   - Blender node graphs

3. **Documentation**
   - Tutorials and guides
   - API documentation
   - Example scenes
   - Troubleshooting tips

4. **Testing**
   - Test coverage improvements
   - Bug reports with reproducible examples
   - Performance benchmarks

### Contribution Workflow

1. **Find or create an issue**:
   - Check [existing issues](https://github.com/princeton-vl/infinigen/issues)
   - Create a new issue for bugs or feature requests
   - Use appropriate issue templates

2. **Fork and create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Make your changes**:
   - Write clean, documented code
   - Follow existing code patterns
   - Add tests for new functionality
   - Update documentation if needed

4. **Test your changes**:
   ```bash
   # Lint your code
   uv run ruff check .
   
   # Run tests
   uv run pytest tests
   
   # Test your specific changes
   uv run pytest tests/path/to/your_test.py
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```
   - Write clear, descriptive commit messages
   - Reference issue numbers (e.g., "Fix #123: description")

6. **Push and create a pull request**:
   ```bash
   git push origin your-branch-name
   ```
   - Go to GitHub and create a pull request
   - Fill out the PR template
   - Link related issues

7. **Respond to review feedback**:
   - Address reviewer comments
   - Update your PR as needed
   - Be patient and respectful

## Code Standards

### Python Style Guide

- **Python Version**: 3.11 only
- **Style**: Follow PEP 8
- **Linting**: Use ruff (configured in `pyproject.toml`)
- **Type Hints**: Use type hints for clarity where appropriate
- **Docstrings**: Document public functions and classes

### Code Quality Checks

Before submitting a PR, ensure your code passes:

```bash
# Check for critical errors
uv run ruff check --select=E9,F63,F7,F82 .

# Full lint check
uv run ruff check .

# Copyright statement check
uv run ruff check --preview --select CPY001 .
```

### Asset Development Guidelines

When creating new procedural assets:

1. **Study existing assets** in `infinigen/assets/`
2. **Follow the factory pattern** used by similar assets
3. **Use gin_config** for configurable parameters
4. **Add tests** in `tests/assets/`
5. **Document parameters** in docstrings
6. **Provide examples** if the asset is complex

### Blender Node Development

When working with Blender nodes:

1. Create your node graph in Blender
2. Use `infinigen/nodes/node_transpiler/dev_script.py` to convert to Python
3. Test the transpiled code
4. Document the original node graph
5. Follow existing transpiled code patterns

## Testing

### Running Tests

```bash
# Run all tests (excluding CI-only tests)
uv run pytest tests -k 'not skip_for_ci'

# Run tests with verbose output
uv run pytest tests -v

# Run specific test file
uv run pytest tests/core/test_something.py

# Run tests matching a pattern
uv run pytest tests -k "test_asset"
```

### Writing Tests

- Place tests in the appropriate `tests/` subdirectory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Test both normal and edge cases
- Mark long-running tests: `@pytest.mark.skip_for_ci`
- Use fixtures from `conftest.py` when appropriate

### Test Structure

```python
import pytest

def test_feature_works_correctly():
    """Test that the feature produces expected output."""
    # Arrange
    input_data = setup_test_data()
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result == expected_output

@pytest.mark.skip_for_ci
def test_expensive_computation():
    """Test that requires significant time/resources."""
    # Long-running test code
    pass
```

## Documentation

### Documentation Structure

- **User Guides**: `docs/` directory
- **API Documentation**: Docstrings in code
- **Examples**: `infinigen_examples/` directory
- **Tutorials**: `docs/Hello*.md` files

### Updating Documentation

When making changes that affect users:

1. Update relevant documentation files in `docs/`
2. Add docstrings to new functions and classes
3. Update README.md for major features
4. Add examples to demonstrate new functionality

### Documentation Style

- Use clear, concise language
- Include code examples
- Provide command-line examples
- Link to related documentation
- Add images/screenshots for visual features

## Getting Help

### Resources

- **Documentation**: Check the [docs/](docs/) directory first
- **Issues**: Search [GitHub Issues](https://github.com/princeton-vl/infinigen/issues)
- **Discussions**: Start a discussion for questions
- **Examples**: Review [infinigen_examples/](infinigen_examples/)

### Asking for Help

When asking for help, please include:

1. **Your setup**: OS, Python version, GPU info
2. **What you tried**: Exact commands and steps
3. **What happened**: Full error messages and logs
4. **What you expected**: Desired behavior
5. **Code version**: Commit hash or branch

Example:
```
**Setup**: Ubuntu 22.04, Python 3.11, NVIDIA RTX 3080
**Command**: python -m infinigen.datagen.manage_jobs --output_folder outputs/test --num_scenes 1
**Error**: [paste full error and stack trace]
**Expected**: Scene should generate successfully
**Version**: commit abc123def
```

### Debug Mode

Run commands with `--debug` flag for more detailed output:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/test --num_scenes 1 --debug
```

## Community Guidelines

- Be respectful and constructive
- Help others when you can
- Share knowledge and learnings
- Credit sources and collaborators
- Follow our code of conduct

## Recognition

Contributors are acknowledged in:
- Commit history
- Pull request discussions
- Project releases
- Research papers (for significant contributions)

## Questions?

If you have questions about contributing:

1. Check this guide and the [documentation](docs/)
2. Search [existing issues](https://github.com/princeton-vl/infinigen/issues)
3. Open a new issue with the "ask-for-help" template
4. Join discussions on GitHub

Thank you for contributing to Infinigen! Your contributions help advance research in computer vision and procedural generation.
