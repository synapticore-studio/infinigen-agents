# GitHub Copilot Instructions for Infinigen

This file contains repository-specific instructions for GitHub Copilot to ensure consistent and accurate assistance when working with the Infinigen codebase.

## Project Overview

Infinigen is a research project for generating infinite photorealistic worlds using procedural generation. It combines computer vision, computer graphics, and machine learning to create realistic 3D environments for various applications.

### Tech Stack

- **Python**: 3.11 (strict requirement, see `pyproject.toml`)
- **Blender**: 4.5.3+ via bpy (3D modeling and rendering)
- **Core Libraries**: 
  - NumPy, SciPy, Pandas for numerical computing
  - OpenCV for image processing
  - Trimesh for 3D mesh operations
  - gin_config for configuration management
- **Build System**: 
  - `uv` for Python dependency management
  - `Makefile` for terrain compilation, docker builds
  - setuptools for package building
- **Testing**: pytest with custom markers
- **Linting**: ruff (configured for E9, F63, F7, F82 checks and copyright statements)
- **CI/CD**: GitHub Actions (see `.github/workflows/`)

## Architecture

The codebase is organized into several key modules:

- `infinigen/` - Main package containing procedural generation code
  - `assets/` - Asset generation (plants, objects, creatures)
  - `core/` - Core utilities and base classes
  - `terrain/` - Terrain generation (compiled C++ extensions)
  - `nodes/` - Blender node transpiler for converting node graphs to Python
- `infinigen_examples/` - Example scenes and usage patterns
- `scripts/` - Build and installation scripts
- `tests/` - Test suite organized by feature
- `docs/` - User documentation (installation, tutorials, configuration)

## Development Workflow

### Setup and Installation

1. Use `uv` for dependency management: `uv sync --extra dev`
2. Compile optional components using Makefile targets:
   - `make terrain` - Compile terrain generation
   - `make customgt` - Compile OpenGL ground truth
   - `make flip_fluids` - Compile fluid simulation

### Building and Testing

- **Lint**: `uv run ruff check .`
  - Critical errors: `uv run ruff check --select=E9,F63,F7,F82 .`
  - Copyright check: `uv run ruff check --preview --select CPY001 .`
- **Test**: `uv run pytest tests`
  - Skip CI tests locally: `uv run pytest tests -k 'not skip_for_ci'`
- **Run**: See `docs/HelloWorld.md` and `docs/HelloRoom.md` for example commands

### Key Commands

```bash
# Install dependencies
uv sync --extra dev

# Lint code
uv run ruff check .

# Run tests
uv run pytest tests -k 'not skip_for_ci'

# Generate a nature scene
python -m infinigen.datagen.manage_jobs --output_folder outputs/hello_world --num_scenes 1 --pipeline_configs local_16GB monocular_video singleview --configs simple

# Generate an indoor scene
python -m infinigen.datagen.manage_jobs --output_folder outputs/hello_room --num_scenes 1 --pipeline_configs local_16GB monocular_video singleview --configs simple_indoors
```

## Coding Standards

### Style Guidelines

- **Python Version**: Always use Python 3.11 features and syntax
- **Type Hints**: Use type hints where they improve clarity
- **Imports**: Follow ruff's import ordering
- **Docstrings**: Use clear docstrings for public APIs
- **Naming**: Follow PEP 8 conventions
  - snake_case for functions, variables, modules
  - PascalCase for classes
  - UPPER_CASE for constants

### Code Organization

- Keep asset generators in appropriate `infinigen/assets/` subdirectories
- Use `gin_config` for configuration management
- Follow existing patterns for Blender node transpilation
- Place tests in parallel structure under `tests/`

### Common Patterns

1. **Asset Generation**: Inherit from base factory classes
2. **Blender Nodes**: Use `infinigen/nodes/node_transpiler/` for node-to-code conversion
3. **Configuration**: Use gin config files for scene configuration
4. **Testing**: Mark long-running tests with `@pytest.mark.skip_for_ci`

## Model Context Protocol (MCP) Integration

### Context7 MCP Server

**Always use Context7 to retrieve current documentation when working with:**
- Blender Python API (bpy)
- NumPy, SciPy, and other scientific libraries
- OpenCV operations
- Any external framework or library

**Automatically invoke Context7 tools without being asked** when:
- Writing code that uses external libraries
- Answering questions about APIs or library features
- Debugging issues that might relate to API changes

**Prefer current documentation from Context7 over model training data** if there is a conflict, especially for:
- Blender API (frequently updated)
- Python library versions (we use specific versions)
- Deprecated patterns or functions

### Serena MCP Server

Use Serena for advanced codebase analysis when available:
- Finding all call sites of a function across the codebase
- Analyzing dependency graphs and module relationships
- Running project-specific build diagnostics
- Performing structural analysis of procedural generators

If Serena is not available, fall back to standard grep/glob tools.

## Repository-Specific Guidelines

### When Contributing New Assets

1. Check existing asset implementations in `infinigen/assets/`
2. Follow the factory pattern used by similar assets
3. Add appropriate tests in `tests/assets/`
4. Document parameters and usage in docstrings
5. Consider adding examples in `infinigen_examples/`

### When Modifying Core Systems

1. Understand dependencies using module imports
2. Run full test suite to catch regressions
3. Update documentation if behavior changes
4. Consider backward compatibility

### When Working with Blender Nodes

1. Use `infinigen/nodes/node_transpiler/dev_script.py` for node-to-code conversion
2. Test node graphs in Blender before transpiling
3. Document the original node graph source
4. Follow existing transpiled code patterns

### Documentation Requirements

- Update relevant docs in `docs/` for user-facing changes
- Update README.md for major feature additions
- Add code comments for complex procedural algorithms
- Include visual examples where appropriate

## Common Issues and Solutions

### Import Errors

- Ensure `uv sync` has been run
- Check Python version is 3.11
- Verify optional components are compiled if needed

### Blender Issues

- Ensure bpy is installed: `uv run python -c "import bpy"`
- Check Blender version compatibility (4.5.3+)
- Some features require X11/display server

### Test Failures

- Use `pytest -v` for verbose output
- Check for `skip_for_ci` markers
- Review test logs in `outputs/MYJOB/MYSEED/logs/`

## Getting Help

- **Issues**: Use GitHub Issues with appropriate template
- **Documentation**: Check `docs/` directory
- **Examples**: See `infinigen_examples/` and `docs/Hello*.md`
- **Community**: Follow [@PrincetonVL](https://twitter.com/PrincetonVL) on Twitter

## References

- [Installation Guide](../docs/Installation.md)
- [Configuration Guide](../docs/ConfiguringInfinigen.md)
- [Asset Implementation Guide](../docs/ImplementingAssets.md)
- [Hello World Tutorial](../docs/HelloWorld.md)
- [Hello Room Tutorial](../docs/HelloRoom.md)
