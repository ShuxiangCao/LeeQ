# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeeQ is a Python package for orchestrating quantum computing experiments with a focus on superconducting circuits. It provides a comprehensive framework for calibration, characterization, and experiment orchestration on both simulated and real quantum hardware.

## Development Commands

### Environment Activation
```bash
# IMPORTANT: Always activate the Python environment before running LeeQ-related code
# The environment is located in the LeeQ folder
source /home/coxious/Projects/LeeQ/venv/bin/activate

# Or if using the symlink from VILA_training folder:
source /home/coxious/Projects/VILA_training/LeeQ/venv/bin/activate

# Alternatively, use the Python binary directly:
/home/coxious/Projects/VILA_training/LeeQ/venv/bin/python
```

### Working with Symlinks
When working in the VILA_training folder, use the symlinks to access related projects:
- `LeeQ` -> Links to ../LeeQ
- `VILA-Internal` -> Links to ../VILA-Internal  
- `leeq-nvidia-deployment` -> Links to ../leeq-nvidia-deployment

These symlinks allow you to run code that depends on LeeQ without changing directories.

### Installation
```bash
# Using Poetry (preferred)
poetry install

# Using pip
pip install -r requirements.txt
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/path/to/test_file.py

# Run specific test
pytest tests/path/to/test_file.py::test_function_name
```

### Linting
```bash
# Run linting checks
bash ./ci_scripts/lint.sh

# Or directly with flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Documentation
```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Code Architecture

### Core Structure
The codebase follows a modular architecture with clear separation of concerns:

1. **Base Classes** (`leeq/core/base.py`): All LeeQ objects inherit from `LeeQObject`, which extends `LoggableObject` from the integrated leeq.chronicle module for persistence and tracking.

2. **Quantum Elements** (`leeq/core/elements/`):
   - `qubit.py`: Basic qubit implementations
   - `transmon.py`: Transmon-specific implementations
   - Elements handle their own calibration data and parameters

3. **Execution Engine** (`leeq/core/engine/`):
   - Manages experiment execution flow
   - Handles measurement collection and data persistence
   - Integrates with hardware backends

4. **Experiment Framework** (`leeq/experiments/`):
   - Pre-built experiment types for common tasks
   - Calibration experiments (Rabi, T1, T2, etc.)
   - Characterization experiments (tomography, RB)
   - All experiments follow a consistent interface pattern

5. **Compiler** (`leeq/compiler/`):
   - Translates high-level operations to hardware pulses
   - LBNL QubiC hardware support
   - Pulse shape generation and optimization

6. **Theory/Simulation** (`leeq/theory/`):
   - Clifford gate implementations
   - Simulation backends (NumPy and QuTiP)
   - Optimal control (GRAPE) implementations

### Key Design Patterns

1. **Dependency Management**: The project uses both Poetry and pip/requirements.txt. Git-based dependencies (k_agents, MinimalLLM) are managed through direct GitHub references. The labchronicle functionality is now integrated as leeq.chronicle.

2. **Hardware Abstraction**: The `setups` module provides a clean interface between experiments and hardware, allowing easy switching between simulation and real devices.

3. **AI/ML Integration**: The `leeq/utils/ai/` module integrates LLMs for experiment generation and translation, using the k_agents framework.

4. **Data Persistence**: Uses the integrated leeq.chronicle module (formerly labchronicle) for automatic logging and tracking of all experiments and results.

### Testing Strategy

- Unit tests mirror the source structure in `/tests/`
- Integration tests in `/tests/integration_tests/`
- Benchmarking suites in `/benchmark/` for performance testing
- All new features should include corresponding tests
- Use pytest fixtures for common test setups

### Important Dependencies

- **numpy < 2.0.0**: Version constraint for compatibility
- **leeq.chronicle**: Integrated experiment logging and persistence module (formerly labchronicle external dependency)
- **k_agents**: For AI/ML experiment generation (GitHub dependency)
- **MinimalLLM**: For LLM integration (GitHub dependency)
- **qutip**: For quantum simulations
- **plotly/dash**: For interactive visualization

### Common Development Tasks

When implementing new experiments:
1. Inherit from appropriate base class in `leeq/experiments/`
2. Follow existing naming conventions and patterns
3. Include proper docstrings and type hints
4. Add corresponding tests in `/tests/experiments/`

When working with hardware interfaces:
1. Check existing setups in `leeq/setups/`
2. Use the abstraction layer rather than direct hardware calls
3. Test with simulation backend first

When modifying core functionality:
1. Be careful with changes to `LeeQObject` as it affects all components
2. Ensure backward compatibility
3. Update relevant documentation in `/docs/`