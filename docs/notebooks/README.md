# LeeQ Notebook Documentation

## Overview
This directory contains comprehensive Jupyter notebooks demonstrating LeeQ quantum experiment framework capabilities, from basic concepts to advanced calibration workflows.

## Completed Notebooks

### Tutorial Notebooks (`tutorials/`)
Progressive learning path for quantum experiment automation:

1. **01_basics.ipynb** ✅
   - Core LeeQ concepts and architecture
   - Chronicle logging integration
   - Virtual transmon configuration
   - Basic simulation setup

2. **02_single_qubit.ipynb** ✅
   - Single-qubit experiments (Rabi, Ramsey, T1/T2)
   - Complete calibration workflow
   - Parameter optimization
   - Coherence characterization

3. **03_multi_qubit.ipynb** ✅
   - Two-qubit gate implementations
   - Entanglement creation and Bell states
   - Crosstalk characterization
   - Multi-qubit calibration

4. **04_calibration.ipynb** ✅
   - Automated calibration workflows
   - Measurement fidelity optimization
   - Parameter persistence with Chronicle
   - Calibration data management

5. **05_ai_integration.ipynb** ⏸️
   - (Placeholder for future AI integration features)

### Shared Utilities (`shared/`)
Reusable components for notebook infrastructure:

- **setup_templates.py**: Standardized simulation configurations
- **experiment_helpers.py**: Common experiment patterns
- **validation_utils.py**: Notebook testing utilities

### Validation Scripts (`../../scripts/`)
Testing and validation infrastructure:

- **validate_notebooks.py**: Notebook execution validation
- **check_outputs.py**: Experiment result validation
- **notebook_test_runner.py**: Automated test orchestration

## Usage

### Running Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Navigate to docs/notebooks/tutorials/
# Open notebooks in sequence (01 → 02 → 03 → 04)
```

### Validating Notebooks
```bash
# Validate single notebook
python scripts/validate_notebooks.py docs/notebooks/tutorials/01_basics.ipynb

# Validate all tutorials
python scripts/validate_notebooks.py docs/notebooks/tutorials/*.ipynb

# Run comprehensive test suite
python scripts/notebook_test_runner.py --all
```

## Key Features Demonstrated

### Quantum Device Control
- Virtual transmon configuration
- Pulse parameter optimization
- Gate calibration sequences
- Measurement discrimination

### Experiment Workflows
- Automated calibration procedures
- Error handling and recovery
- Parameter persistence
- Data analysis and visualization

### LeeQ Framework Integration
- Chronicle logging system
- Experiment management
- High-level simulation backend
- Built-in calibration experiments

## Prerequisites

### Required Knowledge
- Basic quantum computing concepts
- Python programming
- Jupyter notebook usage

### Required Packages
- LeeQ framework
- NumPy, SciPy
- Plotly for visualization
- IPython for display

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Foundation | ✅ Complete | Core tutorials (01-02), shared utilities, validation |
| Phase 2: Advanced | ✅ Complete | Multi-qubit (03), calibration (04) notebooks |
| Phase 3: Examples | ⏸️ Skipped | Per user request |
| Phase 4: Workflows | ⏸️ Skipped | Per user request |

## Architecture

```
notebooks/
├── tutorials/           # Progressive learning notebooks
│   ├── 01_basics       # Core concepts
│   ├── 02_single_qubit # Single-qubit experiments
│   ├── 03_multi_qubit  # Two-qubit operations
│   └── 04_calibration  # Complete workflows
├── shared/             # Reusable utilities
│   ├── setup_templates # Simulation configurations
│   └── experiment_helpers # Common patterns
└── README.md          # This file
```

## Best Practices

1. **Sequential Learning**: Complete notebooks in order for best understanding
2. **Chronicle Logging**: Always initialize Chronicle for data persistence
3. **Parameter Validation**: Verify calibrated parameters are physically reasonable
4. **Error Handling**: Use try-except blocks for robust experiment execution
5. **Visualization**: Use Plotly for interactive data exploration

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure LeeQ is properly installed
- Check Python path includes project root

**Execution Failures**
- Verify simulation setup is initialized
- Check Chronicle logging is started
- Ensure reasonable parameter ranges

**Performance Issues**
- Reduce sweep points for faster execution
- Use simpler experiments for testing
- Enable caching where available

## Contributing

To add new notebooks:
1. Follow established patterns from existing notebooks
2. Use shared utilities for consistency
3. Include comprehensive markdown documentation
4. Validate with testing scripts
5. Update this README

## Support

For issues or questions:
- Check notebook markdown cells for detailed explanations
- Review docstrings in experiment classes
- Consult LeeQ API documentation
- Open issues in project repository

---

*Last Updated: August 2024*
*LeeQ Version: Compatible with current main branch*