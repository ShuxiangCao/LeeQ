# LeeQ Notebooks

This directory contains interactive Jupyter notebooks for learning and working with LeeQ.

## Directory Structure

### `/tutorials/` - Progressive Learning Path
Interactive tutorials that build on each other to teach LeeQ concepts:

1. **01_basics.ipynb** - Introduction to LeeQ concepts and simulation
2. **02_single_qubit.ipynb** - Single qubit experiments and calibration
3. **03_multi_qubit.ipynb** - Two-qubit gates and entanglement
4. **04_calibration.ipynb** - Complete calibration workflows
5. **05_ai_integration.ipynb** - AI-assisted experiment generation

### `/examples/` - Feature-Specific Examples
Focused examples demonstrating specific LeeQ features:

- **rabi_experiments.ipynb** - Comprehensive Rabi experiment examples
- **t1_t2_measurements.ipynb** - Coherence time measurements
- **tomography.ipynb** - Quantum state and process tomography
- **randomized_benchmarking.ipynb** - Gate fidelity characterization
- **custom_experiments.ipynb** - Building custom experiments

### `/workflows/` - Complete Procedures
End-to-end workflows for common tasks:

- **qubit_characterization.ipynb** - Complete single-qubit characterization
- **two_qubit_calibration.ipynb** - Two-qubit gate calibration workflow
- **daily_calibration.ipynb** - Automated daily calibration procedures
- **experiment_analysis.ipynb** - Data analysis and visualization workflows

### Legacy Directories
- **Agent/** - AI agent examples
- **RealSystem/** - Real hardware examples
- **SimulatedSystem/** - Basic simulation examples

## Getting Started

1. **New to LeeQ?** Start with the tutorials in order: `tutorials/01_basics.ipynb`
2. **Need specific examples?** Check the `examples/` directory
3. **Want complete workflows?** See the `workflows/` directory

## Requirements

All notebooks are designed to work with LeeQ's simulation backends by default, so no special hardware is required.

## Testing

To test notebook syntax and structure:
```bash
python scripts/test_notebooks.py
```

To test notebook execution (requires nbval):
```bash
pip install nbval
pytest notebooks/ --nbval --nbval-lax -v
```

## Contributing

When adding new notebooks:
1. Follow the existing structure and naming conventions
2. Include clear learning objectives and prerequisites
3. Use simulation backends for examples when possible
4. Add proper Chronicle logging integration
5. Test your notebooks before committing