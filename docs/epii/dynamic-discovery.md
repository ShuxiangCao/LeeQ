# Dynamic Experiment Discovery

## Overview

The EPII system now features automatic discovery of all experiments that have `EPII_INFO` metadata defined. This eliminates the need to manually register experiments in the router.

## How It Works

1. **Automatic Discovery**: On startup, the `ExperimentRouter` automatically scans all LeeQ experiment modules
2. **EPII_INFO Detection**: Any experiment class with an `EPII_INFO` attribute and a `run` method is automatically registered
3. **Categorized Naming**: Experiments are organized by category (e.g., `calibrations.NormalisedRabi`, `characterizations.SimpleT1`)
4. **Backward Compatibility**: Old simple names (e.g., `rabi`, `t1`) are maintained as aliases

## Adding New Experiments

To make a new experiment available through EPII:

1. Add an `EPII_INFO` attribute to your experiment class:
```python
class MyNewExperiment(ExperimentBase):
    EPII_INFO = {
        'experiment': 'my_new_experiment',
        'description': 'Description of what this experiment does',
        'category': 'calibrations',  # or characterizations, etc.
        'version': '1.0.0'
    }
    
    def run(self, ...):
        """Run the experiment."""
        pass
```

2. Place the experiment in the appropriate LeeQ module under `leeq.experiments.builtin`

3. The experiment will be automatically discovered and available through EPII with no additional configuration needed

## Experiment Naming Convention

Experiments are named using a dot notation:
- `category.ExperimentClassName`
- Examples:
  - `calibrations.NormalisedRabi`
  - `characterizations.SimpleT1`
  - `multi_qubit_gates.ConditionalStarkTuneUp`
  - `tomography.SingleQubitStateTomography`

## Available Categories

- `calibrations`: Basic calibration experiments
- `characterizations`: Qubit characterization experiments  
- `multi_qubit_gates`: Multi-qubit gate experiments
- `tomography`: State and process tomography
- `hamiltonian_tomography`: Hamiltonian reconstruction
- `optimal_control`: Optimal control experiments

## Listing Available Experiments

```python
from leeq.epii.experiments import ExperimentRouter

router = ExperimentRouter()
experiments = router.list_experiments()
print(f"Found {len(experiments)} experiments")

# List by category
for name in sorted(experiments.keys()):
    if name.startswith('calibrations.'):
        print(f"  - {name}")
```

## Metadata Access

Each experiment's metadata can be accessed:

```python
router = ExperimentRouter()
info = router.get_experiment_info('calibrations.NormalisedRabi')
print(info['epii_info'])  # EPII metadata
print(info['run_docstring'])  # Run method documentation
```