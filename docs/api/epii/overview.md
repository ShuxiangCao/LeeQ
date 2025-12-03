# EPII v0.2.0 API Overview

The Experiment Programming Interface for Instruments (EPII) v0.2.0 provides a modern, backend-aware experiment discovery and execution system for LeeQ.

## Architecture Overview

### ExperimentRouter Class
The central component of EPII v0.2.0 is the `ExperimentRouter` class (`leeq.epii.experiments:17`), which provides:

- **Dynamic Experiment Discovery**: Automatically discovers experiments with `EPII_INFO` attributes
- **Backend-Aware Filtering**: Only exposes experiments compatible with current setup (simulation vs hardware)
- **Canonical Naming**: Uses module-qualified names instead of aliases (e.g., `calibrations.NormalisedRabi` instead of `rabi`)

### Key Features

#### 1. Dynamic Discovery (`leeq.epii.experiments:83-148`)
```python
router = ExperimentRouter(setup=my_setup)
experiments = router.list_experiments()  # Returns all discovered experiments
```

The router scans all modules in `leeq.experiments.builtin` and automatically registers classes with:
- `EPII_INFO` attribute (experiment metadata)
- `run` method (execution capability)
- Optional `run_simulated` method for simulation compatibility

#### 2. Backend-Aware Operation (`leeq.epii.experiments:36-51`)
When initialized with a `HighLevelSimulationSetup`, the router:
- Only includes experiments with `run_simulated` implementations (`leeq.epii.experiments:53-81`)
- Filters out hardware-only experiments
- Provides simulation-optimized experiment discovery

#### 3. Canonical Naming System (`leeq.epii.experiments:114-138`)
Experiments are now referenced by their canonical module-qualified names:
- `calibrations.NormalisedRabi` (was `rabi`)
- `characterizations.SimpleT1` (was `t1`)
- `calibrations.SimpleRamseyMultilevel` (was `ramsey`)

### API Methods

#### Core Router Methods
- `get_experiment(name: str)` - Retrieve experiment class by canonical name (`leeq.epii.experiments:172-185`)
- `list_experiments()` - Get all available experiments with descriptions (`leeq.epii.experiments:187-200`)
- `get_experiment_info(name: str)` - Get detailed experiment metadata (`leeq.epii.experiments:150-170`)

#### Backend Detection
- `_detect_simulation_setup()` - Identifies simulation vs hardware setup (`leeq.epii.experiments:36-51`)
- `_has_own_run_simulated()` - Validates simulation compatibility (`leeq.epii.experiments:53-81`)

## Usage Patterns

### Basic Usage
```python
from leeq.epii.experiments import ExperimentRouter

# Initialize router with setup
router = ExperimentRouter(setup=my_setup)

# Get experiment class
experiment_class = router.get_experiment("calibrations.NormalisedRabi")

# List all available experiments
experiments = router.list_experiments()
```

### Simulation-Aware Usage
```python
# Router automatically filters for simulation-compatible experiments
sim_router = ExperimentRouter(setup=simulation_setup)
sim_experiments = sim_router.list_experiments()  # Only includes experiments with run_simulated
```

## Migration from v0.1.x

**Breaking Changes**:
- All experiment aliases removed
- Must use canonical names (module-qualified)
- Backend-aware filtering may limit available experiments

**Migration Steps**:
1. Replace all alias references with canonical names
2. Update configuration files and scripts
3. Test experiment discovery with your setup type

## Module Documentation

::: leeq.epii
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: source
      show_signature_annotations: true
      show_if_no_docstring: true
      separate_signature: true
