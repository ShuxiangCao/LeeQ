# EPII v0.2.0 User Guide

The Experiment Programming Interface for Instruments (EPII) v0.2.0 is a major update that brings backend-aware experiment discovery, canonical naming, and improved integration capabilities to LeeQ.

## What's New in v0.2.0

### Key Features
- **Backend-Aware Discovery**: Automatically filters experiments based on your setup type
- **Canonical Naming**: No more aliases - use full module-qualified experiment names
- **Dynamic Discovery**: Automatically finds all available experiments
- **Enhanced Metadata**: Rich experiment information with EPII_INFO

### Breaking Changes
- All experiment aliases removed (`rabi`, `t1`, `ramsey`, etc.)
- Must use canonical names (`calibrations.NormalisedRabi`, `characterizations.SimpleT1`)
- Backend filtering may limit available experiments

## Getting Started

### Basic Usage

```python
from leeq.epii.experiments import ExperimentRouter

# Initialize with your setup
router = ExperimentRouter(setup=my_setup)

# Discover available experiments
experiments = router.list_experiments()
print(f"Found {len(experiments)} experiments")

# Get a specific experiment
experiment_class = router.get_experiment("calibrations.NormalisedRabi")
```

### Backend-Aware Operation

The router automatically detects your setup type and filters experiments accordingly:

```python
# With simulation setup - only shows experiments with run_simulated
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup

sim_setup = HighLevelSimulationSetup()
sim_router = ExperimentRouter(setup=sim_setup)
sim_experiments = sim_router.list_experiments()  # Filtered for simulation compatibility

# With hardware setup - shows all experiments
hardware_router = ExperimentRouter(setup=hardware_setup)
all_experiments = hardware_router.list_experiments()  # All experiments available
```

## Experiment Naming

### Canonical Names
All experiments now use module-qualified names that reflect their organization:

| Category | Example |
|----------|---------|
| **Calibrations** | `calibrations.NormalisedRabi`, `calibrations.DragCalibrationSingleQubitMultilevel` |
| **Characterizations** | `characterizations.SimpleT1`, `characterizations.SpinEchoMultiLevel` |
| **Multi-Qubit Gates** | `multi_qubit_gates.CrossResonanceCalibration` |
| **Tomography** | `tomography.StateTomography`, `tomography.ProcessTomography` |
| **Optimal Control** | `optimal_control.GrapeOptimization` |

### Finding Experiment Names
```python
# List all available experiments with descriptions
experiments = router.list_experiments()
for name, description in experiments.items():
    print(f"{name}: {description}")

# Get detailed information about an experiment
info = router.get_experiment_info("calibrations.NormalisedRabi")
print(info['epii_info'])  # Experiment metadata
print(info['run_docstring'])  # Run method documentation
```

## Usage Patterns

### Running Experiments
Follow LeeQ's constructor-based pattern - **never call `run()` methods directly**:

```python
# CORRECT: Pass all parameters to constructor
exp = QubitSpectroscopyFrequency(
    dut_qubit=qubit,
    start=4900.0,
    stop=5100.0, 
    step=2.0,
    num_avs=1000
)

# INCORRECT: Never do this
exp = QubitSpectroscopyFrequency()
exp.run_simulated(...)  # WRONG!
```

### Error Handling
```python
try:
    experiment_class = router.get_experiment("calibrations.NormalisedRabi")
    if experiment_class:
        exp = experiment_class(
            dut_qubit=qubit,
            amplitudes=np.linspace(0, 1, 51)
        )
    else:
        print("Experiment not found or not compatible with current setup")
except Exception as e:
    print(f"Experiment failed: {e}")
```

### Checking Compatibility
```python
# Check if an experiment is available (considers backend compatibility)
if "calibrations.NormalisedRabi" in router.experiment_map:
    print("Rabi experiment available")
else:
    print("Rabi experiment not available for this setup")

# Get experiment requirements
info = router.get_experiment_info("calibrations.NormalisedRabi")
epii_info = info['epii_info']
print(f"Description: {epii_info.get('description', 'N/A')}")
print(f"Parameters: {epii_info.get('parameters', {})}")
```

## Advanced Features

### Custom Experiment Discovery
```python
# Initialize without automatic discovery
router = ExperimentRouter()

# Manually discover experiments (useful for debugging)
router._discover_experiments()
print(f"Discovered {len(router.experiment_map)} experiments")
```

### Simulation vs Hardware
```python
# Check if router is in simulation mode
if router.is_simulation:
    print("Running in simulation mode")
    print("Only experiments with run_simulated method are available")
else:
    print("Running on hardware")
    print("All experiments available")

# List simulation capabilities
for name, exp_class in router.experiment_map.items():
    has_sim = router._has_own_run_simulated(exp_class)
    print(f"{name}: {'✓' if has_sim else '✗'} simulation support")
```

## Integration Examples

### With Chronicle Logging
```python
from leeq.experiments.builtin.calibrations import NormalisedRabi

# Experiments automatically integrate with Chronicle
exp = NormalisedRabi(
    dut_qubit=qubit,
    amplitudes=np.linspace(0, 1, 51)
)
# Results automatically logged to Chronicle
```

### Batch Experiment Execution
```python
# Run multiple experiments in sequence
experiment_names = [
    "calibrations.NormalisedRabi",
    "characterizations.SimpleT1", 
    "calibrations.SimpleRamseyMultilevel"
]

results = {}
for exp_name in experiment_names:
    exp_class = router.get_experiment(exp_name)
    if exp_class:
        # Configure experiment based on type
        if "rabi" in exp_name.lower():
            exp = exp_class(dut_qubit=qubit, amplitudes=np.linspace(0, 1, 21))
        elif "t1" in exp_name.lower():
            exp = exp_class(dut_qubit=qubit, delays=np.logspace(-6, -3, 21))
        # Store results
        results[exp_name] = exp
```

## Best Practices

1. **Always initialize router with your setup** for proper backend filtering
2. **Use canonical names consistently** throughout your code
3. **Check experiment availability** before attempting to run
4. **Follow constructor-only pattern** for experiment execution
5. **Handle errors gracefully** with try/catch blocks

## Troubleshooting

### Common Issues

**Experiment not found**:
```python
# Check available experiments
experiments = router.list_experiments()
if "my_experiment" not in experiments:
    print("Available experiments:")
    for name in experiments:
        print(f"  {name}")
```

**No experiments available**:
- Check if your setup is properly configured
- Verify EPII_INFO attributes on custom experiments
- For simulation: ensure experiments have `run_simulated` method

**Import errors**:
- Ensure all LeeQ dependencies are installed
- Check that experiment modules are in Python path

## Migration Guide

See the detailed [Migration Guide](migration-guide.md) for step-by-step instructions on updating from EPII v0.1.x.

## API Reference

For detailed API documentation, see:
- [EPII API Overview](../api/epii/overview.md)
- [ExperimentRouter Reference](../api/epii/experiments.md)
- [Service Integration](../api/epii/service.md)