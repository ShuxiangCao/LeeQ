# EPII v0.2.0 Quick Reference

## Alias to Canonical Name Mapping

| Old Alias | Canonical Name | Description |
|-----------|----------------|-------------|
| `rabi` | `calibrations.NormalisedRabi` | Rabi oscillation amplitude calibration |
| `t1` | `characterizations.SimpleT1` | T1 relaxation time measurement |
| `ramsey` | `calibrations.SimpleRamseyMultilevel` | Ramsey fringe detuning calibration |
| `echo` | `characterizations.SpinEchoMultiLevel` | Spin echo T2 measurement |
| `spin_echo` | `characterizations.SpinEchoMultiLevel` | Spin echo T2 measurement |
| `drag` | `calibrations.DragCalibrationSingleQubitMultilevel` | DRAG pulse calibration |
| `randomized_benchmarking` | `characterizations.RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem` | Gate fidelity benchmarking |
| `multi_qubit_rabi` | `calibrations.MultiQubitRabi` | Multi-qubit Rabi calibration |
| `multi_qubit_t1` | `characterizations.MultiQubitT1` | Multi-qubit T1 measurement |
| `multi_qubit_ramsey` | `calibrations.MultiQubitRamseyMultilevel` | Multi-qubit Ramsey calibration |
| `qubit_spectroscopy_frequency` | `calibrations.QubitSpectroscopyFrequency` | Qubit frequency spectroscopy |

## Common Usage Patterns

### ExperimentRouter Initialization
```python
from leeq.epii.experiments import ExperimentRouter

# With setup for backend-aware filtering
router = ExperimentRouter(setup=my_setup)

# Without setup (discovers all experiments)
router = ExperimentRouter()
```

### Experiment Discovery
```python
# List all available experiments
experiments = router.list_experiments()

# Check if specific experiment is available
if "calibrations.NormalisedRabi" in router.experiment_map:
    print("Rabi experiment available")

# Get experiment class
experiment_class = router.get_experiment("calibrations.NormalisedRabi")
```

### Experiment Execution (Constructor Pattern)
```python
# CORRECT: Pass parameters to constructor
exp = QubitSpectroscopyFrequency(
    dut_qubit=qubit,
    start=4900.0,
    stop=5100.0,
    step=2.0,
    num_avs=1000
)

# INCORRECT: Never call run methods directly
# exp.run_simulated(...)  # DON'T DO THIS
```

## Migration Examples

### Before (v0.1.x with aliases)
```python
# Old approach with aliases
router = ExperimentRouter()
rabi_class = router.get_experiment("rabi")
t1_class = router.get_experiment("t1")

# Configuration with aliases
experiments_to_run = ["rabi", "t1", "ramsey"]

# Client usage with aliases  
result = client.run_experiment("rabi", {...})
```

### After (v0.2.0 with canonical names)
```python
# New approach with canonical names
router = ExperimentRouter(setup=my_setup)
rabi_class = router.get_experiment("calibrations.NormalisedRabi")
t1_class = router.get_experiment("characterizations.SimpleT1")

# Configuration with canonical names
experiments_to_run = [
    "calibrations.NormalisedRabi",
    "characterizations.SimpleT1", 
    "calibrations.SimpleRamseyMultilevel"
]

# Client usage with canonical names
result = client.run_experiment("calibrations.NormalisedRabi", {...})
```

## Error Handling

### Common Errors and Solutions

**1. Experiment not found**
```python
experiment_class = router.get_experiment("rabi")  # Returns None
# Solution: Use canonical name
experiment_class = router.get_experiment("calibrations.NormalisedRabi")
```

**2. No experiments discovered**
```python
router = ExperimentRouter(setup=hardware_setup)
if len(router.experiment_map) == 0:
    print("No experiments found")
# Solution: Check setup compatibility or use simulation setup
```

**3. Simulation experiments not available**
```python
# In simulation mode, only experiments with run_simulated are available
router = ExperimentRouter(setup=simulation_setup)
# Some experiments may be filtered out
```

## Backend Compatibility

### Simulation Setup
```python
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup

setup = HighLevelSimulationSetup()
router = ExperimentRouter(setup=setup)

# Only experiments with run_simulated method are available
print(f"Simulation-compatible experiments: {len(router.experiment_map)}")
```

### Hardware Setup
```python
router = ExperimentRouter(setup=hardware_setup)
# All experiments with EPII_INFO are available
print(f"All available experiments: {len(router.experiment_map)}")
```

## Troubleshooting

### Check Experiment Capabilities
```python
# Check if experiment supports simulation
exp_class = router.get_experiment("calibrations.NormalisedRabi")
has_simulation = router._has_own_run_simulated(exp_class)
print(f"Supports simulation: {has_simulation}")

# Get detailed experiment info
info = router.get_experiment_info("calibrations.NormalisedRabi")
print(f"Description: {info['epii_info'].get('description', 'N/A')}")
print(f"Parameters: {info['epii_info'].get('parameters', {})}")
```

### Debug Discovery Process
```python
# Enable debug logging
import logging
logging.getLogger('leeq.epii.experiments').setLevel(logging.DEBUG)

# Initialize router to see discovery process
router = ExperimentRouter()
```

### Validate Configuration
```python
# Test that aliases don't exist
router = ExperimentRouter()
aliases = ['rabi', 't1', 'ramsey', 'echo', 'drag']
for alias in aliases:
    assert alias not in router.experiment_map, f"Alias {alias} still exists!"
print("✓ All aliases successfully removed")
```

## Best Practices

1. **Always use canonical names** in new code
2. **Initialize router with setup** for proper filtering
3. **Check experiment availability** before execution
4. **Use constructor pattern** for experiment execution
5. **Handle backend compatibility** gracefully

## Links

- [EPII Overview](overview.md) - Complete user guide
- [API Reference](../api/epii/overview.md) - Technical documentation
- [Troubleshooting](troubleshooting.md) - Common issues and solutions