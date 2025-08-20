# Calibrations

This guide covers the calibration procedures and experiments available in LeeQ.

## Overview

Calibration is a crucial step in quantum computing experiments. LeeQ provides a comprehensive set of calibration experiments to characterize and optimize qubit performance.

## Available Calibrations

### Rabi Experiments
- **Purpose**: Determine optimal drive parameters
- **Module**: `leeq.experiments.builtin.basic.calibrations.rabi`
- **Key Parameters**: Drive amplitude, frequency

### Ramsey Experiments  
- **Purpose**: Measure dephasing time and fine-tune frequencies
- **Module**: `leeq.experiments.builtin.basic.calibrations.ramsey`
- **Key Parameters**: Evolution time, detuning

### T1 Measurements
- **Purpose**: Measure relaxation time
- **Module**: `leeq.experiments.builtin.basic.characterizations.t1`
- **Key Parameters**: Delay time range

## Running Calibrations

```python
import leeq

# Example: Rabi calibration
rabi_exp = leeq.experiments.RabiExperiment(
    qubit=my_qubit,
    drive_amplitude_range=(0, 1),
    num_points=50
)

result = rabi_exp.run()
optimal_amplitude = result.fit_result.optimal_amplitude
```

## Best Practices

1. **Regular Calibration**: Run calibrations periodically as system parameters drift
2. **Parameter Ranges**: Use appropriate ranges based on previous calibrations
3. **Fitting**: Always verify fit quality before using calibration results
4. **Documentation**: Log calibration results for tracking system performance

## Related Topics

- [Experiments](experiments.md) - General experiment framework
- [Core Concepts](concepts.md) - Understanding LeeQ architecture