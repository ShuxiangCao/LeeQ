# Experiments Guide

This guide covers how to use and create experiments in LeeQ with EPII v0.2.0 integration.

## EPII v0.2.0 Integration

LeeQ experiments now integrate seamlessly with EPII v0.2.0 for backend-aware discovery and execution.

### Using ExperimentRouter

```python
from leeq.epii.experiments import ExperimentRouter

# Initialize router with your setup for backend-aware filtering
router = ExperimentRouter(setup=my_setup)

# Discover available experiments
experiments = router.list_experiments()
print(f"Found {len(experiments)} experiments")

# Get experiment by canonical name
experiment_class = router.get_experiment("calibrations.NormalisedRabi")
```

### Canonical Naming Convention

All experiments use module-qualified canonical names:

- **Calibrations**: `calibrations.NormalisedRabi`, `calibrations.SimpleRamseyMultilevel`
- **Characterizations**: `characterizations.SimpleT1`, `characterizations.SpinEchoMultiLevel`
- **Multi-Qubit**: `multi_qubit_gates.CrossResonanceCalibration`

### Constructor-Only Execution Pattern

**Important**: Always pass parameters to the constructor - never call `run()` methods directly:

```python
# CORRECT: Constructor-only pattern
exp = QubitSpectroscopyFrequency(
    dut_qubit=qubit,
    start=4900.0,
    stop=5100.0,
    step=2.0,
    num_avs=1000
)
# Experiment automatically executes based on setup type

# INCORRECT: Never do this
exp = QubitSpectroscopyFrequency()
exp.run_simulated(...)  # WRONG - never call run methods directly
```

## Built-in Experiments

LeeQ provides a comprehensive library of built-in experiments for quantum system characterization and calibration.

### Basic Calibrations

#### Resonator Spectroscopy
```python
from leeq.experiments.builtin.basic.calibrations.resonator_spectroscopy import *

# Find resonator frequency
exp = ResonatorSweepTransmissionWithExtraInitialLPB(
    dut,                    # Your quantum element
    start=9.98,            # Start frequency (GHz)
    stop=10.02,            # Stop frequency (GHz) 
    step=0.001,            # Step size (GHz)
    num_avs=1000,          # Number of averages
    mp_width=8             # Measurement pulse width (μs)
)
result = exp.run()
```

#### Qubit Spectroscopy
```python
from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import *

# Find qubit frequency
exp = QubitSpectroscopy(
    dut,
    start=4.8,             # Start frequency (GHz)
    stop=4.9,              # Stop frequency (GHz)
    step=0.001,            # Step size (GHz)
    num_avs=500
)
result = exp.run()
```

#### Rabi Oscillations
```python
from leeq.experiments.builtin.basic.calibrations.rabi import *

# Power Rabi - find π pulse amplitude
exp = PowerRabi(
    dut,
    start=0.0,             # Start amplitude
    stop=0.3,              # Stop amplitude  
    step=0.005,            # Step size
    num_avs=500
)

# Time Rabi - find π pulse width
exp = TimeRabi(
    dut,
    start=0.01,            # Start width (μs)
    stop=0.1,              # Stop width (μs)
    step=0.001,            # Step size (μs)
    num_avs=500
)
```

#### Ramsey Fringes
```python
from leeq.experiments.builtin.basic.calibrations.ramsey import *

# Ramsey experiment for frequency calibration
exp = Ramsey(
    dut,
    start=0.0,             # Start delay (μs)
    stop=10.0,             # Stop delay (μs)
    step=0.1,              # Step size (μs)
    detuning=0.5,          # Detuning frequency (MHz)
    num_avs=500
)
```

#### DRAG Calibration
```python
from leeq.experiments.builtin.basic.calibrations.drag import *

# Calibrate DRAG coefficient
exp = DragCalibration(
    dut,
    start=-2.0,            # Start DRAG coefficient
    stop=2.0,              # Stop DRAG coefficient
    step=0.1,              # Step size
    num_avs=500
)
```

### Characterization Experiments

#### T1 (Energy Relaxation Time)
```python
from leeq.experiments.builtin.basic.characterizations.t1 import *

exp = T1Measurement(
    dut,
    start=0.0,             # Start delay (μs)
    stop=100.0,            # Stop delay (μs)
    step=2.0,              # Step size (μs)
    num_avs=500
)
result = exp.run()
print(f"T1 = {result.fit_params['T1']:.2f} μs")
```

#### T2 (Dephasing Time)  
```python
from leeq.experiments.builtin.basic.characterizations.t2 import *

# T2* measurement (free induction decay)
exp = T2StarMeasurement(
    dut,
    start=0.0,             # Start delay (μs)
    stop=50.0,             # Stop delay (μs)
    step=0.5,              # Step size (μs)
    num_avs=500
)

# T2 Echo measurement (Hahn echo)
exp = T2EchoMeasurement(
    dut,
    start=0.0,
    stop=100.0,
    step=1.0,
    num_avs=500
)
```

#### Randomized Benchmarking
```python
from leeq.experiments.builtin.basic.characterizations.randomized_benchmarking import *

# Single qubit randomized benchmarking
exp = SingleQubitRandomizedBenchmarking(
    dut,
    sequence_lengths=[1, 5, 10, 25, 50, 100, 200],
    num_sequences=20,      # Number of random sequences per length
    num_avs=500
)
result = exp.run()
print(f"Gate fidelity = {result.fit_params['fidelity']:.4f}")
```

### Multi-Qubit Experiments

#### Two-Qubit Calibrations
```python
from leeq.experiments.builtin.multi_qubit_gates import *

# Conditional Stark shift calibration
exp = ConditionalStarkShiftContinuous(
    control_qubit=dut1,
    target_qubit=dut2,
    start_frequency=4.85,
    stop_frequency=4.87,
    step=0.001,
    num_avs=500
)

# Cross-resonance calibration  
exp = CrossResonanceCalibration(
    control_qubit=dut1,
    target_qubit=dut2,
    start_amplitude=0.0,
    stop_amplitude=0.1,
    step=0.002,
    num_avs=500
)
```

### State Discrimination

#### Gaussian Mixture Model (GMM)
```python
from leeq.experiments.builtin.basic.calibrations.state_discrimination import *

# Calibrate measurement discrimination
exp = MeasurementCalibrationMultilevelGMM(
    dut,
    mprim_index=0,         # Measurement primitive index
    sweep_lpb_list=[       # States to prepare and measure
        dut.get_c1('f01')['I'],  # |0⟩ state
        dut.get_c1('f01')['X']   # |1⟩ state  
    ],
    num_avs=1000
)
```

## Custom Experiments

### Creating Your Own Experiment

All experiments inherit from the base experiment class:

```python
from leeq.experiments.experiments import Experiment

class MyCustomExperiment(Experiment):
    def __init__(self, dut, custom_param, **kwargs):
        # Initialize experiment
        super().__init__(dut, **kwargs)
        self.custom_param = custom_param
        
    def _build_sequence(self):
        """Define the pulse sequence"""
        # Build your pulse sequence here
        sequence = []
        
        # Add initialization
        sequence.append(dut.get_c1('f01')['I'])  # Identity
        
        # Add custom operations
        custom_pulse = dut.get_c1('f01')['X'].clone()
        custom_pulse.set_parameter('amp', self.custom_param)
        sequence.append(custom_pulse)
        
        # Add measurement
        sequence.append(dut.get_measurement_prim_intlist(0))
        
        return sequence
    
    def _analyze_data(self, data):
        """Analyze experimental data"""
        # Process your data here
        result = {
            'signal': np.mean(data['I']),
            'std': np.std(data['I'])
        }
        return result
```

### Parameter Sweeps

Use the Sweeper class for parameter variations:

```python
from leeq.experiments.sweeper import Sweeper

# Single parameter sweep
freq_sweep = Sweeper(
    parameter=dut.get_c1('f01')['freq'],
    values=np.linspace(4.8, 4.9, 51)
)

# Multi-dimensional sweeps
amp_sweep = Sweeper(
    parameter=dut.get_c1('f01')['amp'],
    values=np.linspace(0.05, 0.15, 11)
)

# Grid sweep automatically handles combinations
```

### Advanced Features

#### Live Plotting
```python
# Enable live monitoring
from leeq.experiments import setup
setup().start_live_monitor()

# Experiments automatically display live plots
exp = PowerRabi(dut, start=0, stop=0.2, step=0.005)
exp.run()  # Shows live plot as data arrives
```

#### Data Persistence
```python
from leeq.chronicle import Chronicle

# Start logging
Chronicle().start_log()

# All experiments automatically logged
exp = T1Measurement(dut, start=0, stop=100, step=2)
result = exp.run()

# Access logged data later
log_entry = Chronicle().get_last_log_entry()
```

#### AI-Assisted Experiments
```python
from leeq.utils.ai.experiment_generation import ExperimentGenerator

# Generate experiment from description
generator = ExperimentGenerator()
code = generator.generate_experiment(
    description="Measure T2* with varying echo delays",
    requirements=["high precision", "automated fitting"]
)
```

## Best Practices

### 1. Parameter Organization
- Group related parameters in dictionaries
- Use descriptive parameter names
- Document units and ranges

### 2. Error Handling
```python
try:
    result = experiment.run()
    if result.fit_quality < 0.95:
        print("Warning: Poor fit quality")
except ExperimentError as e:
    print(f"Experiment failed: {e}")
```

### 3. Calibration Tracking
```python
# Save calibration after successful experiment
if result.fit_quality > 0.95:
    dut.set_parameter('f01_frequency', result.fit_params['frequency'])
    dut.save_calibration_log()
```

### 4. Reproducibility
- Always set random seeds for stochastic experiments
- Save exact parameter values used
- Include environmental conditions in logs

### 5. Performance Optimization
- Use appropriate number of averages
- Optimize measurement time vs. precision
- Consider parallel execution for multi-qubit systems

## Troubleshooting

### Common Issues

**Poor Signal-to-Noise**: Increase averages or adjust measurement amplitude
**Calibration Drift**: Regular recalibration, track environmental changes  
**Timing Issues**: Check pulse sequence timing and hardware limits
**Fitting Failures**: Verify data quality and fitting bounds

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger('leeq').setLevel(logging.DEBUG)

# Inspect pulse sequences
experiment.sequence.plot()

# Check data quality
result.plot_raw_data()
```

## Next Steps

- Learn about [calibration workflows](calibrations.md)
- Explore [advanced theory](../api/theory/simulation.md)
- Review the [Tutorial](../tutorial.md) for step-by-step examples