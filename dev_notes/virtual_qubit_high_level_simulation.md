# VirtualQubit and HighLevelSimulation Ecosystem in LeeQ

## Overview

The VirtualQubit and HighLevelSimulation ecosystem in LeeQ provides a powerful simulation framework for quantum experiments without requiring actual quantum hardware. This system allows researchers to develop, test, and validate quantum experiments using realistic qubit models that include noise, decoherence, and other physical effects.

## Architecture Components

### 1. VirtualTransmon

**Location**: `leeq/theory/simulation/numpy/rotated_frame_simulator.py:361`

The `VirtualTransmon` class is the core component that simulates a physical transmon qubit with realistic parameters and behavior.

#### Key Features:
- **Multi-level System**: Models transmon as a 4-level system (default) with proper anharmonicity
- **Decoherence**: Implements T1 (relaxation) and T2 (dephasing) dynamics
- **Dispersive Readout**: Simulates realistic readout with IQ noise and state-dependent resonator response
- **Thermal Population**: Models quiescent state distribution for realistic initial conditions
- **Frequency Selectivity**: Implements proper frequency windows for spectroscopy

#### Core Parameters:
```python
VirtualTransmon(
    name="VQubit",
    qubit_frequency=5040.4,        # MHz
    anharmonicity=-198,            # MHz
    t1=70,                        # microseconds
    t2=35,                        # microseconds
    readout_frequency=9645.5,      # MHz
    readout_linewidth=4.0,         # MHz
    readout_dispersive_shift=1.75, # MHz
    readout_fidelity=0.95,
    quiescent_state_distribution=[0.8, 0.15, 0.04, 0.01]
)
```

#### Key Methods:
- `apply_drive(amplitude, frequency, duration)`: Simulates qubit control pulses
- `apply_readout(readout_frequency, readout_duration)`: Returns IQ values with realistic noise
- `get_resonator_response()`: Calculates dispersive readout response
- `reset_state()`: Resets to thermal equilibrium

### 2. HighLevelSimulationSetup

**Location**: `leeq/setups/built_in/setup_simulation_high_level.py:14`

The `HighLevelSimulationSetup` manages multiple VirtualTransmon instances and provides the interface between LeeQ experiments and the simulation backend.

#### Key Features:
- **Channel Mapping**: Maps hardware channel numbers to virtual qubits
- **Coupling Management**: Handles inter-qubit coupling for multi-qubit simulations
- **Calibration Data**: Maintains omega-per-amp calibration values
- **Noise Control**: Configurable sampling noise for realistic simulations

#### Setup Structure:
```python
HighLevelSimulationSetup(
    name='HighLevelSimulationSetup',
    virtual_qubits={
        2: virtual_transmon_a,  # Channel 2 → Qubit A
        4: virtual_transmon_b   # Channel 4 → Qubit B
    }
)
```

## How It Works

### 1. Simulation Flow

```
Experiment Request
       ↓
ExperimentManager checks setup type
       ↓
If HighLevelSimulationSetup:
       ↓
Bypass pulse compilation
       ↓
Map operations to VirtualTransmon methods
       ↓
Apply theoretical formulas with noise
       ↓
Return simulated measurement results
```

### 2. Integration with Experiments

Experiments detect high-level simulation mode by checking:
```python
if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
    # Use theoretical simulation
else:
    # Use pulse-based execution
```

### 3. Multi-Qubit Simulations

For two-qubit gates and experiments:
```python
# Set coupling strength
setup.set_coupling_strength_by_qubit(
    qubit_a, qubit_b, 
    coupling_strength=1.5  # MHz
)

# The coupling affects:
# - CZ gate fidelity
# - Crosstalk effects
# - Spectroscopy line shapes
```

## Supported Experiments

The following experiments have full high-level simulation support:

### Basic Calibrations
- Rabi oscillations (`leeq/experiments/builtin/basic/calibrations/rabi.py`)
- Ramsey fringes (`leeq/experiments/builtin/basic/calibrations/ramsey.py`)
- Qubit/resonator spectroscopy
- DRAG calibration
- Pi/Pi-half pulse calibrations

### Characterizations
- T1 measurement (`leeq/experiments/builtin/basic/characterizations/t1.py`)
- T2 echo/ramsey (`leeq/experiments/builtin/basic/characterizations/t2_echo.py`)
- Randomized benchmarking
- Process/state tomography

### Multi-Qubit
- Two-qubit spectroscopy
- CZ gate calibration
- Cross-resonance experiments

## Usage Examples

### Single Qubit Simulation
```python
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.builtin.basic.calibrations import Rabi

# Create virtual qubit
vqubit = VirtualTransmon(
    name="Q0",
    qubit_frequency=5000,
    anharmonicity=-200,
    t1=50,
    t2=30
)

# Create setup
setup = HighLevelSimulationSetup(
    name="SimSetup",
    virtual_qubits={0: vqubit}
)

# Run experiment
rabi = Rabi(setup)
results = rabi.run(
    qubit=vqubit,
    amplitudes=np.linspace(0, 1, 51)
)
```

### Two-Qubit Simulation with Coupling
```python
# Create two qubits
vqubit_a = VirtualTransmon(name="Q0", qubit_frequency=5000, ...)
vqubit_b = VirtualTransmon(name="Q1", qubit_frequency=5100, ...)

# Setup with coupling
setup = HighLevelSimulationSetup(
    name="SimSetup",
    virtual_qubits={0: vqubit_a, 1: vqubit_b}
)

# Configure coupling
setup.set_coupling_strength_by_name("Q0", "Q1", 2.0)  # 2 MHz coupling

# Run two-qubit experiments
```

## Implementation Details

### Noise Modeling

The simulation includes several noise sources:
1. **Decoherence**: T1/T2 decay during gate operations
2. **Readout Noise**: IQ measurement noise based on fidelity
3. **Thermal Noise**: Initial state preparation errors
4. **Control Noise**: Optional amplitude/frequency fluctuations

### Performance Considerations

- High-level simulation is much faster than pulse-level simulation
- Suitable for rapid prototyping and parameter optimization
- Trade-off: Less accurate for strongly-driven or non-RWA regimes

### Extending the System

To add new virtual qubit types:
1. Inherit from `VirtualTransmon` base class
2. Override relevant methods (e.g., for different qubit architectures)
3. Register with `HighLevelSimulationSetup`

To support new experiments:
1. Add conditional logic checking for `High_Level_Simulation_Mode`
2. Implement theoretical formulas for the experiment
3. Apply appropriate noise models

## Related Components

### Numpy2QVirtualDeviceSetup
- **Location**: `leeq/setups/built_in/setup_numpy_2q_virtual_device.py`
- Lower-level pulse simulation
- Tracks full quantum state evolution
- More accurate but slower

### QuTiP Integration
- **Location**: `leeq/theory/simulation/qutip/`
- Alternative simulation backend
- Supports open quantum system dynamics
- Master equation simulations

## Best Practices

1. **Calibration First**: Always calibrate virtual qubits similar to real hardware
2. **Validate Noise**: Ensure noise parameters match target hardware
3. **Check Limits**: Verify simulation stays within RWA approximation limits
4. **Use for Development**: Ideal for experiment development before hardware testing

## Debugging Tips

1. **Enable Logging**: Virtual qubits support detailed logging
2. **Check State Evolution**: Use `get_state()` to inspect quantum state
3. **Visualize Results**: Plot IQ clouds and state trajectories
4. **Compare Methods**: Validate against pulse-level simulations

## Future Enhancements

- Support for more qubit types (flux-tunable, C-shunt)
- Advanced noise models (1/f, correlated noise)
- Multi-qubit entanglement dynamics
- Integration with quantum error correction simulations