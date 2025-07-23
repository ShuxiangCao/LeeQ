# High-Level Simulation Implementation Plan for Easy Wins

## Overview

This document provides a detailed implementation plan for adding high-level simulation support to four "easy win" experiments in LeeQ:
1. PowerRabi
2. MultiQubitRabi
3. MultiQubitT1
4. MultiQubitRamseyMultilevel

These experiments were chosen because they can leverage existing simulation infrastructure with minimal additional physics modeling.

## Implementation Timeline

**Total Estimated Time**: 8-10 hours (1-2 days of focused work)

| Experiment | Priority | Estimated Time | Dependencies |
|------------|----------|----------------|--------------|
| PowerRabi | 1 | 2-3 hours | None |
| MultiQubitT1 | 2 | 1-2 hours | None |
| MultiQubitRabi | 3 | 2-3 hours | PowerRabi |
| MultiQubitRamseyMultilevel | 4 | 2-3 hours | None |

## Detailed Implementation Plans

### 1. PowerRabi Implementation

**File**: `leeq/experiments/builtin/basic/calibrations/rabi.py:374`

#### Step 1: Add simulation check in `run()` method
```python
def run(self, dut, amp_range, frequency=None, phase=0):
    # Add after initial setup
    if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
        return self.run_simulated(
            dut=dut,
            amp_range=amp_range,
            frequency=frequency,
            phase=phase
        )
```

#### Step 2: Implement `run_simulated()` method
```python
def run_simulated(self, dut, amp_range, frequency=None, phase=0):
    """
    Simulate power Rabi oscillations by sweeping amplitude.
    
    The Rabi frequency is: Ω = amp * omega_per_amp
    The population oscillates as: P = sin²(Ω * π_time / 2)
    """
    import numpy as np
    from leeq.utils.compatibility import get_qubit_pi
    
    # Get virtual qubit
    virtual_qubit = self.setup.get_virtual_qubit(dut)
    if virtual_qubit is None:
        raise ValueError(f"No virtual qubit found for {dut}")
    
    # Get calibration parameters
    pi_pulse = get_qubit_pi(dut)
    pi_time = pi_pulse.get_duration()
    omega_per_amp = self.setup.get_omega_per_amp(dut.get_c1('channel'))
    
    # Use qubit frequency if not specified
    if frequency is None:
        frequency = dut.get_c1('f01')
    
    # Calculate expected oscillations
    amp_range = np.array(amp_range)
    rabi_frequencies = amp_range * omega_per_amp  # MHz
    
    # Calculate rotation angle for each amplitude
    # θ = Ω * t = (amp * omega_per_amp) * pi_time
    rotation_angles = rabi_frequencies * pi_time  # radians
    
    # Calculate excited state population
    # P_e = sin²(θ/2) for starting from ground state
    populations = np.sin(rotation_angles / 2) ** 2
    
    # Add noise if enabled
    if self.setup.status.get('Sampling_Noise', True):
        readout_fidelity = getattr(virtual_qubit, 'readout_fidelity', 0.95)
        noise_level = (1 - readout_fidelity) / 2
        noise = np.random.normal(0, noise_level, len(populations))
        populations = np.clip(populations + noise, 0, 1)
    
    # Format results to match hardware output
    results = {
        'amplitudes': amp_range,
        'populations': populations,
        'fitted_pi_amp': amp_range[np.argmax(populations)],  # Amplitude giving max population
        'frequency': frequency,
        'phase': phase
    }
    
    return results
```

#### Step 3: Add unit test
```python
# In tests/experiments/builtin/basic/calibrations/test_rabi.py
def test_power_rabi_simulation():
    """Test PowerRabi with high-level simulation."""
    from leeq.experiments.builtin.basic.calibrations.rabi import PowerRabi
    from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
    from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
    
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
    exp = PowerRabi(setup)
    amp_range = np.linspace(0, 1, 51)
    results = exp.run(vqubit, amp_range)
    
    # Verify results
    assert 'amplitudes' in results
    assert 'populations' in results
    assert len(results['populations']) == len(amp_range)
    assert 0 <= max(results['populations']) <= 1
```

### 2. MultiQubitT1 Implementation

**File**: `leeq/experiments/builtin/basic/characterizations/t1.py:258`

#### Step 1: Add simulation check
```python
def run(self, duts, collection_name=None, initial_lpb=None, 
        delay_range=None, mprim_indexes=None):
    # Add after parameter setup
    if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
        return self.run_simulated(
            duts=duts,
            delay_range=delay_range,
            initial_lpb=initial_lpb
        )
```

#### Step 2: Implement `run_simulated()` method
```python
def run_simulated(self, duts, delay_range=None, initial_lpb=None):
    """
    Simulate T1 decay for multiple qubits in parallel.
    Each qubit decays independently: P(t) = P(0) * exp(-t/T1)
    """
    import numpy as np
    
    # Default delay range if not specified
    if delay_range is None:
        # Estimate based on average T1
        t1_values = [self.setup.get_virtual_qubit(dut).t1 for dut in duts]
        max_delay = 3 * np.mean(t1_values)  # Go to ~3*T1
        delay_range = np.linspace(0, max_delay, 51)
    
    delay_range = np.array(delay_range)
    results = {}
    
    # Simulate each qubit independently
    for dut in duts:
        virtual_qubit = self.setup.get_virtual_qubit(dut)
        if virtual_qubit is None:
            raise ValueError(f"No virtual qubit found for {dut}")
        
        # Get T1 value
        t1 = virtual_qubit.t1  # in microseconds
        
        # Calculate decay curve
        # For initial excited state (after pi pulse)
        populations = np.exp(-delay_range / t1)
        
        # Add readout noise if enabled
        if self.setup.status.get('Sampling_Noise', True):
            readout_fidelity = getattr(virtual_qubit, 'readout_fidelity', 0.95)
            noise_level = (1 - readout_fidelity) / 2
            noise = np.random.normal(0, noise_level, len(populations))
            populations = np.clip(populations + noise, 0, 1)
        
        # Store results for this qubit
        results[dut.name] = {
            'delays': delay_range,
            'populations': populations,
            'fitted_t1': t1,
            'initial_population': populations[0]
        }
    
    # Combine results in format matching hardware output
    combined_results = {
        'delay_range': delay_range,
        'qubits': results,
        'timestamp': time.time()
    }
    
    return combined_results
```

#### Step 3: Add visualization compatibility
```python
def plot_simulated_results(self, results):
    """Plot T1 decay curves for all qubits."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for qubit_name, data in results['qubits'].items():
        ax.plot(data['delays'], data['populations'], 
                'o-', label=f"{qubit_name}: T1={data['fitted_t1']:.1f} μs")
    
    ax.set_xlabel('Delay (μs)')
    ax.set_ylabel('Excited State Population')
    ax.set_title('Multi-Qubit T1 Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
```

### 3. MultiQubitRabi Implementation

**File**: `leeq/experiments/builtin/basic/calibrations/rabi.py`

#### Step 1: Add simulation check
```python
def run(self, duts, collection_name=None, frequencies=None, 
        phases=None, time_length=None, time_resolution=None):
    # Add after parameter validation
    if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
        return self.run_simulated(
            duts=duts,
            frequencies=frequencies,
            phases=phases,
            time_length=time_length,
            time_resolution=time_resolution
        )
```

#### Step 2: Implement `run_simulated()` method
```python
def run_simulated(self, duts, frequencies=None, phases=None, 
                  time_length=None, time_resolution=None):
    """
    Simulate Rabi oscillations for multiple qubits in parallel.
    Each qubit oscillates independently at its Rabi frequency.
    """
    import numpy as np
    
    # Default parameters
    if time_length is None:
        time_length = 2000  # ns, typical for seeing multiple oscillations
    if time_resolution is None:
        time_resolution = 4  # ns
    
    # Create time array
    time_range = np.arange(0, time_length, time_resolution)
    
    # Handle frequency and phase defaults
    if frequencies is None:
        frequencies = [dut.get_c1('f01') for dut in duts]
    if phases is None:
        phases = [0] * len(duts)
    
    results = {}
    
    # Simulate each qubit
    for i, dut in enumerate(duts):
        virtual_qubit = self.setup.get_virtual_qubit(dut)
        if virtual_qubit is None:
            raise ValueError(f"No virtual qubit found for {dut}")
        
        # Get Rabi frequency for this qubit
        # Assuming we're driving at optimal amplitude
        omega_per_amp = self.setup.get_omega_per_amp(dut.get_c1('channel'))
        drive_amp = 0.5  # Typical amplitude for Rabi oscillations
        rabi_freq = drive_amp * omega_per_amp  # MHz
        
        # Convert to angular frequency
        omega_rabi = 2 * np.pi * rabi_freq  # rad/μs
        
        # Calculate Rabi oscillations
        # P_e(t) = sin²(Ω*t/2) for resonant drive starting from |0⟩
        phase_rad = phases[i] * np.pi / 180  # Convert to radians
        populations = np.sin((omega_rabi * time_range * 1e-3 / 2) + phase_rad) ** 2
        
        # Add decoherence envelope
        # Account for both T1 and T2* effects
        t1 = virtual_qubit.t1
        t2 = virtual_qubit.t2
        
        # Exponential decay due to decoherence
        t_eff = 1 / (1/t1 + 1/(2*t2))  # Effective decay time
        decay_envelope = np.exp(-time_range * 1e-3 / t_eff)
        populations = populations * decay_envelope
        
        # Add measurement noise
        if self.setup.status.get('Sampling_Noise', True):
            readout_fidelity = getattr(virtual_qubit, 'readout_fidelity', 0.95)
            noise_level = (1 - readout_fidelity) / 2
            noise = np.random.normal(0, noise_level, len(populations))
            populations = np.clip(populations + noise, 0, 1)
        
        results[dut.name] = {
            'times': time_range,
            'populations': populations,
            'rabi_frequency': rabi_freq,
            'frequency': frequencies[i],
            'phase': phases[i],
            'decay_time': t_eff
        }
    
    # Format combined results
    combined_results = {
        'time_range': time_range,
        'qubits': results,
        'parameters': {
            'time_length': time_length,
            'time_resolution': time_resolution
        }
    }
    
    return combined_results
```

### 4. MultiQubitRamseyMultilevel Implementation

**File**: `leeq/experiments/builtin/basic/calibrations/ramsey.py`

#### Step 1: Add simulation check
```python
def run(self, duts, collection_name=None, frequencies=None, 
        delay_range=None, initial_lpb=None):
    # Add after parameter setup
    if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
        return self.run_simulated(
            duts=duts,
            frequencies=frequencies,
            delay_range=delay_range
        )
```

#### Step 2: Implement `run_simulated()` method
```python
def run_simulated(self, duts, frequencies=None, delay_range=None):
    """
    Simulate Ramsey fringes for multiple qubits in parallel.
    Each qubit shows oscillations at its detuning frequency with T2* decay.
    """
    import numpy as np
    
    # Default delay range
    if delay_range is None:
        # Base on average T2*
        t2_values = [self.setup.get_virtual_qubit(dut).t2 for dut in duts]
        max_delay = 2 * np.mean(t2_values)
        delay_range = np.linspace(0, max_delay, 101)
    
    delay_range = np.array(delay_range)
    
    # Default frequencies (on resonance)
    if frequencies is None:
        frequencies = [dut.get_c1('f01') for dut in duts]
    
    results = {}
    
    for i, dut in enumerate(duts):
        virtual_qubit = self.setup.get_virtual_qubit(dut)
        if virtual_qubit is None:
            raise ValueError(f"No virtual qubit found for {dut}")
        
        # Calculate detuning
        qubit_freq = dut.get_c1('f01')
        drive_freq = frequencies[i]
        detuning = drive_freq - qubit_freq  # MHz
        
        # Get T2* (dephasing time)
        t2_star = virtual_qubit.t2  # μs
        
        # Ramsey signal: oscillations at detuning frequency with T2* decay
        # P_e(t) = 0.5 * [1 + cos(2π*Δf*t) * exp(-t/T2*)]
        phase_evolution = 2 * np.pi * detuning * delay_range  # rad
        decay = np.exp(-delay_range / t2_star)
        populations = 0.5 * (1 + np.cos(phase_evolution) * decay)
        
        # Add measurement noise
        if self.setup.status.get('Sampling_Noise', True):
            readout_fidelity = getattr(virtual_qubit, 'readout_fidelity', 0.95)
            noise_level = (1 - readout_fidelity) / 2
            noise = np.random.normal(0, noise_level, len(populations))
            populations = np.clip(populations + noise, 0, 1)
        
        # Extract fitted parameters
        fitted_detuning = detuning
        fitted_t2 = t2_star
        
        results[dut.name] = {
            'delays': delay_range,
            'populations': populations,
            'fitted_frequency': qubit_freq + fitted_detuning,
            'fitted_t2': fitted_t2,
            'detuning': fitted_detuning,
            'drive_frequency': drive_freq
        }
    
    # Combined results
    combined_results = {
        'delay_range': delay_range,
        'qubits': results,
        'timestamp': time.time()
    }
    
    return combined_results
```

## Testing Strategy

### 1. Unit Tests Structure
```
tests/experiments/builtin/basic/
├── calibrations/
│   ├── test_rabi_simulation.py
│   └── test_ramsey_simulation.py
└── characterizations/
    └── test_t1_simulation.py
```

### 2. Common Test Pattern
```python
import pytest
import numpy as np
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup

class TestExperimentSimulation:
    @pytest.fixture
    def setup_single_qubit(self):
        """Create single qubit simulation setup."""
        vqubit = VirtualTransmon(
            name="Q0",
            qubit_frequency=5000,
            anharmonicity=-200,
            t1=50,
            t2=30
        )
        return HighLevelSimulationSetup(
            name="TestSetup",
            virtual_qubits={0: vqubit}
        )
    
    @pytest.fixture
    def setup_two_qubits(self):
        """Create two qubit simulation setup."""
        vqubit1 = VirtualTransmon(name="Q0", qubit_frequency=5000, t1=50, t2=30)
        vqubit2 = VirtualTransmon(name="Q1", qubit_frequency=5100, t1=45, t2=25)
        return HighLevelSimulationSetup(
            name="TestSetup",
            virtual_qubits={0: vqubit1, 1: vqubit2}
        )
```

### 3. Validation Tests
- Compare simulation results with theoretical expectations
- Verify noise is applied correctly when enabled
- Check multi-qubit experiments run independently
- Validate fitted parameters match input parameters

## Integration Checklist

- [ ] Implement PowerRabi.run_simulated()
- [ ] Add PowerRabi unit tests
- [ ] Implement MultiQubitT1.run_simulated()
- [ ] Add MultiQubitT1 unit tests
- [ ] Implement MultiQubitRabi.run_simulated()
- [ ] Add MultiQubitRabi unit tests
- [ ] Implement MultiQubitRamseyMultilevel.run_simulated()
- [ ] Add MultiQubitRamseyMultilevel unit tests
- [ ] Update simulation_supported_experiments.md
- [ ] Run full test suite to ensure no regressions
- [ ] Create example notebook demonstrating new simulations

## Code Review Checklist

- [ ] All methods have proper docstrings
- [ ] Type hints are included where appropriate
- [ ] Error handling for missing virtual qubits
- [ ] Consistent with existing simulation patterns
- [ ] Results format matches hardware execution
- [ ] Noise models are physically reasonable
- [ ] No hardcoded parameters (use qubit properties)

## Future Enhancements

After completing these easy wins:
1. Add support for off-resonant driving effects
2. Include power-dependent AC Stark shifts
3. Model crosstalk between qubits
4. Add support for custom noise models
5. Implement batch execution optimization