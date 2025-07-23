# High-Level Simulation Implementation Status

## Overview

This document tracks the current implementation status of high-level simulation support for LeeQ experiments.

**Last Updated**: 2025-01-23

## Implementation Status

### ✅ Fully Implemented (15 experiments)

| Experiment | Location | Test Coverage |
|------------|----------|---------------|
| NormalisedRabi | `leeq/experiments/builtin/basic/calibrations/rabi.py` | ✅ Tested |
| SimpleRamseyMultilevel | `leeq/experiments/builtin/basic/calibrations/ramsey.py` | ✅ Tested |
| QubitSpectroscopyFrequency | `leeq/experiments/builtin/basic/calibrations/spectroscopy.py` | ✅ Tested |
| ResonatorSweepTransmissionWithExtraInitialLPB | `leeq/experiments/builtin/basic/calibrations/spectroscopy.py` | ✅ Tested |
| DragCalibrationSingleQubitMultilevel | `leeq/experiments/builtin/basic/calibrations/drag.py` | ✅ Tested |
| PingPongSingleQubitMultilevel | `leeq/experiments/builtin/basic/calibrations/pingpong.py` | ✅ Tested |
| MeasurementCalibrationMultilevelGMM | `leeq/experiments/builtin/basic/calibrations/calibrate_readout.py` | ✅ Tested |
| SimpleT1 | `leeq/experiments/builtin/basic/characterizations/t1.py` | ✅ Tested |
| SimpleT2 | `leeq/experiments/builtin/basic/characterizations/t2_echo.py` | ✅ Tested |
| StarkRamseyMultilevel | `leeq/experiments/builtin/multi_qubit_gates/stark_tuneup.py` | ✅ Tested |
| ConditionalStarkShiftContinuous | `leeq/experiments/builtin/multi_qubit_gates/conditional_stark.py` | ✅ Tested |
| ConditionalStarkShiftRepeatedGate | `leeq/experiments/builtin/multi_qubit_gates/conditional_stark.py` | ✅ Tested |
| DragPhaseCalibrationMultiQubitsMultilevel | `leeq/experiments/builtin/basic/calibrations/drag.py` | ⚠️ Placeholder |
| AmpPingpongCalibrationSingleQubitMultilevel | `leeq/experiments/builtin/basic/calibrations/pingpong.py` | ⚠️ Placeholder |
| ConditionalStarkEchoTuneUpAI | `leeq/experiments/builtin/multi_qubit_gates/conditional_stark.py` | ⚠️ Placeholder |

### 🚧 Ready for Implementation (4 experiments)

#### 1. PowerRabi

**Location**: `leeq/experiments/builtin/basic/calibrations/rabi.py:374`

**Implementation Status**: Code provided in implementation plan

**Test Cases Needed**:
```python
# tests/experiments/builtin/basic/calibrations/test_power_rabi_simulation.py

def test_power_rabi_finds_pi_amplitude():
    """Test that PowerRabi correctly identifies pi pulse amplitude."""
    # Should find amplitude that gives maximum population transfer
    
def test_power_rabi_with_noise():
    """Test PowerRabi simulation with readout noise enabled."""
    # Verify noise is applied correctly
    
def test_power_rabi_off_resonance():
    """Test PowerRabi with detuned frequency."""
    # Should show reduced oscillation amplitude
    
def test_power_rabi_multiple_amplitudes():
    """Test with fine and coarse amplitude sweeps."""
    # Verify resolution affects accuracy
```

#### 2. MultiQubitT1

**Location**: `leeq/experiments/builtin/basic/characterizations/t1.py:258`

**Implementation Status**: Code provided in implementation plan

**Test Cases Needed**:
```python
# tests/experiments/builtin/basic/characterizations/test_multi_qubit_t1_simulation.py

def test_multi_qubit_t1_independent_decay():
    """Test that each qubit decays independently with its own T1."""
    # Create qubits with different T1 values
    # Verify each follows exponential decay
    
def test_multi_qubit_t1_parallel_execution():
    """Test that all qubits are measured in parallel."""
    # Results should have same time points for all qubits
    
def test_multi_qubit_t1_with_thermal_population():
    """Test T1 measurement with non-zero thermal population."""
    # Should decay to thermal equilibrium, not zero
    
def test_multi_qubit_t1_delay_range_auto():
    """Test automatic delay range calculation."""
    # Should scale with average T1 time
```

#### 3. MultiQubitRabi

**Location**: `leeq/experiments/builtin/basic/calibrations/rabi.py`

**Implementation Status**: Code provided in implementation plan

**Test Cases Needed**:
```python
# tests/experiments/builtin/basic/calibrations/test_multi_qubit_rabi_simulation.py

def test_multi_qubit_rabi_different_frequencies():
    """Test Rabi oscillations with different frequencies per qubit."""
    # Each qubit should oscillate at its own Rabi frequency
    
def test_multi_qubit_rabi_with_decoherence():
    """Test that T1/T2 effects are included."""
    # Oscillations should decay over time
    
def test_multi_qubit_rabi_phase_control():
    """Test phase parameter affects oscillations correctly."""
    # Phase should shift the oscillation pattern
    
def test_multi_qubit_rabi_time_resolution():
    """Test effect of time resolution on results."""
    # Finer resolution should capture oscillations better
```

#### 4. MultiQubitRamseyMultilevel

**Location**: `leeq/experiments/builtin/basic/calibrations/ramsey.py`

**Implementation Status**: Code provided in implementation plan

**Test Cases Needed**:
```python
# tests/experiments/builtin/basic/calibrations/test_multi_qubit_ramsey_simulation.py

def test_multi_qubit_ramsey_detuning():
    """Test Ramsey fringes with known detuning."""
    # Oscillation frequency should match detuning
    
def test_multi_qubit_ramsey_on_resonance():
    """Test Ramsey with no detuning."""
    # Should show only T2* decay, no oscillations
    
def test_multi_qubit_ramsey_t2_extraction():
    """Test that fitted T2 matches input T2."""
    # Decay envelope should give correct T2*
    
def test_multi_qubit_ramsey_different_t2():
    """Test with qubits having different T2 times."""
    # Each qubit should decay at its own rate
```

### ❌ Not Yet Implemented (38 experiments)

#### High Priority Candidates

| Experiment | Complexity | Reason for Priority |
|------------|------------|-------------------|
| QubitSpectroscopyAmplitudeFrequency | Medium | 2D spectroscopy is common calibration tool |
| ZZShiftTwoQubitMultilevel | Medium | Critical for two-qubit gate calibration |
| MeasurementCollectTraces | Low | Useful for debugging readout issues |
| MultiQuditT1Decay | Low | Extension of MultiQubitT1 to multiple levels |
| RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem | High | Important but complex - needs error models |

#### Complete List of Unimplemented Experiments

**Basic Calibrations**:
- QubitSpectroscopyAmplitudeFrequency
- ResonatorSweepAmpFreqWithExtraInitialLPB  
- ResonatorSweepTransmissionXiComparison
- MeasurementScanParams
- CrossAllXYDragMultiRunSingleQubitMultilevel
- PingPongMultiQubitMultilevel
- AmpTuneUpMultiQubitMultilevel
- CalibrateFullAssignmentMatrices
- CalibrateSingleDutAssignmentMatrices
- MeasurementCollectTraces
- CalibrateOptimizedFrequencyWith2QZZShift
- ZZShiftTwoQubitMultilevel
- MultilevelTransmonTuneup

**Basic Characterizations**:
- MultiQuditT1Decay
- RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem

**Multi-Qubit Gates**:
- StarkZZShiftTwoQubitMultilevel
- StarkRepeatedGateRabi
- StarkContinuesRabi
- StarkRepeatedGateDRAGLeakageCalibration
- ConditionalStarkShiftContinuousPhaseSweep
- ConditionalStarkTwoQubitGateAIParameterSearchFull
- ConditionalStarkTwoQubitGateAIParameterSearchBase
- ConditionalStarkTwoQubitGateAmplitudeAdvise
- ConditionalStarkTwoQubitGateFrequencyAdvise
- ConditionalStarkFineFrequencyTuneUp
- ConditionalStarkFineAmpTuneUp
- ConditionalStarkFinePhaseTuneUp
- ConditionalStarkFineRiseTuneUp
- ConditionalStarkFineTruncTuneUp
- ConsidtionalStarkSpectroscopyDifferenceBase
- ConditionalStarkTuneUpRepeatedGateXY
- ConditionalStarkEchoTuneUp
- RandomizedBenchmarking2Qubits
- RandomizedBenchmarking2QubitsInterleavedComparison

**Tomography**:
- GeneralisedSingleDutStateTomography
- GeneralisedSingleDutProcessTomography
- GeneralisedStateTomography
- GeneralisedProcessTomography

**Hamiltonian Tomography**:
- HamiltonianTomographyBaseSingleQudit
- HamiltonianTomographySingleQubitBase

**Optimal Control**:
- GRAPESingleQubitGate

## Test Infrastructure

### Common Test Fixtures

```python
# tests/experiments/builtin/conftest.py

import pytest
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup

@pytest.fixture
def virtual_transmon_typical():
    """Standard virtual transmon for testing."""
    return VirtualTransmon(
        name="Q0",
        qubit_frequency=5000.0,  # MHz
        anharmonicity=-200.0,    # MHz
        t1=50.0,                 # μs
        t2=30.0,                 # μs
        readout_frequency=7000.0, # MHz
        readout_linewidth=5.0,   # MHz
        readout_dispersive_shift=2.0,  # MHz
        readout_fidelity=0.95
    )

@pytest.fixture
def virtual_transmon_noisy():
    """Virtual transmon with more realistic noise."""
    return VirtualTransmon(
        name="Q0_noisy",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=20.0,  # Shorter T1
        t2=15.0,  # Shorter T2
        readout_fidelity=0.85,  # Lower fidelity
        quiescent_state_distribution=[0.85, 0.12, 0.03]  # More thermal population
    )

@pytest.fixture
def two_qubit_setup():
    """Two coupled virtual qubits."""
    q0 = VirtualTransmon(name="Q0", qubit_frequency=5000.0, t1=50.0, t2=30.0)
    q1 = VirtualTransmon(name="Q1", qubit_frequency=5100.0, t1=45.0, t2=25.0)
    
    setup = HighLevelSimulationSetup(
        name="TwoQubitSim",
        virtual_qubits={0: q0, 1: q1}
    )
    setup.set_coupling_strength_by_name("Q0", "Q1", 2.0)  # 2 MHz coupling
    return setup
```

### Test Utilities

```python
# tests/experiments/builtin/utils.py

import numpy as np

def assert_exponential_decay(times, values, expected_tau, tolerance=0.1):
    """Verify data follows exponential decay with given time constant."""
    # Fit exponential and compare tau
    from scipy.optimize import curve_fit
    
    def exp_decay(t, a, tau, c):
        return a * np.exp(-t/tau) + c
    
    popt, _ = curve_fit(exp_decay, times, values, p0=[1, expected_tau, 0])
    fitted_tau = popt[1]
    
    assert abs(fitted_tau - expected_tau) / expected_tau < tolerance, \
        f"Fitted tau {fitted_tau:.2f} differs from expected {expected_tau:.2f}"

def assert_sinusoidal(times, values, expected_frequency, tolerance=0.1):
    """Verify data follows sinusoidal pattern with given frequency."""
    # Use FFT to extract dominant frequency
    from scipy.fft import fft, fftfreq
    
    yf = fft(values - np.mean(values))
    xf = fftfreq(len(times), times[1] - times[0])
    
    # Find peak frequency
    peak_idx = np.argmax(np.abs(yf[:len(yf)//2]))
    measured_freq = abs(xf[peak_idx])
    
    assert abs(measured_freq - expected_frequency) / expected_frequency < tolerance, \
        f"Measured frequency {measured_freq:.3f} differs from expected {expected_frequency:.3f}"
```

## Implementation Checklist

### For Each New Simulation Implementation

- [ ] Add `High_Level_Simulation_Mode` check in `run()` method
- [ ] Implement `run_simulated()` method with proper physics
- [ ] Include noise modeling when `Sampling_Noise` is enabled
- [ ] Match output format of hardware execution
- [ ] Add comprehensive unit tests
- [ ] Update this status document
- [ ] Update `simulation_supported_experiments.md`
- [ ] Verify no regression in existing tests

### Code Structure Requirements

1. **Simulation Check Pattern**:
```python
if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
    return self.run_simulated(...)
```

2. **Virtual Qubit Access**:
```python
virtual_qubit = self.setup.get_virtual_qubit(dut)
if virtual_qubit is None:
    raise ValueError(f"No virtual qubit found for {dut}")
```

3. **Noise Application**:
```python
if self.setup.status.get('Sampling_Noise', True):
    # Apply appropriate noise model
```

4. **Result Format**:
- Must match hardware execution output structure
- Include all fields expected by analysis code
- Use consistent units (MHz, μs, etc.)