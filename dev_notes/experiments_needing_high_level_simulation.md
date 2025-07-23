# Experiments Needing High-Level Simulation Support in LeeQ

## Overview

This document analyzes the built-in experiments in LeeQ that currently lack high-level simulation support and provides recommendations for which experiments should be prioritized for implementation.

## Current Status

### Experiments WITH High-Level Simulation (15 total)
- ✅ NormalisedRabi
- ✅ SimpleRamseyMultilevel
- ✅ QubitSpectroscopyFrequency
- ✅ ResonatorSweepTransmissionWithExtraInitialLPB
- ✅ DragCalibrationSingleQubitMultilevel
- ✅ DragPhaseCalibrationMultiQubitsMultilevel (placeholder)
- ✅ PingPongSingleQubitMultilevel
- ✅ AmpPingpongCalibrationSingleQubitMultilevel (placeholder)
- ✅ MeasurementCalibrationMultilevelGMM
- ✅ SimpleT1
- ✅ SimpleT2
- ✅ StarkRamseyMultilevel
- ✅ ConditionalStarkShiftContinuous
- ✅ ConditionalStarkShiftRepeatedGate
- ✅ ConditionalStarkEchoTuneUpAI (placeholder)

### Experiments WITHOUT High-Level Simulation (42 total)

## Priority Recommendations for Implementation

### 🟢 High Priority (Easy to Implement)

#### 1. **PowerRabi** (`leeq/experiments/builtin/basic/calibrations/rabi.py:374`)
- **Complexity**: Low
- **Description**: Sweeps pulse amplitude to find optimal π pulse power
- **Implementation**: Reuse NormalisedRabi physics, just sweep amplitude instead of duration
- **Estimated effort**: 2-3 hours

#### 2. **MultiQubitRabi** (`leeq/experiments/builtin/basic/calibrations/rabi.py`)
- **Complexity**: Low
- **Description**: Runs Rabi oscillations on multiple qubits simultaneously
- **Implementation**: Parallelize existing NormalisedRabi simulation
- **Estimated effort**: 2-3 hours

#### 3. **MultiQubitT1** (`leeq/experiments/builtin/basic/characterizations/t1.py:258`)
- **Complexity**: Low
- **Description**: Measures T1 on multiple qubits in parallel
- **Implementation**: Run SimpleT1 simulation for each qubit independently
- **Estimated effort**: 1-2 hours

#### 4. **MultiQubitRamseyMultilevel** (`leeq/experiments/builtin/basic/calibrations/ramsey.py`)
- **Complexity**: Low
- **Description**: Ramsey experiment on multiple qubits
- **Implementation**: Parallelize existing SimpleRamseyMultilevel simulation
- **Estimated effort**: 2-3 hours

### 🟡 Medium Priority (Moderate Complexity)

#### 5. **QubitSpectroscopyAmplitudeFrequency** (`leeq/experiments/builtin/basic/calibrations/spectroscopy.py`)
- **Complexity**: Medium
- **Description**: 2D spectroscopy sweep over amplitude and frequency
- **Implementation**: Extend QubitSpectroscopyFrequency to include amplitude dimension
- **Estimated effort**: 4-6 hours

#### 6. **ZZShiftTwoQubitMultilevel** (`leeq/experiments/builtin/basic/calibrations/residual_zz.py:72`)
- **Complexity**: Medium
- **Description**: Measures ZZ coupling between neighboring qubits
- **Implementation**: Model two-qubit Hamiltonian with coupling terms
- **Estimated effort**: 6-8 hours

#### 7. **GeneralisedSingleDutStateTomography** (`leeq/experiments/builtin/tomography/`)
- **Complexity**: Medium
- **Description**: Quantum state tomography for single qubit
- **Implementation**: Simulate measurements in different bases, reconstruct density matrix
- **Estimated effort**: 8-10 hours

#### 8. **MeasurementCollectTraces** (`leeq/experiments/builtin/basic/calibrations/collect_state_disrcrimination_data.py`)
- **Complexity**: Medium
- **Description**: Collects raw IQ measurement traces
- **Implementation**: Generate realistic IQ distributions based on qubit state
- **Estimated effort**: 4-6 hours

### 🔴 Low Priority (High Complexity or Limited Benefit)

#### 9. **RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem**
- **Complexity**: High
- **Description**: Full randomized benchmarking with Clifford sequences
- **Reason for low priority**: Requires complex error modeling and Clifford simulation
- **Estimated effort**: 20+ hours

#### 10. **GeneralisedProcessTomography**
- **Complexity**: High
- **Description**: Full quantum process tomography
- **Reason for low priority**: Very complex, requires full process matrix simulation
- **Estimated effort**: 20+ hours

#### 11. **HamiltonianTomography Experiments**
- **Complexity**: High
- **Description**: Reconstruct system Hamiltonian from measurements
- **Reason for low priority**: Specialized use case, complex implementation
- **Estimated effort**: 30+ hours

#### 12. **GRAPESingleQubitGate**
- **Complexity**: High
- **Description**: Optimal control pulse generation
- **Reason for low priority**: Not a characterization experiment, optimization-focused
- **Estimated effort**: 40+ hours

## Implementation Strategy

### Phase 1: Low-Hanging Fruit (1-2 weeks)
1. Implement PowerRabi - extends existing Rabi simulation
2. Implement MultiQubitT1 - simple parallelization
3. Implement MultiQubitRabi - reuse single-qubit logic
4. Implement MultiQubitRamseyMultilevel - parallel Ramsey

### Phase 2: Important Characterizations (2-3 weeks)
5. Implement QubitSpectroscopyAmplitudeFrequency - 2D spectroscopy
6. Implement ZZShiftTwoQubitMultilevel - crucial for multi-qubit systems
7. Implement MeasurementCollectTraces - useful for debugging

### Phase 3: Advanced Features (1+ months)
8. Implement single-qubit state tomography
9. Consider simplified randomized benchmarking
10. Evaluate need for process tomography

## Technical Considerations

### Common Patterns to Follow

1. **Check for simulation mode**:
```python
if self.setup_manager.status.get('High_Level_Simulation_Mode', False):
    return self.run_simulated(...)
```

2. **Access virtual qubits**:
```python
virtual_qubit = self.setup.get_virtual_qubit(qubit)
```

3. **Apply theoretical formulas with noise**:
```python
if self.setup.status.get('Sampling_Noise', True):
    # Add realistic noise to simulation
```

### Testing Strategy

For each new simulation implementation:
1. Create unit tests comparing to theoretical expectations
2. Validate against existing pulse-level simulations
3. Test with various noise parameters
4. Ensure consistent API with hardware execution

## Benefits of Implementation

1. **Faster Development**: Test experiments without hardware access
2. **Parameter Optimization**: Quickly explore parameter spaces
3. **Education**: Help users understand quantum experiments
4. **Debugging**: Isolate hardware issues from algorithmic issues
5. **CI/CD**: Enable comprehensive testing in continuous integration

## Notes for Developers

- Start with experiments that have clear theoretical models
- Reuse existing simulation components where possible
- Maintain consistency with hardware execution paths
- Document assumptions and approximations
- Include appropriate noise models for realism