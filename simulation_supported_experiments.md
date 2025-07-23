# LeeQ Experiments with Simulation Support

This document provides a comprehensive overview of all experiments in the LeeQ package that support simulations. These experiments can be run without physical quantum hardware, making them useful for testing, education, and parameter optimization.

## Table of Contents
1. [Basic Calibration Experiments](#basic-calibration-experiments)
2. [Characterization Experiments](#characterization-experiments)
3. [Multi-Qubit Gate Experiments](#multi-qubit-gate-experiments)
4. [AI-Assisted Experiments](#ai-assisted-experiments)
5. [How Simulations Work](#how-simulations-work)

## Basic Calibration Experiments

### NormalisedRabi
- **Location**: `leeq/experiments/builtin/basic/calibrations/rabi.py`
- **Purpose**: Rough calibration of driving amplitude for single-qubit gates through Rabi oscillations
- **Parameters**:
  - `dut_qubit`: Device under test qubit
  - `amp`: Amplitude of the Rabi pulse (default: 0.05)
  - `start`: Start time for sweep (default: 0.01)
  - `stop`: Stop time for sweep (default: 0.15)
  - `step`: Time step (default: 0.001)
  - `collection_name`: Pulse collection name (default: 'f01')
- **Simulation**: Calculates Rabi oscillations using the generalized Rabi formula, accounting for detuning and Rabi rate. Adds sampling noise if enabled.

### PowerRabi
- **Location**: `leeq/experiments/builtin/basic/calibrations/rabi.py`
- **Purpose**: Calibration of pi pulse amplitude by sweeping drive power instead of pulse duration
- **Parameters**:
  - `dut_qubit`: Device under test qubit
  - `width`: Pulse width (optional, uses existing pi pulse width if not specified)
  - `amp_start`: Start amplitude (default: 0.01)
  - `amp_stop`: Stop amplitude (default: 0.4)
  - `amp_step`: Amplitude step (default: 0.01)
  - `collection_name`: Pulse collection name (default: 'f01')
- **Simulation**: Calculates population transfer as a function of drive amplitude using P = sin²(Ω*t/2) where Ω = amp * omega_per_amp. Finds optimal amplitude for pi rotation.

### SimpleRamseyMultilevel
- **Location**: `leeq/experiments/builtin/basic/calibrations/ramsey.py`
- **Purpose**: Measures qubit detuning and dephasing time (T2*) through Ramsey interferometry
- **Parameters**:
  - `dut`: Device under test
  - `collection_name`: Transition name (default: 'f01')
  - `freq_offset`: Frequency offset from qubit frequency
  - `wait_time`: Maximum evolution time
  - `time_resolution`: Time step resolution
- **Simulation**: Generates Ramsey fringes with oscillations at the detuning frequency and exponential decay based on T2* values.

### ResonatorSweepTransmissionWithExtraInitialLPB
- **Location**: `leeq/experiments/builtin/basic/calibrations/resonator_spectroscopy.py`
- **Purpose**: Characterizes resonator frequency and transmission properties
- **Parameters**:
  - `start`: Start frequency (default: 8000 MHz)
  - `stop`: Stop frequency (default: 9000 MHz)
  - `step`: Frequency step (default: 5 MHz)
  - `num_avs`: Number of averages
  - `amp`: Drive amplitude
- **Simulation**: Generates resonator response based on virtual transmon parameters with realistic noise characteristics.

### QubitSpectroscopyFrequency
- **Location**: `leeq/experiments/builtin/basic/calibrations/qubit_spectroscopy.py`
- **Purpose**: Finds qubit transition frequency through spectroscopy measurements
- **Parameters**:
  - `start/stop/step`: Frequency sweep parameters
  - `pulse_amp`: Spectroscopy pulse amplitude
  - `pulse_width`: Spectroscopy pulse width
- **Simulation**: Generates spectroscopy peaks centered at the virtual qubit frequency with appropriate linewidth.

### DragCalibrationSingleQubitMultilevel
- **Location**: `leeq/experiments/builtin/basic/calibrations/drag.py`
- **Purpose**: Calibrates DRAG (Derivative Removal by Adiabatic Gate) pulse parameters to minimize leakage
- **Parameters**:
  - `drag_start`: Start value for DRAG parameter sweep
  - `drag_stop`: Stop value for DRAG parameter sweep
  - `drag_step`: Step size for DRAG parameter
  - `collection_name`: Pulse collection to calibrate
- **Simulation**: Simulates DRAG effectiveness by showing characteristic oscillation patterns based on leakage to higher levels.

### PingPongSingleQubitMultilevel
- **Location**: `leeq/experiments/builtin/basic/calibrations/pingpong.py`
- **Purpose**: Fine-tunes pulse amplitude through ping-pong sequences (alternating X and Y rotations)
- **Parameters**:
  - `n_start`: Start value for number of pulse pairs
  - `n_stop`: Stop value for number of pulse pairs
  - `n_step`: Step size for pulse pairs
  - `collection_name`: Pulse collection name
- **Simulation**: Generates oscillation patterns that reveal amplitude errors in the calibrated pulses.

### MeasurementCalibrationMultilevelGMM
- **Location**: `leeq/experiments/builtin/basic/calibrations/state_discrimination/gaussian_mixture.py`
- **Purpose**: Calibrates measurement using Gaussian Mixture Models for optimal state discrimination
- **Parameters**:
  - `sweep_lpb_list`: List of state preparation pulses
  - `mprim_index`: Measurement primitive index
  - `freq`: Optional measurement frequency override
  - `amp`: Optional measurement amplitude override
- **Simulation**: Generates synthetic IQ data for different quantum states with realistic noise and separation characteristics.

## Characterization Experiments

### SimpleT1
- **Location**: `leeq/experiments/builtin/basic/characterizations/t1.py`
- **Purpose**: Measures qubit relaxation time (T1) - the time for excited state to decay to ground state
- **Parameters**:
  - `time_length`: Total experiment duration (default: 100 µs)
  - `time_resolution`: Time step (default: 1 µs)
  - `collection_name`: Transition to measure (default: 'f01')
- **Simulation**: Generates exponential decay curve based on the virtual qubit's T1 value with appropriate noise.

### SpinEchoMultiLevel
- **Location**: `leeq/experiments/builtin/basic/characterizations/t2.py`
- **Purpose**: Measures T2 echo coherence time using Hahn echo sequence (X-wait-Y-wait-X)
- **Parameters**:
  - `free_evolution_time`: Maximum evolution time (default: 100 µs)
  - `time_resolution`: Time step (default: 2 µs)
  - `collection_name`: Transition name (default: 'f01')
- **Simulation**: Generates echo decay curve showing improved coherence compared to T2* due to refocusing of low-frequency noise.

## Multi-Qubit Gate Experiments

### StarkRamseyMultilevel
- **Location**: `leeq/experiments/builtin/multi_qubit_gates/ac_stark/ac_stark_shift.py`
- **Purpose**: Characterizes AC Stark shift induced by off-resonant drives for conditional operations
- **Parameters**:
  - `stark_amp`: Amplitude of the Stark drive
  - `stark_freq_offset`: Frequency offset of Stark drive from qubit
  - `ramsey_freq`: Frequency for Ramsey experiment
  - `echo_time`: Evolution time for measurement
- **Simulation**: Calculates Stark-shifted Ramsey fringes showing frequency shift proportional to drive power.

### ConditionalStarkShiftContinuous
- **Location**: `leeq/experiments/builtin/multi_qubit_gates/conditional_stark_ai.py`
- **Purpose**: Implements continuous conditional Stark shift for two-qubit entangling gates
- **Parameters**:
  - `duts`: List of qubits [control, target]
  - `amp_control`: Drive amplitude for control qubit
  - `amp_target`: Drive amplitude for target qubit
  - `frequency`: Stark drive frequency
  - `phase_diff`: Phase difference between control and target drives
- **Simulation**: Simulates conditional phase accumulation based on qubit states.

### ConditionalStarkShiftRepeatedGate
- **Location**: `leeq/experiments/builtin/multi_qubit_gates/conditional_stark_ai.py`
- **Purpose**: Implements repeated gate version of conditional Stark shift with discrete pulses
- **Parameters**: Same as ConditionalStarkShiftContinuous
- **Simulation**: Simulates effects of applying discrete gate operations rather than continuous drives.

## AI-Assisted Experiments

### ConditionalStarkEchoTuneUpAI
- **Location**: `leeq/experiments/builtin/multi_qubit_gates/conditional_stark_ai.py`
- **Purpose**: AI-assisted tuning of conditional Stark echo sequences for optimal gate performance
- **Simulation**: Provides placeholder for AI-guided optimization routines.

### ConditionalStarkTwoQubitGateAmplitudeAdvise
- **Location**: `leeq/experiments/builtin/multi_qubit_gates/conditional_stark_ai.py`
- **Purpose**: Uses AI models to recommend optimal amplitude parameters for two-qubit gates
- **Simulation**: Uses trained models to suggest parameter values based on system characteristics.

## How Simulations Work

### 1. Simulation Mode Detection
Experiments check if they're running in simulation mode using:
- `setup().is_simulation` flag
- Obtaining a `HighLevelSimulationSetup` instance from the setup manager

### 2. Virtual Qubit Parameters
Each simulated experiment accesses virtual qubit parameters including:
- **Frequency parameters**: Qubit frequency (f01), anharmonicity (f12-f01)
- **Coherence times**: T1 (relaxation), T2 (dephasing), T2* (inhomogeneous dephasing)
- **Readout parameters**: Resonator frequency, dispersive shift, readout fidelity
- **Thermal properties**: Quiescent state distribution (ground/excited state populations)

### 3. Noise Modeling
Simulations include realistic noise sources:
- **Shot noise**: Statistical fluctuations from finite sampling
- **Thermal noise**: Based on qubit temperature and energy levels
- **Measurement noise**: IQ plane rotations and gaussian noise
- **Decoherence**: T1/T2 decay during evolution periods

### 4. Physics-Based Models
Each experiment implements the relevant quantum physics:
- **Rabi oscillations**: Using generalized Rabi formula including detuning effects
- **Ramsey fringes**: Oscillations at detuning frequency with T2* decay envelope
- **Relaxation/Dephasing**: Exponential decay models for T1 and T2 processes
- **Stark shifts**: Hamiltonian evolution under off-resonant drives
- **Multi-level dynamics**: Including leakage to higher transmon levels

### 5. Data Generation
Simulated data includes:
- Appropriate signal shapes based on physics models
- Realistic noise characteristics
- Proper normalization and units
- Complex IQ values for heterodyne measurements

The simulation framework enables:
- **Testing**: Validate experiment code without hardware access
- **Education**: Learn quantum control concepts interactively
- **Optimization**: Pre-optimize parameters before hardware runs
- **Development**: Debug new experiments in a controlled environment