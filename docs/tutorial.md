# LeeQ Tutorial: From Quantum Concepts to Advanced Experiments

Welcome to the comprehensive LeeQ tutorial. This guide will take you on a journey from fundamental quantum computing concepts to advanced experimental procedures. LeeQ is a powerful framework for orchestrating quantum computing experiments on superconducting circuits, designed for both simulation and real hardware.

## Learning Path Overview

This tutorial follows a progressive learning approach:

1. **Quantum Computing Fundamentals in LeeQ** - Understanding qubits, gates, and measurements
2. **LeeQ Architecture and Components** - Core concepts and object model
3. **Single Qubit Experiments** - Basic operations and calibration
4. **Multi-Qubit Systems** - Entanglement and two-qubit gates
5. **Advanced Calibration Procedures** - Theory and practical implementation
6. **Custom Experiment Development** - Building your own experiments

### Interactive Learning Experience

For the best learning experience, read the concepts below and then try the hands-on interactive notebooks:

!!! tip "Interactive Notebooks"
    - **LeeQ Basics** - Set up simulation and run first experiments
    - **Single Qubit Operations** - Practice with quantum gates and calibration
    - **Multi-Qubit Systems** - Explore entanglement and two-qubit gates
    - **Calibration Workflows** - Complete calibration procedures
    - **AI Integration** - Automated experiment generation

!!! note "Additional Resources"
    After completing the tutorial, explore specific techniques in the [User Guide](guide/concepts.md) and learn about [experiments](guide/experiments.md) and [calibrations](guide/calibrations.md).

## Part 1: Quantum Computing Fundamentals in LeeQ

### Understanding Qubits in LeeQ

In quantum computing, a qubit is the fundamental unit of quantum information. Unlike classical bits that exist in states 0 or 1, qubits can exist in a **superposition** of both states simultaneously. In LeeQ, we work primarily with **transmon qubits** - superconducting devices that behave as artificial atoms.

#### Key Quantum Concepts:

**Superposition**: A qubit can be in state |0⟩, |1⟩, or any linear combination α|0⟩ + β|1⟩ where |α|² + |β|² = 1.

**Quantum Gates**: Operations that manipulate qubit states. Common single-qubit gates include:
- **X gate**: Bit flip (|0⟩ ↔ |1⟩)
- **Y gate**: Bit flip with phase
- **Z gate**: Phase flip
- **Hadamard (H)**: Creates superposition

**Measurement**: The process of reading a qubit's state, which collapses the superposition to either |0⟩ or |1⟩.

**Decoherence**: Quantum states are fragile and decay over time:
- **T1 (relaxation time)**: Time for excited state |1⟩ to decay to ground state |0⟩
- **T2 (dephasing time)**: Time for superposition to lose coherence

### Transmon Physics

Transmons are weakly anharmonic oscillators with multiple energy levels. In LeeQ, we typically work with the first few levels:
- |0⟩: Ground state
- |1⟩: First excited state (computational qubit states)
- |2⟩: Second excited state (leakage level)

The **anharmonicity** (difference between energy level spacings) allows us to address qubit transitions selectively.

!!! example "Try It Yourself"
    Ready to see these concepts in action? Continue reading to set up a simulation and explore quantum states interactively.

## Part 2: LeeQ Architecture and Components

### Core Object Model

LeeQ uses an object-oriented approach to quantum experiment orchestration. Understanding the hierarchy is crucial:

```
ExperimentManager
    └── Setup (e.g., HighLevelSimulationSetup)
        └── TransmonElement (DUT - Device Under Test)
            ├── Collections (pulse definitions)
            │   ├── f01 (0→1 transition)
            │   └── f12 (1→2 transition)
            └── Measurement Primitives
                ├── Readout configurations
                └── State discrimination
```

#### Key Components:

**TransmonElement (DUT)**: Represents a physical or virtual qubit with:
- Human-readable ID (hrid)
- Pulse collections for different transitions
- Measurement primitive definitions
- Calibration parameters

**Collections**: Group related pulses sharing common parameters:
- Drive collections (e.g., 'f01' for qubit transitions)
- Virtual operations (phase shifts for Z gates)

**Measurement Primitives**: Define how to read qubit states:
- Readout pulse parameters
- Data acquisition settings
- State discrimination methods

**Logical Primitives (LP)**: Basic operations like single pulses or delays

**Logical Primitive Blocks (LPB)**: Composite operations combining LPs:
- **Series**: Sequential execution (A + B)
- **Parallel**: Simultaneous execution (A * B)
- **Sweep**: Parameter sweeps for measurements

### LeeQ's Pulse-Based Approach

Unlike gate-based quantum computing abstractions, LeeQ works at the pulse level, giving you precise control over:

1. **Pulse shapes**: Gaussian, Blackman-Harris, DRAG, custom shapes
2. **Timing**: Exact pulse scheduling and delays
3. **Phases**: Real-time phase adjustments for gates
4. **Amplitudes**: Power control for Rabi rotations

This low-level control is essential for:
- Calibrating gate fidelities
- Characterizing qubit properties
- Implementing optimal control sequences
- Mitigating crosstalk and errors

!!! example "Practice with LPs and LPBs"
    See logical primitives in action in the sections below, where you'll learn to build and execute quantum gate sequences.

## Parameter Storage and Update

## DUT Object

At the heart of LeeQ is the **DUT (Device Under Test)** object, such as the `TransmonElement` object. The DUT Object represents the central configuration storage mechanism in LeeQ for quantum devices. The `TransmonElement` class specifically implements a superconducting transmon qubit as a DUT.

This object is pivotal for **parameter storage** and maintaining all **configuration** data that describes various elements like the channel, qubit frequency, and pulse shape. The DUT Object encapsulates all the information needed to control and interact with a quantum device, making it the primary interface between your experimental code and the underlying hardware.

To construct a DUT object, you define a dictionary as follows:

```python
from leeq.core.elements.built_in.qudit_transmon import TransmonElement

TransmonElement(name="Q1", parameters={
    'hrid': 'Q1',  # Human-readable ID
    'lpb_collections': lpb_collections,  # LPB collection definition dictionary
    'measurement_primitives': measurement_primitives  # Measurement primitives definition dictionary
})
```

When it comes to saving the calibration log, the configuration of each DUT object is stored on disk. This allows for the object to be reloaded and reconstructed later on.

## Collection

A **Collection** represents a **group of pulses** or virtual operations (like a phase shift for a virtual Z gate) that share **common parameters**. Collections are fundamental organizational units in LeeQ that allow you to define families of related operations with shared characteristics such as frequency, amplitude, and pulse shape.

The most common type is `SimpleDriveCollection`, which groups single-qubit drive pulses that operate on the same transition (e.g., the 0↔1 transition of a transmon). Below is an example of a collection configuration:

```python
lpb_collections = {
    'f01': {
        'type': 'SimpleDriveCollection',  # Class of the collection
        'freq': 4888.20,  # Frequency in MHz
        'channel': 0,  # Refer to QubiC LeeQ channel map for details
        'shape': 'blackman_drag',
        'amp': 0.21,
        'phase': 0.,  # Phase in radians
        'width': 0.05,  # Width in microseconds
        'alpha': 1e9,
        'trunc': 1.2
    }
}
```

You can add more items to the LPB collections to define various pulses. The convention "f<lower level><higher level>" is used to denote a pulse that drives transitions in a subspace, like `f13` for a two-photon transition drive between the 1 and 3 states of a transmon.

## Measurement Primitives

**Measurement Primitives** are definitions for **measurement pulses** that handle quantum state readout. These primitives define how to extract information from quantum systems through dispersive measurements. When a measurement primitive is activated, the **data acquisition** device automatically starts collecting data from the quantum system.

The most common type is `SimpleDispersiveMeasurement`, which implements dispersive readout where the qubit state affects the resonator frequency, allowing state discrimination through phase and amplitude measurements. Here's an example definition for qubit and qutrit readouts:

```python
measurement_primitives = {
    '0': {
        'type': 'SimpleDispersiveMeasurement',
        'freq': 9997.6,  # Frequency in MHz
        'channel': 1,  # Refer to QubiC LeeQ channel map for details
        'shape': 'square',
        'amp': 0.06,
        'phase': 0.,  # Phase in radians
        'width': 8,  # Width in microseconds
        'trunc': 1.2,
        'distinguishable_states': [0, 1]  # Distinguishable states
    },
    '1': {
        'type': 'SimpleDispersiveMeasurement',
        'freq': 9997.55,
        'channel': 1,
        'shape': 'square',
        'amp': 0.06,
        'phase': 0.,
        'width': 8,
        'trunc': 1.2,
        'distinguishable_states': [0, 1, 2]
    }
}
```

Post-experiment, a state classifier, such as one generated by the `MeasurementCalibrationMultilevelGMM` experiment, must be trained. This classifier is then stored in memory and applied to subsequent experiments until the kernel is shut down. To update the classifier, simply rerun the calibration process.

### Customizing Pulse Shapes

In LeeQ, pulse shapes are defined through functions that take a sampling rate as their first argument, followed by several other parameters, including optional ones.

```python
import numpy as np
from leeq.compiler.utils.time_base import get_t_list

def custom_gaussian_func(sampling_rate: int, amp: float, phase: float, width: float, trunc: float) -> np.array:
    """
    Create a custom Gaussian pulse shape with specified parameters.
    
    Args:
        sampling_rate: Sample rate for the pulse
        amp: Pulse amplitude
        phase: Phase in radians
        width: Pulse width in microseconds
        trunc: Truncation factor for the pulse
    
    Returns:
        Complex array representing the pulse shape
    """
    gauss_width = width / 2.0
    t = get_t_list(sampling_rate, width * trunc)
    return amp * np.exp(1.0j * phase) * np.exp(-((t - gauss_width) / gauss_width) ** 2).astype("complex64")
```

To integrate custom pulse shapes into LeeQ, use the `PulseShapeFactory` object. This singleton facilitates the registration of new pulse shapes. Custom pulse shapes can be added to `leeq/compiler/utils/pulse_shapes/basic_shapes.py` and made visible by including their names in the file’s `__all__` list for automatic loading. Alternatively, pulse shapes can be registered manually as needed:

```python
from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory

PulseShapeFactory().register_pulse_shape(
    pulse_shape_name='custom_gaussian_func',
    pulse_shape_function=custom_gaussian_func
)
```

## Orchestrating Pulses

LeeQ employs a tree structure for scheduling rather than a predefined schedule, introducing two key concepts:

**Logical Primitive**: The **basic operation**, typically a single pulse or delay, serving as a **tree leaf** in the pulse scheduling hierarchy. Each Logical Primitive represents an atomic operation that cannot be further subdivided.

**Logical Primitive Block (LPB)**: A composite element within the tree, including `LogicalPrimitiveBlockSeries`, `LogicalPrimitiveBlockParallel`, and `LogicalPrimitiveBlockSweep`.

`LogicalPrimitiveBlockSeries` signifies sequential execution of its children. It can be constructed using the `+` operator to combine LPs or LPBs.

`LogicalPrimitiveBlockParallel` indicates simultaneous start times for its children, created using the `*` operator.

`LogicalPrimitiveBlockSweep` pairs with a `Sweeper` for dynamic pulse adjustments during a sequence sweep.

Example:

```python
from leeq.core.primitives.logical_primitive_block import LogicalPrimitiveBlockSeries, LogicalPrimitiveBlockParallel

lpb_1 = LogicalPrimitiveBlockSeries([lp_1, lp_2, lp_3])

lpb_2 = LogicalPrimitiveBlockParallel([lpb_1, lp_4])  # Mixing LPBs and LPs

lpb_3 = lpb_1 + lpb_2  # Series combination

lpb_4 = lpb_1 * lpb_2  # Parallel combination
```

### Single Qubit Operations

Single qubit gates are accessible through the DUT object's collection, which organizes the operations by subspace. For instance:

```python
dut = duts_dict['Q1']
c1 = dut.get_c1('f01')  # Access the single qubit drive collection for subspace 0,1
lp = c1['X']  # X gate
lp = c1['Y']  # Y gate
lp = c1['Yp']  # +pi/2 Y gate
lp = c1['Ym']  # -pi/2 Y gate
```

Shortcut methods are available for composite gates, like:

```python
gate = dut.get_gate('qutrit_hadamard')
```

The returned object, typically an LPB, consists of a sequence of gates. Detailed documentation is available for `get_gate`.

To access measurement primitives:

```python
mprim = dut.get_measurement_prim_intlist(name='0')
```

`get_measurement_prim_intlist` offers single-shot, demodulated, and aggregated readouts, among other options detailed in the documentation.

## Adjusting Runtime Parameters

You can **update parameters** for gates or measurement primitives on-the-fly during experiments, with changes stored in memory. This dynamic parameter adjustment capability is crucial for real-time calibration and optimization workflows.

**Parameter updates** can be applied to any collection or measurement primitive using the `set_parameters()` method:

```python
# Update pulse amplitude
qubit.get_c1('f01').set_parameters({'amp': 0.52})

# Update frequency and phase simultaneously  
qubit.get_c1('f01').set_parameters({'freq': 5040.2, 'phase': 0.1})
```

To achieve **calibration persistence** and save adjustments permanently:

```python
dut.save_calibration_log()
```

This saves the configuration to disk, enabling **calibration persistence** across experimental sessions. To load a saved configuration:

```python
from leeq.core.elements.built_in.qudit_transmon import TransmonElement

dut = TransmonElement.load_from_calibration_log('<Qubit hrid>')
```

This method retrieves the latest calibration log, restoring all previously saved parameter values. If `LEEQ_CALIBRATION_LOG_PATH` is unset, logs are saved in the default `.\calibration_log` directory. This calibration persistence mechanism ensures that experimental setups can be reliably reproduced and builds upon previous optimization work.

## Part 4: Single Qubit Experiments and Calibration

!!! tip "Interactive Learning"
    Follow along with this section for hands-on practice with the concepts described below.

### Setting Up Your First Experiment

Before running experiments, you need to set up LeeQ with either a simulation or hardware backend. Here's how to create a simulated two-qubit system:

```python
from leeq.chronicle import Chronicle
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
import numpy as np

# Start data logging
Chronicle().start_log()

# Create experiment manager and clear any existing setups
manager = ExperimentManager()
manager.clear_setups()

# Define virtual transmons with realistic parameters
virtual_qubit_a = VirtualTransmon(
    name="VQubitA",
    qubit_frequency=5040.4,  # MHz
    anharmonicity=-198,      # MHz
    t1=70,                   # microseconds
    t2=35,                   # microseconds  
    readout_frequency=9645.4,
    quiescent_state_distribution=np.array([0.8, 0.15, 0.04, 0.01])
)

# Create high-level simulation setup
setup = HighLevelSimulationSetup(
    name='TutorialSimulation',
    virtual_qubits={2: virtual_qubit_a}  # Channel 2 for drive
)

# Register the setup
manager.register_setup(setup)

# Create DUT configuration
qubit_config = {
    'hrid': 'Q1',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.5487,
            'phase': 0.,
            'width': 0.05,
            'alpha': 500,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9645.5,
            'channel': 1,
            'shape': 'square',
            'amp': 0.15,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

# Create the qubit DUT
qubit = TransmonElement(name="Q1", parameters=qubit_config)
```

### Fundamental Single-Qubit Experiments

#### 1. Rabi Oscillations

Rabi experiments measure the relationship between pulse amplitude/duration and qubit state. This is fundamental for calibrating π-pulses (X gates).

**Theory**: When you apply a resonant drive to a qubit, it oscillates between |0⟩ and |1⟩ states. The frequency of oscillation (Rabi frequency) is proportional to the drive amplitude.

```python
from leeq.experiments.builtin.basic.calibrations.rabi import RabiAmpExperiment
from leeq.experiments.experiments import basic
import numpy as np

# Amplitude Rabi: Sweep drive amplitude
rabi_amp = RabiAmpExperiment(
    qubit=qubit,
    collection_name='f01',
    amp_range=np.linspace(0, 1.0, 51),
    width=0.05  # Pulse width in microseconds
)

# The experiment will show oscillations - the first π-pulse amplitude gives maximum excitation
```

#### 2. T1 Relaxation Measurement  

T1 measures how long an excited qubit stays in |1⟩ before decaying to |0⟩.

**Theory**: After exciting a qubit with a π-pulse, the population decays exponentially: P(1) = e^(-t/T1)

```python
from leeq.experiments.builtin.basic.characterizations.t1 import SimpleT1
from leeq.experiments.experiments import basic

# Measure T1 relaxation time
t1_exp = SimpleT1(
    qubit=qubit,
    collection_name='f01', 
    time_length=200.0,      # Total measurement time (μs)
    time_resolution=2.0     # Time step (μs)
)

# Expected: Exponential decay with time constant ~70 μs
```

#### 3. T2* Dephasing Measurement

T2* measures how quickly superposition states lose coherence due to dephasing.

**Theory**: After creating superposition with π/2-pulse, the coherence decays as cos(ωt)e^(-t/T2*)

```python
from leeq.experiments.builtin.basic.characterizations.t2 import SimpleT2
from leeq.experiments.experiments import basic

# Measure T2* dephasing time  
t2_exp = SimpleT2(
    qubit=qubit,
    collection_name='f01',
    time_length=100.0,
    time_resolution=1.0
)

# Expected: Oscillating decay with envelope ~35 μs
```

#### 4. Qubit Spectroscopy

Finds the precise qubit transition frequency by sweeping drive frequency.

```python
from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import QubitSpectroscopy
from leeq.experiments.experiments import basic
import numpy as np

# Find the exact qubit frequency
spec_exp = QubitSpectroscopy(
    qubit=qubit,
    collection_name='f01',
    freq_range=np.linspace(5035, 5045, 101),  # MHz around expected frequency
    amp=0.5
)

# Expected: Peak at ~5040.4 MHz
```

### Understanding the Results

Each experiment provides both data and fitted parameters:

```python
# After running T1 experiment
print(f"Measured T1: {t1_exp.fit_params['tau']:.1f} μs")
print(f"Fit quality R²: {t1_exp.fit_params['r_squared']:.3f}")

# Access raw data
time_points = t1_exp.get_run_args_dict()['time_length']
populations = t1_exp.trace
```

## Part 5: Multi-Qubit Systems and Entanglement

!!! tip "Interactive Multi-Qubit Learning"
    Explore entanglement and two-qubit gates hands-on in the sections below.

### Setting Up Two-Qubit Systems

Multi-qubit experiments enable quantum entanglement and two-qubit gates. Let's extend our setup to include a second qubit:

```python
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.experiments import basic
import numpy as np

# Add a second virtual transmon
virtual_qubit_b = VirtualTransmon(
    name="VQubitB", 
    qubit_frequency=4855.3,
    anharmonicity=-197,
    t1=60,
    t2=30,
    readout_frequency=9025.1,
    quiescent_state_distribution=np.array([0.75, 0.18, 0.05, 0.02])
)

# Create two-qubit setup with coupling
setup = HighLevelSimulationSetup(
    name='TwoQubitSystem',
    virtual_qubits={
        2: virtual_qubit_a,  # Q1 on channel 2
        4: virtual_qubit_b   # Q2 on channel 4  
    }
)

# Set coupling strength between qubits (MHz)
setup.set_coupling_strength_by_qubit(
    virtual_qubit_a, virtual_qubit_b, 
    coupling_strength=1.5
)

# Create second qubit configuration
qubit_b_config = {
    'hrid': 'Q2',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4855.3,
            'channel': 4,
            'shape': 'blackman_drag',
            'amp': 0.5234,
            'phase': 0.,
            'width': 0.05,
            'alpha': 480,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement', 
            'freq': 9025.1,
            'channel': 3,
            'shape': 'square',
            'amp': 0.12,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

qubit_b = TransmonElement(name="Q2", parameters=qubit_b_config)
```

### Understanding Quantum Entanglement

**Entanglement** is a uniquely quantum phenomenon where qubits become correlated in ways impossible classically. Key concepts:

- **Separable states**: |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ (qubits independent)
- **Entangled states**: Cannot be written as a product, e.g., |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
- **Bell states**: Maximally entangled two-qubit states

### Two-Qubit Gate Implementations

#### 1. CNOT Gate via Cross-Resonance

The most common two-qubit gate in superconducting systems uses cross-resonance coupling:

```python
from leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai import ConditionalStarkShiftAI
from leeq.experiments.experiments import basic
import numpy as np

# Calibrate CNOT gate using AI-assisted optimization
cnot_cal = ConditionalStarkShiftAI(
    control_qubit=qubit,      # Q1 as control
    target_qubit=qubit_b,     # Q2 as target
    stark_drive_freq=4855.3,  # Drive at target frequency
    amp_range=np.linspace(0, 0.3, 31),
    width_range=np.linspace(0.1, 0.5, 21)
)

# This experiment finds optimal parameters for CNOT gate
```

#### 2. Bell State Creation

Create maximally entangled Bell states:

```python
from leeq.core import primitives as prims
from leeq.experiments.sweeper import Sweeper
from leeq.experiments.experiments import basic

def create_bell_state():
    """
    Create a Bell state preparation sequence.
    
    Returns:
        Bell state preparation LPB sequence
    """
    # Get collections for both qubits
    c1_q1 = qubit.get_c1('f01')
    c1_q2 = qubit_b.get_c1('f01')
    
    # Get measurement primitives
    mp_q1 = qubit.get_measurement_prim_intlist(0)
    mp_q2 = qubit_b.get_measurement_prim_intlist(0)
    
    # Bell state sequence: H(Q1) + CNOT(Q1,Q2)
    hadamard_q1 = c1_q1['Yp']  # π/2 Y rotation = Hadamard
    
    # For CNOT, we need the calibrated two-qubit gate
    # This is a simplified example - real implementation requires calibration
    cnot_gate = get_calibrated_cnot_gate(qubit, qubit_b)
    
    # Parallel measurement of both qubits
    measurement = mp_q1 * mp_q2
    
    # Complete sequence
    bell_sequence = hadamard_q1 + cnot_gate + measurement
    
    return bell_sequence

def measure_bell_state():
    """
    Run Bell state measurement experiment.
    
    Expected: ~50% |00⟩ and ~50% |11⟩, very little |01⟩ or |10⟩
    """
    lpb = create_bell_state()
    
    # Single shot measurement (no sweep)
    swp = Sweeper([0], params=[])
    
    # Measure both qubits
    basic(lpb, swp, ['p(1)', 'p(1)'])
```

#### 3. Quantum Process Tomography

Characterize two-qubit gate fidelity using process tomography:

```python
from leeq.experiments.builtin.tomography.qubits import ProcessTomographyTwoQubit

# Measure the actual gate implemented vs. ideal CNOT
process_tomo = ProcessTomographyTwoQubit(
    control_qubit=qubit,
    target_qubit=qubit_b,
    gate_lpb=cnot_gate,  # The gate to characterize
    num_preparations=16,  # Different input states
    num_measurements=16   # Different measurement bases
)

# Results include:
# - Process fidelity (how close to ideal CNOT)
# - Gate error rates
# - Systematic errors (e.g., over/under-rotation)
```

### Multi-Qubit Measurement and Correlation

When measuring multiple qubits, you get correlated results:

```python
import numpy as np

# Simultaneous measurement
results_q1 = mp_q1.result()  # Shape: (shots,)
results_q2 = mp_q2.result()  # Shape: (shots,)

def calculate_correlations(r1, r2):
    """
    Calculate correlation functions for two-qubit measurements.
    
    Args:
        r1: Results from qubit 1
        r2: Results from qubit 2
    
    Returns:
        Dictionary with joint probabilities and entanglement witness
    """
    # Joint probabilities
    p_00 = np.mean((r1 == 0) & (r2 == 0))
    p_01 = np.mean((r1 == 0) & (r2 == 1))
    p_10 = np.mean((r1 == 1) & (r2 == 0))
    p_11 = np.mean((r1 == 1) & (r2 == 1))
    
    # Entanglement witness: for Bell state, expect p_00 + p_11 ≈ 1
    entanglement_witness = p_00 + p_11
    
    return {
        'joint_probs': [p_00, p_01, p_10, p_11],
        'entanglement_witness': entanglement_witness
    }

corr_results = calculate_correlations(results_q1, results_q2)
print(f"Entanglement witness: {corr_results['entanglement_witness']:.3f}")
```

## Part 6: Advanced Calibration Procedures

!!! tip "Complete Calibration Workflows"
    Practice advanced calibration techniques with the [Calibrations Guide](guide/calibrations.md), which demonstrates systematic calibration procedures for real quantum devices.

### Systematic Calibration Workflow

Real quantum devices require careful calibration of all parameters. Here's a systematic approach:

#### 1. Frequency Calibration

```python
from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import QubitSpectroscopy
from leeq.experiments.experiments import basic
import numpy as np

def calibrate_frequencies(qubit, freq_range_mhz=10):
    """
    Find precise qubit and readout frequencies.
    
    Args:
        qubit: Qubit element to calibrate
        freq_range_mhz: Frequency range to scan around current value
    
    Returns:
        Calibrated frequency in MHz
    """
    
    # 1. Rough frequency scan
    current_freq = qubit.get_c1('f01').get_parameters()['freq']
    rough_range = np.linspace(
        current_freq - freq_range_mhz/2,
        current_freq + freq_range_mhz/2, 
        101
    )
    
    rough_spec = QubitSpectroscopy(
        qubit=qubit,
        freq_range=rough_range,
        amp=0.5
    )
    
    # 2. Fine frequency scan around peak
    peak_freq = rough_spec.fit_params['center']
    fine_range = np.linspace(peak_freq - 1, peak_freq + 1, 101)
    
    fine_spec = QubitSpectroscopy(
        qubit=qubit,
        freq_range=fine_range,
        amp=0.5
    )
    
    # 3. Update qubit frequency
    calibrated_freq = fine_spec.fit_params['center']
    qubit.get_c1('f01').set_parameters({'freq': calibrated_freq})
    
    return calibrated_freq
```

#### 2. Rabi Calibration for Gates

```python
from leeq.experiments.builtin.basic.calibrations.rabi import RabiAmpExperiment
import numpy as np

def calibrate_pi_pulse(qubit):
    """
    Calibrate π-pulse amplitude for X gate.
    
    Args:
        qubit: Qubit element to calibrate
    
    Returns:
        Dictionary with π and π/2 pulse amplitudes
    """
    
    # Amplitude sweep for Rabi oscillations
    rabi_exp = RabiAmpExperiment(
        qubit=qubit,
        amp_range=np.linspace(0, 1.2, 61),
        width=0.05
    )
    
    # Find first maximum (π-pulse amplitude)
    pi_amp = rabi_exp.fit_params['pi_amp']
    
    # Update collection parameters
    qubit.get_c1('f01').set_parameters({'amp': pi_amp})
    
    # Calibrate π/2 pulses
    half_pi_amp = pi_amp / 2
    
    return {
        'pi_amp': pi_amp,
        'half_pi_amp': half_pi_amp
    }
```

#### 3. DRAG Pulse Optimization

DRAG (Derivative Removal by Adiabatic Gating) pulses reduce leakage to higher levels:

```python
from leeq.experiments.builtin.basic.calibrations.drag import DragCalibration
import numpy as np

def optimize_drag_parameter(qubit):
    """
    Optimize DRAG parameter to minimize leakage.
    
    Args:
        qubit: Qubit element to optimize
    
    Returns:
        Optimal DRAG alpha parameter
    """
    
    drag_cal = DragCalibration(
        qubit=qubit,
        alpha_range=np.linspace(-1000, 1000, 41),
        num_cliffords=20  # Test with multiple Clifford gates
    )
    
    # Optimal alpha minimizes population in |2⟩ state
    optimal_alpha = drag_cal.fit_params['optimal_alpha']
    
    # Update DRAG parameter
    qubit.get_c1('f01').set_parameters({'alpha': optimal_alpha})
    
    return optimal_alpha
```

#### 4. Complete Calibration Sequence

```python
from leeq.experiments.builtin.basic.characterizations.t1 import SimpleT1
from leeq.experiments.builtin.basic.characterizations.t2 import SimpleT2

def full_single_qubit_calibration(qubit):
    """
    Complete single-qubit calibration sequence.
    
    Args:
        qubit: Qubit element to calibrate completely
    
    Returns:
        Dictionary with all calibration results
    """
    
    print("Starting single-qubit calibration")
    
    # 1. Find precise frequency
    print("1. Calibrating qubit frequency")
    freq = calibrate_frequencies(qubit)
    print(f"   Calibrated frequency: {freq:.3f} MHz")
    
    # 2. Calibrate π-pulse amplitude
    print("2. Calibrating Rabi amplitude")
    rabi_params = calibrate_pi_pulse(qubit)
    print(f"   π-pulse amplitude: {rabi_params['pi_amp']:.4f}")
    
    # 3. Optimize DRAG parameter
    print("3. Optimizing DRAG parameter")
    alpha = optimize_drag_parameter(qubit)
    print(f"   Optimal DRAG α: {alpha:.1f}")
    
    # 4. Characterize coherence times
    print("4. Measuring coherence times")
    t1_exp = SimpleT1(qubit=qubit, time_length=300, time_resolution=3)
    t2_exp = SimpleT2(qubit=qubit, time_length=150, time_resolution=1.5)
    
    print(f"   T1: {t1_exp.fit_params['tau']:.1f} μs")
    print(f"   T2*: {t2_exp.fit_params['tau']:.1f} μs")
    
    # 5. Save calibration
    qubit.save_calibration_log()
    print("Calibration complete and saved!")
    
    return {
        'frequency': freq,
        'rabi_params': rabi_params,
        'drag_alpha': alpha,
        't1': t1_exp.fit_params['tau'],
        't2_star': t2_exp.fit_params['tau']
    }
```

### Two-Qubit Gate Calibration

```python
from leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai import ConditionalStarkShiftAI
from leeq.experiments.builtin.tomography.qubits import ProcessTomographyTwoQubit
import numpy as np

def calibrate_two_qubit_gate(control_qubit, target_qubit):
    """
    Calibrate CNOT gate between two qubits.
    
    Args:
        control_qubit: Control qubit element
        target_qubit: Target qubit element
    
    Returns:
        Dictionary with optimal parameters and gate fidelity
    """
    
    print("Calibrating two-qubit CNOT gate")
    
    # 1. Find optimal cross-resonance parameters
    cr_cal = ConditionalStarkShiftAI(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        amp_range=np.linspace(0, 0.5, 26),
        width_range=np.linspace(0.1, 1.0, 19)
    )
    
    optimal_params = cr_cal.get_optimal_parameters()
    
    # 2. Characterize gate with process tomography
    cnot_lpb = create_cnot_gate(control_qubit, target_qubit, optimal_params)
    
    process_tomo = ProcessTomographyTwoQubit(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        gate_lpb=cnot_lpb
    )
    
    gate_fidelity = process_tomo.results['process_fidelity']
    
    print(f"CNOT gate fidelity: {gate_fidelity:.3f}")
    
    return {
        'optimal_params': optimal_params,
        'gate_fidelity': gate_fidelity,
        'gate_lpb': cnot_lpb
    }
```

## Customizing Your Setup

LeeQ provides integrated support for setups utilizing Single Board RPC control within the LBNL QubiC system. Here's a straightforward example of how to define your setup:

```python
class QubiCDemoSetup(QubiCSingleBoardRemoteRPCSetup):

    def __init__(self):
        """Initialize the QubiC demo setup with default configuration."""
        super().__init__(
            name='qubic_demo_setup',
            rpc_uri='http://192.168.1.80:9095' # The RPC address for QubiC system
        )

```

The system can be configured to adjust pulse parameters before they are submitted to the compiler. This feature is particularly useful, for instance, when integrating the QubiC system to synthesize an Intermediate Frequency (IF) signal that will be mixed with an external Local Oscillator (LO).

```python
class QubiCSetup(QubiCSingleBoardRemoteRPCSetup):

    @staticmethod
    def _readout_frequency_mixing_callback(parameters: dict):
        """
        This function changes the frequency of the lpb parameters of the readout channel,
        considering we have a mixing of 15GHz signal.
        """

        if 'freq' not in parameters:
            return parameters

        modified_parameters = parameters.copy()
        modified_parameters['freq'] = 15000-parameters['freq']  # 4-8 GHz for IF Readout
        return modified_parameters

    def __init__(self):
        """Initialize the QubiC setup with frequency mixing callback."""
        super().__init__(
            name='qubic_setup',
            rpc_uri='http://192.168.1.80:9095'
        )

        # Register call function for all readout channels

        for i in range(8):
            readout_channel = 2 * i + 1
            self._status.register_compile_lpb_callback(
                channel=readout_channel,
                callback=self._readout_frequency_mixing_callback
            )
```

## Creating a Customized Experiment

Customizing an experiment can generally be segmented into three main components: (1) the setup and execution of the experiment, including data acquisition; (2) data processing and analysis; and (3) data visualization.

Below, we present an example outlining the implementation of a basic experiment aimed at measuring the T1 relaxation time of a qubit. For the sake of brevity and clarity, certain details in the data visualization section are simplified.

```python
from leeq.experiments.experiments import Experiment
from leeq.chronicle import log_and_record
from leeq.experiments.plots.live_dash_app import register_browser_function
from leeq.experiments.sweeper import Sweeper
from leeq.experiments.experiments import basic
from leeq.core import primitives as prims
from leeq.experiments import sweeper as sparam
import numpy as np
import plotly.graph_objects as go
from typing import Any, Optional

class SimpleT1(Experiment):
    """
    Custom T1 relaxation measurement experiment.
    
    This experiment measures the T1 relaxation time by applying a π-pulse
    followed by a variable delay before measurement.
    """
    
    @log_and_record # This decorator is used to log the experiment and record the data
    def run(self, # The run function should be defined to carry out the experiment
            qubit: Any,  # Qubit element to measure
            collection_name: str = 'f01',
            initial_lpb: Optional[Any] = None,  # Optional preparation sequence
            mprim_index: int = 0,
            time_length: float = 100.0,
            time_resolution: float = 1.0
            ) -> None:
        """
        Run T1 relaxation measurement.
        
        Args:
            qubit: Qubit element to measure
            collection_name: Collection name for pulses
            initial_lpb: Optional initial preparation sequence
            mprim_index: Measurement primitive index
            time_length: Maximum delay time in microseconds
            time_resolution: Time step in microseconds
        """
        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(mprim_index)
        
        self.mp = mp # Store the measurement primitive for the live data plot
        delay = prims.Delay(0)

        lpb = c1['X'] + delay + mp

        if initial_lpb:
            lpb = initial_lpb + lpb

        sweep_range = np.arange(0.0, time_length, time_resolution)
        swp = Sweeper(sweep_range,
                      params=[sparam.func(delay.set_delay, {}, 'delay')])

        # The basic function is used to run the experiment
        # The swp is used to sweep the delay time
        # The basis parameter is used to set what to return from the experiment, 
        # here we return the probability of the qubit in the excited state 
        basic(lpb, swp, 'p(1)') 
        
        self.trace = np.squeeze(mp.result())
        
    # This decorator is used to register the function for visualization. 
    #It will be shown after the experiment is finished.
    @register_browser_function() 
    def plot_t1(self, fit=True, step_no=None) -> go.Figure:
        """
        Plot T1 decay curve with optional fitting.
        
        Args:
            fit: Whether to perform exponential fit
            step_no: Current step for live plotting
        
        Returns:
            Plotly figure object
        """
        self.trace = np.squeeze(self.mp.result()) if hasattr(self, 'mp') else []
        self.fit_params = {}  # Initialize as an empty dictionary or suitable default value

        args = self.get_run_args_dict() # Retrieve the arguments from the run function

        t = np.arange(0, args['time_length'], args['time_resolution'])
        trace = self.trace

        if step_no is not None: # The step number is used to plot the live data
            t = t[:step_no[0]]
            trace = trace[:step_no[0]]

        # Create plot data
        data = [go.Scatter(x=t, y=trace, mode='markers+lines', name='T1 Data')]
        layout = go.Layout(title='T1 Relaxation Measurement', 
                          xaxis_title='Time (μs)', 
                          yaxis_title='P(|1⟩)')

        fig = go.Figure(data=data, layout=layout)

        return fig

    def live_plots(self, step_no):
        """Generate live plots during experiment execution."""
        return self.plot_t1(fit=step_no[0] > 10, step_no=step_no) # The live plot function is used to plot the live data
```

To initiate and execute the experiment, use the following snippet:

```python
from leeq.core.elements.built_in.qudit_transmon import TransmonElement

# Assuming you have a qubit already configured
qubit = TransmonElement(name="Q1", parameters=qubit_config)
t1_exp = SimpleT1(qubit=qubit, time_length=100, time_resolution=0.5)
```

This command automatically runs the experiment and presents the results.

## Data Persistence

LeeQ leverages the integrated leeq.chronicle module for data persistence. To ensure proper data logging from the outset, initiate the logging process at the beginning of your notebook as shown below. Additionally, remember to annotate the `run` method of your experiment class with `@log_and_record` to enable experiment logging.

```python
from leeq.chronicle import Chronicle
Chronicle().start_log()
```

For each experiment, leeq.chronicle automatically generates a data path and experiment ID, which can be used to access the recorded data. For comprehensive information on data retrieval and additional functionalities, consult the leeq.chronicle documentation.

## Part 7: Custom Experiment Development

!!! tip "Build Your Own Experiments"
    Learn to create custom experiments with the examples below, which show both automated experiment generation and manual custom experiment creation.

### Anatomy of a LeeQ Experiment

Building custom experiments follows a structured pattern. Here's the template for creating your own experiments:

```python
from leeq.chronicle import log_and_record
from leeq.experiments.plots.live_dash_app import register_browser_function
from leeq.experiments.experiments import Experiment
from leeq.experiments.sweeper import Sweeper
from leeq.experiments.experiments import basic
from leeq.core import primitives as prims
from leeq.experiments import sweeper as sparam
import numpy as np
from plotly import graph_objects as go

class MyCustomExperiment(Experiment):
    """
    Template for custom experiments in LeeQ.
    
    This example implements a custom Ramsey experiment with adjustable detuning.
    """
    
    @log_and_record  # Essential: enables data logging
    def run(self, 
            qubit,                    # Target qubit
            collection_name='f01',    # Pulse collection to use
            detuning_freq=0.1,       # Frequency detuning (MHz)
            time_length=50.0,        # Max evolution time (μs)
            time_resolution=1.0,     # Time step (μs)
            initial_lpb=None         # Optional preparation sequence
           ):
        """
        Run custom Ramsey experiment with detuning.
        
        Sequence: π/2 - delay - π/2(φ) - measurement
        where φ = 2π * detuning * delay
        """
        
        # Get pulse collection and measurement primitive
        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(0)
        
        # Store for plotting
        self.mp = mp
        self.detuning = detuning_freq
        
        # Create delay primitive
        from leeq.core import primitives as prims
        delay = prims.Delay(0)  # Will be swept
        
        # Create Ramsey sequence: π/2 - delay - π/2 - measure
        ramsey_lpb = c1['Yp'] + delay + c1['Yp'] + mp
        
        # Add initial preparation if provided
        if initial_lpb:
            ramsey_lpb = initial_lpb + ramsey_lpb
        
        # Create time sweep
        sweep_times = np.arange(0.0, time_length, time_resolution)
        
        # Create phase sweep for detuning effect
        phases = 2 * np.pi * detuning_freq * sweep_times
        
        # Create sweeper for both delay and phase
        from leeq.experiments.sweeper import SweepParametersSideEffect
        
        delay_sweep = sparam.func(delay.set_delay, {}, 'delay')
        phase_sweep = sparam.func(c1['Yp'].set_phase, {}, 'phase')
        
        swp = Sweeper(
            param_list=list(zip(sweep_times, phases)),
            params=[delay_sweep, phase_sweep]
        )
        
        # Execute experiment
        basic(ramsey_lpb, swp, 'p(1)')
        
        # Store results
        self.trace = np.squeeze(mp.result())
        self.times = sweep_times
    
    @register_browser_function()  # Makes plot available in web interface
    def plot_ramsey(self, fit=True):
        """Plot Ramsey oscillations with optional fitting."""
        
        if not hasattr(self, 'trace'):
            return go.Figure()
        
        # Create plot data
        trace_data = go.Scatter(
            x=self.times,
            y=self.trace,
            mode='markers+lines',
            name=f'Ramsey (δf={self.detuning:.2f} MHz)',
            marker=dict(size=4)
        )
        
        data = [trace_data]
        
        # Optional fitting
        if fit and len(self.times) > 10:
            try:
                # Fit oscillating decay: A*cos(2πft + φ)*exp(-t/T2) + offset
                from scipy.optimize import curve_fit
                
                def ramsey_func(t, amp, freq, phase, t2, offset):
                    """Ramsey oscillation function with exponential decay."""
                    return amp * np.cos(2*np.pi*freq*t + phase) * np.exp(-t/t2) + offset
                
                # Initial guess
                p0 = [0.4, self.detuning, 0, 20, 0.5]
                
                popt, _ = curve_fit(ramsey_func, self.times, self.trace, p0=p0)
                
                # Generate fit curve
                t_fit = np.linspace(0, self.times[-1], 200)
                y_fit = ramsey_func(t_fit, *popt)
                
                fit_trace = go.Scatter(
                    x=t_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'Fit (T2*={popt[3]:.1f}μs)',
                    line=dict(color='red', dash='dash')
                )
                data.append(fit_trace)
                
                # Store fit parameters
                self.fit_params = {
                    'amplitude': popt[0],
                    'frequency': popt[1], 
                    'phase': popt[2],
                    't2_star': popt[3],
                    'offset': popt[4]
                }
                
            except Exception as e:
                print(f"Fitting failed: {e}")
        
        # Create layout
        layout = go.Layout(
            title=f'Custom Ramsey Experiment (Detuning: {self.detuning:.2f} MHz)',
            xaxis=dict(title='Time (μs)'),
            yaxis=dict(title='P(|1⟩)'),
            hovermode='closest'
        )
        
        return go.Figure(data=data, layout=layout)
    
    def live_plots(self, step_no):
        """Real-time plotting during experiment."""
        return self.plot_ramsey(fit=step_no[0] > 10)
```

### Running Your Custom Experiment

```python
# Run the custom experiment
custom_exp = MyCustomExperiment(
    qubit=qubit,
    detuning_freq=0.2,  # 200 kHz detuning
    time_length=80.0,
    time_resolution=1.0
)

# Access results
print(f"Measured T2*: {custom_exp.fit_params['t2_star']:.1f} μs")
print(f"Observed frequency: {custom_exp.fit_params['frequency']:.3f} MHz")
```

### Advanced Experiment Features

#### 1. Multi-Parameter Sweeps

```python
from leeq.experiments.experiments import Experiment
from leeq.chronicle import log_and_record
from leeq.experiments.sweeper import Sweeper
from leeq.experiments.experiments import basic
from leeq.experiments import sweeper as sparam
import numpy as np

class ParameterSweep2D(Experiment):
    """
    Example of 2D parameter sweep experiment.
    
    This experiment demonstrates how to sweep two parameters simultaneously
    and collect data in a 2D grid format.
    """
    
    @log_and_record
    def run(self, qubit, amp_range, freq_range):
        """
        Run 2D parameter sweep over amplitude and frequency.
        
        Args:
            qubit: Qubit element to use
            amp_range: Array of amplitude values to sweep
            freq_range: Array of frequency values to sweep
        """
        c1 = qubit.get_c1('f01')
        mp = qubit.get_measurement_prim_intlist(0)
        
        # Create 2D parameter grid
        amp_grid, freq_grid = np.meshgrid(amp_range, freq_range)
        param_pairs = list(zip(amp_grid.flatten(), freq_grid.flatten()))
        
        # Define parameter sweeps
        amp_sweep = sparam.func(c1.set_amp, {}, 'amplitude')
        freq_sweep = sparam.func(c1.set_freq, {}, 'frequency')
        
        swp = Sweeper(param_pairs, params=[amp_sweep, freq_sweep])
        
        # Simple pulse-measure sequence
        lpb = c1['X'] + mp
        basic(lpb, swp, 'p(1)')
        
        # Reshape results back to 2D
        self.results_2d = mp.result().reshape(len(freq_range), len(amp_range))
        self.amp_range = amp_range
        self.freq_range = freq_range
```

#### 2. AI-Assisted Optimization

```python
from k_agents.inspection.decorator import text_inspection
from leeq.experiments.experiments import Experiment
from leeq.chronicle import log_and_record
from leeq.experiments.experiments import basic

class AIOptimizedExperiment(Experiment):
    """
    Example using LeeQ's AI capabilities for parameter optimization.
    
    This experiment demonstrates integration with AI agents for automated
    parameter optimization and experimental feedback.
    """
    
    @text_inspection(agent_name="experiment_optimizer")
    @log_and_record
    def run(self, qubit, optimization_target='fidelity'):
        """
        AI agent can analyze results and suggest parameter adjustments.
        
        Args:
            qubit: Qubit element to optimize
            optimization_target: Target metric to optimize
        """
        # Implementation with AI feedback loop
        c1 = qubit.get_c1('f01')
        mp = qubit.get_measurement_prim_intlist(0)
        
        # Run initial measurement
        lpb = c1['X'] + mp
        basic(lpb, None, 'p(1)')
        
        # Store results for AI analysis
        self.initial_results = mp.result()
        
        # AI agent will analyze these results and suggest optimizations
        print(f"Optimizing {optimization_target} for qubit {qubit.hrid}")
```

## Next Steps and Learning Resources

### Immediate Next Steps

1. **Explore the User Guide**: Understand LeeQ's architecture and capabilities:
   - [**Core Concepts**](guide/concepts.md) - LeeQ design principles and architecture
   - [**Experiments Guide**](guide/experiments.md) - Available experiment types
   - [**Calibrations Guide**](guide/calibrations.md) - Calibration procedures

2. **Dive into the API Documentation**:
   - [**Core API**](api/core/base.md) - Base classes and functionality
   - [**Experiments API**](api/experiments/builtin.md) - Built-in experiments
   - [**Theory API**](api/theory/simulation.md) - Simulation backends
   - [**Compiler API**](api/compiler/base.md) - Pulse compilation

3. **Apply Your Knowledge**:
   - Build custom experiments using the patterns from this tutorial
   - Implement calibration workflows for your quantum system
   - Use AI assistance for experiment generation
   - Analyze results with Chronicle logging

4. **Join the Community**: Connect with other LeeQ users for support and collaboration

### Advanced Topics to Explore

1. **Optimal Control**: Use GRAPE and other techniques for pulse optimization
2. **Error Mitigation**: Implement error correction and mitigation strategies  
3. **Machine Learning**: Integrate ML models for automated calibration
4. **Real Hardware**: Transition from simulation to actual quantum devices
5. **Large-Scale Systems**: Scale to many-qubit experiments and algorithms

### Learning Resources

- **[User Guide](guide/concepts.md)**: Core concepts and architecture
- **[Experiments Guide](guide/experiments.md)**: Available experiment types  
- **[Calibrations Guide](guide/calibrations.md)**: Calibration procedures
- **[API Documentation](api/core/base.md)**: Complete technical documentation
- **Community Forums**: Q&A and discussion with experts
- **Research Papers**: Scientific background on implemented techniques

### Contributing to LeeQ

LeeQ is an open framework that benefits from community contributions:

1. **Report Issues**: Help improve LeeQ by reporting bugs or requesting features
2. **Contribute Code**: Add new experiments, pulse shapes, or analysis tools
3. **Share Notebooks**: Create tutorials and examples for others to learn from
4. **Documentation**: Help improve guides and API documentation

### Final Tips for Success

1. **Start Simple**: Begin with basic experiments before attempting complex protocols
2. **Understand the Physics**: Strong quantum mechanics knowledge aids troubleshooting
3. **Calibrate Carefully**: Good calibration is essential for meaningful results
4. **Validate Results**: Always sanity-check your experimental outcomes
5. **Keep Learning**: Quantum computing is rapidly evolving - stay updated!

Welcome to the LeeQ community! We're excited to see what you'll build with this powerful quantum experiment orchestration framework.