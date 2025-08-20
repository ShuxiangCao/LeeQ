# Core Concepts

This guide explains the fundamental concepts and architecture of LeeQ.

## Architecture Overview

LeeQ follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Experiments   │    │    Elements     │    │    Compiler     │
│                 │    │                 │    │                 │
│ - Calibrations  │────│ - Qubits        │────│ - Pulse Gen     │
│ - Measurements  │    │ - Transmons     │    │ - Sequencing    │
│ - Protocols     │    │ - Parameters    │    │ - Hardware API  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Engine      │
                    │                 │
                    │ - Execution     │
                    │ - Data Flow     │
                    │ - Logging       │
                    └─────────────────┘
```

## Core Components

### 1. LeeQObject Base Class

All LeeQ components inherit from `LeeQObject`, which provides:

- **Persistence**: Automatic logging through leeq.chronicle (integrated module)
- **Configuration**: Parameter management and serialization
- **Tracking**: Experiment history and reproducibility

```python
from leeq.core.base import LeeQObject

class MyCustomComponent(LeeQObject):
    def __init__(self, name, parameters):
        super().__init__(name=name, parameters=parameters)
```

### 2. Quantum Elements

Elements represent physical or simulated quantum systems:

#### Basic Qubit
```python
from leeq.core.elements.built_in.qudit_transmon import TransmonElement

# Create a transmon qubit
qubit = TransmonElement(
    name="Q1",
    parameters={
        'hrid': 'Q1',
        'lpb_collections': {...},  # Pulse definitions
        'measurement_primitives': {...}  # Measurement configs
    }
)
```

#### Key Features:
- **Calibration Management**: Store and retrieve calibrated parameters
- **Pulse Collections**: Define different types of control pulses (f01, f12, etc.)
- **Measurement Primitives**: Configure readout and state discrimination

### 3. Execution Engine

The engine manages experiment execution flow:

```python
from leeq.core.engine.engine_base import EngineBase

# Engines handle:
# - Experiment scheduling
# - Data collection
# - Result processing
# - Hardware synchronization
```

### 4. Primitives

Primitives are the building blocks of quantum operations:

#### Logical Primitives
```python
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep

# Define parameter sweeps
sweep = LogicalPrimitiveBlockSweep(
    lpb_list=[...],  # List of primitive blocks
    param_name="frequency",
    param_values=[4.8, 4.9, 5.0]  # GHz
)
```

#### Physical Primitives
```python
from leeq.core.primitives.built_in.simple_drive import SimpleDrive

# Single qubit gate
x_gate = SimpleDrive(
    channel=0,
    frequency=4.85,  # GHz
    amplitude=0.1,
    width=0.05,  # μs
    shape='gaussian'
)
```

### 5. Compiler

The compiler translates high-level operations to hardware instructions:

```python
from leeq.compiler.compiler_base import CompilerBase

# Compiler features:
# - Pulse shape generation
# - Timing optimization
# - Hardware-specific adaptation
# - Sequence validation
```

## Key Design Patterns

### 1. Setup Pattern

LeeQ uses a setup pattern to abstract hardware differences:

```python
from leeq.setups.setup_base import SetupBase
from leeq.experiments import setup

# Register your hardware configuration
my_setup = MyHardwareSetup()
setup().register_setup(my_setup)

# Setup provides unified interface regardless of backend
```

### 2. Collection Pattern

Related primitives are grouped into collections:

```python
# Pulse collections for different transitions
lpb_collections = {
    'f01': {  # 0→1 transition
        'type': 'SimpleDriveCollection',
        'freq': 4.85,
        'amp': 0.1
    },
    'f12': {  # 1→2 transition  
        'type': 'SimpleDriveCollection',
        'freq': 4.65,
        'amp': 0.08
    }
}
```

### 3. Sweep Pattern

Parameter sweeps are first-class objects:

```python
from leeq.experiments.sweeper import Sweeper

# Create parameter sweeps
freq_sweep = Sweeper(
    parameter=dut.get_c1('f01')['freq'],
    values=np.linspace(4.8, 4.9, 51)
)

# Combine multiple sweeps
amp_sweep = Sweeper(
    parameter=dut.get_c1('f01')['amp'], 
    values=np.linspace(0.05, 0.15, 11)
)

# Grid sweep automatically generated
```

## Data Flow

```
Experiment Definition
        │
        ▼
Parameter Sweeps ──┐
        │         │
        ▼         │
Primitive Blocks  │
        │         │
        ▼         │
Compiler ─────────┘
        │
        ▼
Hardware Execution
        │
        ▼
Data Collection
        │
        ▼
Result Processing
        │
        ▼
Logging & Storage
```

## Integration with AI/ML

LeeQ includes built-in AI capabilities:

### Experiment Generation
```python
from leeq.utils.ai.experiment_generation import ExperimentGenerator

# AI-assisted experiment design
generator = ExperimentGenerator()
experiment_code = generator.generate_experiment(
    description="Optimize qubit frequency with Ramsey fringes",
    qubit_params=qubit.get_parameters()
)
```

### Translation Agent
```python
from leeq.utils.ai.translation_agent import TranslationAgent

# Convert between different quantum languages
agent = TranslationAgent()
qiskit_code = agent.translate_to_qiskit(leeq_experiment)
```

## Best Practices

### 1. Configuration Management
- Store all parameters in structured dictionaries
- Use version control for configuration files
- Maintain separate configs for different setups

### 2. Calibration Workflow
- Regularly save calibration states
- Track calibration history
- Automate recalibration procedures

### 3. Error Handling
- Implement proper exception handling
- Use logging for debugging
- Validate parameters before execution

### 4. Testing
- Write unit tests for custom components
- Use simulation backends for development
- Validate against known results

## Next Steps

- Follow the [experiments guide](experiments.md) to learn about built-in experiments
- Read the [calibrations guide](calibrations.md) for calibration workflows
- Explore the [API reference](../api/core/base.md) for detailed documentation