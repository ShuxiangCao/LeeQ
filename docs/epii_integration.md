# LeeQ EPII Integration Guide

This document describes how to use LeeQ experiments with the Quantum Calibration Agent (qca) through the EPII interface.

## Overview

The EPII (Experiment Programming Interface for Instruments) integration allows the CalibrationNTKAgent to discover and execute LeeQ experiments through a declarative service pattern. An LLM-based agent can then orchestrate quantum calibration workflows by selecting and running experiments based on results.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CalibrationNTKAgent                          │
│  ┌───────────┐    ┌─────────────┐    ┌───────────────────────┐  │
│  │    LLM    │───▶│ EPII Client │───▶│   Backend Factory     │  │
│  │  (GPT-4o) │    └─────────────┘    └───────────────────────┘  │
└──│───────────│────────────────────────────────│─────────────────┘
   └───────────┘                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LeeQ                                     │
│  ┌───────────────────┐    ┌──────────────────────────────────┐  │
│  │    LeeQBackend    │───▶│  DeclarativeServiceAdapter       │  │
│  │ (leeq.epii.backend)│    │  (leeq.epii.leeq_epii_service)  │  │
│  └───────────────────┘    └──────────────────────────────────┘  │
│                                           │                      │
│                                           ▼                      │
│                           ┌──────────────────────────────────┐  │
│                           │      ExperimentRouter            │  │
│                           │   (leeq.epii.experiments)        │  │
│                           └──────────────────────────────────┘  │
│                                           │                      │
│                                           ▼                      │
│                           ┌──────────────────────────────────┐  │
│                           │   leeq.experiments.builtin.*     │  │
│                           │   (Auto-discovered experiments)  │  │
│                           └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- CalibrationNTKAgent repository
- LeeQ repository
- OpenAI API key (for GPT-4o)

## Installation

### 1. Set up CalibrationNTKAgent

```bash
cd /path/to/CalibrationNTKAgent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Install LeeQ in the same virtualenv

```bash
pip install -e /path/to/LeeQ
```

### 3. Install additional dependencies

```bash
pip install python-frontmatter langchain-sandbox
```

### 4. Fix mllm compatibility (if needed)

If you encounter `ImportError: cannot import name 'p_map' from 'mllm.utils'`, add the following to `.venv/lib/python3.12/site-packages/mllm/utils/__init__.py`:

```python
from .maps import parallel_map, p_map
```

## Configuration

### EPII Backend Configuration

Create or edit `config/quantum_calibration/epii_config.yaml`:

```yaml
backend:
  type: leeq
  config:
    service_module: leeq.epii.leeq_epii_service
```

### LLM Configuration

Edit `config/quantum_calibration/config.yml` to configure the LLM:

```yaml
llms:
  orchestrator_llm:
    _type: openai
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0
    max_tokens: 8192
  workflow_llm:
    _type: openai
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0
    max_tokens: 8192
  experiment_llm:
    _type: openai
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0
    max_tokens: 8192
  vlm_llm:
    _type: openai
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0
    max_tokens: 4096
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-proj-your-api-key"
```

## Usage

### Command Line Interface

```bash
cd /path/to/CalibrationNTKAgent
source .venv/bin/activate

# List available experiments
python -m quantum_calibration_agent.cli.main exec "list all available experiments"

# Run a specific experiment
python -m quantum_calibration_agent.cli.main exec "run a normalized Rabi experiment on qubit 0"

# Full calibration workflow
python -m quantum_calibration_agent.cli.main exec "calibrate qubit 0 starting with resonator spectroscopy, then do Rabi and Ramsey"
```

### Web Interface

```bash
python -m quantum_calibration_agent.cli.main serve
# Open http://localhost:8000 in browser
```

## Available Experiments

The EPII integration auto-discovers experiments from `leeq.experiments.builtin`. Experiments must have:

1. `EPII_INFO` class attribute with metadata
2. `run_simulated()` method for simulation mode
3. At least one `@text_inspection` decorated method

### Currently Available Experiments

| Category | Experiment | Description |
|----------|------------|-------------|
| calibrations | DragCalibrationSingleQubitMultilevel | DRAG coefficient calibration |
| calibrations | NormalisedRabi | Driving amplitude calibration |
| calibrations | SimpleRamseyMultilevel | Frequency detuning measurement |
| calibrations | ResonatorSweepTransmissionWithExtraInitialLPB | Resonator frequency sweep |
| calibrations | ResonatorThreeRegimeCharacterization | Power regime characterization |
| calibrations | MeasurementCalibrationMultilevelGMM | GMM measurement calibration |
| characterizations | SingleQubitRandomizedBenchmarking | Gate fidelity benchmarking |
| characterizations | SimpleT1 | T1 relaxation measurement |
| characterizations | SpinEchoMultiLevel | T2 coherence measurement |
| ac_stark | StarkRamseyMultilevel | AC Stark shift measurement |

## Adding New Experiments

To make a LeeQ experiment available through EPII:

### 1. Add EPII_INFO to your experiment class

```python
from leeq.experiments import Experiment

class MyNewExperiment(Experiment):
    EPII_INFO = {
        "name": "MyNewExperiment",
        "description": "Description of what this experiment does",
        "category": "calibrations",
        "parameters": {
            "qubit": {"type": "qubit", "description": "Target qubit"},
            "start": {"type": "float", "description": "Start frequency (MHz)"},
            "stop": {"type": "float", "description": "Stop frequency (MHz)"},
        },
        "outputs": {
            "resonance_freq": {"type": "float", "description": "Measured resonance frequency"},
        }
    }
```

### 2. Implement run_simulated() method

```python
def run_simulated(self, **kwargs):
    """Run the experiment in simulation mode."""
    # Simulation logic here
    self.result = simulated_result
    return self.result
```

### 3. Add text inspection method

```python
from k_agents.inspection.decorator import text_inspection

@text_inspection("Analyze the experiment results")
def analyze_results(self) -> str:
    """Return a text description of the results for the LLM."""
    return f"The resonance frequency was found at {self.result['freq']:.2f} MHz"
```

## Troubleshooting

### "Unknown backend type: 'leeq'"

LeeQ is not installed in the CalibrationNTKAgent virtualenv:
```bash
pip install -e /path/to/LeeQ
```

### "No experiments defined"

The ExperimentRouter couldn't find any valid experiments. Check that:
- Experiments have `EPII_INFO` attribute
- Experiments have `run_simulated()` method
- Experiments have at least one `@text_inspection` method

### "cannot import name 'p_map' from 'mllm.utils'"

The mllm package needs patching. Add to `mllm/utils/__init__.py`:
```python
from .maps import parallel_map, p_map
```

### 401 Authentication Error

Invalid or expired OpenAI API key:
```bash
export OPENAI_API_KEY="sk-proj-your-valid-key"
```

## Development

### Key Files

| File | Purpose |
|------|---------|
| `leeq/epii/backend.py` | LeeQBackend class bridging EPII to LeeQ |
| `leeq/epii/leeq_epii_service.py` | Declarative service with experiment wrappers |
| `leeq/epii/experiments.py` | ExperimentRouter for auto-discovery |
| `leeq/epii/config.py` | EPII configuration management |

### Testing the Integration

```python
# Test experiment discovery
from leeq.epii.experiments import ExperimentRouter
router = ExperimentRouter()
print(f"Found {len(router.experiment_map)} experiments")
for name in router.experiment_map:
    print(f"  - {name}")
```

## References

- [CalibrationNTKAgent Repository](https://github.com/your-org/CalibrationNTKAgent)
- [LeeQ Documentation](https://leeq.readthedocs.io/)
- [EPII Protocol Specification](https://github.com/your-org/epii-spec)
