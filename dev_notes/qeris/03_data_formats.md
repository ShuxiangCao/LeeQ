# Standard Data Formats

## 1. Experiment Status Schema

All quantum experiment frameworks must provide status in this format:

```json
{
  "experiment": {
    "id": "exp_123456",
    "type": "rabi",
    "state": "running",  // "pending", "running", "completed", "failed"
    "progress": 0.45,    // 0.0 to 1.0
    "start_time": "2024-01-20T10:30:00Z",
    "parameters": {
      // Experiment-specific parameters
    }
  },
  "data": {
    "points_collected": 45,
    "total_points": 100,
    "latest_value": 0.823,
    "dimensions": ["time", "amplitude"],
    "units": ["us", "mV"]
  },
  "device": {
    "backend": "quantum_device_1",
    "qubits": ["q0", "q1"],
    "ready": true
  }
}
```

## 2. Live Data Stream Format

```json
{
  "timestamp": "2024-01-20T10:30:45.123Z",
  "type": "data_point",  // "data_point", "status_update", "error"
  "data": {
    "x": [0.1, 0.2, 0.3],
    "y": [0.5, 0.7, 0.9],
    "index": 45
  }
}
```

## 3. Experiment List Format

```json
{
  "experiments": [
    {
      "name": "RabiExperiment",
      "category": "calibration",
      "description": "Measures Rabi oscillations to calibrate qubit drive amplitude",
      "supported_qubits": ["single", "multiple"],
      "required_parameters": ["start", "stop", "step"],
      "optional_parameters": ["shots", "update"]
    },
    {
      "name": "T1Experiment",
      "category": "characterization",
      "description": "Measures qubit relaxation time (T1)",
      "supported_qubits": ["single", "multiple"],
      "required_parameters": ["delays"],
      "optional_parameters": ["shots", "repetitions"]
    }
  ]
}
```

## 4. Experiment Details Format

```json
{
  "name": "RabiExperiment",
  "full_name": "leeq.experiments.builtin.basic.calibrations.rabi.RabiExperiment",
  "category": "calibration",
  "description": "Measures Rabi oscillations by varying the drive pulse amplitude or duration to calibrate the pi pulse",
  "documentation": "The Rabi experiment applies drive pulses of varying amplitude...",
  "parameters": {
    "required": {
      "start": {
        "type": "float",
        "description": "Starting amplitude or time",
        "units": "a.u. or us"
      },
      "stop": {
        "type": "float",
        "description": "Ending amplitude or time",
        "units": "a.u. or us"
      },
      "step": {
        "type": "float",
        "description": "Step size",
        "units": "a.u. or us"
      }
    },
    "optional": {
      "shots": {
        "type": "int",
        "description": "Number of measurement shots",
        "default": 1024
      },
      "update": {
        "type": "bool",
        "description": "Update qubit parameters after fit",
        "default": false
      }
    }
  },
  "returns": {
    "data": "Array of measured expectation values",
    "fit_params": {
      "frequency": "Rabi frequency",
      "amplitude": "Oscillation amplitude",
      "phase": "Phase offset",
      "offset": "DC offset"
    }
  },
  "example": {
    "experiment_name": "RabiExperiment",
    "qubits": ["q0"],
    "parameters": {
      "start": 0,
      "stop": 1,
      "step": 0.05,
      "shots": 1024,
      "update": true
    }
  }
}
```

## 5. Device Information Format

```json
{
  "device_name": "quantum_processor_1",
  "total_qubits": 5,
  "architecture": "linear",
  "backend_type": "superconducting",
  "measurement_basis": ["z"],
  "max_shots": 10000,
  "max_experiments": 100
}
```

## 6. Qubit Parameters Format

The parameters returned are backend-specific. Each parameter includes metadata about its type, unit, and description.

```json
{
  "q0": {
    "parameters": {
      "frequency": {
        "value": 4.562e9,
        "unit": "Hz",
        "type": "float",
        "category": "hamiltonian",
        "description": "Qubit transition frequency"
      },
      "pulse_shape": {
        "value": {
          "type": "gaussian",
          "sigma": 10e-9,
          "truncation": 4
        },
        "unit": null,
        "type": "dict",
        "category": "control",
        "description": "Pulse shape configuration"
      },
      "t1": {
        "value": 45.2e-6,
        "unit": "s",
        "type": "float",
        "category": "coherence",
        "description": "Relaxation time"
      },
      "readout_kernel": {
        "value": [0.98, 0.02, 0.03, 0.97],
        "unit": null,
        "type": "list",
        "category": "readout",
        "description": "2x2 readout confusion matrix"
      }
    },
    "metadata": {
      "last_updated": "2024-01-20T10:30:00Z",
      "status": "ready"
    }
  }
}
```

## 7. Parameter Schema Format

```json
{
  "parameter_schema": {
    "categories": ["hamiltonian", "coherence", "control", "readout", "coupling", "custom"],
    "parameter_types": {
      "supported": ["float", "int", "str", "bool", "list", "dict", "ndarray", "complex", "object"],
      "serialization": "JSON-compatible or base64-encoded for complex objects"
    },
    "examples": {
      "simple_value": {
        "name": "frequency",
        "type": "float",
        "value": 4.5e9
      },
      "complex_value": {
        "name": "pulse_envelope", 
        "type": "ndarray",
        "value": "[base64-encoded-numpy-array]"
      },
      "structured_value": {
        "name": "error_model",
        "type": "dict",
        "value": {
          "T1": {"distribution": "exponential", "rate": 1/50e-6},
          "T2": {"distribution": "gaussian", "mean": 70e-6, "std": 5e-6}
        }
      }
    }
  }
}
```

## 8. Reset Result Format

```json
{
  "reset_type": "full",  // "full", "qubits", "instruments", "calibrations"
  "status": "success",   // "success", "partial", "failed"
  "details": {
    "qubits": "Reset 5 qubits to defaults",
    "instruments": "All instruments reset",
    "calibrations": "Calibration data cleared",
    "setup_status": "Reset to default parameters"
  },
  "error": null  // Error message if status is not "success"
}
```

## 9. Multi-Qubit Parameters Format

Coupling and crosstalk information are just parameters with appropriate metadata:

```json
{
  "device": {
    "parameters": {
      "coupling_map": {
        "value": {
          "q0-q1": {"strength": 0.012e9, "type": "capacitive"},
          "q1-q2": {"strength": 0.011e9, "type": "capacitive"}
        },
        "unit": null,
        "type": "dict",
        "category": "coupling",
        "description": "Physical coupling between qubits"
      },
      "crosstalk_matrix": {
        "value": {
          "_type": "ndarray",
          "data": "[base64-encoded-5x5-matrix]",
          "shape": [5, 5],
          "dtype": "float64"
        },
        "unit": null,
        "type": "ndarray",
        "category": "coupling",
        "description": "Crosstalk coefficients between all qubit pairs"
      }
    },
    "metadata": {
      "parameter_level": "device",
      "last_updated": "2024-01-20T10:30:00Z"
    }
  }
}
```