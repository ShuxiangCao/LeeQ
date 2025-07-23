# MCP Interface Specification

## MCP Tools

Standard MCP tools that all quantum experiment frameworks must implement:

### Experiment Control

#### list_experiments
```python
{
    "name": "list_experiments",
    "description": "List all available experiments with documentation",
    "parameters": {
        "category": "string"  # Optional: "calibration", "characterization", "gates", etc.
    }
}
```

#### get_experiment_info
```python
{
    "name": "get_experiment_info",
    "description": "Get detailed information about a specific experiment",
    "parameters": {
        "experiment_name": "string"
    }
}
```

#### run_experiment
```python
{
    "name": "run_experiment",
    "description": "Start a quantum experiment",
    "parameters": {
        "experiment_name": "string",  # Full experiment class name
        "qubits": "array",  # ["q0", "q1", ...] 
        "parameters": "object"  # Experiment-specific parameters
    }
}
```

#### get_status
```python
{
    "name": "get_status",
    "description": "Get current experiment status",
    "parameters": {}
}
```

#### stop_experiment
```python
{
    "name": "stop_experiment",
    "description": "Stop running experiment",
    "parameters": {
        "experiment_id": "string"
    }
}
```

#### get_results
```python
{
    "name": "get_results",
    "description": "Retrieve experiment results",
    "parameters": {
        "experiment_id": "string",
        "format": "string"  # "raw", "processed", "plot"
    }
}
```

### Qubit Configuration

#### get_device_info
```python
{
    "name": "get_device_info",
    "description": "Get quantum device configuration and topology",
    "parameters": {}
}
```

#### reset_hardware
```python
{
    "name": "reset_hardware",
    "description": "Reset hardware to default state (qubits, instruments, etc.)",
    "parameters": {
        "reset_type": "string"  # "full", "qubits", "instruments", "calibrations"
    }
}
```

#### reset_server
```python
{
    "name": "reset_server",
    "description": "Reset MCP server state (clear cache, reset connections, etc.)",
    "parameters": {
        "keep_hardware_state": "boolean"  # Whether to preserve hardware state
    }
}
```

#### list_qubits
```python
{
    "name": "list_qubits",
    "description": "List all available qubits and their status",
    "parameters": {
        "include_parameters": "boolean"  # Include full parameter details
    }
}
```

#### get_parameter_schema
```python
{
    "name": "get_parameter_schema",
    "description": "Get schema of available parameters for this backend",
    "parameters": {}
}
```

#### get_qubit_parameters
```python
{
    "name": "get_qubit_parameters",
    "description": "Get parameters for specific qubit(s) or device",
    "parameters": {
        "qubits": "array",  # ["q0", "q1", ...] or ["all"] or ["device"]
        "parameter_categories": "array"  # Optional: ["hamiltonian", "coherence", "control", "coupling", ...] 
    }
}
```

#### set_qubit_parameters
```python
{
    "name": "set_qubit_parameters",
    "description": "Update parameters for specific qubit(s)",
    "parameters": {
        "updates": "array"  # [{"qubit": "q0", "parameter": "frequency", "value": 4.5e9}, ...]
    }
}
```

## MCP Resources

Standard MCP resources for real-time data:

### live_data
```python
{
    "uri": "qeris://live_data",
    "name": "Live Experiment Data",
    "description": "Real-time data stream from running experiment",
    "mimeType": "application/json"
}
```

### experiment_status
```python
{
    "uri": "qeris://experiment_status", 
    "name": "Experiment Status",
    "description": "Current experiment status and progress",
    "mimeType": "application/json"
}
```

### device_status
```python
{
    "uri": "qeris://device_status",
    "name": "Device Status", 
    "description": "Quantum device status and configuration",
    "mimeType": "application/json"
}
```

### qubit_parameters
```python
{
    "uri": "qeris://qubit_parameters",
    "name": "Qubit Parameters",
    "description": "Real-time qubit parameter values",
    "mimeType": "application/json"
}
```

## Resource Update Intervals

- `live_data`: 0.5 seconds (configurable)
- `experiment_status`: 1.0 second
- `device_status`: 5.0 seconds
- `qubit_parameters`: 5.0 seconds

## Error Handling

All MCP tools should return errors in standard format:

```json
{
    "error": {
        "code": "EXPERIMENT_NOT_FOUND",
        "message": "Experiment 'UnknownExperiment' not found",
        "details": {
            "available_experiments": ["RabiExperiment", "T1Experiment", ...]
        }
    }
}
```

Common error codes:
- `EXPERIMENT_NOT_FOUND`
- `INVALID_PARAMETERS`
- `QUBIT_NOT_AVAILABLE`
- `HARDWARE_ERROR`
- `EXPERIMENT_RUNNING`
- `NO_ACTIVE_EXPERIMENT`
- `PARAMETER_READ_ONLY`
- `RESET_FAILED`