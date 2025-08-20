# EPII Protocol Documentation

## Overview

The EPII (Experiment Python Interface and Infrastructure) protocol defines the communication interface between EPII clients and the EPII daemon service.

## Protocol Version

Current protocol version: 1.0

## Communication Model

EPII uses a request-response model over TCP sockets with JSON-encoded messages.

### Message Format

All messages are JSON objects with the following structure:

```json
{
    "method": "method_name",
    "params": {...},
    "id": "unique_request_id"
}
```

### Response Format

```json
{
    "result": {...},
    "error": null,
    "id": "unique_request_id"
}
```

## Available Methods

### Core Methods

- `list_experiments`: List available experiments
- `run_experiment`: Execute an experiment
- `get_experiment_status`: Check experiment status
- `cancel_experiment`: Cancel a running experiment

### Parameter Methods

- `list_parameters`: List available parameters
- `get_parameter`: Get parameter value
- `set_parameter`: Set parameter value

### Capability Methods

- `get_capabilities`: Get service capabilities
- `get_version`: Get protocol version

## Error Handling

Errors are returned in the response with an error object:

```json
{
    "error": {
        "code": -32000,
        "message": "Error description",
        "data": {...}
    }
}
```

## Examples

### List Experiments Request

```json
{
    "method": "list_experiments",
    "params": {},
    "id": "req-001"
}
```

### Run Experiment Request

```json
{
    "method": "run_experiment",
    "params": {
        "experiment": "PowerRabi",
        "kwargs": {
            "qubit": 0,
            "amplitude_range": [0, 1],
            "num_points": 51
        }
    },
    "id": "req-002"
}
```

## See Also

- [EPII Client Usage](client-usage.md)
- [EPII README](README.md)