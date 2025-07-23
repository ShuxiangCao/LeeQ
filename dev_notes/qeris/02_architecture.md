# Architecture Overview

## System Architecture

```
┌────────────────────┐     ┌────────────────────┐
│   Web Client       │     │    AI Agent        │
│  (MCP via HTTP)    │     │  (MCP Client)      │
└─────────┬──────────┘     └─────────┬──────────┘
          │                          │
          │ MCP/HTTP                 │ MCP Protocol
          │                          │
          ▼                          ▼
┌─────────────────────────────────────────────────┐
│         QERIS MCP Server (Port 8765)            │
├─────────────────────────────────────────────────┤
│  Tools: run_experiment, get_status, etc.        │
│  Resources: live_data, experiment_history       │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│          Quantum Experiment Adapter              │
│         (Framework-specific implementation)      │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│     Quantum Experiment Framework                 │
│   (LeeQ, Qiskit, Cirq, PyQuil, etc.)           │
└─────────────────────────────────────────────────┘
```

## Component Descriptions

### QERIS MCP Server

The central server component that:
- Hosts MCP tools for experiment control
- Provides MCP resources for real-time data
- Manages client connections
- Handles protocol translation

### Quantum Experiment Adapter

The abstraction layer that:
- Implements the `QERISAdapter` interface
- Translates framework-specific operations to QERIS standard
- Handles data serialization/deserialization
- Manages experiment lifecycle

### Quantum Experiment Framework

The underlying quantum control software:
- Executes experiments on hardware or simulators
- Manages qubit calibrations and parameters
- Provides measurement data
- Controls hardware instruments

## Data Flow

### Experiment Execution

1. Client calls `run_experiment` MCP tool
2. Server invokes adapter's `run_experiment` method
3. Adapter translates to framework-specific calls
4. Framework executes on quantum hardware
5. Results flow back through the same path

### Real-time Monitoring

1. Client subscribes to `qeris://live_data` resource
2. Server polls adapter's `get_live_data` method
3. Adapter retrieves current data from framework
4. Server streams updates to all subscribers
5. Clients receive data via MCP resource updates

### Parameter Management

1. Client requests parameters via `get_qubit_parameters`
2. Adapter queries framework for current values
3. Values are serialized to JSON-compatible format
4. Metadata (type, unit, category) is included
5. Client receives structured parameter data

## Communication Protocols

### MCP Tools (Request/Response)

- Synchronous operations
- JSON-RPC style communication
- Error handling with standard responses
- Typed parameters and returns

### MCP Resources (Streaming)

- Asynchronous updates
- Subscribe/unsubscribe model
- Server-sent events for HTTP clients
- Automatic reconnection support

### HTTP Bridge

- RESTful endpoints for MCP tools
- EventSource for resource subscriptions
- CORS support for web clients
- JSON content type throughout