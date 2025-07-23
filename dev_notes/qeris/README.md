# Quantum Experiment Remote Interface Standard (QERIS)

## Overview

QERIS is a universal standard for remote interfaces to quantum experiments. It provides a simple, extensible protocol that combines MCP (Model Context Protocol) for programmatic access with real-time updates for monitoring. This standard is designed to work with any quantum experiment framework (LeeQ, Qiskit, Cirq, etc.) while maintaining simplicity as a Minimum Viable Product (MVP).

## Table of Contents

1. [Core Principles](./01_core_principles.md)
2. [Architecture Overview](./02_architecture.md)
3. [Standard Data Formats](./03_data_formats.md)
4. [MCP Interface Specification](./04_mcp_interface.md)
5. [Implementation Guide](./05_implementation_guide.md)
6. [Adapter Examples](./06_adapter_examples.md)
7. [Client Examples](./07_client_examples.md)
8. [Reset Commands](./08_reset_commands.md)
9. [Parameter Management](./09_parameter_management.md)
10. [Experiment Discovery](./10_experiment_discovery.md)
11. [Real-time Monitoring](./11_realtime_monitoring.md)
12. [Integration Guide](./12_integration_guide.md)

## Quick Start

1. Implement the `QERISAdapter` interface for your quantum framework
2. Start the QERIS server:
   ```python
   adapter = YourFrameworkAdapter()
   server = QERISServer(adapter)
   await server.start()
   ```
3. Connect with any MCP client or web interface

## Benefits

1. **Universal**: One protocol for all quantum frameworks
2. **Simple**: Core adapter methods to implement with clear abstractions
3. **MCP Native**: Leverages MCP for both control and monitoring
4. **Real-time**: MCP resources provide efficient live data streaming
5. **AI-Ready**: Standard tools and resources for autonomous agents
6. **Web Compatible**: MCP can be exposed via HTTP for browser clients
7. **Robust Recovery**: Built-in reset commands for hardware and server state management
8. **Maintenance-Friendly**: Support for calibration clearing and clean session initialization

## License

This standard is open for implementation by any quantum computing framework.