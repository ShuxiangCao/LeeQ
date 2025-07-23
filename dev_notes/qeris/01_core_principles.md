# Core Principles

## 1. Framework Agnostic

QERIS is designed to work with any quantum experiment software, providing a common interface regardless of the underlying implementation. Whether you're using LeeQ, Qiskit, Cirq, PyQuil, or a custom framework, QERIS provides the same standard interface.

## 2. Simple MVP

As a Minimum Viable Product, QERIS focuses on essential functionality only:
- No authentication or encryption (handled at a different layer)
- No complex state management beyond basic experiment tracking
- Simple JSON-based data formats
- Clear, minimal API surface

## 3. Standard Data Format

All quantum experiments communicate using a common JSON schema that includes:
- Experiment status and progress
- Real-time data points
- Device configuration
- Qubit parameters

## 4. Real-time Updates

MCP resources with subscription provide live data streaming:
- No need for WebSocket implementation
- Built-in support for multiple concurrent subscribers
- Efficient updates only when data changes

## 5. MCP Native

The entire protocol is built on top of Model Context Protocol (MCP):
- Standard tools for experiment control
- Resources for real-time data streaming
- HTTP bridge for web clients
- Native support for AI agents

## 6. Backend Agnostic Parameters

Parameters can represent any type of data:
- Simple values (float, int, string, bool)
- Complex structures (dictionaries, arrays)
- Serialized objects (numpy arrays, custom types)
- Metadata includes type, unit, category, and description

## 7. Dynamic Discovery

No fixed experiment mappings:
- Experiments are discovered at runtime
- Full documentation and parameter schemas exposed
- AI agents can understand and choose experiments autonomously

## 8. Extensible Design

While keeping the core simple, QERIS allows for:
- Custom parameter categories
- Backend-specific experiment types
- Additional metadata fields
- Future protocol extensions