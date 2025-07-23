# Implementation Guide

## Base QERIS Server

A minimal implementation that any quantum framework can extend:

```python
# qeris/server.py
import asyncio
import json
from abc import ABC, abstractmethod
from mcp import Server, Resource
from typing import AsyncGenerator

class QERISAdapter(ABC):
    """Abstract adapter that quantum frameworks must implement"""
    
    # Experiment Discovery
    @abstractmethod
    async def list_experiments(self, category: str = None) -> dict:
        """List all available experiments with basic info"""
        pass
    
    @abstractmethod
    async def get_experiment_info(self, experiment_name: str) -> dict:
        """Get detailed information about a specific experiment"""
        pass
    
    # Experiment Control
    @abstractmethod
    async def run_experiment(self, experiment_name: str, qubits: list, parameters: dict) -> str:
        """Start an experiment and return experiment ID"""
        pass
    
    @abstractmethod
    async def get_status(self) -> dict:
        """Return status in QERIS format"""
        pass
    
    @abstractmethod
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment"""
        pass
    
    @abstractmethod
    async def get_results(self, experiment_id: str, format: str) -> dict:
        """Get experiment results"""
        pass
    
    @abstractmethod
    async def get_live_data(self) -> dict:
        """Get latest data point for streaming"""
        pass
    
    # Qubit Configuration
    @abstractmethod
    async def get_device_info(self) -> dict:
        """Get device configuration and capabilities"""
        pass
    
    @abstractmethod
    async def reset_hardware(self, reset_type: str = "full") -> dict:
        """Reset hardware to default state"""
        pass
    
    @abstractmethod
    async def reset_server(self, keep_hardware_state: bool = True) -> dict:
        """Reset MCP server state"""
        pass
    
    @abstractmethod
    async def list_qubits(self, include_parameters: bool = False) -> dict:
        """List all qubits and optionally their parameters"""
        pass
    
    @abstractmethod
    async def get_parameter_schema(self) -> dict:
        """Get schema of available parameters for this backend"""
        pass
    
    @abstractmethod
    async def get_qubit_parameters(self, qubits: list, parameter_categories: list = None) -> dict:
        """Get parameters for specified qubits"""
        pass
    
    @abstractmethod
    async def set_qubit_parameters(self, updates: list) -> dict:
        """Update qubit parameters"""
        pass

class QERISServer:
    def __init__(self, adapter: QERISAdapter, port: int = 8765):
        self.adapter = adapter
        self.port = port
        self.mcp_server = Server("qeris")
        self._setup_mcp_interface()
        
    def _setup_mcp_interface(self):
        """Register MCP tools and resources"""
        
        # Register experiment discovery tools
        @self.mcp_server.tool()
        async def list_experiments(category: str = None):
            return await self.adapter.list_experiments(category)
            
        @self.mcp_server.tool()
        async def get_experiment_info(experiment_name: str):
            return await self.adapter.get_experiment_info(experiment_name)
        
        # Register experiment control tools
        @self.mcp_server.tool()
        async def run_experiment(experiment_name: str, qubits: list, parameters: dict):
            return await self.adapter.run_experiment(experiment_name, qubits, parameters)
            
        @self.mcp_server.tool()
        async def get_status():
            return await self.adapter.get_status()
            
        @self.mcp_server.tool()
        async def stop_experiment(experiment_id: str):
            return await self.adapter.stop_experiment(experiment_id)
            
        @self.mcp_server.tool()
        async def get_results(experiment_id: str, format: str = "processed"):
            return await self.adapter.get_results(experiment_id, format)
        
        # Register qubit configuration tools
        @self.mcp_server.tool()
        async def get_device_info():
            return await self.adapter.get_device_info()
        
        @self.mcp_server.tool()
        async def reset_hardware(reset_type: str = "full"):
            return await self.adapter.reset_hardware(reset_type)
            
        @self.mcp_server.tool()
        async def reset_server(keep_hardware_state: bool = True):
            return await self.adapter.reset_server(keep_hardware_state)
            
        @self.mcp_server.tool()
        async def list_qubits(include_parameters: bool = False):
            return await self.adapter.list_qubits(include_parameters)
            
        @self.mcp_server.tool()
        async def get_parameter_schema():
            return await self.adapter.get_parameter_schema()
            
        @self.mcp_server.tool()
        async def get_qubit_parameters(qubits: list, parameter_categories: list = None):
            return await self.adapter.get_qubit_parameters(qubits, parameter_categories)
            
        @self.mcp_server.tool()
        async def set_qubit_parameters(updates: list):
            return await self.adapter.set_qubit_parameters(updates)
        
        # Register resources for real-time data
        @self.mcp_server.resource("qeris://live_data")
        async def live_data_resource() -> AsyncGenerator[Resource, None]:
            """Stream live experiment data"""
            while True:
                data = await self.adapter.get_live_data()
                yield Resource(
                    uri="qeris://live_data",
                    name="Live Experiment Data",
                    description="Real-time data from running experiment",
                    mimeType="application/json",
                    text=json.dumps(data)
                )
                await asyncio.sleep(0.5)  # Update interval
        
        @self.mcp_server.resource("qeris://experiment_status")
        async def status_resource() -> AsyncGenerator[Resource, None]:
            """Stream experiment status updates"""
            while True:
                status = await self.adapter.get_status()
                yield Resource(
                    uri="qeris://experiment_status",
                    name="Experiment Status",
                    description="Current experiment status and progress",
                    mimeType="application/json",
                    text=json.dumps(status)
                )
                await asyncio.sleep(1.0)  # Status update interval
        
        @self.mcp_server.resource("qeris://qubit_parameters")
        async def qubit_parameters_resource() -> AsyncGenerator[Resource, None]:
            """Stream qubit parameter updates"""
            while True:
                params = await self.adapter.get_qubit_parameters(["all"])
                yield Resource(
                    uri="qeris://qubit_parameters",
                    name="Qubit Parameters",
                    description="Real-time qubit parameter values",
                    mimeType="application/json",
                    text=json.dumps(params)
                )
                await asyncio.sleep(5.0)  # Parameter update interval
    
    async def start(self):
        """Start MCP server"""
        print(f"QERIS MCP Server starting on port {self.port}")
        await self.mcp_server.run(port=self.port)
```

## Implementation Steps

### 1. Create Your Adapter

Implement the `QERISAdapter` abstract class for your quantum framework:

```python
from qeris.server import QERISAdapter

class MyQuantumAdapter(QERISAdapter):
    def __init__(self, quantum_backend):
        self.backend = quantum_backend
        self.experiments = self._discover_experiments()
        
    async def list_experiments(self, category=None):
        # Return list of available experiments
        pass
        
    async def run_experiment(self, experiment_name, qubits, parameters):
        # Start experiment on your backend
        pass
        
    # ... implement all required methods
```

### 2. Handle Data Serialization

For complex data types, implement serialization:

```python
def serialize_value(self, value):
    if isinstance(value, np.ndarray):
        return {
            "_type": "ndarray",
            "data": base64.b64encode(value.tobytes()).decode('utf-8'),
            "shape": value.shape,
            "dtype": str(value.dtype)
        }
    elif isinstance(value, complex):
        return {"_type": "complex", "real": value.real, "imag": value.imag}
    # ... handle other types
    return value
```

### 3. Start the Server

```python
async def main():
    # Initialize your quantum backend
    backend = MyQuantumBackend()
    
    # Create adapter
    adapter = MyQuantumAdapter(backend)
    
    # Create and start server
    server = QERISServer(adapter, port=8765)
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Error Handling**: Always return structured errors with helpful messages
2. **Parameter Validation**: Validate parameters before passing to backend
3. **Resource Cleanup**: Properly handle experiment cancellation and cleanup
4. **Caching**: Cache expensive operations like experiment discovery
5. **Logging**: Log all operations for debugging and monitoring
6. **Type Safety**: Use type hints throughout your implementation
7. **Testing**: Write unit tests for your adapter implementation

## Testing Your Implementation

Use the QERIS compliance test suite:

```python
from qeris.tests import compliance_test

async def test_my_adapter():
    adapter = MyQuantumAdapter(test_backend)
    await compliance_test(adapter)
```