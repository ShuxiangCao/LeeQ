# Adapter Examples

## Simple Quantum Adapter

A minimal adapter showing parameter exposure:

```python
# example_setup.py
import asyncio
from qeris.server import QERISServer, QERISAdapter

class SimpleQuantumAdapter(QERISAdapter):
    """Minimal adapter showing parameter exposure"""
    
    def __init__(self):
        # Define what parameters this backend supports
        self.parameter_definitions = {
            # Single qubit parameters
            "qubit": {
                "frequency": {
                    "type": "float",
                    "unit": "Hz",
                    "category": "hamiltonian",
                    "description": "Qubit transition frequency",
                    "settable": True
                },
                "t1": {
                    "type": "float", 
                    "unit": "s",
                    "category": "coherence",
                    "description": "Relaxation time",
                    "settable": False  # Read-only
                },
                "pi_pulse": {
                    "type": "dict",
                    "unit": None,
                    "category": "control",
                    "description": "Pi pulse parameters",
                    "settable": True,
                    "schema": {
                        "amplitude": "float",
                        "duration": "float",
                        "shape": "str"
                    }
                }
            },
            # Device-level parameters
            "device": {
                "coupling_map": {
                    "type": "dict",
                    "unit": "Hz",
                    "category": "coupling",
                    "description": "Qubit coupling strengths",
                    "settable": False
                },
                "topology": {
                    "type": "str",
                    "unit": None,
                    "category": "coupling",
                    "description": "Device topology type",
                    "settable": False
                }
            }
        }
        
        # Simple in-memory storage for demo
        self.qubit_values = {
            "q0": {
                "frequency": 5.0e9,
                "t1": 50e-6,
                "pi_pulse": {"amplitude": 0.5, "duration": 20e-9, "shape": "gaussian"}
            },
            "q1": {
                "frequency": 5.1e9,
                "t1": 45e-6,
                "pi_pulse": {"amplitude": 0.48, "duration": 20e-9, "shape": "gaussian"}
            }
        }
        
        self.device_values = {
            "coupling_map": {"q0-q1": 0.01e9},
            "topology": "linear"
        }
    
    async def get_parameter_schema(self) -> dict:
        """Return what parameters are available"""
        return {
            "parameter_schema": {
                "categories": list(set(
                    p["category"] 
                    for params in self.parameter_definitions.values()
                    for p in params.values()
                )),
                "qubit_parameters": list(self.parameter_definitions["qubit"].keys()),
                "device_parameters": list(self.parameter_definitions["device"].keys()),
                "parameter_info": self.parameter_definitions
            }
        }
    
    async def get_qubit_parameters(self, qubits: list, parameter_categories: list = None) -> dict:
        """Get parameter values"""
        result = {}
        
        for qubit in qubits:
            if qubit == "device":
                result["device"] = {
                    "parameters": {},
                    "metadata": {"parameter_level": "device"}
                }
                for param_name, param_def in self.parameter_definitions["device"].items():
                    if parameter_categories and param_def["category"] not in parameter_categories:
                        continue
                    result["device"]["parameters"][param_name] = {
                        "value": self.device_values.get(param_name),
                        **param_def
                    }
            elif qubit in self.qubit_values:
                result[qubit] = {
                    "parameters": {},
                    "metadata": {"status": "ready"}
                }
                for param_name, param_def in self.parameter_definitions["qubit"].items():
                    if parameter_categories and param_def["category"] not in parameter_categories:
                        continue
                    result[qubit]["parameters"][param_name] = {
                        "value": self.qubit_values[qubit].get(param_name),
                        **param_def
                    }
                    
        return result
    
    async def set_qubit_parameters(self, updates: list) -> dict:
        """Update parameters"""
        results = {"updates": {}}
        
        for update in updates:
            qubit = update["qubit"]
            param = update["parameter"]
            value = update["value"]
            
            # Check if parameter is settable
            if param in self.parameter_definitions["qubit"]:
                if self.parameter_definitions["qubit"][param]["settable"]:
                    self.qubit_values[qubit][param] = value
                    results["updates"][qubit] = {
                        "parameter": param,
                        "status": "updated"
                    }
                else:
                    results["updates"][qubit] = {
                        "parameter": param,
                        "status": "failed",
                        "error": "Parameter is read-only"
                    }
                    
        return results
    
    async def reset_hardware(self, reset_type: str = "full") -> dict:
        """Reset hardware state"""
        results = {"reset_type": reset_type, "status": "success", "details": {}}
        
        if reset_type in ["full", "qubits"]:
            # Reset qubit parameters to defaults
            self.qubit_values = {
                "q0": {
                    "frequency": 5.0e9,
                    "t1": 50e-6,
                    "pi_pulse": {"amplitude": 0.5, "duration": 20e-9, "shape": "gaussian"}
                },
                "q1": {
                    "frequency": 5.1e9,
                    "t1": 45e-6,
                    "pi_pulse": {"amplitude": 0.48, "duration": 20e-9, "shape": "gaussian"}
                }
            }
            results["details"]["qubits"] = "Reset to default values"
            
        return results
    
    async def reset_server(self, keep_hardware_state: bool = True) -> dict:
        """Reset MCP server state"""
        results = {"status": "success", "details": {}}
        
        # Clear any cached data
        results["details"]["cache"] = "Cleared"
        
        if not keep_hardware_state:
            hw_reset = await self.reset_hardware("full")
            results["details"]["hardware"] = hw_reset["details"]
        else:
            results["details"]["hardware"] = "Preserved"
            
        return results
    
    # Implement other required methods...
    async def list_experiments(self, category: str = None) -> dict:
        return {"experiments": []}  # Simplified
        
    async def get_experiment_info(self, experiment_name: str) -> dict:
        return {}  # Simplified
        
    async def run_experiment(self, experiment_name: str, qubits: list, parameters: dict) -> str:
        return "exp_001"  # Simplified
        
    async def get_status(self) -> dict:
        return {"experiment": {"state": "idle"}}  # Simplified
        
    async def get_live_data(self) -> dict:
        return {"type": "idle"}  # Simplified
        
    async def stop_experiment(self, experiment_id: str) -> bool:
        return True  # Simplified
        
    async def get_results(self, experiment_id: str, format: str) -> dict:
        return {"data": []}  # Simplified
        
    async def get_device_info(self) -> dict:
        return {"device_name": "demo_device", "total_qubits": 2}  # Simplified
        
    async def list_qubits(self, include_parameters: bool = False) -> dict:
        return {"q0": {"status": "ready"}, "q1": {"status": "ready"}}  # Simplified

# Simple script to start the server
async def main():
    # Create adapter with your quantum system
    adapter = SimpleQuantumAdapter()
    
    # Create and start QERIS server
    server = QERISServer(adapter, port=8765)
    
    print("Starting QERIS server...")
    print("Available parameters:")
    schema = await adapter.get_parameter_schema()
    print(f"  Qubit parameters: {schema['parameter_schema']['qubit_parameters']}")
    print(f"  Device parameters: {schema['parameter_schema']['device_parameters']}")
    
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## Quick Start Script

For the absolute simplest setup:

```python
# quick_start_qeris.py
import asyncio
from qeris.server import QERISServer, QERISAdapter

class MyQuantumSystem(QERISAdapter):
    """Your quantum system adapter"""
    
    def __init__(self, backend):
        self.backend = backend  # Your actual quantum backend
        
    async def get_parameter_schema(self):
        # Return what parameters your system has
        return {
            "parameter_schema": {
                "qubit_parameters": ["frequency", "t1", "t2"],
                "device_parameters": ["coupling_strengths"],
                "categories": ["hamiltonian", "coherence", "coupling"]
            }
        }
    
    async def get_qubit_parameters(self, qubits, categories=None):
        # Get values from your backend
        params = {}
        for q in qubits:
            if q == "device":
                params["device"] = {
                    "parameters": {
                        "coupling_strengths": {
                            "value": self.backend.get_couplings(),
                            "type": "dict",
                            "category": "coupling"
                        }
                    }
                }
            else:
                params[q] = {
                    "parameters": {
                        "frequency": {
                            "value": self.backend.get_frequency(q),
                            "type": "float",
                            "unit": "Hz",
                            "category": "hamiltonian"
                        }
                    }
                }
        return params
    
    # ... implement other required methods

# Start server
async def start_server(quantum_backend, port=8765):
    adapter = MyQuantumSystem(quantum_backend)
    server = QERISServer(adapter, port)
    print(f"QERIS server starting on port {port}")
    await server.start()

# Usage
# quantum_backend = YourQuantumSystem()
# asyncio.run(start_server(quantum_backend))
```

## Framework-Specific Examples

### Qiskit Adapter Example

```python
from qeris.server import QERISAdapter

class QiskitAdapter(QERISAdapter):
    def __init__(self, backend):
        self.backend = backend
        self.jobs = {}
        
    async def run_experiment(self, type: str, qubits: list, parameters: dict) -> str:
        # Create Qiskit circuit based on experiment type
        circuit = self._create_circuit(type, qubits, parameters)
        job = self.backend.run(circuit, shots=parameters.get('shots', 1024))
        job_id = str(job.job_id())
        self.jobs[job_id] = job
        return job_id
        
    async def get_status(self) -> dict:
        # Convert Qiskit job status to QERIS format
        pass
```

### Integration with Existing Frameworks

For frameworks with existing experiment managers:

```python
class ExistingFrameworkAdapter(QERISAdapter):
    def __init__(self, experiment_manager, device_manager):
        self.exp_mgr = experiment_manager
        self.dev_mgr = device_manager
        
    async def list_experiments(self, category=None):
        # Wrap existing experiment registry
        experiments = []
        for name, exp_class in self.exp_mgr.registry.items():
            if category and exp_class.category != category:
                continue
            experiments.append({
                'name': name,
                'category': exp_class.category,
                'description': exp_class.__doc__.split('\n')[0]
            })
        return {'experiments': experiments}
```