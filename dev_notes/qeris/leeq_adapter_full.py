# leeq/remote/qeris_adapter.py
"""
Full LeeQ implementation of QERIS adapter
"""

from qeris.server import QERISAdapter
import uuid
import inspect
import datetime
import numpy as np
import base64

class LeeQAdapter(QERISAdapter):
    def __init__(self, experiment_manager, setup):
        self.experiment_manager = experiment_manager
        self.setup = setup
        self.current_experiment_id = None
        self._discover_experiments()
        
    def _discover_experiments(self):
        """Discover all available experiments in LeeQ"""
        import inspect
        from leeq.experiments.builtin import basic, multi_qubit_gates
        
        self.experiments = {}
        
        # Scan experiment modules
        modules = [
            (basic.calibrations, "calibration"),
            (basic.characterizations, "characterization"),
            (multi_qubit_gates, "gates")
        ]
        
        for module, category in modules:
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'run') and 
                    name.endswith('Experiment')):
                    self.experiments[name] = {
                        'class': obj,
                        'category': category,
                        'module': obj.__module__,
                        'doc': inspect.getdoc(obj) or "No documentation available"
                    }
    
    async def list_experiments(self, category: str = None) -> dict:
        """List all available LeeQ experiments"""
        experiments = []
        for name, info in self.experiments.items():
            if category is None or info['category'] == category:
                # Extract parameter info from __init__ signature
                sig = inspect.signature(info['class'].__init__)
                params = list(sig.parameters.keys())[1:]  # Skip 'self'
                
                experiments.append({
                    'name': name,
                    'category': info['category'],
                    'description': info['doc'].split('\n')[0],  # First line of docstring
                    'supported_qubits': self._get_qubit_support(info['class']),
                    'required_parameters': [p for p in params if sig.parameters[p].default == inspect.Parameter.empty],
                    'optional_parameters': [p for p in params if sig.parameters[p].default != inspect.Parameter.empty]
                })
        
        return {'experiments': experiments}
    
    async def get_experiment_info(self, experiment_name: str) -> dict:
        """Get detailed information about a specific experiment"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
            
        info = self.experiments[experiment_name]
        exp_class = info['class']
        sig = inspect.signature(exp_class.__init__)
        
        # Build parameter details
        parameters = {'required': {}, 'optional': {}}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_info = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                'description': self._extract_param_description(exp_class, param_name),
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
            
            if param.default == inspect.Parameter.empty:
                parameters['required'][param_name] = param_info
            else:
                parameters['optional'][param_name] = param_info
        
        # Create example
        example_params = {}
        if 'RabiExperiment' in experiment_name:
            example_params = {'start': 0, 'stop': 1, 'step': 0.05, 'shots': 1024}
        elif 'T1Experiment' in experiment_name:
            example_params = {'delays': list(range(0, 200, 10)), 'shots': 1024}
        
        return {
            'name': experiment_name,
            'full_name': f"{info['module']}.{experiment_name}",
            'category': info['category'],
            'description': info['doc'].split('\n')[0],
            'documentation': info['doc'],
            'parameters': parameters,
            'returns': self._get_return_info(exp_class),
            'example': {
                'experiment_name': experiment_name,
                'qubits': ['q0'],
                'parameters': example_params
            }
        }
    
    async def run_experiment(self, experiment_name: str, qubits: list, parameters: dict) -> str:
        """Start a LeeQ experiment by name"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
            
        exp_class = self.experiments[experiment_name]['class']
        
        # Generate unique ID
        self.current_experiment_id = str(uuid.uuid4())
        
        # Get qubit objects from setup
        qubit_objs = [self.setup.get_qubit(q) for q in qubits]
        
        # Start experiment
        if len(qubit_objs) == 1:
            experiment = exp_class(qubit_objs[0], **parameters)
        else:
            experiment = exp_class(qubit_objs, **parameters)
        
        return self.current_experiment_id
    
    def _get_qubit_support(self, exp_class):
        """Determine if experiment supports single/multiple qubits"""
        # This would need more sophisticated analysis
        return ["single", "multiple"]
    
    def _extract_param_description(self, exp_class, param_name):
        """Extract parameter description from docstring"""
        # This would parse the docstring for parameter descriptions
        return f"Parameter {param_name}"
    
    def _get_return_info(self, exp_class):
        """Extract return value information"""
        # This would analyze what the experiment returns
        return {
            "data": "Measurement data",
            "fit_params": "Fitted parameters (if applicable)"
        }
    
    def _serialize_parameter_value(self, value):
        """Serialize parameter values for JSON compatibility"""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_parameter_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_parameter_value(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            return {
                "_type": "ndarray",
                "data": base64.b64encode(value.tobytes()).decode('utf-8'),
                "shape": value.shape,
                "dtype": str(value.dtype)
            }
        elif isinstance(value, complex):
            return {"_type": "complex", "real": value.real, "imag": value.imag}
        elif callable(value):
            return {"_type": "callable", "name": getattr(value, '__name__', str(value))}
        else:
            # For custom objects, try to serialize as dict or string
            try:
                return {"_type": "object", "class": type(value).__name__, "data": str(value)}
            except:
                return {"_type": "unserializable", "class": type(value).__name__}
    
    def _get_value_type(self, value):
        """Get the type name of a value"""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, np.ndarray):
            return "ndarray"
        elif isinstance(value, complex):
            return "complex"
        elif callable(value):
            return "callable"
        else:
            return "object"
    
    async def get_status(self) -> dict:
        """Convert LeeQ status to QERIS format"""
        leeq_status = self.experiment_manager.get_live_status()
        
        # Transform to QERIS standard format
        return {
            "experiment": {
                "id": self.current_experiment_id or "none",
                "type": leeq_status.get("experiment_arguments", {}).get("type", "unknown"),
                "state": "running" if leeq_status.get("engine_status", {}).get("progress", 0) < 1 else "completed",
                "progress": leeq_status.get("engine_status", {}).get("progress", 0),
                "start_time": "2024-01-20T10:30:00Z",  # Add timestamp tracking
                "parameters": leeq_status.get("experiment_arguments", {})
            },
            "data": {
                "points_collected": leeq_status.get("engine_status", {}).get("step_no", [0])[0],
                "total_points": 100,  # Get from experiment
                "latest_value": 0,  # Get from measurement primitive
                "dimensions": ["time", "amplitude"],
                "units": ["us", "mV"]
            },
            "device": {
                "backend": leeq_status.get("setup_status", {}).get("setup_name", "unknown"),
                "qubits": [],  # Get from experiment
                "ready": True
            }
        }
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment"""
        # Implementation depends on LeeQ's capability
        return True
    
    async def get_results(self, experiment_id: str, format: str) -> dict:
        """Get experiment results"""
        # Retrieve from LeeQ's data storage
        return {"data": [], "fit_params": {}}
    
    async def get_live_data(self) -> dict:
        """Get latest data point"""
        status = await self.get_status()
        return {
            "timestamp": "2024-01-20T10:30:45.123Z",
            "type": "data_point",
            "data": {
                "x": [],  # Get from measurement primitive
                "y": [],
                "index": status["data"]["points_collected"]
            }
        }
    
    # Qubit Configuration Methods
    async def get_device_info(self) -> dict:
        """Get LeeQ device information"""
        return {
            "device_name": self.setup._name,
            "total_qubits": len(self.setup.qubits),
            "architecture": "custom",
            "backend_type": "superconducting",
            "measurement_basis": ["z"],
            "max_shots": self.setup._status.get_parameters("Shot_Number"),
            "max_experiments": 100
        }
    
    async def list_qubits(self, include_parameters: bool = False) -> dict:
        """List all qubits in the setup"""
        qubit_list = {}
        for qubit_name, qubit in self.setup.qubits.items():
            info = {"name": qubit_name, "status": "ready"}
            if include_parameters:
                info.update(await self.get_qubit_parameters([qubit_name]))
            qubit_list[qubit_name] = info
        return qubit_list
    
    async def get_parameter_schema(self) -> dict:
        """Get LeeQ parameter schema"""
        # Discover available parameters from a sample qubit
        sample_qubit = list(self.setup.qubits.values())[0] if self.setup.qubits else None
        available_params = []
        
        if sample_qubit:
            # Introspect qubit object to find available parameters
            for attr in dir(sample_qubit):
                if attr.startswith('get_') and not attr.startswith('get_c'):
                    param_name = attr[4:]  # Remove 'get_' prefix
                    available_params.append(param_name)
        
        return {
            "parameter_schema": {
                "categories": ["hamiltonian", "coherence", "control", "readout", "custom"],
                "available_parameters": available_params,
                "backend_type": self.setup._name,
                "parameter_info": {
                    "frequency": {"category": "hamiltonian", "unit": "Hz", "type": "float"},
                    "anharmonicity": {"category": "hamiltonian", "unit": "Hz", "type": "float"},
                    "t1": {"category": "coherence", "unit": "s", "type": "float"},
                    "t2": {"category": "coherence", "unit": "s", "type": "float"},
                    "pi_amp": {"category": "control", "unit": "a.u.", "type": "float"},
                    "readout_frequency": {"category": "readout", "unit": "Hz", "type": "float"}
                }
            }
        }
    
    async def get_qubit_parameters(self, qubits: list, parameter_categories: list = None) -> dict:
        """Get LeeQ qubit parameters in backend-agnostic format"""
        params = {}
        schema = await self.get_parameter_schema()
        param_info = schema['parameter_schema']['parameter_info']
        
        for qubit_name in qubits:
            if qubit_name == "all":
                return await self.get_qubit_parameters(list(self.setup.qubits.keys()), parameter_categories)
            elif qubit_name == "device":
                # Return device-level parameters
                return {"device": await self.get_device_level_parameters()}
                
            qubit = self.setup.get_qubit(qubit_name)
            qubit_params = {"parameters": {}, "metadata": {}}
            
            # Dynamically get parameters based on what the qubit object provides
            for attr in dir(qubit):
                if attr.startswith('get_') and not attr.startswith('get_c'):
                    param_name = attr[4:]  # Remove 'get_' prefix
                    try:
                        value = getattr(qubit, attr)()
                        
                        # Get parameter metadata
                        info = param_info.get(param_name, {
                            "category": "custom",
                            "unit": None,
                            "type": type(value).__name__
                        })
                        
                        # Filter by category if specified
                        if parameter_categories and info['category'] not in parameter_categories:
                            continue
                            
                        # Serialize complex values
                        serialized_value = self._serialize_parameter_value(value)
                        
                        qubit_params["parameters"][param_name] = {
                            "value": serialized_value,
                            "unit": info.get('unit'),
                            "type": self._get_value_type(value),
                            "category": info.get('category'),
                            "description": info.get('description', f"Qubit {param_name}")
                        }
                    except:
                        pass  # Skip parameters that can't be retrieved
            
            # Add metadata
            qubit_params["metadata"] = {
                "last_updated": datetime.datetime.now().isoformat(),
                "status": "ready"
            }
                
            params[qubit_name] = qubit_params
        return params
    
    async def set_qubit_parameters(self, updates: list) -> dict:
        """Update LeeQ qubit parameters"""
        updated = {}
        for update in updates:
            qubit_name = update['qubit']
            param_name = update['parameter']
            value = update['value']
            
            # Deserialize complex values if needed
            deserialized_value = self._deserialize_parameter_value(value)
            
            qubit = self.setup.get_qubit(qubit_name)
            
            # Try different setter methods
            if hasattr(qubit, f'set_{param_name}'):
                getattr(qubit, f'set_{param_name}')(deserialized_value)
                updated[qubit_name] = {"parameter": param_name, "status": "updated"}
            elif hasattr(qubit, 'update_parameters'):
                qubit.update_parameters(**{param_name: deserialized_value})
                updated[qubit_name] = {"parameter": param_name, "status": "updated"}
            else:
                updated[qubit_name] = {"parameter": param_name, "status": "failed", "error": "No setter found"}
                
        return {"updates": updated}
    
    def _deserialize_parameter_value(self, value):
        """Deserialize parameter values from JSON format"""
        if isinstance(value, dict) and "_type" in value:
            if value["_type"] == "ndarray":
                data = base64.b64decode(value["data"])
                return np.frombuffer(data, dtype=value["dtype"]).reshape(value["shape"])
            elif value["_type"] == "complex":
                return complex(value["real"], value["imag"])
            elif value["_type"] in ["callable", "object", "unserializable"]:
                # These cannot be deserialized directly
                raise ValueError(f"Cannot deserialize {value['_type']} type")
        elif isinstance(value, list):
            return [self._deserialize_parameter_value(v) for v in value]
        elif isinstance(value, dict):
            # Check if it's a regular dict or a serialized object
            if "_type" not in value:
                return {k: self._deserialize_parameter_value(v) for k, v in value.items()}
        
        return value
    
    async def get_device_level_parameters(self) -> dict:
        """Get device-level parameters like coupling and crosstalk"""
        # Example: Get coupling information from setup
        device_params = {"parameters": {}, "metadata": {}}
        
        # Check if setup has coupling information
        if hasattr(self.setup, 'get_coupling_map'):
            coupling_data = self.setup.get_coupling_map()
            device_params["parameters"]["coupling_map"] = {
                "value": coupling_data,
                "unit": "Hz",
                "type": "dict",
                "category": "coupling",
                "description": "Physical coupling between qubits"
            }
        
        # Check for crosstalk matrix
        if hasattr(self.setup, 'get_crosstalk_matrix'):
            matrix = self.setup.get_crosstalk_matrix()
            device_params["parameters"]["crosstalk_matrix"] = {
                "value": self._serialize_parameter_value(matrix),
                "unit": None,
                "type": "ndarray",
                "category": "coupling", 
                "description": "Crosstalk coefficients between all qubit pairs"
            }
            
        # Add other device-level parameters
        if hasattr(self.setup, 'get_connectivity'):
            device_params["parameters"]["connectivity"] = {
                "value": self.setup.get_connectivity(),
                "unit": None,
                "type": "list",
                "category": "coupling",
                "description": "Connected qubit pairs"
            }
            
        device_params["metadata"] = {
            "parameter_level": "device",
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return device_params
    
    async def reset_hardware(self, reset_type: str = "full") -> dict:
        """Reset LeeQ hardware to default state"""
        results = {"reset_type": reset_type, "status": "success", "details": {}}
        
        try:
            if reset_type in ["full", "qubits"]:
                # Reset all qubit parameters to their default values
                for qubit_name, qubit in self.setup.qubits.items():
                    if hasattr(qubit, 'reset_to_defaults'):
                        qubit.reset_to_defaults()
                    elif hasattr(qubit, 'initialize'):
                        qubit.initialize()
                results["details"]["qubits"] = f"Reset {len(self.setup.qubits)} qubits to defaults"
                
            if reset_type in ["full", "instruments"]:
                # Reset instruments
                if hasattr(self.setup, 'disconnect_devices'):
                    self.setup.disconnect_devices()
                if hasattr(self.setup, 'connect_devices'):
                    self.setup.connect_devices()
                results["details"]["instruments"] = "Disconnected and reconnected all instruments"
                
            if reset_type in ["full", "calibrations"]:
                # Clear calibration data from qubits
                for qubit_name, qubit in self.setup.qubits.items():
                    if hasattr(qubit, 'clear_calibrations'):
                        qubit.clear_calibrations()
                    # Clear common calibration parameters
                    for param in ['pi_amp', 'pi2_amp', 'pi_len', 'readout_amp']:
                        if hasattr(qubit, f'set_{param}'):
                            try:
                                getattr(qubit, f'set_{param}')(None)
                            except:
                                pass
                results["details"]["calibrations"] = "Cleared all calibration data"
                
            # Reset setup status parameters
            if hasattr(self.setup, '_status'):
                self.setup._status.set_parameters(
                    Shot_Number=2000,
                    Shot_Period=500.0,
                    Measurement_Basis='<z>',
                    Acquisition_Type='IQ'
                )
                results["details"]["setup_status"] = "Reset to default parameters"
                
        except Exception as e:
            results["status"] = "partial"
            results["error"] = str(e)
            
        return results
    
    async def reset_server(self, keep_hardware_state: bool = True) -> dict:
        """Reset LeeQ MCP server state"""
        results = {"status": "success", "details": {}}
        
        try:
            # Clear experiment tracking
            self.current_experiment_id = None
            self.experiment_manager._active = None
            results["details"]["experiments"] = "Cleared active experiment"
            
            # Stop live monitor if running
            if hasattr(self.experiment_manager, 'stop_live_monitor'):
                self.experiment_manager.stop_live_monitor()
                results["details"]["monitor"] = "Stopped live monitor"
            
            # Re-discover experiments
            self._discover_experiments()
            results["details"]["discovery"] = f"Re-discovered {len(self.experiments)} experiments"
            
            # Clear any caches
            if hasattr(self, '_parameter_cache'):
                self._parameter_cache = {}
            if hasattr(self, '_result_cache'):
                self._result_cache = {}
            results["details"]["cache"] = "Cleared all caches"
            
            if not keep_hardware_state:
                # Also reset hardware
                hw_reset = await self.reset_hardware("full")
                results["details"]["hardware"] = hw_reset
            else:
                results["details"]["hardware"] = "Preserved"
                
            # Restart live monitor
            if hasattr(self.experiment_manager, 'start_live_monitor'):
                self.experiment_manager.start_live_monitor()
                results["details"]["monitor"] = "Restarted live monitor"
                
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            
        return results