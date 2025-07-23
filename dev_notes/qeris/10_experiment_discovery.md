# Experiment Discovery

## Overview

QERIS provides dynamic experiment discovery, allowing AI agents and users to explore available experiments without hardcoded mappings. This enables autonomous decision-making about which experiments to run.

## Discovery Tools

### list_experiments

Lists all available experiments with basic information:

```python
# List all experiments
all_experiments = await client.call_tool('list_experiments')

# List by category
calibration_exps = await client.call_tool('list_experiments', {
    'category': 'calibration'
})
```

Response format:
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
    }
  ]
}
```

### get_experiment_info

Gets detailed information about a specific experiment:

```python
info = await client.call_tool('get_experiment_info', {
    'experiment_name': 'RabiExperiment'
})
```

Response includes:
- Full documentation
- Parameter schemas with types and descriptions
- Return value information
- Usage examples

## Implementation Examples

### Discovering Experiments in Your Framework

```python
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
```

### Extracting Parameter Information

```python
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
    example_params = self._generate_example_params(experiment_name)
    
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
```

## AI Agent Usage

### Intelligent Experiment Selection

```python
async def intelligent_experiment_selection():
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Discover available experiments
    all_experiments = await client.call_tool('list_experiments')
    print(f"Found {len(all_experiments['experiments'])} experiments")
    
    # Find calibration experiments
    calibration_exps = await client.call_tool('list_experiments', {
        'category': 'calibration'
    })
    
    # Get detailed info about Rabi experiment
    rabi_info = await client.call_tool('get_experiment_info', {
        'experiment_name': 'RabiExperiment'
    })
    
    print(f"RabiExperiment documentation: {rabi_info['documentation']}")
    print(f"Required parameters: {list(rabi_info['parameters']['required'].keys())}")
    print(f"Example: {rabi_info['example']}")
    
    # AI can now make informed decision about which experiment to run
    # based on documentation and parameter requirements
    
    # Run the selected experiment
    exp_id = await client.call_tool('run_experiment', {
        'experiment_name': 'RabiExperiment',
        'qubits': ['q0'],
        'parameters': {
            'start': 0,
            'stop': 1, 
            'step': 0.05,
            'shots': 1024,
            'update': True
        }
    })
    
    return exp_id
```

### Adaptive Characterization

```python
async def adaptive_characterization():
    """AI agent that chooses experiments based on qubit state"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Get current qubit parameters
    params = await client.call_tool('get_qubit_parameters', {
        'qubits': ['q0'],
        'parameter_categories': ['coherence']
    })
    
    # List characterization experiments
    char_exps = await client.call_tool('list_experiments', {
        'category': 'characterization'
    })
    
    # Choose experiment based on what needs updating
    q0_params = params['q0']['parameters']
    
    if 't1' not in q0_params or q0_params['t1']['value'] is None:
        # T1 unmeasured, run T1 experiment
        exp_name = 'T1Experiment'
    elif q0_params['t1']['value'] < 20e-6:
        # T1 seems low, re-measure
        exp_name = 'T1Experiment'
    elif 't2' not in q0_params or q0_params['t2']['value'] is None:
        # T1 is good, measure T2
        exp_name = 'T2Experiment'
    else:
        # Both are good, run more advanced characterization
        exp_name = 'RandomizedBenchmarking'
    
    # Get experiment details
    exp_info = await client.call_tool('get_experiment_info', {
        'experiment_name': exp_name
    })
    
    print(f"Selected {exp_name}: {exp_info['description']}")
    print(f"Using parameters: {exp_info['example']['parameters']}")
    
    # Run the experiment
    exp_id = await client.call_tool('run_experiment', {
        'experiment_name': exp_name,
        'qubits': ['q0'],
        'parameters': exp_info['example']['parameters']
    })
    
    return exp_id
```

### Experiment Recommendation System

```python
async def recommend_next_experiment():
    """Recommend the next experiment based on system state"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Get all experiments
    experiments = await client.call_tool('list_experiments')
    
    # Get current system state
    device_info = await client.call_tool('get_device_info')
    qubit_params = await client.call_tool('get_qubit_parameters', {
        'qubits': ['all']
    })
    
    # Analyze what's missing or outdated
    recommendations = []
    
    # Check each qubit
    for qubit_name, qubit_data in qubit_params.items():
        if qubit_name == 'device':
            continue
            
        params = qubit_data['parameters']
        
        # Check if basic calibrations exist
        if 'frequency' not in params or params['frequency']['value'] is None:
            recommendations.append({
                'experiment': 'QubitSpectroscopy',
                'qubit': qubit_name,
                'reason': 'No frequency calibration found'
            })
        
        if 'pi_amp' not in params or params['pi_amp']['value'] is None:
            recommendations.append({
                'experiment': 'RabiExperiment',
                'qubit': qubit_name,
                'reason': 'No pi pulse calibration found'
            })
        
        # Check coherence times
        if 't1' in params:
            t1_age = calculate_age(params['t1'].get('last_updated'))
            if t1_age > 3600:  # More than 1 hour old
                recommendations.append({
                    'experiment': 'T1Experiment',
                    'qubit': qubit_name,
                    'reason': f'T1 measurement is {t1_age/3600:.1f} hours old'
                })
    
    # Sort by priority
    recommendations.sort(key=lambda x: experiment_priority(x['experiment']))
    
    return recommendations
```

## Experiment Categories

Standard categories help organize experiments:

### Calibration
- `QubitSpectroscopy` - Find qubit frequency
- `RabiExperiment` - Calibrate drive amplitude
- `DragCalibration` - Optimize DRAG parameters
- `ReadoutCalibration` - Optimize readout

### Characterization
- `T1Experiment` - Measure relaxation time
- `T2Experiment` - Measure dephasing time
- `T2EchoExperiment` - Measure echo coherence
- `RandomizedBenchmarking` - Gate fidelity

### Gates
- `SingleQubitGateTune` - Optimize single-qubit gates
- `TwoQubitGateTune` - Optimize two-qubit gates
- `CrosstalkCompensation` - Compensate for crosstalk

### Verification
- `ProcessTomography` - Full process characterization
- `StateTomography` - Quantum state reconstruction
- `GateBenchmarking` - Comprehensive gate testing

## Custom Experiment Discovery

For custom experiments, provide rich metadata:

```python
class CustomExperiment:
    """
    Measures custom quantum property.
    
    This experiment performs a specialized measurement sequence
    to characterize a specific quantum behavior.
    
    Parameters:
        custom_param1 (float): Description of parameter 1
        custom_param2 (list): Description of parameter 2
        
    Returns:
        dict: Results containing measured values and fit parameters
    """
    
    category = "custom"
    supported_qubits = ["single"]
    
    def __init__(self, qubit, custom_param1, custom_param2, shots=1024):
        self.qubit = qubit
        self.custom_param1 = custom_param1
        self.custom_param2 = custom_param2
        self.shots = shots
```

## Best Practices

1. **Provide comprehensive documentation** - Include parameter descriptions and examples
2. **Use standard categories** - Helps with organization and discovery
3. **Include parameter validation** - Document valid ranges and types
4. **Generate realistic examples** - Help users understand typical usage
5. **Version experiments** - Track changes in experiment implementations
6. **Support both single and multi-qubit** - When applicable
7. **Return structured results** - Use consistent format for results