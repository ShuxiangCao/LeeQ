# Parameter Management

## Overview

QERIS provides a flexible, backend-agnostic parameter management system that supports any type of quantum system. Parameters can represent simple values or complex data structures.

## Parameter Categories

Standard categories that help organize parameters:

- **`hamiltonian`** - Fundamental system parameters (frequency, anharmonicity, etc.)
- **`coherence`** - Decoherence times (T1, T2, T2*)
- **`control`** - Control parameters (pulse shapes, amplitudes, durations)
- **`readout`** - Measurement parameters (readout frequency, kernels)
- **`coupling`** - Inter-qubit coupling parameters
- **`custom`** - Backend-specific parameters

## Parameter Types

### Simple Types
- `float` - Numeric values (frequencies, times, amplitudes)
- `int` - Integer values (shots, repetitions)
- `str` - String values (pulse shapes, modes)
- `bool` - Boolean flags (enable/disable features)

### Complex Types
- `list` - Arrays of values
- `dict` - Structured data
- `ndarray` - NumPy arrays (serialized as base64)
- `complex` - Complex numbers
- `object` - Custom objects (serialized as strings)

## Parameter Schema Discovery

### Getting Available Parameters

```python
async def discover_backend_capabilities():
    """Discover what parameters and experiments are available on any backend"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Get device info
    device = await client.call_tool('get_device_info')
    print(f"Connected to {device['backend_type']} backend: {device['device_name']}")
    
    # Discover parameter schema
    schema = await client.call_tool('get_parameter_schema')
    categories = schema['parameter_schema']['categories']
    available_params = schema['parameter_schema']['available_parameters']
    
    print(f"\nAvailable parameter categories: {categories}")
    print(f"Total parameters available: {len(available_params)}")
    
    # Get all parameters for one qubit to see what's available
    sample_params = await client.call_tool('get_qubit_parameters', {
        'qubits': ['q0']
    })
    
    # Organize parameters by category
    params_by_category = {}
    for param_name, param_data in sample_params['q0']['parameters'].items():
        category = param_data['category']
        if category not in params_by_category:
            params_by_category[category] = []
        params_by_category[category].append({
            'name': param_name,
            'value': param_data['value'],
            'unit': param_data['unit']
        })
    
    # Display organized parameters
    for category, params in params_by_category.items():
        print(f"\n{category.upper()} parameters:")
        for p in params:
            unit_str = f" {p['unit']}" if p['unit'] else ""
            print(f"  - {p['name']}: {p['value']}{unit_str}")
```

## Working with Parameters

### Getting Parameters

```python
# Get all parameters for specific qubits
params = await client.call_tool('get_qubit_parameters', {
    'qubits': ['q0', 'q1', 'q2']
})

# Get only specific categories
coherence_params = await client.call_tool('get_qubit_parameters', {
    'qubits': ['q0'],
    'parameter_categories': ['coherence']
})

# Get device-level parameters
device_params = await client.call_tool('get_qubit_parameters', {
    'qubits': ['device']
})

# Get all qubit parameters
all_params = await client.call_tool('get_qubit_parameters', {
    'qubits': ['all']
})
```

### Setting Parameters

```python
# Update single parameter
await client.call_tool('set_qubit_parameters', {
    'updates': [
        {'qubit': 'q0', 'parameter': 'frequency', 'value': 4.5e9}
    ]
})

# Update multiple parameters
await client.call_tool('set_qubit_parameters', {
    'updates': [
        {'qubit': 'q0', 'parameter': 'frequency', 'value': 4.5e9},
        {'qubit': 'q0', 'parameter': 'pi_amp', 'value': 0.248},
        {'qubit': 'q1', 'parameter': 'frequency', 'value': 4.6e9}
    ]
})
```

## Complex Parameter Examples

### Dictionary Parameters

```python
# Set pulse shape configuration
await client.call_tool('set_qubit_parameters', {
    'updates': [{
        'qubit': 'q0',
        'parameter': 'pulse_shape',
        'value': {
            'type': 'gaussian',
            'sigma': 10e-9,
            'truncation': 4,
            'drag_coefficient': 0.5
        }
    }]
})

# Set error model
await client.call_tool('set_qubit_parameters', {
    'updates': [{
        'qubit': 'q0',
        'parameter': 'error_model',
        'value': {
            'T1': {'distribution': 'exponential', 'rate': 1/50e-6},
            'T2': {'distribution': 'gaussian', 'mean': 70e-6, 'std': 5e-6},
            'gate_errors': {
                'X': 0.001,
                'Y': 0.001,
                'Z': 0.0005
            }
        }
    }]
})
```

### Array Parameters

```python
# Set readout confusion matrix
await client.call_tool('set_qubit_parameters', {
    'updates': [{
        'qubit': 'q0',
        'parameter': 'readout_kernel',
        'value': [[0.98, 0.02], [0.03, 0.97]]  # 2x2 matrix
    }]
})

# Set calibration data array
await client.call_tool('set_qubit_parameters', {
    'updates': [{
        'qubit': 'q0',
        'parameter': 'calibration_points',
        'value': [0.1, 0.2, 0.3, 0.4, 0.5]
    }]
})
```

### NumPy Array Parameters

```python
import numpy as np
import base64

# Create a pulse envelope as numpy array
t = np.linspace(0, 100e-9, 1000)
gaussian_envelope = np.exp(-(t - 50e-9)**2 / (2 * (10e-9)**2))

# Set numpy array parameter
await client.call_tool('set_qubit_parameters', {
    'updates': [{
        'qubit': 'q0',
        'parameter': 'pulse_envelope',
        'value': {
            '_type': 'ndarray',
            'data': base64.b64encode(gaussian_envelope.tobytes()).decode('utf-8'),
            'shape': gaussian_envelope.shape,
            'dtype': str(gaussian_envelope.dtype)
        }
    }]
})

# Retrieve and use the array
params = await client.call_tool('get_qubit_parameters', {
    'qubits': ['q0'],
    'parameter_categories': ['control']
})

if 'pulse_envelope' in params['q0']['parameters']:
    envelope_data = params['q0']['parameters']['pulse_envelope']['value']
    # Reconstruct numpy array from serialized data
    if envelope_data['_type'] == 'ndarray':
        data = base64.b64decode(envelope_data['data'])
        envelope = np.frombuffer(data, dtype=envelope_data['dtype']).reshape(envelope_data['shape'])
        print(f"Pulse envelope shape: {envelope.shape}")
        print(f"Peak amplitude: {np.max(envelope)}")
```

## Device-Level Parameters

### Coupling Information

```python
# Different backends represent coupling differently
device_params = await client.call_tool('get_qubit_parameters', {
    'qubits': ['device']
})

device_info = device_params['device']['parameters']

# Example 1: Superconducting qubits with capacitive coupling
if 'capacitive_coupling' in device_info:
    cap_coupling = device_info['capacitive_coupling']['value']
    print(f"Capacitive coupling unit: {device_info['capacitive_coupling']['unit']}")

# Example 2: Generic coupling strength matrix
elif 'coupling_strengths' in device_info:
    coupling = device_info['coupling_strengths']['value']
    for pair, strength in coupling.items():
        print(f"{pair}: {strength} Hz")

# Example 3: Crosstalk matrix
elif 'crosstalk_matrix' in device_info:
    crosstalk = device_info['crosstalk_matrix']['value']
    if crosstalk['_type'] == 'ndarray':
        # Deserialize the matrix
        data = base64.b64decode(crosstalk['data'])
        matrix = np.frombuffer(data, dtype=crosstalk['dtype']).reshape(crosstalk['shape'])
        print(f"Crosstalk matrix shape: {matrix.shape}")
```

## Parameter Metadata

Each parameter includes metadata for proper interpretation:

```json
{
    "frequency": {
        "value": 4.562e9,
        "unit": "Hz",
        "type": "float",
        "category": "hamiltonian",
        "description": "Qubit transition frequency",
        "settable": true,
        "range": [4.0e9, 5.0e9]
    }
}
```

## AI Agent Parameter Management

```python
async def quantum_device_manager():
    client = Client()
    await client.connect('http://localhost:8765')
    
    # First, get the parameter schema to understand this backend
    schema = await client.call_tool('get_parameter_schema')
    print(f"Backend type: {schema['parameter_schema']['backend_type']}")
    print(f"Available parameters: {schema['parameter_schema']['available_parameters']}")
    
    # Get specific parameter categories
    params = await client.call_tool('get_qubit_parameters', {
        'qubits': ['q0', 'q1', 'q2'],
        'parameter_categories': ['coherence', 'control']
    })
    
    # Analyze qubit quality based on available coherence parameters
    high_quality_qubits = []
    for qubit, data in params.items():
        qubit_params = data['parameters']
        # Check T1 if available
        if 't1' in qubit_params and qubit_params['t1']['value'] > 50e-6:
            # Check T2 if available
            if 't2' in qubit_params and qubit_params['t2']['value'] > 70e-6:
                high_quality_qubits.append(qubit)
    
    print(f"High quality qubits: {high_quality_qubits}")
    
    # Update specific qubit parameters based on measurements
    # Check what control parameters are available
    control_params = [p for p in schema['parameter_schema']['available_parameters'] 
                     if p in ['pi_amp', 'pi_pulse_amplitude', 'drive_amplitude']]
    
    if control_params:
        await client.call_tool('set_qubit_parameters', {
            'updates': [
                {'qubit': 'q0', 'parameter': control_params[0], 'value': 0.248},
                {'qubit': 'q1', 'parameter': control_params[0], 'value': 0.236}
            ]
        })
```

## Best Practices

1. **Always check parameter schema first** - Different backends have different parameters
2. **Use categories to filter** - Reduces data transfer and processing
3. **Handle read-only parameters** - Check `settable` flag before updating
4. **Validate complex values** - Ensure proper structure for dicts and arrays
5. **Use appropriate serialization** - Base64 for binary data, JSON for structured data
6. **Monitor parameter drift** - Subscribe to `qeris://qubit_parameters` resource
7. **Document custom parameters** - Include clear descriptions and units