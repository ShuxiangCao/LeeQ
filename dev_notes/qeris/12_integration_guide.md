# Integration Guide

## For Quantum Framework Developers

### Step 1: Implement the QERISAdapter

Create an adapter for your quantum framework:

```python
from qeris.server import QERISAdapter
import your_quantum_framework as qf

class YourFrameworkAdapter(QERISAdapter):
    def __init__(self):
        self.backend = qf.get_backend()
        self.experiment_registry = qf.experiments.registry
        self.device_manager = qf.devices.manager
        
    # Implement all required methods...
```

### Step 2: Map Your Experiments

Map your framework's experiments to QERIS standard:

```python
async def list_experiments(self, category=None):
    experiments = []
    
    for name, exp_class in self.experiment_registry.items():
        # Extract metadata from your experiment class
        exp_info = {
            'name': name,
            'category': self._categorize_experiment(exp_class),
            'description': exp_class.__doc__.split('\n')[0] if exp_class.__doc__ else '',
            'supported_qubits': self._get_qubit_support(exp_class),
            'required_parameters': self._get_required_params(exp_class),
            'optional_parameters': self._get_optional_params(exp_class)
        }
        
        if category is None or exp_info['category'] == category:
            experiments.append(exp_info)
    
    return {'experiments': experiments}

def _categorize_experiment(self, exp_class):
    # Map your experiment types to standard categories
    if hasattr(exp_class, 'category'):
        return exp_class.category
    elif 'calibration' in exp_class.__name__.lower():
        return 'calibration'
    elif 'tomography' in exp_class.__name__.lower():
        return 'verification'
    # ... more mappings
    return 'custom'
```

### Step 3: Handle Parameter Translation

Convert between your framework's parameters and QERIS format:

```python
async def get_qubit_parameters(self, qubits, parameter_categories=None):
    result = {}
    
    for qubit_name in qubits:
        if qubit_name == "all":
            # Get all qubits
            all_qubits = self.device_manager.get_all_qubits()
            return await self.get_qubit_parameters(all_qubits, parameter_categories)
        
        elif qubit_name == "device":
            # Get device-level parameters
            result["device"] = self._get_device_parameters(parameter_categories)
        
        else:
            # Get specific qubit parameters
            qubit_obj = self.device_manager.get_qubit(qubit_name)
            result[qubit_name] = self._translate_qubit_params(qubit_obj, parameter_categories)
    
    return result

def _translate_qubit_params(self, qubit_obj, categories):
    params = {"parameters": {}, "metadata": {}}
    
    # Map your framework's parameter names to QERIS standard
    param_mapping = {
        'transition_freq': ('frequency', 'hamiltonian', 'Hz'),
        'relax_time': ('t1', 'coherence', 's'),
        'dephase_time': ('t2', 'coherence', 's'),
        'pi_pulse_amp': ('pi_amp', 'control', 'a.u.'),
        # ... more mappings
    }
    
    for internal_name, (standard_name, category, unit) in param_mapping.items():
        if categories and category not in categories:
            continue
            
        if hasattr(qubit_obj, f'get_{internal_name}'):
            value = getattr(qubit_obj, f'get_{internal_name}')()
            params["parameters"][standard_name] = {
                "value": value,
                "unit": unit,
                "type": type(value).__name__,
                "category": category,
                "description": f"Qubit {standard_name}"
            }
    
    params["metadata"] = {
        "last_updated": qubit_obj.last_calibration_time.isoformat(),
        "status": "ready" if qubit_obj.is_ready() else "offline"
    }
    
    return params
```

### Step 4: Start the Server

```python
# start_qeris_server.py
import asyncio
from your_framework_adapter import YourFrameworkAdapter
from qeris.server import QERISServer

async def main():
    # Initialize your framework
    import your_quantum_framework as qf
    qf.initialize()
    
    # Create adapter
    adapter = YourFrameworkAdapter()
    
    # Create and start server
    server = QERISServer(adapter, port=8765)
    
    print("Starting QERIS server for YourFramework...")
    print(f"Server running at http://localhost:8765")
    print("MCP tools available at /tools/")
    print("MCP resources available at /resources/")
    
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## For Existing LeeQ Users

### Enable QERIS in LeeQ

```python
# In your LeeQ setup script
from leeq.experiments.experiments import ExperimentManager
from leeq.remote.qeris_adapter import LeeQAdapter
from qeris.server import QERISServer

# Start QERIS server alongside LeeQ
def start_leeq_with_qeris(setup, port=8765):
    # Start normal LeeQ monitoring
    ExperimentManager().start_live_monitor()
    
    # Start QERIS server
    adapter = LeeQAdapter(ExperimentManager(), setup)
    server = QERISServer(adapter, port)
    
    # Run in background thread
    import threading
    import asyncio
    def run_server():
        asyncio.run(server.start())
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    print(f"QERIS server started on port {port}")
    print("Connect with any MCP client or visit http://localhost:8765")

# Usage in your experiment script
from leeq.setups import built_in
setup = built_in.setup_from_yaml('config.yaml')
start_leeq_with_qeris(setup)
```

### Migrating from LeeQ Live Monitor

If you're currently using LeeQ's built-in monitoring:

```python
# Before (LeeQ specific)
from leeq.experiments.experiments import ExperimentManager
ExperimentManager().start_live_monitor()
# Visit http://localhost:8050

# After (QERIS universal)
from leeq.remote.qeris_adapter import start_qeris_server
start_qeris_server(setup, port=8765)
# Connect with any MCP client
# Visit http://localhost:8765 for web interface
```

## For AI/ML Researchers

### Using QERIS with Your AI Agent

```python
# ai_agent.py
from mcp import Client
import asyncio

class QuantumExperimentAgent:
    def __init__(self, qeris_url='http://localhost:8765'):
        self.client = Client()
        self.qeris_url = qeris_url
        
    async def connect(self):
        await self.client.connect(self.qeris_url)
        
        # Discover capabilities
        self.device_info = await self.client.call_tool('get_device_info')
        self.experiments = await self.client.call_tool('list_experiments')
        self.parameter_schema = await self.client.call_tool('get_parameter_schema')
        
        print(f"Connected to {self.device_info['device_name']}")
        print(f"Found {len(self.experiments['experiments'])} experiments")
        
    async def optimize_qubit(self, qubit_name):
        """Example: Optimize a qubit through calibration"""
        # Get current parameters
        params = await self.client.call_tool('get_qubit_parameters', {
            'qubits': [qubit_name]
        })
        
        # Run calibration sequence
        calibrations = [
            'QubitSpectroscopy',
            'RabiExperiment',
            'T1Experiment',
            'T2Experiment'
        ]
        
        for exp_name in calibrations:
            # Get experiment details
            exp_info = await self.client.call_tool('get_experiment_info', {
                'experiment_name': exp_name
            })
            
            # Run with appropriate parameters
            exp_id = await self.client.call_tool('run_experiment', {
                'experiment_name': exp_name,
                'qubits': [qubit_name],
                'parameters': exp_info['example']['parameters']
            })
            
            # Monitor until complete
            await self._wait_for_completion(exp_id)
            
            # Get and process results
            results = await self.client.call_tool('get_results', {
                'experiment_id': exp_id,
                'format': 'processed'
            })
            
            # AI logic to analyze results and decide next steps
            self._analyze_results(exp_name, results)
```

### Connecting Multiple Quantum Systems

```python
class MultiSystemController:
    """Control multiple quantum systems through QERIS"""
    
    def __init__(self):
        self.systems = {}
        
    async def add_system(self, name, url):
        client = Client()
        await client.connect(url)
        
        # Get system info
        device_info = await client.call_tool('get_device_info')
        
        self.systems[name] = {
            'client': client,
            'device_info': device_info,
            'url': url
        }
        
        print(f"Added system '{name}': {device_info['device_name']}")
        
    async def run_parallel_experiments(self, experiment_name, parameters):
        """Run same experiment on all systems"""
        tasks = []
        
        for system_name, system in self.systems.items():
            client = system['client']
            
            # Start experiment
            task = asyncio.create_task(
                client.call_tool('run_experiment', {
                    'experiment_name': experiment_name,
                    'qubits': ['q0'],  # Use first qubit on each system
                    'parameters': parameters
                })
            )
            tasks.append((system_name, task))
        
        # Wait for all to start
        results = {}
        for system_name, task in tasks:
            exp_id = await task
            results[system_name] = exp_id
            
        return results

# Usage
controller = MultiSystemController()
await controller.add_system('ibm_system', 'http://ibm-qeris:8765')
await controller.add_system('google_system', 'http://google-qeris:8765')
await controller.add_system('rigetti_system', 'http://rigetti-qeris:8765')

# Run T1 experiment on all systems
results = await controller.run_parallel_experiments(
    'T1Experiment',
    {'delays': list(range(0, 200, 10))}
)
```

## Testing Your Integration

### QERIS Compliance Test

```python
# test_qeris_compliance.py
import pytest
from qeris.tests import QERISComplianceTest

@pytest.mark.asyncio
async def test_adapter_compliance():
    """Test that your adapter meets QERIS requirements"""
    from your_framework_adapter import YourFrameworkAdapter
    
    adapter = YourFrameworkAdapter()
    compliance = QERISComplianceTest(adapter)
    
    # Test all required methods
    await compliance.test_experiment_discovery()
    await compliance.test_experiment_execution()
    await compliance.test_parameter_management()
    await compliance.test_reset_commands()
    await compliance.test_real_time_data()
    
    print("All compliance tests passed!")
```

### Integration Test

```python
# test_integration.py
import asyncio
from mcp import Client

async def test_full_workflow():
    """Test complete workflow through QERIS"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # 1. Discover experiments
    experiments = await client.call_tool('list_experiments')
    assert len(experiments['experiments']) > 0
    
    # 2. Get experiment info
    exp_name = experiments['experiments'][0]['name']
    info = await client.call_tool('get_experiment_info', {
        'experiment_name': exp_name
    })
    assert 'parameters' in info
    
    # 3. Run experiment
    exp_id = await client.call_tool('run_experiment', {
        'experiment_name': exp_name,
        'qubits': ['q0'],
        'parameters': info['example']['parameters']
    })
    assert exp_id is not None
    
    # 4. Monitor status
    status = await client.call_tool('get_status')
    assert status['experiment']['id'] == exp_id
    
    print("Integration test passed!")

asyncio.run(test_full_workflow())
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app

# Install your framework and QERIS
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy adapter implementation
COPY your_framework_adapter.py .
COPY start_server.py .

# Expose QERIS port
EXPOSE 8765

# Start server
CMD ["python", "start_server.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  qeris:
    build: .
    ports:
      - "8765:8765"
    environment:
      - QUANTUM_BACKEND=simulator
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    restart: unless-stopped
    
  web-monitor:
    image: qeris/web-client:latest
    ports:
      - "3000:3000"
    environment:
      - QERIS_SERVER=http://qeris:8765
    depends_on:
      - qeris
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```python
   # Check if port is available
   import socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   result = sock.connect_ex(('localhost', 8765))
   if result == 0:
       print("Port 8765 is already in use")
   ```

2. **Adapter method not implemented**
   ```python
   # Use base class with NotImplementedError
   class PartialAdapter(QERISAdapter):
       async def list_experiments(self, category=None):
           # Implement gradually
           return {"experiments": []}
   ```

3. **Serialization errors**
   ```python
   # Add custom serialization
   def serialize_custom_type(self, obj):
       if isinstance(obj, MyCustomType):
           return {"_type": "custom", "data": str(obj)}
       return obj
   ```

## Next Steps

1. Implement the adapter for your framework
2. Test with the compliance suite
3. Deploy your QERIS server
4. Share with the community!

For more examples and support, visit the QERIS repository.