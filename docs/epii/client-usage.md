# EPII gRPC Client Usage Guide

This guide demonstrates how to use the LeeQ EPII service from client applications using gRPC.

## Quick Start

### Python Client Setup
```python
import grpc
from leeq.epii.proto import epii_pb2, epii_pb2_grpc
import numpy as np

# Connect to EPII service
channel = grpc.insecure_channel('localhost:50051')
stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)

# Test connection
response = stub.Ping(epii_pb2.PingRequest())
print(f"Service online: {response.message}")
```

### Basic Experiment Execution
```python
# Prepare experiment request
request = epii_pb2.ExperimentRequest(
    experiment_name="rabi",
    parameters={
        "qubit": "q0",
        "amplitudes": serialize_array(np.linspace(0, 1, 21)),
        "num_shots": 1000
    }
)

# Run experiment
response = stub.RunExperiment(request)

# Deserialize results
data = deserialize_array(response.data)
fit_params = dict(response.fit_params)
print(f"Rabi frequency: {fit_params.get('frequency', 'N/A')}")
```

## Common Usage Patterns

### 1. Parameter Management

#### Get Parameters
```python
# Get single parameter
request = epii_pb2.ParameterRequest(name="q0.frequency")
response = stub.GetParameter(request)
frequency = response.value

# List all parameters
response = stub.ListParameters(epii_pb2.Empty())
for param in response.parameters:
    print(f"{param.name}: {param.value} ({param.type})")
```

#### Set Parameters
```python
# Set qubit frequency
request = epii_pb2.SetParameterRequest(
    name="q0.frequency",
    value="5.1e9"
)
response = stub.SetParameter(request)
if response.success:
    print("Parameter updated successfully")
```

### 2. Experiment Workflows

#### Rabi Experiment
```python
def run_rabi_experiment(qubit, amplitudes, num_shots=1000):
    """Run a Rabi experiment and return results."""
    request = epii_pb2.ExperimentRequest(
        experiment_name="rabi",
        parameters={
            "qubit": qubit,
            "amplitudes": serialize_array(np.array(amplitudes)),
            "num_shots": str(num_shots)
        }
    )
    
    try:
        response = stub.RunExperiment(request)
        return {
            "data": deserialize_array(response.data),
            "fit_params": dict(response.fit_params),
            "success": True
        }
    except grpc.RpcError as e:
        return {
            "error": str(e),
            "success": False
        }

# Usage
result = run_rabi_experiment("q0", np.linspace(0, 1, 21))
if result["success"]:
    print(f"Pi pulse amplitude: {result['fit_params'].get('pi_amplitude')}")
```

#### T1 Measurement
```python
def measure_t1(qubit, delays, num_shots=1000):
    """Measure T1 relaxation time."""
    request = epii_pb2.ExperimentRequest(
        experiment_name="t1",
        parameters={
            "qubit": qubit,
            "delays": serialize_array(np.array(delays)),
            "num_shots": str(num_shots)
        }
    )
    
    response = stub.RunExperiment(request)
    return {
        "delays": delays,
        "populations": deserialize_array(response.data),
        "t1": float(response.fit_params.get("t1", 0))
    }

# Usage
delays = np.logspace(-6, -3, 20)  # 1μs to 1ms
result = measure_t1("q0", delays)
print(f"T1 = {result['t1']*1e6:.1f} μs")
```

#### Ramsey Experiment
```python
def run_ramsey(qubit, delays, detuning=0, num_shots=1000):
    """Run Ramsey experiment to measure T2*."""
    request = epii_pb2.ExperimentRequest(
        experiment_name="ramsey",
        parameters={
            "qubit": qubit,
            "delays": serialize_array(np.array(delays)),
            "detuning": str(detuning),
            "num_shots": str(num_shots)
        }
    )
    
    response = stub.RunExperiment(request)
    return {
        "delays": delays,
        "populations": deserialize_array(response.data),
        "t2_star": float(response.fit_params.get("t2_star", 0)),
        "frequency": float(response.fit_params.get("frequency", 0))
    }
```

### 3. Advanced Patterns

#### Calibration Sequence
```python
def full_qubit_calibration(qubit):
    """Perform complete qubit calibration sequence."""
    results = {}
    
    # 1. Rabi calibration
    print("Running Rabi calibration...")
    rabi_result = run_rabi_experiment(qubit, np.linspace(0, 1, 51))
    results["rabi"] = rabi_result
    
    # 2. Update pi pulse amplitude
    if rabi_result["success"]:
        pi_amp = rabi_result["fit_params"].get("pi_amplitude")
        if pi_amp:
            set_request = epii_pb2.SetParameterRequest(
                name=f"{qubit}.pi_amplitude",
                value=str(pi_amp)
            )
            stub.SetParameter(set_request)
    
    # 3. T1 measurement
    print("Measuring T1...")
    delays = np.logspace(-6, -3, 30)
    results["t1"] = measure_t1(qubit, delays)
    
    # 4. Ramsey for T2*
    print("Measuring T2*...")
    delays = np.linspace(0, 50e-6, 51)
    results["ramsey"] = run_ramsey(qubit, delays)
    
    return results
```

#### Error Handling
```python
def robust_experiment_runner(experiment_func, max_retries=3):
    """Run experiment with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            return experiment_func()
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print(f"Experiment timeout, attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                print(f"Service unavailable, attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == max_retries - 1:
                    raise
            else:
                raise  # Don't retry other errors
```

### 4. Asynchronous Operations

#### Async Client
```python
import asyncio
import grpc.aio

async def async_experiment_client():
    """Example of asynchronous experiment execution."""
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
        
        # Run multiple experiments concurrently
        tasks = []
        for qubit in ["q0", "q1"]:
            request = epii_pb2.ExperimentRequest(
                experiment_name="t1",
                parameters={
                    "qubit": qubit,
                    "delays": serialize_array(np.logspace(-6, -3, 20)),
                    "num_shots": "1000"
                }
            )
            task = stub.RunExperiment(request)
            tasks.append(task)
        
        # Wait for all experiments to complete
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            t1 = float(result.fit_params.get("t1", 0))
            print(f"q{i} T1 = {t1*1e6:.1f} μs")

# Run async client
asyncio.run(async_experiment_client())
```

## Utility Functions

### Data Serialization
```python
def serialize_array(array):
    """Convert NumPy array to protobuf bytes."""
    return array.astype(np.float64).tobytes()

def deserialize_array(data, shape=None):
    """Convert protobuf bytes back to NumPy array."""
    array = np.frombuffer(data, dtype=np.float64)
    if shape:
        array = array.reshape(shape)
    return array

def serialize_complex_array(array):
    """Serialize complex array as interleaved real/imag."""
    complex_array = array.astype(np.complex128)
    real_imag = np.empty(complex_array.size * 2, dtype=np.float64)
    real_imag[0::2] = complex_array.real
    real_imag[1::2] = complex_array.imag
    return real_imag.tobytes()

def deserialize_complex_array(data):
    """Deserialize complex array from interleaved real/imag."""
    real_imag = np.frombuffer(data, dtype=np.float64)
    complex_array = real_imag[0::2] + 1j * real_imag[1::2]
    return complex_array
```

### Connection Management
```python
class EPIIClient:
    """Managed EPII client with connection pooling."""
    
    def __init__(self, address='localhost:50051', timeout=60):
        self.address = address
        self.timeout = timeout
        self.channel = None
        self.stub = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def connect(self):
        """Establish connection to EPII service."""
        self.channel = grpc.insecure_channel(self.address)
        self.stub = epii_pb2_grpc.ExperimentPlatformServiceStub(self.channel)
        
        # Test connection
        try:
            self.stub.Ping(epii_pb2.PingRequest(), timeout=5)
        except grpc.RpcError:
            raise ConnectionError(f"Cannot connect to EPII service at {self.address}")
    
    def disconnect(self):
        """Close connection."""
        if self.channel:
            self.channel.close()

# Usage
with EPIIClient() as client:
    result = client.stub.RunExperiment(request)
```

## Configuration Examples

### Client Configuration
```python
# config.py
EPII_CONFIG = {
    "address": "localhost:50051",
    "timeout": 300,  # 5 minutes
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "compression": grpc.Compression.Gzip
}

# client.py
def create_channel(config):
    """Create gRPC channel with configuration."""
    options = [
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.http2.min_ping_interval_without_data_ms', 300000)
    ]
    
    if config.get("compression"):
        return grpc.insecure_channel(
            config["address"], 
            options=options,
            compression=config["compression"]
        )
    else:
        return grpc.insecure_channel(config["address"], options=options)
```

## Best Practices

1. **Connection Management**: Use context managers or connection pooling
2. **Error Handling**: Always wrap gRPC calls in try-catch blocks
3. **Timeouts**: Set appropriate timeouts for long-running experiments
4. **Data Serialization**: Use provided utility functions for NumPy arrays
5. **Parameter Validation**: Validate parameters client-side when possible
6. **Logging**: Log all experiment requests and responses for debugging
7. **Concurrency**: Use async clients for parallel experiment execution
8. **Resource Cleanup**: Always close channels and clean up resources

## See Also

- [EPII Protocol Documentation](protocol.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Deployment Guide](deployment-guide.md)