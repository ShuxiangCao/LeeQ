# LeeQ EPII Documentation

EPII (Experiment Platform Intelligence Interface) is a gRPC service that provides standardized access to LeeQ quantum experiment capabilities. This documentation covers deployment, usage, and troubleshooting of the EPII service.

## Quick Start

### Installation
```bash
# Run the installation script
sudo ./scripts/install-epii.sh --config simulation_2q

# Or manually deploy following the deployment guide
```

### Basic Usage
```python
import grpc
from leeq.epii.proto import epii_pb2, epii_pb2_grpc

# Connect and run experiment
channel = grpc.insecure_channel('localhost:50051')
stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
response = stub.Ping(epii_pb2.PingRequest())
print(f"Service: {response.message}")
```

## Documentation

### Deployment & Administration
- **[Deployment Guide](deployment-guide.md)** - Complete deployment instructions for production environments
- **[Operations Guide](../epii-operations-guide.md)** - Comprehensive production operations and maintenance guide
- **[Deployment Checklist](../epii-deployment-checklist.md)** - Step-by-step production deployment validation
- **[Configuration Examples](../configs/epii/)** - Ready-to-use configuration files for different setups
- **[Systemd Service](../scripts/systemd/)** - Service templates and installation scripts

### Development & Usage
- **[Client Usage Guide](client-usage.md)** - Comprehensive guide for gRPC client development
- **[Example Clients](../examples/epii/)** - Working example applications and usage patterns
- **[API Reference](protocol.md)** - Complete gRPC protocol documentation (if available)

### Operations & Support
- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
- **[Monitoring & Logging](deployment-guide.md#monitoring-and-logging)** - Service monitoring setup
- **[Security Considerations](deployment-guide.md#security-considerations)** - Production security guidelines

## Configuration Templates

### Available Configurations
- **`simulation_2q.json`** - 2-qubit simulation setup for development and testing
- **`hardware_lab1.json`** - Hardware setup template for production quantum devices  
- **`minimal.json`** - Minimal configuration for basic testing

### Configuration Structure
```json
{
  "setup_type": "simulation|hardware",
  "setup_class": "HighLevelSimulationSetup|QubicLBNLSetup",
  "config": { /* Setup-specific configuration */ },
  "grpc": {
    "port": 50051,
    "max_workers": 10,
    "max_message_length": 104857600
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/leeq/epii.log"
  },
  "experiment_timeout": 300,
  "parameter_validation": true
}
```

## Supported Experiments

EPII supports the following quantum experiments:

### Single-Qubit Experiments
- **Rabi** - Calibrate pulse amplitudes and measure Rabi frequency
- **T1** - Measure qubit relaxation time
- **Ramsey** - Measure dephasing time (T2*) and frequency detuning
- **Echo** - Measure coherence time (T2) with echo sequences

### Advanced Experiments  
- **DRAG** - Optimize pulse shapes to reduce leakage
- **Randomized Benchmarking** - Measure average gate fidelity

### Multi-Qubit Experiments (Future)
- Two-qubit gate calibration
- Process tomography
- Quantum error correction protocols

## Service Management

### Systemd Commands
```bash
# Start service
sudo systemctl start leeq-epii@<config>

# Check status  
sudo systemctl status leeq-epii@<config>

# View logs
sudo journalctl -u leeq-epii@<config> -f

# Multiple configurations
sudo systemctl start leeq-epii@simulation_2q
sudo systemctl start leeq-epii@hardware_lab1
```

### Health Checks
```bash
# Test gRPC endpoint
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext localhost:50051 ExperimentPlatformService/Ping

# Check process status
ps aux | grep leeq-epii
```

## Development Examples

### Simple Experiment
```python
# Run a Rabi experiment
request = epii_pb2.ExperimentRequest(
    experiment_name="rabi",
    parameters={
        "qubit": "q0",
        "amplitudes": serialize_array(np.linspace(0, 1, 21)),
        "num_shots": "1000"
    }
)
response = stub.RunExperiment(request)
data = deserialize_array(response.data)
```

### Parameter Management
```python
# Get parameter
request = epii_pb2.ParameterRequest(name="q0.frequency")
response = stub.GetParameter(request)

# Set parameter
request = epii_pb2.SetParameterRequest(name="q0.frequency", value="5.1e9")
response = stub.SetParameter(request)
```

## Architecture Overview

```
┌─────────────────┐    gRPC     ┌─────────────────┐
│   Client Apps   │ ◄────────► │  EPII Service   │
│                 │             │                 │
│ - Lab Software  │             │ - Experiment    │
│ - Jupyter       │             │   Router        │
│ - Automation    │             │ - Parameter     │
│ - External APIs │             │   Manager       │
└─────────────────┘             │ - Serialization │
                                └─────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   LeeQ Core     │
                                │                 │
                                │ - Experiments   │
                                │ - Setups        │
                                │ - Hardware      │
                                │ - Simulation    │
                                └─────────────────┘
```

## Support & Community

### Getting Help
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/ShuxiangCao/LeeQ/issues)
- **Documentation**: Check the troubleshooting guide for common problems
- **Support**: Contact the development team for enterprise support

### Contributing
- Follow the contribution guidelines in the main LeeQ repository
- Add tests for new experiment types and features
- Update documentation for any API changes

### License
LeeQ EPII is part of the LeeQ project and follows the same licensing terms.

## Version Information

- **EPII Protocol Version**: 1.0
- **Supported LeeQ Version**: Latest (check requirements.txt)
- **Python Version**: 3.8+
- **gRPC Version**: See requirements.txt

## Migration & Upgrades

### From Direct LeeQ Usage
If migrating from direct LeeQ experiment scripts:
1. Review the [Client Usage Guide](client-usage.md) for equivalent patterns
2. Use the example clients as templates
3. Consider the standardized parameter naming in EPII

### Service Upgrades
Follow the upgrade procedures in the [Deployment Guide](deployment-guide.md#upgrades) for safe service updates.