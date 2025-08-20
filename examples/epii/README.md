# EPII Client Examples

This directory contains example client applications demonstrating how to use the LeeQ EPII service.

## Prerequisites

1. LeeQ EPII service running on localhost:50051 (or modify the address in examples)
2. Python with grpcio installed: `pip install grpcio`
3. NumPy installed: `pip install numpy`

## Examples

### simple_client.py
Basic client demonstrating fundamental EPII operations:
- Connection testing with Ping
- Service capability discovery
- Parameter listing
- Simple experiment execution (Rabi)

**Usage:**
```bash
python simple_client.py
```

### calibration_client.py
Advanced client showing a complete qubit calibration workflow:
- Rabi calibration for pi pulse amplitude
- T1 measurement (relaxation time)
- Ramsey experiment for T2* (dephasing time)
- Echo experiment for T2 (coherence time)
- Randomized benchmarking for gate fidelity
- Complete calibration sequence with results summary

**Usage:**
```bash
python calibration_client.py
```

## Running the Examples

1. **Start EPII service:**
   ```bash
   # Using systemd
   sudo systemctl start leeq-epii@simulation_2q
   
   # Or manually
   python -m leeq.epii.daemon --config configs/epii/simulation_2q.json
   ```

2. **Run examples:**
   ```bash
   cd examples/epii
   python simple_client.py
   python calibration_client.py
   ```

## Expected Output

### simple_client.py
```
Testing connection...
✓ Service online: LeeQ EPII Service v1.0

Getting capabilities...
✓ Available experiments: rabi, t1, ramsey, echo, drag, randomized_benchmarking

Listing parameters...
✓ Found 15 parameters
  - q0.frequency: 5000000000.0 (float)
  - q0.anharmonicity: -250000000.0 (float)
  - q0.t1: 5e-05 (float)
  - q0.t2: 3e-05 (float)
  - q0.pi_amplitude: 0.5 (float)

Running Rabi experiment...
✓ Experiment completed
  - Data points: 11
  - Population range: 0.023 - 0.956
  - Fit parameters:
    pi_amplitude: 0.4987
    frequency: 12.56e6

✓ All tests passed!
```

### calibration_client.py
```
Connecting to EPII service...
✓ Connected: LeeQ EPII Service v1.0

==================================================
Full calibration for q0
==================================================

Running Rabi calibration for q0...
  ✓ Pi amplitude: 0.4987

Measuring T1 for q0...
  ✓ T1: 50.2 μs

Measuring T2* for q0...
  ✓ T2*: 29.8 μs
  ✓ Detuning: 500.1 kHz

Measuring T2 echo for q0...
  ✓ T2 echo: 45.6 μs

Running randomized benchmarking for q0...
  ✓ Gate fidelity: 99.85%

==================================================
Calibration Summary for q0
==================================================
Pi amplitude: 0.4987
T1: 50.2 μs
T2*: 29.8 μs
T2: 45.6 μs
T2/T1 ratio: 0.91
Gate fidelity: 99.85%

============================================================
OVERALL CALIBRATION SUMMARY
============================================================

q0:
  pi_amplitude: 0.4987
  t1: 50.2 μs
  t2_star: 29.8 μs
  t2: 45.6 μs
  fidelity: 99.85%

✓ Calibration complete!
```

## Customization

### Changing Server Address
Modify the client initialization:
```python
client = EPIIClient('your-server:50051')
```

### Adding New Experiments
Extend the client classes with new experiment methods:
```python
def run_custom_experiment(self, qubit, custom_params):
    return self.run_experiment("custom_experiment", {
        "qubit": qubit,
        "custom_param": custom_params
    })
```

### Error Handling
Both examples include basic error handling. For production use, consider:
- Retry logic for transient failures
- Timeout handling for long experiments
- Logging for debugging
- Graceful degradation when experiments fail

## Troubleshooting

### Connection Issues
- Verify EPII service is running: `systemctl status leeq-epii@simulation_2q`
- Check port accessibility: `netstat -tlnp | grep :50051`
- Test with grpcurl: `grpcurl -plaintext localhost:50051 list`

### Import Errors
```bash
# Install required packages
pip install grpcio numpy

# If using virtual environment
source /opt/leeq/venv/bin/activate
pip install grpcio numpy
```

### Experiment Failures
- Check EPII service logs: `journalctl -u leeq-epii@simulation_2q -f`
- Verify configuration is valid
- Ensure sufficient timeout for long experiments

## See Also

- [Client Usage Guide](../../docs/epii/client-usage.md)
- [Troubleshooting Guide](../../docs/epii/troubleshooting.md)
- [Deployment Guide](../../docs/epii/deployment-guide.md)