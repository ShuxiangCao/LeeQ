# EPII Troubleshooting Guide

This guide helps diagnose and resolve common issues with the LeeQ EPII service.

## Quick Diagnostics

### Service Status Check
```bash
# Check if service is running
sudo systemctl status leeq-epii@<config>

# Check service logs
sudo journalctl -u leeq-epii@<config> --no-pager -l

# Check process information
ps aux | grep leeq-epii
```

### Network Connectivity
```bash
# Test if gRPC port is listening
netstat -tlnp | grep :50051

# Test gRPC endpoint (requires grpcurl)
grpcurl -plaintext localhost:50051 list

# Basic ping test
grpcurl -plaintext localhost:50051 ExperimentPlatformService/Ping
```

### Configuration Validation
```bash
# Validate configuration file
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq/configs/epii/<config>.json --validate

# Check JSON syntax
python -m json.tool /etc/leeq/configs/epii/<config>.json
```

## Common Issues

### 1. Service Won't Start

#### Symptoms
- `systemctl start` fails
- Service immediately exits
- "Failed to start" in system logs

#### Diagnostics
```bash
# Check detailed service logs
sudo journalctl -u leeq-epii@<config> -f

# Try manual start for more detailed errors
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq/configs/epii/<config>.json --verbose
```

#### Common Causes & Solutions

**Configuration File Issues**
```bash
# Symptom: JSON parsing errors in logs
# Solution: Validate JSON syntax
python -m json.tool /etc/leeq/configs/epii/<config>.json

# Common fixes:
# - Remove trailing commas
# - Check quote marks are balanced
# - Verify all brackets are closed
```

**Permission Issues**
```bash
# Symptom: "Permission denied" errors
# Solution: Fix file ownership
sudo chown -R leeq:leeq /opt/leeq /var/log/leeq /etc/leeq
sudo chmod 755 /opt/leeq
sudo chmod 644 /etc/leeq/configs/epii/<config>.json
```

**Missing Dependencies**
```bash
# Symptom: ImportError or ModuleNotFoundError
# Solution: Reinstall dependencies
cd /opt/leeq
sudo -u leeq ./venv/bin/pip install -r requirements.txt
sudo -u leeq ./venv/bin/pip install -e .
```

**Port Already in Use**
```bash
# Symptom: "Address already in use" error
# Solution: Find and kill conflicting process
sudo netstat -tlnp | grep :50051
sudo kill <PID>

# Or change port in configuration
```

### 2. gRPC Connection Issues

#### Symptoms
- Client cannot connect
- "Connection refused" errors
- Timeouts on requests

#### Diagnostics
```bash
# Check if service is listening
sudo netstat -tlnp | grep :50051

# Test from local machine
grpcurl -plaintext localhost:50051 list

# Test from remote machine
grpcurl -plaintext <server-ip>:50051 list
```

#### Solutions

**Firewall Issues**
```bash
# Check firewall status
sudo ufw status

# Open gRPC port
sudo ufw allow 50051/tcp

# For iptables
sudo iptables -A INPUT -p tcp --dport 50051 -j ACCEPT
```

**Service Not Listening**
```bash
# Check service is actually running
sudo systemctl status leeq-epii@<config>

# Check logs for startup errors
sudo journalctl -u leeq-epii@<config> --no-pager -l
```

**Network Configuration**
```bash
# Verify service binds to correct interface
# In config.json, ensure proper binding:
{
  "grpc": {
    "host": "0.0.0.0",  // Listen on all interfaces
    "port": 50051
  }
}
```

### 3. Experiment Execution Failures

#### Symptoms
- Experiments timeout
- gRPC INTERNAL errors
- Invalid parameter errors

#### Diagnostics
```bash
# Check experiment-specific logs
sudo journalctl -u leeq-epii@<config> | grep "experiment"

# Enable debug logging
# In config.json:
{
  "logging": {
    "level": "DEBUG"
  }
}
```

#### Solutions

**LeeQ Setup Issues**
```bash
# Symptom: Setup initialization errors
# Solution: Verify LeeQ configuration

# Check if simulation setup works
python3 -c "
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
setup = HighLevelSimulationSetup.build_setup_from_config({
    'platform': 'numpy',
    'num_qubits': 2
})
print('Setup OK')
"
```

**Parameter Validation Errors**
```bash
# Symptom: "Invalid parameter" in logs
# Solution: Check parameter names and types

# List available parameters
grpcurl -plaintext localhost:50051 ExperimentPlatformService/ListParameters

# Verify parameter format matches expected types
```

**Memory Issues**
```bash
# Symptom: Out of memory errors, experiments killed
# Solution: Increase memory limits

# Check current memory usage
ps aux | grep leeq-epii
free -h

# Increase systemd memory limit
sudo systemctl edit leeq-epii@<config>
# Add:
[Service]
MemoryLimit=8G
```

### 4. Performance Issues

#### Symptoms
- Slow experiment execution
- High CPU/memory usage
- Client timeouts

#### Diagnostics
```bash
# Monitor resource usage
htop
systemd-cgtop

# Check gRPC worker threads
ps -eLf | grep leeq-epii

# Monitor network connections
netstat -an | grep :50051
```

#### Solutions

**Worker Thread Tuning**
```json
// In config.json
{
  "grpc": {
    "max_workers": 4,  // Reduce for low-memory systems
    "max_message_length": 52428800  // 50MB, reduce if needed
  }
}
```

**Experiment Timeout Adjustment**
```json
{
  "experiment_timeout": 600,  // Increase for slow experiments
  "grpc": {
    "client_timeout": 300
  }
}
```

**Hardware-Specific Optimizations**
```json
// For hardware setups
{
  "config": {
    "parallel_execution": false,  // Disable for stability
    "safety_checks": {
      "enabled": true,
      "max_experiment_duration": 1800
    }
  }
}
```

### 5. Data Serialization Issues

#### Symptoms
- "Failed to serialize data" errors
- Corrupted experiment results
- gRPC message size errors

#### Solutions

**Large Data Handling**
```json
{
  "grpc": {
    "max_message_length": 104857600,  // 100MB
    "compression": "gzip"
  }
}
```

**NumPy Compatibility**
```python
# Ensure NumPy arrays are contiguous
data = np.ascontiguousarray(data)

# Check data types
assert data.dtype == np.float64
```

## Advanced Troubleshooting

### Debug Mode Setup
```bash
# Create debug configuration
sudo cp /etc/leeq/configs/epii/production.json /etc/leeq/configs/epii/debug.json

# Edit debug config
sudo nano /etc/leeq/configs/epii/debug.json
```

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
  },
  "grpc": {
    "port": 50052,  // Use different port
    "max_workers": 1  // Single worker for easier debugging
  }
}
```

### Manual Testing
```python
# Test individual components
from leeq.epii.service import EPIIService
from leeq.epii.config import load_config

# Load configuration
config = load_config("/etc/leeq/configs/epii/debug.json")

# Test service initialization
service = EPIIService(config)

# Test experiment router
from leeq.epii.experiments import ExperimentRouter
router = ExperimentRouter()
print(router.list_experiments())
```

### Log Analysis
```bash
# Extract error patterns
sudo journalctl -u leeq-epii@<config> | grep ERROR

# Extract performance metrics
sudo journalctl -u leeq-epii@<config> | grep "duration"

# Extract client requests
sudo journalctl -u leeq-epii@<config> | grep "RunExperiment"
```

## Environment-Specific Issues

### Docker Deployment
```bash
# Check container logs
docker logs leeq-epii

# Verify port mapping
docker port leeq-epii

# Check resource limits
docker stats leeq-epii
```

### Kubernetes Deployment
```bash
# Check pod status
kubectl describe pod leeq-epii-pod

# Check service endpoints
kubectl get endpoints leeq-epii-service

# View logs
kubectl logs leeq-epii-pod -f
```

### Hardware Lab Environment
```bash
# Check hardware connectivity
ping <instrument-ip>

# Verify hardware configuration
cat /etc/leeq/hardware/lab_config.json

# Test hardware interfaces
python3 -c "
from leeq.setups.qubic_lbnl_setups import QubicLBNLSetup
# Test hardware connection
"
```

## Getting Help

### Information to Collect
When reporting issues, include:

1. **System Information**
   ```bash
   uname -a
   python3 --version
   pip list | grep leeq
   ```

2. **Service Logs**
   ```bash
   sudo journalctl -u leeq-epii@<config> --since "1 hour ago" > epii-logs.txt
   ```

3. **Configuration**
   ```bash
   sudo cat /etc/leeq/configs/epii/<config>.json > config.txt
   ```

4. **Error Details**
   - Exact error messages
   - Steps to reproduce
   - Expected vs actual behavior

### Debug Information Script
```bash
#!/bin/bash
# debug-info.sh - Collect EPII debugging information

echo "=== System Information ==="
uname -a
python3 --version
pip list | grep -E "(leeq|grpc|numpy)"

echo "=== Service Status ==="
sudo systemctl status leeq-epii@* --no-pager

echo "=== Recent Logs ==="
sudo journalctl -u leeq-epii@* --since "1 hour ago" --no-pager

echo "=== Network Status ==="
netstat -tlnp | grep :50051

echo "=== Resource Usage ==="
ps aux | grep leeq-epii
free -h
df -h /var/log/leeq
```

### Contact Information
- GitHub Issues: https://github.com/ShuxiangCao/LeeQ/issues
- Documentation: https://leeq.readthedocs.io/
- Support Forum: [Add forum link if available]

## Preventive Measures

### Regular Maintenance
```bash
# Weekly log rotation
sudo logrotate /etc/logrotate.d/leeq-epii

# Monthly configuration backup
sudo tar -czf /backup/leeq-config-$(date +%Y%m).tar.gz /etc/leeq/

# Quarterly dependency updates
cd /opt/leeq
sudo -u leeq ./venv/bin/pip list --outdated
```

### Monitoring Setup
```bash
# Setup log monitoring
sudo systemctl enable --now systemd-journald

# Configure log retention
sudo mkdir -p /etc/systemd/journald.conf.d/
echo "[Journal]
SystemMaxUse=1G
RuntimeMaxUse=100M
MaxRetentionSec=1month" | sudo tee /etc/systemd/journald.conf.d/leeq.conf
```

### Health Checks
```python
# health-check.py - Regular service health verification
import grpc
from leeq.epii.proto import epii_pb2, epii_pb2_grpc

def health_check():
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
            response = stub.Ping(epii_pb2.PingRequest(), timeout=5)
            return True
    except:
        return False

if __name__ == "__main__":
    if health_check():
        print("EPII service is healthy")
        exit(0)
    else:
        print("EPII service is unhealthy")
        exit(1)
```