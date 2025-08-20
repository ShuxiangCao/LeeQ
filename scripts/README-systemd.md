# LeeQ EPII Systemd Integration

This directory contains scripts and configuration files for running LeeQ EPII as a systemd service on Linux systems.

## Overview

The systemd integration allows you to:
- Run LeeQ EPII daemon as a system service with automatic startup
- Manage multiple daemon instances with different configurations
- Monitor service health and logs through systemd/journald
- Perform graceful shutdowns and restarts
- Maintain proper process isolation and security

## Files

### Core Files
- `systemd/leeq-epii@.service` - Systemd service template
- `install-systemd-service.sh` - Installation script
- `uninstall-systemd-service.sh` - Uninstallation script

### Management Tools
- `leeq-epii-service.sh` - Service management utility
- `test-systemd-integration.sh` - Integration test suite

### Documentation
- `README-systemd.md` - This file

## Installation

### Prerequisites
- Linux system with systemd
- Root/sudo access
- LeeQ installed with dependencies

### Quick Installation
```bash
# Install the systemd service
sudo ./install-systemd-service.sh

# This will:
# - Create leeq user and group
# - Install LeeQ to /opt/leeq
# - Create configuration directories
# - Install systemd service template
# - Create sample configurations
```

## Configuration

### Configuration Files
Configuration files are stored in `/etc/leeq-epii/` and use JSON format:

```json
{
    "setup_type": "simulation",
    "setup_name": "simulation_2q",
    "description": "2-qubit simulation setup",
    "max_workers": 10,
    "timeout": 300,
    "logging": {
        "level": "INFO"
    },
    "simulation": {
        "backend": "numpy",
        "qubits": 2
    }
}
```

### Multiple Instances
You can run multiple daemon instances by creating different configuration files:

```bash
# Create configurations
sudo cp /etc/leeq-epii/simulation_2q.json /etc/leeq-epii/lab1_hardware.json
sudo cp /etc/leeq-epii/simulation_2q.json /etc/leeq-epii/lab2_hardware.json

# Edit configurations as needed
sudo nano /etc/leeq-epii/lab1_hardware.json
```

Each configuration file corresponds to a service instance:
- `simulation_2q.json` → `leeq-epii@simulation_2q.service`
- `lab1_hardware.json` → `leeq-epii@lab1_hardware.service`

## Service Management

### Using the Management Script (Recommended)
```bash
# List available configurations
sudo ./leeq-epii-service.sh list

# Start a service instance
sudo ./leeq-epii-service.sh start simulation_2q

# Check status
sudo ./leeq-epii-service.sh status simulation_2q

# View logs
sudo ./leeq-epii-service.sh logs simulation_2q

# Follow logs in real-time
sudo ./leeq-epii-service.sh logs simulation_2q --follow

# Restart service
sudo ./leeq-epii-service.sh restart simulation_2q

# Stop service
sudo ./leeq-epii-service.sh stop simulation_2q

# Validate configuration
sudo ./leeq-epii-service.sh validate simulation_2q
```

### Using systemctl Directly
```bash
# Enable and start service
sudo systemctl enable leeq-epii@simulation_2q.service
sudo systemctl start leeq-epii@simulation_2q.service

# Check status
sudo systemctl status leeq-epii@simulation_2q.service

# View logs
sudo journalctl -u leeq-epii@simulation_2q.service

# Follow logs
sudo journalctl -u leeq-epii@simulation_2q.service -f

# Restart service
sudo systemctl restart leeq-epii@simulation_2q.service

# Stop and disable service
sudo systemctl stop leeq-epii@simulation_2q.service
sudo systemctl disable leeq-epii@simulation_2q.service
```

## Logging

### Log Configuration
The service is configured to log to systemd journal with structured output:

- **Format**: `[LEVEL] module: message` (for systemd)
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Identifier**: `leeq-epii-<instance>`

### Viewing Logs
```bash
# Recent logs
sudo journalctl -u leeq-epii@simulation_2q.service

# Follow logs
sudo journalctl -u leeq-epii@simulation_2q.service -f

# Logs from specific time
sudo journalctl -u leeq-epii@simulation_2q.service --since "1 hour ago"

# All EPII services
sudo journalctl -u "leeq-epii@*.service"

# Filter by log level
sudo journalctl -u leeq-epii@simulation_2q.service -p err
```

### Log Rotation
systemd/journald handles log rotation automatically. Configure in `/etc/systemd/journald.conf`:

```ini
[Journal]
SystemMaxUse=1G
SystemKeepFree=1G
SystemMaxFileSize=100M
MaxRetentionSec=1month
```

## Security Features

The systemd service includes several security features:

### Process Isolation
- Runs as dedicated `leeq` user/group
- No new privileges allowed
- Private temporary directories
- Protected system directories

### File System Access
- Read-only access to most system paths
- Write access only to required directories:
  - `/var/log/leeq-epii/` - Log files
  - `/var/run/leeq-epii/` - Runtime files
  - `/tmp` - Temporary files

### Resource Limits
Configure in the service file if needed:
```ini
[Service]
MemoryLimit=2G
CPUQuota=200%
TasksMax=100
```

## Troubleshooting

### Service Won't Start
1. Check configuration validation:
   ```bash
   sudo ./leeq-epii-service.sh validate simulation_2q
   ```

2. Check service status:
   ```bash
   sudo systemctl status leeq-epii@simulation_2q.service
   ```

3. Check logs for errors:
   ```bash
   sudo journalctl -u leeq-epii@simulation_2q.service -n 50
   ```

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulpn | grep :50051

# Use different port in configuration
{
    "port": 50052,
    ...
}
```

#### Permission Denied
```bash
# Check file permissions
ls -la /etc/leeq-epii/
ls -la /opt/leeq/

# Fix ownership if needed
sudo chown -R leeq:leeq /etc/leeq-epii/
sudo chown -R leeq:leeq /opt/leeq/
```

#### Service Fails to Stop
```bash
# Force stop if needed
sudo systemctl kill leeq-epii@simulation_2q.service

# Check for zombie processes
ps aux | grep leeq-epii
```

### Health Checks
The daemon includes built-in health checks:

```bash
# Run health check
cd /opt/leeq
sudo -u leeq python3 -m leeq.epii.daemon --config /etc/leeq-epii/simulation_2q.json --health-check
```

## Testing

### Run Integration Tests
```bash
# Install first, then test
sudo ./install-systemd-service.sh
sudo ./test-systemd-integration.sh
```

The test suite verifies:
- Service installation and recognition
- Enable/disable operations
- Start/stop/restart functionality
- Logging configuration
- Graceful shutdown
- Configuration validation

### Manual Testing
```bash
# Start service and test gRPC connection
sudo ./leeq-epii-service.sh start simulation_2q

# Test with grpcurl (if available)
grpcurl -plaintext localhost:50051 list

# Or test with Python client
python3 -c "
import grpc
from leeq.epii.proto import epii_pb2_grpc, epii_pb2
channel = grpc.insecure_channel('localhost:50051')
stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
response = stub.Ping(epii_pb2.PingRequest())
print('Service is responding:', response.message)
"
```

## Uninstallation

### Remove Service Only
```bash
sudo ./uninstall-systemd-service.sh
```

### Remove Everything
```bash
sudo ./uninstall-systemd-service.sh --all
```

Options:
- `--remove-data` - Remove configuration and log directories
- `--remove-user` - Remove leeq user and group
- `--all` - Remove everything

## Advanced Configuration

### Custom Service Settings
Edit `/etc/systemd/system/leeq-epii@.service` for custom settings:

```ini
[Service]
# Custom memory limit
MemoryLimit=4G

# Custom restart policy
Restart=on-failure
RestartSec=10

# Custom timeout
TimeoutStartSec=60
TimeoutStopSec=60
```

After editing, reload systemd:
```bash
sudo systemctl daemon-reload
```

### Environment Variables
Add environment variables in the service file:

```ini
[Service]
Environment=LEEQ_EPII_LOG_LEVEL=DEBUG
Environment=PYTHONPATH=/opt/leeq:/custom/path
Environment=CUDA_VISIBLE_DEVICES=0
```

### Network Configuration
For remote access, modify firewall and service binding:

```bash
# Open firewall port
sudo ufw allow 50051

# For external access, modify service to bind to all interfaces
# Edit configuration file to include:
{
    "bind_address": "0.0.0.0",
    "port": 50051
}
```

## Best Practices

1. **Use unique ports** for different instances
2. **Monitor logs** regularly for errors
3. **Test configurations** before deploying
4. **Use health checks** for monitoring
5. **Keep configurations** in version control
6. **Regular backups** of configuration directory
7. **Monitor resource usage** (CPU, memory, disk)
8. **Use log rotation** to manage disk space

## Support

For issues with systemd integration:

1. Check this documentation
2. Run the test suite
3. Check systemd/journald logs
4. Verify LeeQ installation
5. Check network connectivity and ports
6. Review file permissions and ownership

For LeeQ-specific issues, refer to the main LeeQ documentation.