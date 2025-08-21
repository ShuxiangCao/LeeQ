# LeeQ EPII Deployment Guide

This guide covers deploying the LeeQ EPII (Experiment Platform Intelligence Interface) service in production environments.

## Prerequisites

### System Requirements
- Linux server (Ubuntu 20.04+ or CentOS 8+ recommended)
- Python 3.8+ with virtual environment support
- systemd for service management
- Minimum 4GB RAM, 8GB recommended for hardware setups
- Network access to quantum hardware (for hardware configurations)

### User Setup
Create a dedicated user for running the EPII service:

```bash
sudo useradd -r -s /bin/bash -d /opt/leeq leeq
sudo mkdir -p /opt/leeq /var/log/leeq /var/lib/leeq /etc/leeq/configs/epii
sudo chown -R leeq:leeq /opt/leeq /var/log/leeq /var/lib/leeq
sudo chown -R leeq:leeq /etc/leeq
```

## Installation

### 1. Install LeeQ EPII
```bash
# Clone the repository
cd /opt/leeq
sudo -u leeq git clone https://github.com/ShuxiangCao/LeeQ.git .

# Create virtual environment
sudo -u leeq python3 -m venv venv
sudo -u leeq ./venv/bin/pip install --upgrade pip

# Install dependencies
sudo -u leeq ./venv/bin/pip install -r requirements.txt
sudo -u leeq ./venv/bin/pip install -e .
```

### 2. Configure the Service
```bash
# Copy configuration files
sudo cp configs/epii/*.json /etc/leeq/configs/epii/

# Edit configuration for your environment
sudo -u leeq nano /etc/leeq/configs/epii/production.json
```

### 3. Install systemd Service
```bash
# Copy service template
sudo cp scripts/systemd/leeq-epii@.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable leeq-epii@production
sudo systemctl start leeq-epii@production
```

## Configuration

### Configuration File Structure
```json
{
  "setup_type": "simulation|hardware",
  "setup_class": "HighLevelSimulationSetup|QubicLBNLSetup",
  "config": {
    // Setup-specific configuration
  },
  "grpc": {
    "port": 50051,
    "max_workers": 10,
    "max_message_length": 104857600
  },
  "logging": {
    "level": "INFO|WARNING|ERROR",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "/var/log/leeq/epii.log"
  },
  "experiment_timeout": 300,
  "parameter_validation": true
}
```

### Configuration Templates

#### Simulation Setup
Use `simulation_2q.json` for development and testing with simulated qubits.

#### Hardware Setup  
Use `hardware_lab1.json` as a template for real quantum hardware deployments.

#### Minimal Setup
Use `minimal.json` for basic testing and validation.

## Service Management

### Basic Commands
```bash
# Start service
sudo systemctl start leeq-epii@<config-name>

# Stop service
sudo systemctl stop leeq-epii@<config-name>

# Restart service
sudo systemctl restart leeq-epii@<config-name>

# Check status
sudo systemctl status leeq-epii@<config-name>

# View logs
sudo journalctl -u leeq-epii@<config-name> -f
```

### Multiple Configurations
You can run multiple EPII instances with different configurations:

```bash
# Start simulation instance
sudo systemctl start leeq-epii@simulation_2q

# Start hardware instance
sudo systemctl start leeq-epii@hardware_lab1

# Check all running instances
sudo systemctl list-units "leeq-epii@*"
```

## Security Considerations

### Network Security
- Configure firewall to restrict access to gRPC port (default 50051)
- Use TLS/SSL for production deployments
- Consider VPN access for remote clients

### System Security
The systemd service includes security hardening:
- Runs with restricted user privileges
- Isolated temporary directories
- Protected system directories
- Resource limits to prevent abuse

### Configuration Security
- Store sensitive configuration in `/etc/leeq/configs/` with restricted permissions
- Use environment variables for secrets when possible
- Regularly rotate any embedded credentials

## Monitoring and Logging

### Log Files
- Service logs: `journalctl -u leeq-epii@<config>`
- Application logs: `/var/log/leeq/epii.log` (if configured)
- System logs: `/var/log/syslog`

### Health Checks
```bash
# Check service status
sudo systemctl status leeq-epii@<config>

# Test gRPC endpoint
grpcurl -plaintext localhost:50051 list

# Check process resources
ps aux | grep leeq-epii
```

### Performance Monitoring
- Monitor memory usage with `htop` or `systemd-cgtop`
- Track gRPC request metrics in application logs
- Use `netstat` to monitor network connections

## Backup and Recovery

### Configuration Backup
```bash
# Backup configurations
sudo tar -czf leeq-config-backup-$(date +%Y%m%d).tar.gz /etc/leeq/

# Backup application
sudo tar -czf leeq-app-backup-$(date +%Y%m%d).tar.gz /opt/leeq/
```

### Disaster Recovery
1. Reinstall system packages
2. Restore application from backup
3. Restore configurations
4. Restart services
5. Validate functionality with test experiments

## Upgrades

### Application Upgrades
```bash
# Stop service
sudo systemctl stop leeq-epii@<config>

# Backup current installation
sudo cp -r /opt/leeq /opt/leeq.backup.$(date +%Y%m%d)

# Update code
cd /opt/leeq
sudo -u leeq git pull origin main
sudo -u leeq ./venv/bin/pip install --upgrade -r requirements.txt

# Test configuration
sudo -u leeq ./venv/bin/python -m leeq.epii.daemon --config /etc/leeq/configs/epii/<config>.json --validate

# Restart service
sudo systemctl start leeq-epii@<config>
```

### Configuration Updates
```bash
# Validate new configuration
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq/configs/epii/new-config.json --validate

# Reload service (for minor changes)
sudo systemctl reload leeq-epii@<config>

# Or restart for major changes
sudo systemctl restart leeq-epii@<config>
```

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.