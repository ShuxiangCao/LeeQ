# LeeQ EPII Operations Guide

## Overview

This guide provides comprehensive instructions for deploying, managing, and monitoring the LeeQ EPII (Experiment Platform Intelligence Interface) service in production environments. The EPII service exposes LeeQ's quantum experiments through a standardized gRPC interface for integration with external orchestration systems.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Service Management](#service-management)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Security](#security)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Backup and Recovery](#backup-and-recovery)
10. [Maintenance](#maintenance)

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 20 GB free space
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8+ GB
- Storage: 100+ GB SSD
- Network: 1 Gbps

### Software Requirements

**Operating System:**
- Ubuntu 20.04 LTS or later
- RHEL/CentOS 8 or later
- SUSE Linux Enterprise 15 or later

**Dependencies:**
- Python 3.8 or later
- systemd (for service management)
- gRPC runtime libraries

**Network Requirements:**
- Port 50051-50099 (configurable)
- Firewall configured for gRPC traffic
- DNS resolution for external dependencies

## Installation

### 1. Automated Installation

The recommended way to install LeeQ EPII in production is using the automated installer:

```bash
# Clone the LeeQ repository
git clone https://github.com/ShuxiangCao/LeeQ.git /opt/leeq-source
cd /opt/leeq-source

# Run the installation script as root
sudo ./scripts/install-systemd-service.sh
```

This script will:
- Create system user and group (`leeq`)
- Install LeeQ to `/opt/leeq`
- Create configuration directories
- Install systemd service templates
- Set up logging directories

### 2. Manual Installation

For custom installations, follow these steps:

```bash
# Create system user
sudo groupadd --system leeq
sudo useradd --system --gid leeq --shell /bin/false \
    --home-dir /opt/leeq --no-create-home \
    --comment "LeeQ EPII daemon user" leeq

# Create directories
sudo mkdir -p /opt/leeq /etc/leeq-epii /var/log/leeq-epii /var/run/leeq-epii
sudo chown -R leeq:leeq /opt/leeq /etc/leeq-epii /var/log/leeq-epii /var/run/leeq-epii
sudo chmod 755 /etc/leeq-epii
sudo chmod 750 /var/log/leeq-epii /var/run/leeq-epii

# Install LeeQ
sudo cp -r leeq /opt/leeq/
sudo cp -r venv /opt/leeq/  # If using virtual environment
sudo chown -R leeq:leeq /opt/leeq

# Install systemd service
sudo cp scripts/systemd/leeq-epii@.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 3. Verification

Verify the installation:

```bash
# Check service template
sudo systemctl list-unit-files "leeq-epii@*.service"

# Validate sample configuration
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
    --config /etc/leeq-epii/simulation_2q.json --validate
```

## Configuration

### 1. Configuration File Structure

Configuration files are stored in `/etc/leeq-epii/` and use JSON format:

```json
{
    "setup_type": "simulation|hardware",
    "setup_name": "unique_setup_identifier",
    "description": "Human-readable description",
    "port": 50051,
    "max_workers": 10,
    "timeout": 300,
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "simulation": {
        "backend": "numpy|qutip",
        "qubits": 2
    },
    "hardware": {
        "type": "qubic_lbnl",
        "config_file": "/etc/leeq-epii/hardware/setup_config.json"
    }
}
```

### 2. Configuration Examples

#### Simulation Setup
```json
{
    "setup_type": "simulation",
    "setup_name": "sim_2q_development",
    "description": "2-qubit simulation for development and testing",
    "port": 50051,
    "max_workers": 5,
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

#### Hardware Setup
```json
{
    "setup_type": "hardware",
    "setup_name": "lab1_quantum_device",
    "description": "Lab 1 quantum device with 4 qubits",
    "port": 50052,
    "max_workers": 3,
    "timeout": 600,
    "logging": {
        "level": "WARNING"
    },
    "hardware": {
        "type": "qubic_lbnl",
        "config_file": "/etc/leeq-epii/hardware/lab1_config.json"
    }
}
```

### 3. Environment Variables

Configure system-wide settings using environment variables:

```bash
# /etc/environment or /etc/systemd/system/leeq-epii@.service
LEEQ_EPII_CONFIG_DIR=/etc/leeq-epii
LEEQ_EPII_LOG_LEVEL=INFO
LEEQ_EPII_MAX_MESSAGE_SIZE=104857600  # 100MB
PYTHONPATH=/opt/leeq
```

### 4. Port Management

Configure ports to avoid conflicts:

- **50051**: Default simulation setup
- **50052-50059**: Additional simulation instances
- **50060-50069**: Hardware setups
- **50070-50099**: Development/testing

### 5. Configuration Validation

Always validate configurations before deployment:

```bash
# Validate specific configuration
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
    --config /etc/leeq-epii/your_setup.json --validate

# Validate all configurations
for config in /etc/leeq-epii/*.json; do
    echo "Validating $config..."
    sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
        --config "$config" --validate
done
```

## Service Management

### 1. Service Instances

Each configuration file creates a service instance:

```bash
# Enable and start simulation setup
sudo systemctl enable leeq-epii@simulation_2q.service
sudo systemctl start leeq-epii@simulation_2q.service

# Enable and start hardware setup
sudo systemctl enable leeq-epii@hardware_lab1.service
sudo systemctl start leeq-epii@hardware_lab1.service
```

### 2. Service Management Commands

```bash
# Start service
sudo systemctl start leeq-epii@<setup_name>.service

# Stop service
sudo systemctl stop leeq-epii@<setup_name>.service

# Restart service
sudo systemctl restart leeq-epii@<setup_name>.service

# Check status
sudo systemctl status leeq-epii@<setup_name>.service

# Enable auto-start on boot
sudo systemctl enable leeq-epii@<setup_name>.service

# Disable auto-start
sudo systemctl disable leeq-epii@<setup_name>.service

# Reload configuration (after config changes)
sudo systemctl reload-or-restart leeq-epii@<setup_name>.service
```

### 3. Service Management Script

Use the included management script for easier operations:

```bash
# List all configured services
sudo /opt/leeq-source/scripts/leeq-epii-service.sh list

# Start a service
sudo /opt/leeq-source/scripts/leeq-epii-service.sh start simulation_2q

# Check status
sudo /opt/leeq-source/scripts/leeq-epii-service.sh status simulation_2q

# View logs
sudo /opt/leeq-source/scripts/leeq-epii-service.sh logs simulation_2q

# Validate configuration
sudo /opt/leeq-source/scripts/leeq-epii-service.sh validate simulation_2q
```

### 4. Health Checks

Monitor service health:

```bash
# Built-in health check
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
    --config /etc/leeq-epii/simulation_2q.json --health-check

# Check if service is listening
sudo ss -tlnp | grep :50051

# Test gRPC connectivity
grpcurl -plaintext localhost:50051 epii.v1.ExperimentPlatformService/Ping
```

## Monitoring and Logging

### 1. Systemd Logs

View service logs using journalctl:

```bash
# View recent logs
sudo journalctl -u leeq-epii@simulation_2q.service

# Follow logs in real-time
sudo journalctl -u leeq-epii@simulation_2q.service -f

# View logs since last boot
sudo journalctl -u leeq-epii@simulation_2q.service -b

# View logs for specific time range
sudo journalctl -u leeq-epii@simulation_2q.service \
    --since "2024-01-01 00:00:00" --until "2024-01-02 00:00:00"

# Export logs to file
sudo journalctl -u leeq-epii@simulation_2q.service > epii_logs.txt
```

### 2. Log File Management

Configure log rotation to prevent disk space issues:

```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/leeq-epii << EOF
/var/log/leeq-epii/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 640 leeq leeq
    postrotate
        systemctl reload leeq-epii@*.service
    endscript
}
EOF
```

### 3. Monitoring Metrics

Key metrics to monitor:

**Service Health:**
- Service uptime
- Restart frequency
- Memory usage
- CPU usage

**gRPC Metrics:**
- Request rate
- Response time
- Error rate
- Active connections

**Experiment Metrics:**
- Experiment success rate
- Average execution time
- Queue depth
- Resource utilization

### 4. Monitoring Integration

#### Prometheus Integration

Create a monitoring endpoint:

```python
# Add to service configuration for metrics export
from prometheus_client import start_http_server, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('epii_requests_total', 'Total requests', ['method', 'status'])
REQUEST_DURATION = Histogram('epii_request_duration_seconds', 'Request duration')

# Start metrics server on different port
start_http_server(8000)
```

#### Grafana Dashboard

Monitor key metrics with these queries:

```promql
# Request rate
rate(epii_requests_total[5m])

# Error rate
rate(epii_requests_total{status="error"}[5m]) / rate(epii_requests_total[5m])

# Average response time
rate(epii_request_duration_seconds_sum[5m]) / rate(epii_request_duration_seconds_count[5m])
```

## Security

### 1. Network Security

**Firewall Configuration:**

```bash
# Allow EPII ports
sudo ufw allow 50051:50099/tcp comment "LeeQ EPII services"

# Restrict to specific networks
sudo ufw allow from 192.168.1.0/24 to any port 50051:50099 proto tcp
```

**TLS/SSL Setup:**

For production, enable TLS encryption:

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt \
    -days 365 -nodes -subj "/C=US/ST=CA/L=Berkeley/O=LeeQ/CN=epii-server"

# Update service configuration
{
    "tls": {
        "enabled": true,
        "cert_file": "/etc/leeq-epii/certs/server.crt",
        "key_file": "/etc/leeq-epii/certs/server.key"
    }
}
```

### 2. Access Control

**File Permissions:**

```bash
# Configuration files
sudo chmod 640 /etc/leeq-epii/*.json
sudo chown root:leeq /etc/leeq-epii/*.json

# Log files
sudo chmod 640 /var/log/leeq-epii/*.log
sudo chown leeq:leeq /var/log/leeq-epii/*.log

# Service files
sudo chmod 644 /etc/systemd/system/leeq-epii@.service
sudo chown root:root /etc/systemd/system/leeq-epii@.service
```

**User Isolation:**

The service runs as the `leeq` user with minimal privileges:

```bash
# Verify user configuration
id leeq
# Should show: uid=999(leeq) gid=999(leeq) groups=999(leeq)

# Check service security
sudo systemctl show leeq-epii@simulation_2q.service | grep -E "User|Group|NoNewPrivileges|PrivateTmp"
```

### 3. Authentication

For production deployments, implement authentication:

```python
# Example gRPC interceptor for API key authentication
class AuthInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        api_key = metadata.get('x-api-key')
        
        if not self.validate_api_key(api_key):
            context = grpc.ServicerContext()
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")
        
        return continuation(handler_call_details)
```

## Performance Tuning

### 1. System Optimization

**Memory Settings:**

```bash
# Increase shared memory for large experiments
echo "kernel.shmmax = 134217728" | sudo tee -a /etc/sysctl.conf
echo "kernel.shmall = 32768" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**File Descriptor Limits:**

```bash
# Increase file descriptor limits
echo "leeq soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "leeq hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

### 2. gRPC Optimization

**Configuration Tuning:**

```json
{
    "grpc": {
        "max_workers": 10,
        "max_message_size": 104857600,
        "keepalive_time_ms": 30000,
        "keepalive_timeout_ms": 5000,
        "keepalive_permit_without_calls": true
    }
}
```

**Connection Pooling:**

For clients connecting to EPII:

```python
# Optimize client connections
channel = grpc.insecure_channel(
    'localhost:50051',
    options=[
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.http2.min_ping_interval_without_data_ms', 300000)
    ]
)
```

### 3. Hardware Optimization

**CPU Affinity:**

```bash
# Pin service to specific CPUs
sudo systemctl edit leeq-epii@simulation_2q.service

# Add in override file:
[Service]
ExecStart=
ExecStart=/usr/bin/taskset -c 0,1 /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq-epii/simulation_2q.json
```

**NUMA Optimization:**

```bash
# Check NUMA topology
numactl --hardware

# Run on specific NUMA node
sudo systemctl edit leeq-epii@hardware_lab1.service

# Add in override file:
[Service]
ExecStart=
ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq-epii/hardware_lab1.json
```

## Troubleshooting

### 1. Common Issues

**Service Won't Start:**

```bash
# Check service status
sudo systemctl status leeq-epii@simulation_2q.service

# Check detailed logs
sudo journalctl -u leeq-epii@simulation_2q.service -n 50

# Validate configuration
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
    --config /etc/leeq-epii/simulation_2q.json --validate

# Check port availability
sudo ss -tlnp | grep :50051
```

**Connection Refused:**

```bash
# Check if service is listening
sudo netstat -tlnp | grep :50051

# Test local connectivity
telnet localhost 50051

# Check firewall
sudo ufw status
```

**High Memory Usage:**

```bash
# Monitor memory usage
sudo systemctl status leeq-epii@simulation_2q.service | grep Memory

# Check process details
sudo ps aux | grep leeq

# Monitor in real-time
sudo htop -u leeq
```

### 2. Debug Mode

Enable debug logging for troubleshooting:

```bash
# Temporary debug mode
sudo systemctl edit leeq-epii@simulation_2q.service

# Add in override file:
[Service]
Environment=LEEQ_EPII_LOG_LEVEL=DEBUG
```

### 3. Diagnostic Tools

**Generate diagnostic report:**

```bash
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
    --config /etc/leeq-epii/simulation_2q.json --diagnostic-report
```

**Performance profiling:**

```bash
# Profile service startup
sudo -u leeq python -m cProfile -o epii_profile.prof \
    -m leeq.epii.daemon --config /etc/leeq-epii/simulation_2q.json

# Analyze profile
python -c "
import pstats
p = pstats.Stats('epii_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

### 4. Recovery Procedures

**Service Recovery:**

```bash
# Stop all EPII services
sudo systemctl stop leeq-epii@*.service

# Clear any stuck processes
sudo pkill -f "leeq.epii.daemon"

# Remove stale PID files
sudo rm -f /var/run/leeq-epii/*.pid

# Restart services
sudo systemctl start leeq-epii@simulation_2q.service
```

**Configuration Recovery:**

```bash
# Backup current config
sudo cp /etc/leeq-epii/simulation_2q.json /etc/leeq-epii/simulation_2q.json.backup

# Restore from backup
sudo cp /etc/leeq-epii/simulation_2q.json.backup /etc/leeq-epii/simulation_2q.json

# Regenerate default config
sudo /opt/leeq-source/scripts/install-systemd-service.sh
```

## Backup and Recovery

### 1. Backup Strategy

**Configuration Backup:**

```bash
#!/bin/bash
# /opt/leeq/scripts/backup-config.sh

BACKUP_DIR="/backup/leeq-epii/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
cp -r /etc/leeq-epii "$BACKUP_DIR/"

# Backup service files
cp /etc/systemd/system/leeq-epii@.service "$BACKUP_DIR/"

# Backup logs (last 7 days)
journalctl -u 'leeq-epii@*.service' --since "7 days ago" > "$BACKUP_DIR/logs.txt"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"
```

**Automated Backup:**

```bash
# Add to crontab
sudo crontab -e

# Daily backup at 2 AM
0 2 * * * /opt/leeq/scripts/backup-config.sh
```

### 2. Recovery Procedures

**Configuration Recovery:**

```bash
# Extract backup
tar -xzf /backup/leeq-epii/20240101.tar.gz -C /tmp/restore

# Stop services
sudo systemctl stop leeq-epii@*.service

# Restore configuration
sudo cp -r /tmp/restore/leeq-epii/* /etc/leeq-epii/
sudo chown -R root:leeq /etc/leeq-epii
sudo chmod 640 /etc/leeq-epii/*.json

# Restore service file
sudo cp /tmp/restore/leeq-epii@.service /etc/systemd/system/
sudo systemctl daemon-reload

# Restart services
sudo systemctl start leeq-epii@*.service
```

**Disaster Recovery:**

```bash
#!/bin/bash
# Complete system recovery script

# 1. Reinstall LeeQ EPII
sudo /opt/leeq-source/scripts/install-systemd-service.sh

# 2. Restore configurations
sudo tar -xzf /backup/leeq-epii/latest.tar.gz -C /
sudo chown -R root:leeq /etc/leeq-epii
sudo chmod 640 /etc/leeq-epii/*.json

# 3. Restart services
sudo systemctl daemon-reload
sudo systemctl start leeq-epii@*.service

# 4. Verify operation
for service in /etc/leeq-epii/*.json; do
    setup_name=$(basename "$service" .json)
    echo "Testing $setup_name..."
    sudo systemctl status leeq-epii@$setup_name.service
done
```

## Maintenance

### 1. Regular Maintenance Tasks

**Weekly Tasks:**

```bash
#!/bin/bash
# Weekly maintenance script

# Check service health
systemctl status leeq-epii@*.service

# Check disk space
df -h /var/log/leeq-epii

# Rotate logs if needed
sudo logrotate -f /etc/logrotate.d/leeq-epii

# Check for configuration changes
sudo find /etc/leeq-epii -name "*.json" -mtime -7 -ls
```

**Monthly Tasks:**

```bash
#!/bin/bash
# Monthly maintenance script

# Update LeeQ
cd /opt/leeq-source
sudo git pull origin main
sudo /opt/leeq/venv/bin/pip install -e .

# Restart services to pick up updates
sudo systemctl restart leeq-epii@*.service

# Cleanup old logs
sudo find /var/log/leeq-epii -name "*.gz" -mtime +30 -delete

# Performance review
sudo journalctl -u 'leeq-epii@*.service' --since "1 month ago" | grep -E "ERROR|WARNING" | wc -l
```

### 2. Updates and Upgrades

**LeeQ Updates:**

```bash
# Stop all services
sudo systemctl stop leeq-epii@*.service

# Backup current installation
sudo tar -czf /backup/leeq-$(date +%Y%m%d).tar.gz -C /opt leeq

# Update LeeQ
cd /opt/leeq-source
sudo git pull origin main
sudo cp -r leeq /opt/leeq/
sudo chown -R leeq:leeq /opt/leeq

# Update Python dependencies
sudo -u leeq /opt/leeq/venv/bin/pip install -r requirements.txt

# Test configuration
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon \
    --config /etc/leeq-epii/simulation_2q.json --validate

# Restart services
sudo systemctl start leeq-epii@*.service
```

**System Updates:**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Restart services if kernel was updated
if [ -f /var/run/reboot-required ]; then
    echo "System restart required after kernel update"
    # Schedule restart during maintenance window
fi
```

### 3. Performance Monitoring

**Resource Usage:**

```bash
#!/bin/bash
# Monitor resource usage

echo "=== LeeQ EPII Resource Usage ==="
echo "Memory Usage:"
sudo systemctl status leeq-epii@*.service | grep Memory

echo "CPU Usage:"
sudo ps -u leeq -o pid,pcpu,pmem,command

echo "Network Connections:"
sudo ss -tlnp | grep -E ":5005[0-9]"

echo "Disk Usage:"
df -h /var/log/leeq-epii /etc/leeq-epii /opt/leeq
```

**Performance Baseline:**

```bash
# Establish performance baseline
for i in {1..10}; do
    grpcurl -plaintext localhost:50051 epii.v1.ExperimentPlatformService/Ping | \
    jq '.timestamp' | xargs -I {} date -d @{}
    sleep 1
done
```

### 4. Capacity Planning

Monitor these metrics for capacity planning:

- **Concurrent experiments**: Track number of simultaneous experiment executions
- **Memory per experiment**: Monitor memory usage patterns
- **Experiment duration**: Track typical execution times
- **Queue depth**: Monitor experiment request queuing
- **Error rates**: Track failure patterns and recovery times

**Capacity Monitoring Script:**

```bash
#!/bin/bash
# Capacity monitoring

echo "=== Capacity Metrics $(date) ==="

# Active connections
echo "Active gRPC connections:"
sudo ss -tn state established '( dport = :50051 or sport = :50051 )' | wc -l

# Memory usage trend
echo "Memory usage (MB):"
sudo ps -u leeq -o rss= | awk '{sum+=$1} END {print sum/1024}'

# Service uptime
echo "Service uptime:"
sudo systemctl show leeq-epii@simulation_2q.service --property=ActiveEnterTimestamp

# Request rate (if metrics are available)
echo "Request rate (last hour):"
sudo journalctl -u leeq-epii@simulation_2q.service --since "1 hour ago" | \
grep "Experiment request" | wc -l
```

---

## Quick Reference

### Essential Commands

```bash
# Service Management
sudo systemctl {start|stop|restart|status} leeq-epii@<setup>.service

# Configuration Validation
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq-epii/<setup>.json --validate

# Health Check
sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq-epii/<setup>.json --health-check

# View Logs
sudo journalctl -u leeq-epii@<setup>.service -f

# Test Connectivity
grpcurl -plaintext localhost:<port> epii.v1.ExperimentPlatformService/Ping
```

### Important Paths

- **Configuration**: `/etc/leeq-epii/`
- **Installation**: `/opt/leeq/`
- **Logs**: `/var/log/leeq-epii/` (systemd logs via journalctl)
- **Service Files**: `/etc/systemd/system/leeq-epii@.service`
- **Runtime**: `/var/run/leeq-epii/`

### Support and Resources

- **Documentation**: `/opt/leeq-source/docs/`
- **Issue Tracking**: GitHub Issues
- **Configuration Examples**: `/opt/leeq-source/configs/epii/`
- **Test Scripts**: `/opt/leeq-source/scripts/`

---

*This operations guide provides comprehensive coverage of LeeQ EPII deployment and management. For additional support, consult the development team or submit issues through the project's GitHub repository.*