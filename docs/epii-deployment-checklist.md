# LeeQ EPII Production Deployment Checklist

## Pre-Deployment Checklist

### System Requirements Verification
- [ ] Operating system meets requirements (Ubuntu 20.04+ / RHEL 8+ / SUSE 15+)
- [ ] Minimum hardware requirements met (2 cores, 4GB RAM, 20GB storage)
- [ ] Python 3.8+ installed and available
- [ ] systemd service manager available
- [ ] Network ports 50051-50099 available
- [ ] Firewall configured to allow gRPC traffic

### Security Preparation
- [ ] System user `leeq` created with appropriate permissions
- [ ] File system permissions configured correctly
- [ ] Network security policies reviewed and approved
- [ ] TLS certificates prepared (for production environments)
- [ ] Access control mechanisms defined

## Installation Checklist

### Automated Installation
- [ ] LeeQ source code downloaded to `/opt/leeq-source`
- [ ] Installation script executed: `sudo ./scripts/install-systemd-service.sh`
- [ ] Installation completed without errors
- [ ] System directories created with correct ownership:
  - [ ] `/opt/leeq` (leeq:leeq)
  - [ ] `/etc/leeq-epii` (root:leeq)
  - [ ] `/var/log/leeq-epii` (leeq:leeq)
  - [ ] `/var/run/leeq-epii` (leeq:leeq)

### Service Registration
- [ ] Systemd service template installed: `/etc/systemd/system/leeq-epii@.service`
- [ ] Systemd daemon reloaded: `sudo systemctl daemon-reload`
- [ ] Service template recognized: `sudo systemctl list-unit-files "leeq-epii@*.service"`

## Configuration Checklist

### Configuration Files
- [ ] Configuration files created in `/etc/leeq-epii/`
- [ ] Configuration syntax validated for each setup:
  ```bash
  sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq-epii/<setup>.json --validate
  ```
- [ ] Port assignments documented and non-conflicting
- [ ] Logging levels appropriate for environment (INFO for production)
- [ ] Resource limits set appropriately (max_workers, timeout)

### Environment Configuration
- [ ] Environment variables configured:
  - [ ] `LEEQ_EPII_CONFIG_DIR=/etc/leeq-epii`
  - [ ] `LEEQ_EPII_LOG_LEVEL=INFO`
  - [ ] `PYTHONPATH=/opt/leeq`
- [ ] System limits configured (file descriptors, memory)

### Setup-Specific Configuration
#### For Simulation Setups:
- [ ] Backend specified (numpy/qutip)
- [ ] Number of qubits defined
- [ ] Resource allocation appropriate for simulation complexity

#### For Hardware Setups:
- [ ] Hardware type specified correctly
- [ ] Hardware configuration files present and accessible
- [ ] Connection parameters validated
- [ ] Calibration data available

## Service Deployment Checklist

### Service Enablement
- [ ] Services enabled for desired setups:
  ```bash
  sudo systemctl enable leeq-epii@<setup_name>.service
  ```
- [ ] Auto-start behavior configured as required

### Service Startup
- [ ] Services started successfully:
  ```bash
  sudo systemctl start leeq-epii@<setup_name>.service
  ```
- [ ] Service status verified:
  ```bash
  sudo systemctl status leeq-epii@<setup_name>.service
  ```
- [ ] No error messages in startup logs

### Health Verification
- [ ] Health checks pass for all services:
  ```bash
  sudo -u leeq /opt/leeq/venv/bin/python -m leeq.epii.daemon --config /etc/leeq-epii/<setup>.json --health-check
  ```
- [ ] Services listening on configured ports:
  ```bash
  sudo ss -tlnp | grep :<port>
  ```
- [ ] gRPC connectivity verified:
  ```bash
  grpcurl -plaintext localhost:<port> epii.v1.ExperimentPlatformService/Ping
  ```

## Testing Checklist

### Basic Functionality Tests
- [ ] Ping service responds correctly
- [ ] GetCapabilities returns expected experiment list
- [ ] ListParameters returns setup parameters
- [ ] Configuration validation passes
- [ ] Service handles invalid requests gracefully

### Experiment Execution Tests
#### For Simulation Setups:
- [ ] Basic Rabi experiment executes successfully
- [ ] T1 experiment executes successfully
- [ ] Ramsey experiment executes successfully
- [ ] Results returned in expected format
- [ ] No memory leaks during repeated execution

#### For Hardware Setups:
- [ ] Hardware connection verified
- [ ] Basic experiment execution confirmed
- [ ] Parameter setting/getting works correctly
- [ ] Hardware safety checks functioning

### Load Testing
- [ ] Service handles concurrent requests
- [ ] Memory usage stable under load
- [ ] Response times within acceptable limits
- [ ] No resource exhaustion under sustained load

### Error Handling Tests
- [ ] Invalid experiment parameters rejected gracefully
- [ ] Service recovers from experiment failures
- [ ] Network interruptions handled correctly
- [ ] Configuration errors reported clearly

## Monitoring Setup Checklist

### Logging Configuration
- [ ] Log rotation configured: `/etc/logrotate.d/leeq-epii`
- [ ] Log levels appropriate for environment
- [ ] Structured logging format verified
- [ ] Log aggregation configured (if applicable)

### Health Monitoring
- [ ] Service health monitoring configured
- [ ] Resource usage monitoring active
- [ ] Performance metrics collection enabled
- [ ] Alert thresholds defined
- [ ] Notification mechanisms tested

### Backup Configuration
- [ ] Configuration backup strategy implemented
- [ ] Backup automation configured
- [ ] Recovery procedures documented
- [ ] Backup restoration tested

## Security Validation Checklist

### Access Control
- [ ] Service runs as non-privileged user (`leeq`)
- [ ] File permissions restrict access appropriately
- [ ] Network access limited to required ports
- [ ] API authentication implemented (if required)

### Communication Security
- [ ] TLS encryption enabled (for production)
- [ ] Certificate validity verified
- [ ] Cipher suites appropriate for security requirements
- [ ] Man-in-the-middle protections verified

### System Security
- [ ] No unnecessary privileges granted
- [ ] Security patches applied to system
- [ ] Audit logging enabled
- [ ] Security scanning completed (if required)

## Performance Optimization Checklist

### System Optimization
- [ ] CPU affinity configured (if needed)
- [ ] NUMA topology optimized (if applicable)
- [ ] Memory allocation tuned for workload
- [ ] I/O scheduling optimized

### Application Optimization
- [ ] gRPC message size limits configured
- [ ] Connection pooling parameters tuned
- [ ] Worker thread counts optimized
- [ ] Timeout values appropriate for experiments

### Resource Management
- [ ] Resource limits prevent system overload
- [ ] Memory usage patterns acceptable
- [ ] CPU utilization within expected ranges
- [ ] Network bandwidth sufficient for load

## Documentation Checklist

### Deployment Documentation
- [ ] Configuration parameters documented
- [ ] Service dependencies identified
- [ ] Deployment procedure recorded
- [ ] Rollback procedure defined

### Operational Documentation
- [ ] Service management procedures documented
- [ ] Monitoring and alerting procedures defined
- [ ] Troubleshooting guides created
- [ ] Maintenance schedules established

### User Documentation
- [ ] API documentation available and current
- [ ] Client connection examples provided
- [ ] Experiment parameter documentation complete
- [ ] Error code reference available

## Post-Deployment Validation

### Operational Validation
- [ ] Services running continuously for 24+ hours without issues
- [ ] Log analysis shows no unexpected errors
- [ ] Performance metrics within acceptable ranges
- [ ] Client connections successful from expected sources

### Integration Testing
- [ ] External orchestration systems connect successfully
- [ ] End-to-end experiment workflows complete
- [ ] Data serialization/deserialization working correctly
- [ ] Error propagation functioning as expected

### User Acceptance
- [ ] User training completed (if applicable)
- [ ] User feedback collected and addressed
- [ ] Known limitations documented
- [ ] Support procedures communicated

## Maintenance Preparation

### Ongoing Maintenance
- [ ] Maintenance schedule defined
- [ ] Update procedures documented
- [ ] Backup and recovery tested
- [ ] Contact information for support updated

### Change Management
- [ ] Change control procedures established
- [ ] Configuration management in place
- [ ] Version control for configurations implemented
- [ ] Testing procedures for changes defined

## Sign-off Checklist

### Technical Sign-off
- [ ] System administrator approves deployment
- [ ] Security team approves configuration
- [ ] Network team confirms connectivity
- [ ] Performance testing results acceptable

### Business Sign-off
- [ ] Operations team trained and ready
- [ ] Documentation reviewed and approved
- [ ] Support procedures in place
- [ ] Service level agreements defined

### Go-Live Authorization
- [ ] All checklist items completed
- [ ] Risk assessment completed and accepted
- [ ] Rollback plan confirmed and tested
- [ ] Go-live authorization obtained

---

## Emergency Procedures

### Service Failure Response
1. Check service status: `sudo systemctl status leeq-epii@<setup>.service`
2. Review logs: `sudo journalctl -u leeq-epii@<setup>.service -n 50`
3. Attempt restart: `sudo systemctl restart leeq-epii@<setup>.service`
4. If restart fails, check configuration validation
5. Escalate to development team if configuration is valid

### Performance Degradation Response
1. Check resource usage: `sudo systemctl status leeq-epii@<setup>.service`
2. Monitor active connections: `sudo ss -tlnp | grep :<port>`
3. Review recent logs for errors or warnings
4. Consider temporary worker limit reduction
5. Monitor for recovery, escalate if issues persist

### Security Incident Response
1. Immediately stop affected services
2. Preserve logs for analysis
3. Check for unauthorized access attempts
4. Review and update security configurations
5. Coordinate with security team for incident response

---

*Use this checklist to ensure comprehensive deployment validation and preparation for production operation of LeeQ EPII services.*