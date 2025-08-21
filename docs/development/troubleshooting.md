# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: Git-based dependencies fail to install
**Solution**: Ensure you have git installed and accessible from your command line. For GitHub dependencies, you may need to configure SSH keys or use HTTPS URLs.

#### Problem: LabChronicle module not found
**Solution**: As of the latest version, LabChronicle is now integrated as `leeq.chronicle`. Update your imports:
```python
# Old
from labchronicle import LoggableObject

# New
from leeq.chronicle import LoggableObject
```

### Runtime Issues

#### Problem: EPII daemon connection refused
**Solution**: 
1. Check if the EPII daemon is running: `systemctl status epii-daemon`
2. Verify the port is not blocked by firewall
3. Check the daemon logs: `journalctl -u epii-daemon -f`

#### Problem: Experiment execution timeout
**Solution**:
1. Increase timeout in experiment configuration
2. Check if hardware connections are stable
3. Verify that measurement devices are responsive

### Import Errors

#### Problem: Circular import detected
**Solution**: 
1. Use lazy imports where possible
2. Move shared utilities to a common module
3. Refactor to reduce module interdependencies

#### Problem: Type hints cause import errors
**Solution**: Use `TYPE_CHECKING` for forward references:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leeq.core.elements import TransmonElement
```

### Performance Issues

#### Problem: Slow experiment execution
**Solution**:
1. Enable parallel execution where possible
2. Optimize sweep parameters
3. Use batch operations for multiple qubits
4. Check network latency to hardware

#### Problem: High memory usage
**Solution**:
1. Limit data retention in Chronicle logger
2. Clear experiment results after processing
3. Use generators for large data sequences

### Data Issues

#### Problem: Chronicle log files growing too large
**Solution**:
1. Set `LAB_CHRONICLE_LOG_DIR` to a disk with sufficient space
2. Implement log rotation
3. Clean old log files periodically

#### Problem: Experiment results not saved
**Solution**:
1. Verify Chronicle is properly initialized
2. Check file permissions in log directory
3. Ensure decorators are properly applied

## Debug Tips

### Enable Verbose Logging
```python
from leeq.utils import setup_logging
logger = setup_logging(__name__, level='DEBUG')
```

### Check LeeQ Version
```python
import leeq
print(leeq.__version__)
```

### Verify Hardware Connection
```python
from leeq import setup
setup.status()  # Check all connections
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/ShuxiangCao/LeeQ/issues)
2. Review the API Documentation
3. Contact the development team

## See Also

- [Installation Guide](../getting-started/installation.md)
- [Experiments Guide](../guide/experiments.md)
- [EPII Troubleshooting](../epii/troubleshooting.md)