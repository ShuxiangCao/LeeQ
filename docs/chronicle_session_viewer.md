# Chronicle Session Viewer Documentation

## Overview

The Chronicle Session Viewer is a real-time monitoring dashboard for active Chronicle sessions in LeeQ. Unlike the historical viewer that loads completed chronicle files, the session viewer connects directly to the Chronicle singleton instance to display experiments as they complete during calibration workflows.

## Purpose and Benefits

### What is the Session Viewer?

The session viewer is a web-based dashboard that provides:
- **Real-time monitoring** of experiments during active calibration sessions
- **Automatic updates** every 5 seconds showing newly completed experiments
- **Direct integration** with Chronicle singleton - no file loading required
- **Familiar interface** using the same tree view and layout as the historical viewer

### How it Differs from the Historical Viewer

| Feature | Session Viewer | Historical Viewer |
|---------|---------------|-------------------|
| **Data Source** | Live Chronicle singleton | HDF5 chronicle files |
| **Use Case** | Active monitoring during calibration | Post-session analysis |
| **Launch Method** | `chronicle.launch_viewer()` | Standalone script or import |
| **Updates** | Auto-refresh every 5 seconds | Static file content |
| **Default Port** | 8051 | 8050 |

## Installation and Requirements

The session viewer is included with the LeeQ package. No additional installation is required beyond the standard LeeQ dependencies:

```bash
# Standard LeeQ installation includes viewer
pip install -e .
```

Required packages (automatically installed with LeeQ):
- dash >= 2.0.0
- dash-bootstrap-components >= 1.0.0
- plotly >= 5.0.0
- h5py (for chronicle data)

## Usage Patterns and Examples

### Basic Usage - Launching the Viewer

The simplest way to launch the session viewer is through the Chronicle singleton:

```python
from leeq.chronicle import Chronicle

# Get Chronicle singleton instance
chronicle = Chronicle()

# Launch the viewer dashboard
chronicle.launch_viewer()
```

This opens a web browser to `http://localhost:8051` showing the current session.

### Example 1: Monitoring During Calibration Script

```python
"""calibration_session.py - Example calibration workflow with monitoring"""

from leeq.chronicle import Chronicle
from leeq.setups import setup_from_config
import time

def run_calibration():
    # Initialize setup
    setup = setup_from_config("config.json")
    
    # Launch viewer for monitoring
    chronicle = Chronicle()
    chronicle.launch_viewer(debug=False)  # Run in production mode
    
    print("Chronicle viewer launched at http://localhost:8051")
    print("Starting calibration experiments...")
    
    # Run experiments - they'll appear in viewer as they complete
    for qubit in setup.get_qubits():
        # Resonator spectroscopy
        exp = ResonatorSpectroscopy(qubit=qubit)
        print(f"Completed resonator spectroscopy for {qubit.name}")
        time.sleep(2)  # Viewer will update within 5 seconds
        
        # Qubit spectroscopy  
        exp = QubitSpectroscopy(qubit=qubit)
        print(f"Completed qubit spectroscopy for {qubit.name}")
        time.sleep(2)
    
    print("Check the viewer to see all completed experiments!")
    input("Press Enter to continue...")

if __name__ == "__main__":
    run_calibration()
```

### Example 2: Interactive Jupyter Notebook Usage

```python
# Cell 1 - Setup and launch viewer
from leeq.chronicle import Chronicle
from leeq.setups import setup_from_config

# Initialize
setup = setup_from_config("my_setup.json")
chronicle = Chronicle()

# Launch viewer in background
chronicle.launch_viewer(port=8051, debug=True)
print("Viewer running at http://localhost:8051")
```

```python
# Cell 2 - Run experiments (viewer updates automatically)
qubit = setup.get_qubit("Q1")

# These experiments will appear in the viewer as they complete
exp1 = ResonatorSpectroscopy(qubit=qubit, start=6000, stop=6200, step=2)
exp2 = QubitSpectroscopy(qubit=qubit, start=4900, stop=5100, step=1)
```

```python
# Cell 3 - Check specific experiments in viewer
print("Open http://localhost:8051 to see your experiments")
print("The viewer refreshes every 5 seconds automatically")
```

### Example 3: Custom Port and Configuration

```python
from leeq.chronicle import Chronicle

chronicle = Chronicle()

# Launch with custom settings
chronicle.launch_viewer(
    port=8055,           # Custom port to avoid conflicts
    debug=False,         # Production mode (no auto-reload)
    host='0.0.0.0'      # Allow external connections
)
```

### Example 4: Multi-Session Workflow

```python
"""multi_session.py - Running multiple calibration sessions"""

from leeq.chronicle import Chronicle
import subprocess
import time

def monitor_sessions():
    chronicle = Chronicle()
    
    # Launch viewer once
    chronicle.launch_viewer(port=8051)
    
    # Run multiple calibration sessions
    sessions = ["morning_cal.py", "afternoon_cal.py", "evening_cal.py"]
    
    for session_script in sessions:
        print(f"Starting {session_script}")
        subprocess.run(["python", session_script])
        
        print(f"Completed {session_script}")
        print("Check viewer for results at http://localhost:8051")
        time.sleep(5)  # Allow viewer to update
        
        response = input("Continue to next session? (y/n): ")
        if response.lower() != 'y':
            break

if __name__ == "__main__":
    monitor_sessions()
```

## Configuration Options

The `launch_viewer()` method accepts several configuration parameters:

```python
chronicle.launch_viewer(
    debug=True,          # Enable debug mode with auto-reload (default: True)
    port=8051,          # Port number for the server (default: 8051)
    host='127.0.0.1',   # Host address (default: localhost only)
    **kwargs            # Additional Dash server arguments
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug` | bool | True | Enable debug mode with auto-reload and error messages |
| `port` | int | 8051 | Port number for the web server |
| `host` | str | '127.0.0.1' | Host address ('0.0.0.0' for external access) |
| `dev_tools_hot_reload` | bool | False | Enable hot reload in development |
| `use_reloader` | bool | True (if debug) | Auto-reload on code changes |

### Port Selection Guidelines

- **8051**: Default for session viewer (recommended)
- **8050**: Reserved for historical chronicle viewer
- **8000-8010**: Common range for development
- **Custom ports**: Use if running multiple viewers simultaneously

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: "Address already in use" Error

**Symptom:** 
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```python
# Use a different port
chronicle.launch_viewer(port=8052)

# Or kill the existing process
# In terminal: lsof -i :8051
# Then: kill -9 <PID>
```

#### Issue 2: Viewer Shows "No experiments in current session"

**Symptom:** Empty viewer despite running experiments

**Possible Causes and Solutions:**

1. **Chronicle not initialized properly**
```python
# Ensure Chronicle singleton is initialized
from leeq.chronicle import Chronicle
chronicle = Chronicle()
print(chronicle.is_active())  # Should be True
```

2. **Experiments not being logged**
```python
# Check chronicle logging is enabled
import os
os.environ['CHRONICLE_LOGGING'] = 'True'
```

3. **Session not active**
```python
# Start a new session if needed
chronicle.start_new_session()
```

#### Issue 3: Viewer Not Updating Automatically

**Symptom:** New experiments don't appear without manual refresh

**Solution:**
```python
# Check browser console for errors
# Ensure interval component is working
# Try manual refresh button as fallback
# Check network tab in browser dev tools
```

#### Issue 4: Cannot Connect to Viewer

**Symptom:** Browser shows "Unable to connect"

**Solutions:**

1. **Check if server is running**
```bash
# Check if process is listening on port
lsof -i :8051
```

2. **Firewall or network issues**
```python
# Try using localhost explicitly
# Open http://127.0.0.1:8051 instead of localhost:8051
```

3. **Remote access needed**
```python
# Allow external connections
chronicle.launch_viewer(host='0.0.0.0', port=8051)
```

#### Issue 5: Slow Performance with Many Experiments

**Symptom:** Dashboard becomes sluggish with 100+ experiments

**Solutions:**

1. **Increase polling interval**
```python
# Modify in session_dashboard.py if needed
# Change interval from 5000ms to 10000ms
```

2. **Clear old sessions**
```python
# Start fresh session
chronicle.end_current_session()
chronicle.start_new_session()
```

3. **Use filtering in viewer**
   - Focus on specific experiment types
   - Collapse unused tree branches

### Debug Mode Tips

When running in debug mode (`debug=True`):
- Server auto-reloads on code changes
- Detailed error messages in browser
- Console output shows all callbacks
- Performance may be slower

For production use:
```python
chronicle.launch_viewer(debug=False)
```

## API Reference

### Chronicle.launch_viewer()

Launch the session viewer dashboard for monitoring active experiments.

```python
def launch_viewer(self, **kwargs) -> None:
    """
    Launch chronicle viewer dashboard for current session.
    
    This method launches a web-based dashboard for monitoring experiments
    in the active Chronicle session. The dashboard polls every 5 seconds
    to display newly completed experiments.
    
    Args:
        debug (bool): Whether to run in debug mode (default: True)
        port (int): Port to run the server on (default: 8051)
        host (str): Host address (default: '127.0.0.1')
        **kwargs: Additional arguments passed to the Dash server
    
    Returns:
        None (launches server in current thread)
    
    Raises:
        RuntimeError: If Chronicle is not properly initialized
        OSError: If port is already in use
    
    Example:
        >>> from leeq.chronicle import Chronicle
        >>> chronicle = Chronicle()
        >>> chronicle.launch_viewer(port=8051, debug=False)
    """
```

### Session Dashboard Callbacks

The session viewer implements several Dash callbacks:

#### update_session_view()
Updates the experiment tree and display panels every 5 seconds or on manual refresh.

#### toggle_tree_node()  
Handles expanding/collapsing tree nodes for navigation.

#### display_experiment_info()
Shows experiment details when an experiment is selected.

#### generate_plot()
Creates visualizations when plot buttons are clicked.

### Chronicle Session Methods

Related Chronicle methods for session management:

```python
# Check if session is active
chronicle.is_active() -> bool

# Get current session entries
chronicle.get_current_session_entries() -> List[RecordEntry]

# Start new session
chronicle.start_new_session() -> None

# End current session  
chronicle.end_current_session() -> None
```

## Best Practices

### 1. Launch Early in Workflow
Start the viewer at the beginning of calibration sessions:
```python
def main():
    chronicle = Chronicle()
    chronicle.launch_viewer()  # Launch first
    
    # Then run experiments
    run_calibrations()
```

### 2. Use Appropriate Ports
Keep standard port assignments:
- 8051 for session viewer
- 8050 for historical viewer
- Custom ports for multiple instances

### 3. Handle Errors Gracefully
```python
try:
    chronicle.launch_viewer(port=8051)
except OSError as e:
    print(f"Port 8051 in use, trying 8052...")
    chronicle.launch_viewer(port=8052)
```

### 4. Production vs Development
```python
# Development - with debug info
chronicle.launch_viewer(debug=True)

# Production - optimized performance  
chronicle.launch_viewer(debug=False)
```

### 5. Document Viewer URLs
```python
# Always inform users where to find the viewer
port = 8051
chronicle.launch_viewer(port=port)
print(f"Chronicle viewer running at http://localhost:{port}")
print("Open this URL in your browser to monitor experiments")
```

## Integration with LeeQ Workflow

The session viewer integrates seamlessly with the LeeQ calibration workflow:

1. **Setup Phase**: Initialize hardware and Chronicle
2. **Launch Viewer**: Start monitoring dashboard
3. **Run Experiments**: Execute calibration routines
4. **Monitor Progress**: View completed experiments in real-time
5. **Analyze Results**: Use historical viewer for detailed analysis later

```python
# Complete integrated workflow
from leeq.chronicle import Chronicle
from leeq.setups import setup_from_config

# 1. Setup
setup = setup_from_config("config.json")
chronicle = Chronicle()

# 2. Launch monitoring
chronicle.launch_viewer()

# 3. Run calibrations
calibrate_resonators(setup)
calibrate_qubits(setup)

# 4. Monitor in browser at http://localhost:8051

# 5. Later analysis with historical viewer
# python scripts/chronicle_viewer.py
```

## Additional Resources

- [Chronicle API Documentation](../api/chronicle.html)
- [LeeQ Experiments Guide](../experiments/index.html)
- [Historical Viewer Documentation](./chronicle_viewer.md)
- [LeeQ Setup Configuration](../setups/configuration.html)

## Support

For issues or questions:
1. Check this troubleshooting guide
2. Review the [LeeQ GitHub Issues](https://github.com/leeq-framework/leeq/issues)
3. Contact the LeeQ development team