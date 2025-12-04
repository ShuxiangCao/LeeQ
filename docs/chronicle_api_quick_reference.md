# Chronicle Session Viewer - Quick Reference

## Launch Methods

### Basic Launch
```python
from leeq.chronicle import Chronicle
chronicle = Chronicle()
chronicle.launch_viewer()  # Opens at http://localhost:8051
```

### Custom Port
```python
chronicle.launch_viewer(port=8055)
```

### Production Mode
```python
chronicle.launch_viewer(debug=False)
```

### Remote Access
```python
chronicle.launch_viewer(host='0.0.0.0', port=8051)
```

### Non-Blocking Launch
```python
import threading
viewer_thread = threading.Thread(
    target=chronicle.launch_viewer,
    kwargs={'port': 8051, 'debug': False},
    daemon=True
)
viewer_thread.start()
```

## Common Patterns

### In Calibration Script
```python
# Start of calibration
chronicle = Chronicle()
chronicle.launch_viewer()
print("Monitor at http://localhost:8051")

# Run experiments - they appear automatically
run_calibrations()
```

### In Jupyter Notebook
```python
# Cell 1: Launch viewer
chronicle = Chronicle()
chronicle.launch_viewer(debug=False)

# Cell 2: Run experiments
exp = QubitSpectroscopy(...)
# Appears in viewer within 5 seconds
```

### With Error Handling
```python
try:
    chronicle.launch_viewer(port=8051)
except OSError:
    chronicle.launch_viewer(port=8052)  # Try alternative
```

## Troubleshooting

### Port In Use
```python
# Try different port
chronicle.launch_viewer(port=8052)
```

### No Experiments Showing
```python
# Check environment
import os
os.environ['CHRONICLE_LOGGING'] = 'True'

# Verify chronicle is active
chronicle.get_current_session_entries()
```

### Manual Refresh
- Click "Refresh" button in viewer
- Or wait 5 seconds for auto-update

## Key Differences

| Feature | Session Viewer | Historical Viewer |
|---------|---------------|-------------------|
| **Launch** | `chronicle.launch_viewer()` | `python chronicle_viewer.py` |
| **Data** | Live session | HDF5 files |
| **Port** | 8051 | 8050 |
| **Updates** | Auto (5 sec) | Static |
| **Use** | During calibration | Post-analysis |

## Parameters

```python
chronicle.launch_viewer(
    debug=True,      # Debug mode with auto-reload
    port=8051,       # Server port
    host='127.0.0.1' # Host address
)
```

## Related Methods

```python
# Chronicle session management
chronicle.is_active()                    # Check if session active
chronicle.get_current_session_entries()  # Get session experiments
chronicle.start_new_session()            # Start new session
chronicle.end_current_session()          # End current session
```