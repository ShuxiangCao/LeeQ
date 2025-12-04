# Chronicle Viewer Package

Interactive visualization tools for LeeQ chronicle experiment data.

## Overview

The Chronicle Viewer package provides two complementary web-based dashboards for visualizing LeeQ experiments:

1. **Historical Viewer** (`dashboard.py`): Loads and analyzes completed chronicle HDF5 files
2. **Session Viewer** (`session_dashboard.py`): Real-time monitoring of active Chronicle sessions

Both viewers feature an intuitive three-panel interface with resizable layout and comprehensive experiment data display.

## Features

- **Interactive Dashboard**: Web-based interface using Dash and Plotly
- **Hierarchical Experiment Selection**: Tree view with timestamp ordering
- **Resizable Layout**: Adjustable left panel width for optimal viewing
- **Parent Experiment Selection**: Both parent and child experiments are selectable
- **Comprehensive Attributes Display**: Shows run arguments, kwargs, and experiment properties
- **Plot Generation**: Automatic discovery and execution of experiment browser functions
- **Error Handling**: Robust error messages and loading states

## Usage

### Historical Viewer (Post-Session Analysis)

#### From Package
```python
from leeq.chronicle.viewer import app, main

# Run with default settings
main()

# Or customize the app
app.run_server(debug=True, port=8050)
```

#### From Command Line
```bash
# Using the entry point script
python scripts/chronicle_viewer.py

# With custom port
python scripts/chronicle_viewer.py --port 8080

# Production mode
python scripts/chronicle_viewer.py --no-debug
```

#### From Package Import
```python
import leeq.chronicle.viewer.dashboard as viewer
viewer.main()
```

### Session Viewer (Real-Time Monitoring)

#### From Chronicle Instance
```python
from leeq.chronicle import Chronicle

# Get Chronicle singleton
chronicle = Chronicle()

# Launch session viewer
chronicle.launch_viewer()  # Opens at http://localhost:8051

# With custom settings
chronicle.launch_viewer(port=8055, debug=False)
```

#### Use Cases
- **Session Viewer**: Monitor experiments during active calibration sessions
- **Historical Viewer**: Analyze completed chronicle files after sessions

## Interface Layout

### Three-Panel Design:
1. **Left Panel (Resizable)**: File selection and experiment tree
2. **Center Panel**: Experiment info, plot controls, and plot display
3. **Right Panel**: Experiment attributes with run args/kwargs

### Key Components:
- **File Input**: Load chronicle HDF5 files
- **Experiment Tree**: Hierarchical view ordered by timestamp
- **Plot Controls**: Dynamic buttons for available browser functions
- **Attributes Panel**: Complete experiment metadata and parameters

## Requirements

- dash >= 2.0.0
- dash-bootstrap-components >= 1.0.0
- plotly >= 5.0.0
- leeq (with chronicle support)

## File Structure

```
leeq/chronicle/viewer/
├── __init__.py          # Package exports
├── dashboard.py         # Historical viewer for HDF5 files
├── session_dashboard.py # Live session viewer with polling
├── utils.py            # Helper functions (future)
└── README.md           # This file
```

## Testing

```bash
# Run viewer-specific tests
pytest tests/chronicle/viewer/ -v

# Run with coverage
pytest tests/chronicle/viewer/ --cov=leeq.chronicle.viewer
```

## Development

The viewer integrates seamlessly with the LeeQ package structure and follows established patterns for chronicle data handling.