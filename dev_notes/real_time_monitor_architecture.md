# LeeQ Real-Time Monitor Architecture

## Overview

The LeeQ real-time monitoring system provides live visibility into running quantum experiments through a web-based dashboard. It allows researchers to track experiment progress, view live data visualization, and monitor experiment parameters without interfering with the measurement process.

## Architecture Components

### 1. Dash Web Application (`leeq/experiments/plots/live_dash_app.py`)

The frontend is built using the Dash/Plotly framework and provides:

- **Web Server**: Runs on http://localhost:8050 by default
- **Auto-refresh**: Updates every 1 second via `dcc.Interval` component
- **UI Components**:
  - Progress bar showing percentage and step number
  - Live plot visualization area
  - Experiment details in JSON format
  - Bootstrap-styled responsive layout

Key callbacks:
- `update_experiment_details()`: Updates status JSON and progress bar
- `update_figure()`: Updates the live plot visualization

### 2. ExperimentManager (`leeq/experiments/experiments.py`)

A singleton class that serves as the central hub for experiment monitoring:

**Key Methods**:
- `start_live_monitor(**kwargs)`: Starts the Dash web server
- `stop_live_monitor()`: Stops the monitoring server
- `get_live_status()`: Aggregates status from experiment and setup
- `get_live_plots()`: Retrieves live visualization from active experiment

**Responsibilities**:
- Manages active experiment instance
- Bridges experiment data to the web interface
- Handles experiment registration and tracking

### 3. Data Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌─────────┐
│  Experiment │────▶│ ExperimentManager│────▶│  Dash App   │────▶│ Browser │
│   Instance  │     │   (Singleton)    │     │ (Web Server)│     │         │
└─────────────┘     └──────────────────┘     └─────────────┘     └─────────┘
       │                     │                       ▲
       │                     │                       │
       ▼                     ▼                   1s intervals
   ┌────────┐         ┌─────────────┐
   │ Engine │         │ Live Status │
   │        │         │   + Plots   │
   └────────┘         └─────────────┘
       │
       ▼
  ┌──────────┐
  │ Progress │
  │  Status  │
  └──────────┘
```

### 4. Integration Details

#### Starting the Monitor

```python
from leeq.experiments.experiments import ExperimentManager

# Start the monitoring server
ExperimentManager().start_live_monitor()

# Optional parameters:
# - jupyter_mode: 'external' (default)
# - host: '0.0.0.0' (for network access)
# - debug: True (default)
```

#### Status Collection (`get_live_status()`)

The method aggregates information from multiple sources:

1. **Experiment Details** (if active experiment exists):
   - Record details from labchronicle
   - Experiment arguments
   - Experiment-specific information

2. **Setup Status**:
   - Setup name
   - Shot number
   - Shot period

3. **Engine Status**:
   - Progress (0-1 float value)
   - Step number (tuple indicating current position)

#### Live Plotting (`get_live_plots()`)

- Checks if an active experiment exists
- Verifies the experiment has a `live_plots()` method
- Passes current step number to show partial data
- Returns empty Plotly figure if no data available

### 5. Engine Support (`leeq/core/engine/engine_base.py`)

The base engine class provides monitoring hooks:

- **Tracked Properties**:
  - `_progress`: Float between 0 and 1
  - `_step_no`: Tuple indicating current step position
  
- **Method**:
  - `get_live_status()`: Returns dict with step_no and progress

### 6. Experiment Implementation

Experiments can support live monitoring by implementing a `live_plots(step_no=None)` method:

```python
def live_plots(self, step_no=None) -> go.Figure:
    """
    Generate live plot during experiment execution.
    
    Parameters:
        step_no (tuple): Current step number from engine
        
    Returns:
        go.Figure: Plotly figure with current data
    """
    args = self._get_run_args_dict()
    data = np.squeeze(self.mp.result())
    
    fig = go.Figure()
    # Plot data up to current step
    fig.add_trace(go.Scatter(
        x=t[:step_no[0]],
        y=data[:step_no[0]],
        mode='lines',
        name='data'
    ))
    
    return fig
```

### 7. Example Usage

```python
from leeq.experiments.experiments import ExperimentManager
from leeq.experiments.builtin.basic.calibrations.rabi import RabiExperiment

# Start monitoring server
ExperimentManager().start_live_monitor()

# Run experiment - automatically tracked
experiment = RabiExperiment(
    qubit=q0,
    start=0,
    stop=5,
    step=0.1,
    update=True
)

# View live updates at http://localhost:8050
# Monitor shows:
# - Real-time progress bar
# - Live Rabi oscillation plot
# - Experiment parameters
# - Current step information

# Stop monitoring when done
ExperimentManager().stop_live_monitor()
```

## Supported Experiments

Experiments with `live_plots()` method implementation:
- `RabiExperiment` - Rabi oscillations
- `RamseyExperiment` - Ramsey fringes
- `T1Experiment` - T1 decay
- `T2Experiment` - T2 decay
- `QubitSpectroscopy` - Qubit frequency sweep
- `ResonatorSpectroscopy` - Resonator frequency sweep
- Various multi-qubit experiments

## Technical Details

- **Thread Safety**: ExperimentManager uses Singleton pattern
- **Error Handling**: Graceful degradation if experiment doesn't support live plots
- **Performance**: Minimal overhead - only active when monitor is running
- **Network Access**: Can be configured for remote access with `host='0.0.0.0'`

## Future Enhancements

Potential improvements to the monitoring system:
- WebSocket support for more efficient updates
- Multiple experiment tracking
- Historical data viewing
- Export functionality for live data
- Customizable update intervals
- Enhanced error reporting and diagnostics