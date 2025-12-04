# Chronicle Log Tools

This directory contains tools for working with LeeQ chronicle logs.

## Files Created

### 1. `generate_chronicle_logs.py`
**Purpose**: Generates sample chronicle log files by running the same experiments as the TuneUpDemo.ipynb notebook.

**Usage**:
```bash
# Generate chronicle logs (requires leeq-nvidia environment)
python generate_chronicle_logs.py
```

**What it does**:
- Sets up a simulated 2-qubit system
- Runs various experiments:
  - Resonator spectroscopy
  - Rabi oscillations
  - Ramsey experiments (multiple configurations)
  - Pingpong calibration
  - DRAG calibration
  - T1 measurements
  - Spin echo experiments
- Saves all results to chronicle logs

### 2. `chronicle_viewer.py`
**Purpose**: Web dashboard for visualizing chronicle log files.

**Usage**:
```bash
# Start the dashboard
python chronicle_viewer.py

# Or with custom options
python chronicle_viewer.py --port 8051 --host 0.0.0.0
```

**Features**:
- Load chronicle log files by path
- Automatic experiment type detection
- Dynamic plot discovery using `experiment.get_browser_functions()`
- Interactive visualizations
- Error handling for corrupted files

### 3. Supporting Files
- `test_chronicle_viewer.py` - Comprehensive test suite (26 tests)
- `chronicle_viewer_example.py` - Usage examples and documentation
- `requirements_chronicle_viewer.txt` - Dependencies

## Workflow

1. **Generate Test Data**:
   ```bash
   python generate_chronicle_logs.py
   ```
   This creates chronicle logs in the `./log/` directory.

2. **View the Data**:
   ```bash
   python chronicle_viewer.py
   ```
   Navigate to http://localhost:8050 and enter the path to a chronicle log file.

3. **Find Log Files**:
   Chronicle logs are typically saved to:
   ```
   ./log/{username}/{YYYY-MM}/{YYYY-MM-DD}/{HH.MM.SS}{experiment_name}
   ```

## Example Chronicle Log Paths

After running `generate_chronicle_logs.py`, you'll have logs like:
- ResonatorSweepTransmissionWithExtraInitialLPB
- NormalisedRabi  
- SimpleRamseyMultilevel (multiple instances)
- AmpPingpongCalibrationSingleQubitMultilevel
- CrossAllXYDragMultiRunSingleQubitMultilevel
- SimpleT1
- SpinEchoMultiLevel

Each experiment object will have plot methods that can be automatically discovered and displayed in the dashboard.

## Requirements

- Python 3.10+ with LeeQ environment (leeq-nvidia)
- For the dashboard: dash, dash-bootstrap-components, plotly
- All experiments use LeeQ's built-in functions and simulated hardware

## Notes

The chronicle viewer leverages LeeQ's existing infrastructure:
- Uses `load_object()` to load experiments from HDF5 files
- Uses `experiment.get_browser_functions()` to discover available plots
- All plot methods are decorated with `@register_browser_function()`
- No custom parsing needed - works with any LeeQ experiment type