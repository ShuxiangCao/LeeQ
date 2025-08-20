# Installation

## Requirements

- Python 3.8 or higher
- Git (for installing dependencies from GitHub)

## Installation Methods

### Option 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/ShuxiangCao/LeeQ
```

### Option 2: Install from Source

1. Clone the repository:
```bash
git clone https://github.com/ShuxiangCao/LeeQ.git
cd LeeQ
```

2. Install using pip:
```bash
pip install -e .
```

### Option 3: Using Poetry

If you prefer to use Poetry for dependency management:

```bash
git clone https://github.com/ShuxiangCao/LeeQ.git
cd LeeQ
poetry install
```

## Dependencies

LeeQ requires the following key dependencies:

### Core Dependencies
- **numpy < 2.0.0**: Numerical computing library
- **scipy**: Scientific computing library 
- **matplotlib**: Plotting library
- **qutip**: Quantum simulation library
- **plotly**: Interactive plotting
- **dash**: Web-based dashboards for live monitoring

### External Dependencies (GitHub)
- **k_agents**: AI/ML experiment generation framework
- **MinimalLLM**: LLM integration utilities

### Integrated Modules
- **leeq.chronicle**: Experiment logging and persistence (integrated from labchronicle)

### Optional Dependencies
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **h5py**: HDF5 file format support
- **ipywidgets**: Jupyter notebook widgets

## Docker Installation

LeeQ provides a Docker image with all dependencies pre-installed:

```bash
docker run -p 8888:8888 -p 8050:8050 -v /path/to/local/folder:/home/jovyan/work ghcr.io/shuxiangcao/leeq:latest
```

This will:
- Start a Jupyter notebook server on port 8888
- Enable live plotting dashboard on port 8050
- Mount your local directory for persistent storage

## Environment Configuration

### Optional Environment Variables

Set these environment variables to customize LeeQ behavior:

```bash
# Directory for experiment logs (leeq.chronicle)
export LAB_CHRONICLE_LOG_DIR=/path/to/experiment/logs

# Directory for calibration logs
export LEEQ_CALIBRATION_LOG_PATH=/path/to/calibration/logs
```

If not set, LeeQ will create default directories in your working folder.

### Virtual Environment (Recommended)

It's recommended to install LeeQ in a virtual environment:

```bash
python -m venv leeq_env
source leeq_env/bin/activate  # On Windows: leeq_env\Scripts\activate
pip install git+https://github.com/ShuxiangCao/LeeQ
```

## Verification

To verify your installation works correctly:

```python
import leeq
print(f"LeeQ version: {leeq.__version__}")

# Test basic functionality
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
```

## Troubleshooting

### Common Issues

**Dependency Conflicts**: If you encounter version conflicts, try installing in a fresh virtual environment.

**GitHub Dependencies**: If installation of GitHub dependencies fails, ensure you have Git installed and can access GitHub.

**NumPy Version**: LeeQ requires numpy < 2.0.0 for compatibility. If you have numpy 2.x installed, downgrade with:
```bash
pip install "numpy<2.0.0"
```

### Getting Help

- Check the [GitHub Issues](https://github.com/ShuxiangCao/LeeQ/issues) for known problems
- Review the [troubleshooting guide](../development/troubleshooting.md)
- Join our community discussions for support