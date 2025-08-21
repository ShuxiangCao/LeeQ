# LeeQ

[![main](https://github.com/ShuxiangCao/LeeQ/actions/workflows/test.yaml/badge.svg)](https://github.com/ShuxiangCao/LeeQ/actions/workflows/test.yaml) [![Docker Image](https://github.com/ShuxiangCao/LeeQ/actions/workflows/docker_image.yaml/badge.svg)](https://github.com/ShuxiangCao/LeeQ/actions/workflows/docker_image.yaml)

LeeQ is a Python package for orchestrating quantum computing experiments with easy-to-use syntax, with a specific focus
on superconducting circuits-based quantum computing systems.

## Quick Start 

### Docker image
To use the Docker image, run the following command:

```bash
docker run -p 8888:8888 -p 8050:8050 -v /path/to/local/folder:/home/jovyan/work ghcr.io/shuxiangcao/leeq:latest
```

Then, open the browser and go to `http://localhost:8888` to access the Jupyter notebook. The port `8050` is used for the
live plotting server. To mount the local folder, replace `/path/to/local/folder` with the path to the local folder.

## Documentation

### Getting Started
- **[Quick Start Guide](docs/quick_start.md)** - Get your first quantum experiment running in 10 minutes
- **[Comprehensive Tutorial](docs/tutorial.md)** - In-depth tutorial covering all major LeeQ features
- **[Installation Guide](docs/getting-started/installation.md)** - Detailed installation instructions

### User Guides
- **[Core Concepts](docs/guide/concepts.md)** - Understand LeeQ's architecture and design principles
- **[Experiments Guide](docs/guide/experiments.md)** - Learn about available experiment types
- **[Calibrations Guide](docs/guide/calibrations.md)** - Master calibration procedures

### API Documentation
- **[Core API](docs/api/core/base.md)** - Base classes and core functionality
- **[Experiments API](docs/api/experiments/builtin.md)** - Built-in experiment implementations
- **[Theory API](docs/api/theory/simulation.md)** - Simulation and theory modules

### Advanced Topics
- **[EPII Service](docs/epii/deployment-guide.md)** - Deploy LeeQ as a gRPC service
- **[Development Guide](docs/development/contributing.md)** - Contribute to LeeQ development
- **[Architecture Overview](docs/development/architecture.md)** - Deep dive into LeeQ's design

### Examples and Resources
Configuration examples and sample experiments are available in the [notebooks](notebooks) folder.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgement

This project was developed by Shuxiang Cao at the University of Oxford [Quantum Superconducting Circuits Research Group](https://leeklab.org/)  
