# LeeQ

[![main](https://github.com/ShuxiangCao/LeeQ/actions/workflows/test.yaml/badge.svg)](https://github.com/ShuxiangCao/LeeQ/actions/workflows/test.yaml)

LeeQ is a Python package for orchestrating quantum computing experiments with easy-to-use syntax, with a specific focus
on superconducting circuits-based quantum computing systems.

## Quick Start 

### Docker image
To use the Docker image, run the following command:

```bash
docker run -p 8888:8888 -p 5000:5000 -v /path/to/local/folder:/home/jovyan/ ghcr.io/shuxiangcao/leeq:latest
```

Then, open the browser and go to `http://localhost:8888` to access the Jupyter notebook. The port 5000 is used for the
live plotting server. To mount the local folder, replace `/path/to/local/folder` with the path to the local folder.

### Tutorial

See the [Quick Start Guide](docs/quick_start.md) for a 10-minute guide to launch LeeQ. Or the [One-hour full tutorial tutorial](docs/tutorial.md).

A notebook example and configuration example are prepared in the [notebook](notebooks) folder.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgement

This project was developed by Shuxiang Cao at the University of Oxford [Quantum Superconducting Circuits Research Group](https://leeklab.org/)  
