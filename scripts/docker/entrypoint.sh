#!/bin/bash
# Docker entrypoint script for LeeQ container

set -e

echo "Starting Jupyter notebook mode..."
exec jupyter notebook --NotebookApp.token='' --NotebookApp.password='' --ip=0.0.0.0 --allow-root
