#!/bin/bash
# Docker entrypoint script for LeeQ container

set -euo pipefail

echo "Starting Jupyter notebook mode..."

NOTEBOOK_ARGS=(
  "--ip=${LEEQ_JUPYTER_IP:-0.0.0.0}"
  "--allow-root"
)

if [[ "${LEEQ_DISABLE_JUPYTER_AUTH:-false}" == "true" ]]; then
  NOTEBOOK_ARGS+=("--NotebookApp.token=" "--NotebookApp.password=")
elif [[ -n "${LEEQ_JUPYTER_TOKEN:-}" ]]; then
  NOTEBOOK_ARGS+=("--NotebookApp.token=${LEEQ_JUPYTER_TOKEN}")
fi

exec jupyter notebook "${NOTEBOOK_ARGS[@]}"
