#!/bin/bash

set -euo pipefail

PYTHON=${PYTHON:-python}

echo "Running linting checks..."
echo "========================="

echo "Running Python compile check..."
"${PYTHON}" -m compileall -q src tests

echo "Running Ruff critical checks..."
"${PYTHON}" -m ruff check src tests --select=E9,F63,F7,F82 --statistics

echo "Running Ruff advisory scan..."
"${PYTHON}" -m ruff check src tests --statistics --exit-zero

echo "Running mypy advisory scan..."
if "${PYTHON}" -m mypy --version >/dev/null 2>&1; then
  "${PYTHON}" -m mypy src/leeq --ignore-missing-imports || true
else
  echo "mypy is not installed; skipping advisory type scan."
fi

echo "========================="
echo "Blocking linting checks passed."
