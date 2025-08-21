#!/bin/bash

# Exit on any error
set -e

echo "Running linting checks..."
echo "========================="

# Run flake8 for Python syntax errors and undefined names
echo "Running flake8 (critical errors)..."
flake8 leeq/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run flake8 with extended checks (warnings)
echo "Running flake8 (extended checks)..."
flake8 leeq/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run ruff for code quality checks
echo "Running ruff..."
ruff check leeq/

# Run mypy for type checking
echo "Running mypy..."
mypy leeq/ --ignore-missing-imports

echo "========================="
echo "All linting checks passed!"