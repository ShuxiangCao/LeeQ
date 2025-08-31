#!/bin/bash
# Docker entrypoint script for LeeQ container
# Supports both Jupyter notebook and EPII daemon modes

set -e

# Default to Jupyter mode for backward compatibility
MODE="${CONTAINER_MODE:-jupyter}"

if [ "$MODE" = "daemon" ]; then
    echo "Starting EPII daemon mode..."
    
    # Ensure we're in the right directory
    cd /home/jovyan/packages/LeeQ
    
    # Build command with optional arguments
    CMD="python scripts/epii_chronicle_daemon.py"
    
    # Add config if specified
    if [ -n "$EPII_CONFIG" ]; then
        CMD="$CMD --config $EPII_CONFIG"
    else
        CMD="$CMD --config /home/jovyan/config/example_daemon.json"
    fi
    
    # Add port if specified
    if [ -n "$EPII_PORT" ]; then
        CMD="$CMD --port $EPII_PORT"
    else
        CMD="$CMD --port 50051"
    fi
    
    # Add log level if specified
    if [ -n "$EPII_LOG_LEVEL" ]; then
        CMD="$CMD --log-level $EPII_LOG_LEVEL"
    fi
    
    # Add viewer flag if requested
    if [ "$EPII_LAUNCH_VIEWER" = "true" ]; then
        CMD="$CMD --launch-viewer"
    fi
    
    # Add viewer port if specified
    if [ -n "$EPII_VIEWER_PORT" ]; then
        CMD="$CMD --viewer-port $EPII_VIEWER_PORT"
    fi
    
    echo "Executing: $CMD"
    exec $CMD
else
    echo "Starting Jupyter notebook mode..."
    # Start Jupyter with no token for easier access in development
    exec jupyter notebook --NotebookApp.token='' --NotebookApp.password='' --ip=0.0.0.0 --allow-root
fi