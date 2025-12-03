"""
Chronicle Viewer Package

This package provides visualization tools for LeeQ chronicle data,
including interactive dashboards and data exploration utilities.

Main Components:
    - dashboard: Interactive Dash-based web viewer for chronicle files
    - utils: Helper functions for data formatting and processing

Usage:
    from leeq.chronicle.viewer import dashboard

    # Run the dashboard
    dashboard.main()

    # Or use the app object directly
    app = dashboard.app
    app.run_server(debug=True)
"""

from .dashboard import app, main

__all__ = ['app', 'main']
