#!/usr/bin/env python3
"""
Example usage of the Chronicle Log Viewer application.

This script demonstrates how to use the Chronicle Log Viewer to visualize
LeeQ experiment chronicle files. The viewer provides a web-based interface
for loading and exploring chronicle log data.

Requirements:
    - Python 3.7+
    - dash
    - dash-bootstrap-components
    - plotly
    - leeq (with chronicle support)

Installation:
    pip install dash dash-bootstrap-components plotly
    # Ensure leeq is installed with chronicle support

Basic Usage:
    1. Start the viewer:
       python chronicle_viewer.py
    
    2. Open your browser to http://localhost:8050
    
    3. Enter the full path to a chronicle log file (*.hdf5)
    
    4. Click on plot buttons to visualize different aspects of the experiment

Command Line Options:
    --host HOST     : Host to run server on (default: 0.0.0.0)
    --port PORT     : Port number (default: 8050)
    --no-debug      : Disable debug mode for production
    --help          : Show help message

Examples:
    # Start with default settings (debug mode on port 8050)
    python chronicle_viewer.py
    
    # Start on a different port
    python chronicle_viewer.py --port 8080
    
    # Start in production mode (no debug, no auto-reload)
    python chronicle_viewer.py --no-debug
    
    # Start on specific host and port
    python chronicle_viewer.py --host 127.0.0.1 --port 9000

Programmatic Usage:
    You can also import and use the viewer programmatically:
    
    from chronicle_viewer import app
    
    # Customize the app if needed
    app.title = "My Custom Chronicle Viewer"
    
    # Run the server
    app.run_server(debug=True, port=8050)

Troubleshooting:
    - File Not Found: Ensure the path is absolute and the file exists
    - Permission Denied: Check file permissions
    - Invalid Chronicle: Ensure the file is a valid LeeQ chronicle HDF5 file
    - No Plots Available: Some experiments may not have browser functions
    - Port Already in Use: Try a different port with --port option

For more information about LeeQ chronicle files, refer to the LeeQ documentation.
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nTo run the viewer, execute:")
    print("  python chronicle_viewer.py")
    print("\nFor help on command line options:")
    print("  python chronicle_viewer.py --help")