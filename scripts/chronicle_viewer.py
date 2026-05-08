#!/usr/bin/env python3
"""
Chronicle Log Viewer Entry Point

Standalone script to launch the Chronicle Log Viewer dashboard.
This script provides a convenient way to start the viewer from the command line
while the main implementation is properly organized within the LeeQ package.

Usage:
    python scripts/chronicle_viewer.py
    python scripts/chronicle_viewer.py --port 8080 --no-debug
    
The script imports and runs the dashboard from leeq.apps.chronicle_viewer.dashboard.
"""

import sys
from pathlib import Path

# Allow direct execution from a source checkout before editable install.
source_root = Path(__file__).parent.parent / "src"
if source_root.exists():
    sys.path.insert(0, str(source_root))

try:
    from leeq.apps.chronicle_viewer.dashboard import main
    
    if __name__ == "__main__":
        print("Chronicle Log Viewer - Starting from package location")
        print("Module: leeq.apps.chronicle_viewer.dashboard")
        print("=" * 50)
        main()
        
except ImportError as e:
    print(f"Error importing Chronicle Viewer from package: {e}")
    print(f"Make sure you have installed LeeQ in development mode:")
    print(f"  pip install -e .")
    print(f"Or run from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error starting Chronicle Viewer: {e}")
    sys.exit(1)
