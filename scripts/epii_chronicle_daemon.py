#!/usr/bin/env python3
"""
EPII Daemon with Chronicle Integration

This script starts an EPII daemon with a simulated quantum setup and Chronicle
session tracking for live experiment monitoring.

Usage:
    python epii_chronicle_daemon.py [--config CONFIG] [--launch-viewer]
"""

import argparse
import sys
import os
import threading
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from leeq.epii.daemon import EPIIDaemon, load_config, setup_logging
from leeq.chronicle import Chronicle


def start_chronicle_session(config):
    """Start Chronicle session for experiment tracking."""
    print("Initializing Chronicle session...")
    chronicle = Chronicle()
    
    session_name = config.get("chronicle", {}).get("session_name", "epii_session")
    chronicle.start_log(session_name)
    print(f"✓ Chronicle session started: {session_name}")
    
    return chronicle


def launch_chronicle_viewer(chronicle, port=8051):
    """Launch Chronicle viewer in a separate thread."""
    def viewer_thread():
        try:
            print(f"\nLaunching Chronicle viewer at http://localhost:{port}")
            print("  You can monitor experiments in real-time")
            chronicle.launch_viewer(port=port, debug=False)
        except Exception as e:
            print(f"Warning: Could not launch Chronicle viewer: {e}")
            print("  You can launch it manually with: chronicle.launch_viewer()")
    
    thread = threading.Thread(target=viewer_thread, daemon=True)
    thread.start()
    time.sleep(2)  # Give viewer time to start


def main():
    """Main entry point for EPII daemon with Chronicle."""
    parser = argparse.ArgumentParser(
        description="LeeQ EPII daemon with Chronicle integration"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/epii/simulation_chronicle.json",
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port for EPII gRPC server (default: 50051)"
    )
    
    parser.add_argument(
        "--launch-viewer",
        action="store_true",
        help="Launch Chronicle viewer dashboard"
    )
    
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=8051,
        help="Port for Chronicle viewer (default: 8051)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        print(f"Loading configuration from: {config_path}")
        config = load_config(str(config_path))
        
        # Override port if specified
        if args.port:
            config["port"] = args.port
        
        # Start Chronicle session if enabled
        chronicle = None
        if config.get("chronicle", {}).get("enabled", False):
            chronicle = start_chronicle_session(config)
            
            # Launch viewer if requested
            if args.launch_viewer or config.get("chronicle", {}).get("auto_launch", False):
                viewer_port = args.viewer_port or config.get("chronicle", {}).get("viewer_port", 8051)
                launch_chronicle_viewer(chronicle, viewer_port)
        
        # Print startup information
        print("\n" + "="*60)
        print("EPII DAEMON WITH CHRONICLE")
        print("="*60)
        print(f"Configuration: {config_path.name}")
        print(f"Setup Type: {config.get('setup_type', 'simulation')}")
        print(f"Number of Qubits: {config.get('num_qubits', 2)}")
        print(f"EPII Port: {config.get('port', 50051)}")
        if chronicle:
            print(f"Chronicle Session: Active")
            if args.launch_viewer:
                print(f"Chronicle Viewer: http://localhost:{viewer_port}")
        print("="*60)
        
        # Create and start daemon
        print("\nStarting EPII daemon...")
        daemon = EPIIDaemon(
            config=config,
            port=config.get("port", 50051)
        )
        
        print("✓ EPII daemon ready")
        print("\nPress Ctrl+C to stop the daemon\n")
        
        # Start the daemon (blocks until shutdown)
        daemon.start()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if chronicle and chronicle.is_recording():
            chronicle.end_log()
            print("✓ Chronicle session ended")
        print("✓ Daemon stopped")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()