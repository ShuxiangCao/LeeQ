#!/usr/bin/env python3
"""
Simple daemon startup script for LeeQ EPII service.

This script provides an easy way to start the LeeQ daemon with proper
PID file management and health checking, as required by Phase 1 Task 1.3.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add LeeQ to path if running from scripts directory
script_dir = Path(__file__).parent
leeq_root = script_dir.parent
if leeq_root not in sys.path:
    sys.path.insert(0, str(leeq_root))

from leeq.epii.daemon import main as daemon_main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_config_file(config_name: str) -> Path:
    """Find configuration file in standard locations."""
    search_paths = [
        leeq_root / "configs" / "epii" / f"{config_name}.json",
        Path(f"/etc/leeq-epii/{config_name}.json"),
        Path(f"./{config_name}.json"),
        Path(f"./configs/epii/{config_name}.json")
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(f"Configuration '{config_name}' not found in any of: {[str(p) for p in search_paths]}")


def get_default_pid_dir() -> Path:
    """Get default PID file directory."""
    if os.getuid() == 0:  # Running as root
        return Path("/var/run/leeq-epii")
    else:  # Running as user
        return Path.home() / ".local" / "run" / "leeq-epii"


def main():
    """Main entry point for daemon startup script."""
    parser = argparse.ArgumentParser(
        description="Start LeeQ EPII daemon with simplified configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s simulation_2q                    # Start with simulation_2q config
  %(prog)s simulation_2q --port 50052      # Override port
  %(prog)s simulation_2q --validate-only   # Just validate config and exit
        """
    )
    
    parser.add_argument(
        'config',
        help='Configuration name (e.g., simulation_2q, hardware_lab1)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Override port number from config'
    )
    parser.add_argument(
        '--pid-dir',
        type=Path,
        default=get_default_pid_dir(),
        help='Directory for PID files (default: %(default)s)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Just validate configuration and exit'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Find configuration file
        config_file = find_config_file(args.config)
        logger.info(f"Using configuration: {config_file}")
        
        # Build daemon arguments
        daemon_args = [
            'leeq.epii.daemon',
            '--config', str(config_file)
        ]
        
        # Set PID file path
        port = args.port or 50051  # Default port
        pid_file_path = args.pid_dir / f"leeq-epii-{port}.pid"
        daemon_args.extend(['--pid-file', str(pid_file_path)])
        
        if args.port:
            daemon_args.extend(['--port', str(args.port)])
            
        if args.validate_only:
            daemon_args.append('--validate')
            
        if args.verbose:
            daemon_args.extend(['--log-level', 'DEBUG'])
        
        logger.info(f"Starting daemon with args: {' '.join(daemon_args[1:])}")
        
        # Replace current process with daemon
        sys.argv = daemon_args
        daemon_main()
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()