#!/usr/bin/env python3
"""
Daemon control utility for LeeQ EPII service.

This script provides PID file management and daemon control functionality
as required by Phase 1 Task 1.3.
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add LeeQ to path if running from scripts directory
script_dir = Path(__file__).parent
leeq_root = script_dir.parent
if leeq_root not in sys.path:
    sys.path.insert(0, str(leeq_root))

from leeq.epii.daemon import PIDFileManager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DaemonController:
    """Controls daemon processes using PID files."""
    
    def __init__(self, pid_dir: Path):
        """Initialize daemon controller.
        
        Args:
            pid_dir: Directory containing PID files
        """
        self.pid_dir = pid_dir
        self.pid_dir.mkdir(parents=True, exist_ok=True)
    
    def get_running_daemons(self) -> Dict[str, Dict]:
        """Get information about running daemon instances.
        
        Returns:
            Dictionary mapping instance names to daemon info
        """
        running_daemons = {}
        
        for pid_file in self.pid_dir.glob("*.pid"):
            instance_name = pid_file.stem
            
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is still running
                try:
                    os.kill(pid, 0)  # Signal 0 doesn't actually send a signal
                    
                    # Get process info
                    proc_info = self._get_process_info(pid)
                    
                    running_daemons[instance_name] = {
                        'pid': pid,
                        'pid_file': str(pid_file),
                        'status': 'running',
                        'cmdline': proc_info.get('cmdline', 'unknown'),
                        'start_time': proc_info.get('start_time', 'unknown'),
                        'memory_mb': proc_info.get('memory_mb', 'unknown')
                    }
                    
                except OSError:
                    # Process doesn't exist, mark as stale
                    running_daemons[instance_name] = {
                        'pid': pid,
                        'pid_file': str(pid_file),
                        'status': 'stale',
                        'error': 'Process not found'
                    }
                    
            except (ValueError, IOError) as e:
                logger.warning(f"Invalid PID file {pid_file}: {e}")
                running_daemons[instance_name] = {
                    'pid': None,
                    'pid_file': str(pid_file),
                    'status': 'invalid',
                    'error': str(e)
                }
        
        return running_daemons
    
    def stop_daemon(self, instance_name: str, force: bool = False) -> bool:
        """Stop a daemon instance.
        
        Args:
            instance_name: Name of daemon instance to stop
            force: Use SIGKILL instead of SIGTERM
            
        Returns:
            True if daemon was stopped successfully
        """
        pid_file = self.pid_dir / f"{instance_name}.pid"
        
        if not pid_file.exists():
            logger.error(f"PID file not found: {pid_file}")
            return False
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            try:
                os.kill(pid, 0)
            except OSError:
                logger.warning(f"Process {pid} for {instance_name} not found, removing stale PID file")
                pid_file.unlink()
                return True
            
            # Send termination signal
            sig = signal.SIGKILL if force else signal.SIGTERM
            signal_name = "SIGKILL" if force else "SIGTERM"
            
            logger.info(f"Sending {signal_name} to {instance_name} (PID {pid})")
            os.kill(pid, sig)
            
            # Wait for process to exit
            max_wait = 30 if not force else 5
            for _ in range(max_wait * 10):  # Check every 0.1 seconds
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    # Process has exited
                    break
            else:
                if not force:
                    logger.warning(f"Process {pid} didn't exit after {max_wait}s, trying SIGKILL")
                    return self.stop_daemon(instance_name, force=True)
                else:
                    logger.error(f"Process {pid} didn't exit after SIGKILL")
                    return False
            
            # Clean up PID file
            if pid_file.exists():
                pid_file.unlink()
                logger.info(f"Removed PID file {pid_file}")
            
            logger.info(f"Successfully stopped {instance_name}")
            return True
            
        except (ValueError, IOError) as e:
            logger.error(f"Error stopping {instance_name}: {e}")
            return False
    
    def stop_all_daemons(self, force: bool = False) -> List[str]:
        """Stop all running daemon instances.
        
        Args:
            force: Use SIGKILL instead of SIGTERM
            
        Returns:
            List of successfully stopped daemon names
        """
        running_daemons = self.get_running_daemons()
        active_daemons = [name for name, info in running_daemons.items() if info['status'] == 'running']
        
        if not active_daemons:
            logger.info("No active daemons to stop")
            return []
        
        stopped_daemons = []
        
        for daemon_name in active_daemons:
            if self.stop_daemon(daemon_name, force):
                stopped_daemons.append(daemon_name)
        
        return stopped_daemons
    
    def cleanup_stale_pids(self) -> List[str]:
        """Clean up stale PID files.
        
        Returns:
            List of cleaned up PID files
        """
        running_daemons = self.get_running_daemons()
        stale_pids = []
        
        for instance_name, info in running_daemons.items():
            if info['status'] in ['stale', 'invalid']:
                pid_file = Path(info['pid_file'])
                try:
                    pid_file.unlink()
                    stale_pids.append(instance_name)
                    logger.info(f"Cleaned up stale PID file for {instance_name}")
                except Exception as e:
                    logger.error(f"Failed to clean up PID file for {instance_name}: {e}")
        
        return stale_pids
    
    def send_signal_to_daemon(self, instance_name: str, signal_num: int) -> bool:
        """Send a signal to a daemon instance.
        
        Args:
            instance_name: Name of daemon instance
            signal_num: Signal number to send
            
        Returns:
            True if signal was sent successfully
        """
        pid_file = self.pid_dir / f"{instance_name}.pid"
        
        if not pid_file.exists():
            logger.error(f"PID file not found: {pid_file}")
            return False
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            os.kill(pid, signal_num)
            logger.info(f"Sent signal {signal_num} to {instance_name} (PID {pid})")
            return True
            
        except (ValueError, IOError, OSError) as e:
            logger.error(f"Failed to send signal to {instance_name}: {e}")
            return False
    
    def _get_process_info(self, pid: int) -> Dict:
        """Get information about a process.
        
        Args:
            pid: Process ID
            
        Returns:
            Dictionary with process information
        """
        info = {}
        
        try:
            # Try to read /proc/PID/stat for basic info
            stat_file = Path(f"/proc/{pid}/stat")
            if stat_file.exists():
                with open(stat_file, 'r') as f:
                    stat_data = f.read().strip().split()
                    
                # Get memory usage (RSS in pages, convert to MB)
                rss_pages = int(stat_data[23]) if len(stat_data) > 23 else 0
                page_size = os.sysconf(os.sysconf_names.get('SC_PAGE_SIZE', 'SC_PAGESIZE'))
                info['memory_mb'] = (rss_pages * page_size) / (1024 * 1024)
            
            # Try to read command line
            cmdline_file = Path(f"/proc/{pid}/cmdline")
            if cmdline_file.exists():
                with open(cmdline_file, 'rb') as f:
                    cmdline_data = f.read()
                    # Command line arguments are null-separated
                    cmdline = cmdline_data.replace(b'\x00', b' ').decode('utf-8', errors='ignore').strip()
                    info['cmdline'] = cmdline
            
        except Exception as e:
            logger.debug(f"Failed to get process info for PID {pid}: {e}")
        
        return info


def print_daemon_status(daemons: Dict, use_colors: bool = True) -> None:
    """Print daemon status in a readable format."""
    if not daemons:
        print("No daemon instances found")
        return
    
    colors = {
        'running': '\033[92m',  # Green
        'stale': '\033[93m',    # Yellow
        'invalid': '\033[91m'   # Red
    }
    end_color = '\033[0m'
    
    print(f"\n{'Instance':<20} {'Status':<10} {'PID':<10} {'Memory (MB)':<12} {'Command'}")
    print("-" * 80)
    
    for instance_name, info in daemons.items():
        status = info['status']
        
        if use_colors and status in colors:
            status_str = f"{colors[status]}{status}{end_color}"
        else:
            status_str = status
        
        pid_str = str(info['pid']) if info['pid'] else 'N/A'
        memory_str = f"{info.get('memory_mb', 0):.1f}" if isinstance(info.get('memory_mb'), (int, float)) else 'N/A'
        cmdline = info.get('cmdline', info.get('error', 'N/A'))
        
        # Truncate long command lines
        if len(cmdline) > 40:
            cmdline = cmdline[:37] + "..."
        
        print(f"{instance_name:<20} {status_str:<10} {pid_str:<10} {memory_str:<12} {cmdline}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Control daemon processes using PID files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                       # Show status of all daemons
  %(prog)s stop simulation_2q           # Stop specific daemon
  %(prog)s stop-all                     # Stop all daemons
  %(prog)s cleanup                      # Clean up stale PID files
  %(prog)s signal simulation_2q SIGUSR1 # Send signal to daemon
        """
    )
    
    parser.add_argument(
        'action',
        choices=['status', 'stop', 'stop-all', 'cleanup', 'signal'],
        help='Action to perform'
    )
    parser.add_argument(
        'instance',
        nargs='?',
        help='Daemon instance name (required for stop and signal actions)'
    )
    parser.add_argument(
        'signal_name',
        nargs='?',
        help='Signal name (required for signal action)'
    )
    parser.add_argument(
        '--pid-dir',
        type=Path,
        default=Path.home() / ".local" / "run" / "leeq-epii",
        help='Directory containing PID files'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Use SIGKILL instead of SIGTERM for stop operations'
    )
    parser.add_argument(
        '--no-colors',
        action='store_true',
        help='Disable colored output'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for required arguments
    if args.action in ['stop', 'signal'] and not args.instance:
        parser.error(f"Instance name is required for {args.action} action")
    
    if args.action == 'signal' and not args.signal_name:
        parser.error("Signal name is required for signal action")
    
    # Initialize controller
    controller = DaemonController(args.pid_dir)
    
    try:
        if args.action == 'status':
            daemons = controller.get_running_daemons()
            print_daemon_status(daemons, not args.no_colors)
            
        elif args.action == 'stop':
            success = controller.stop_daemon(args.instance, args.force)
            sys.exit(0 if success else 1)
            
        elif args.action == 'stop-all':
            stopped = controller.stop_all_daemons(args.force)
            if stopped:
                print(f"Stopped {len(stopped)} daemon(s): {', '.join(stopped)}")
            else:
                print("No daemons were stopped")
                
        elif args.action == 'cleanup':
            cleaned = controller.cleanup_stale_pids()
            if cleaned:
                print(f"Cleaned up {len(cleaned)} stale PID file(s): {', '.join(cleaned)}")
            else:
                print("No stale PID files found")
                
        elif args.action == 'signal':
            # Convert signal name to number
            signal_name = args.signal_name.upper()
            if not signal_name.startswith('SIG'):
                signal_name = f'SIG{signal_name}'
            
            try:
                signal_num = getattr(signal, signal_name)
            except AttributeError:
                logger.error(f"Unknown signal: {signal_name}")
                sys.exit(1)
            
            success = controller.send_signal_to_daemon(args.instance, signal_num)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()