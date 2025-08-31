"""
EPII daemon implementation for LeeQ.

This module provides the gRPC server daemon that hosts the EPII service.
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, Optional

import grpc

from .proto import epii_pb2_grpc
from .service import ExperimentPlatformService
from .utils import DiagnosticTool, create_debug_interceptor, format_diagnostic_report

logger = logging.getLogger(__name__)


class PIDFileManager:
    """
    Manages PID file creation, validation, and cleanup for daemon processes.
    """

    def __init__(self, pid_file: Path):
        """
        Initialize PID file manager.

        Args:
            pid_file: Path to the PID file
        """
        self.pid_file = pid_file
        self.pid = os.getpid()

    def create(self) -> bool:
        """
        Create PID file with current process ID.

        Returns:
            True if created successfully, False if another instance is running
        """
        if self.pid_file.exists():
            # Check if process is still running
            try:
                with open(self.pid_file, 'r') as f:
                    existing_pid = int(f.read().strip())

                # Check if process exists
                try:
                    os.kill(existing_pid, 0)  # Signal 0 doesn't actually send a signal
                    logger.error(f"Another instance is already running with PID {existing_pid}")
                    return False
                except OSError:
                    # Process doesn't exist, remove stale PID file
                    logger.warning(f"Removing stale PID file for process {existing_pid}")
                    self.pid_file.unlink()
            except (ValueError, IOError) as e:
                logger.warning(f"Invalid PID file {self.pid_file}: {e}")
                self.pid_file.unlink()

        # Create PID file directory if it doesn't exist
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

        # Write current PID
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(self.pid))
            logger.info(f"Created PID file {self.pid_file} with PID {self.pid}")

            # Register cleanup on exit
            atexit.register(self.cleanup)
            return True
        except IOError as e:
            logger.error(f"Failed to create PID file {self.pid_file}: {e}")
            return False

    def cleanup(self):
        """Remove PID file if it exists and contains our PID."""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    file_pid = int(f.read().strip())

                if file_pid == self.pid:
                    self.pid_file.unlink()
                    logger.info(f"Removed PID file {self.pid_file}")
                else:
                    logger.warning(f"PID file contains different PID {file_pid}, not removing")
            except (ValueError, IOError) as e:
                logger.warning(f"Error cleaning up PID file {self.pid_file}: {e}")


class HealthChecker:
    """
    Provides health checking functionality for the daemon.
    """

    def __init__(self, service: Optional['ExperimentPlatformService'] = None):
        """
        Initialize health checker.

        Args:
            service: EPII service instance to check
        """
        self.service = service
        self.start_time = time.time()

    def check_startup_health(self, config: Dict[str, Any]) -> bool:
        """
        Perform health checks during daemon startup.

        Args:
            config: Configuration dictionary

        Returns:
            True if all checks pass, False otherwise
        """
        checks = [
            ("Configuration validation", lambda: validate_config(config)),
            ("Port availability", lambda: self._check_port_available(config.get("port", 50051))),
            ("Disk space", lambda: self._check_disk_space()),
            ("Memory availability", lambda: self._check_memory()),
        ]

        logger.info("Performing startup health checks...")
        all_passed = True

        for check_name, check_func in checks:
            try:
                if check_func():
                    logger.info(f"✓ {check_name}: PASS")
                else:
                    logger.error(f"✗ {check_name}: FAIL")
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ {check_name}: ERROR - {e}")
                all_passed = False

        if all_passed:
            logger.info("All startup health checks passed")
        else:
            logger.error("One or more startup health checks failed")

        return all_passed

    def check_runtime_health(self) -> Dict[str, Any]:
        """
        Perform runtime health checks.

        Returns:
            Dictionary with health check results
        """
        uptime = time.time() - self.start_time

        health_status = {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "checks": {}
        }

        # Basic runtime checks
        checks = [
            ("service_available", lambda: self.service is not None),
            ("disk_space", lambda: self._check_disk_space()),
            ("memory", lambda: self._check_memory()),
        ]

        for check_name, check_func in checks:
            try:
                result = check_func()
                health_status["checks"][check_name] = "pass" if result else "fail"
                if not result:
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["checks"][check_name] = f"error: {str(e)}"
                health_status["status"] = "degraded"

        return health_status

    def _check_port_available(self, port: int) -> bool:
        """Check if the specified port is available."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except OSError:
            return False

    def _check_disk_space(self, min_free_mb: int = 100) -> bool:
        """Check if there's enough free disk space."""
        try:
            import shutil
            _, _, free = shutil.disk_usage(Path.cwd())
            free_mb = free // (1024 * 1024)
            return free_mb >= min_free_mb
        except Exception:
            return False

    def _check_memory(self, min_free_mb: int = 50) -> bool:
        """Check if there's enough available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available // (1024 * 1024)
            return available_mb >= min_free_mb
        except ImportError:
            # psutil not available, assume memory is OK
            return True
        except Exception:
            return False


class EPIIDaemon:
    """
    EPII daemon that manages the gRPC server lifecycle.
    """

    def __init__(self, config: Dict[str, Any], port: int = 50051, pid_file: Optional[str] = None):
        """
        Initialize the EPII daemon.

        Args:
            config: Configuration dictionary
            port: Port to bind the gRPC server to
            pid_file: Path to PID file (default: leeq-epii-{port}.pid)
        """
        self.config = config
        self.port = port
        self.server: Optional[grpc.Server] = None
        self.service: Optional[ExperimentPlatformService] = None
        self._stop_requested = False

        # Initialize PID file manager
        if pid_file is None:
            pid_file = f"leeq-epii-{port}.pid"
        self.pid_manager = PIDFileManager(Path(pid_file))

        # Initialize health checker
        self.health_checker = HealthChecker()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._health_check_handler)  # Health check signal

        logger.info(f"Initialized EPII daemon on port {port}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}
        signal_name = signal_names.get(signum, f"signal {signum}")
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._stop_requested = True
        if self.server:
            self.server.stop(grace=30)  # 30 second grace period

    def _health_check_handler(self, signum, frame):
        """Handle health check signal (SIGUSR1)."""
        health_status = self.health_checker.check_runtime_health()
        logger.info(f"Health check requested via signal: {health_status}")
        # Could also write to a status file here if needed

    def _load_setup(self):
        """
        Load the LeeQ setup based on configuration.

        Returns:
            Setup instance or None for simulation mode
        """
        setup_type = self.config.get("setup_type", "simulation")

        if setup_type == "simulation":
            # Create actual simulation setup using config module
            from .config import create_setup_from_config
            logger.info("Creating simulation setup from configuration")
            try:
                setup = create_setup_from_config(self.config)
                logger.info(f"Created simulation setup: {setup.name if hasattr(setup, 'name') else 'simulation'}")
                return setup
            except Exception as e:
                logger.error(f"Failed to create simulation setup: {e}")
                return None
        else:
            # Hardware setup loading will be implemented in Phase 2
            logger.warning(f"Setup type {setup_type} not yet implemented")
            return None

    def start(self) -> None:
        """
        Start the gRPC server and begin serving requests.
        """
        # Perform startup health checks (unless disabled)
        if not self.config.get("skip_health_checks", False):
            if not self.health_checker.check_startup_health(self.config):
                logger.error("Startup health checks failed, aborting daemon start")
                sys.exit(1)
        else:
            logger.warning("Startup health checks skipped")

        # Create PID file
        if not self.pid_manager.create():
            logger.error("Failed to create PID file, aborting daemon start")
            sys.exit(1)

        try:
            # Load setup
            setup = self._load_setup()

            # Create service
            self.service = ExperimentPlatformService(setup=setup, config=self.config)

            # Update health checker with service instance
            self.health_checker.service = self.service

            # Create gRPC server with thread pool
            max_workers = self.config.get("max_workers", 10)

            # Add debug interceptor if enabled
            interceptors = []
            if self.config.get("debug_grpc", False):
                interceptors.append(create_debug_interceptor())
                logger.info("gRPC debug interceptor enabled")

            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=max_workers),
                interceptors=interceptors
            )

            # Add service to server
            epii_pb2_grpc.add_ExperimentPlatformServiceServicer_to_server(
                self.service, self.server
            )

            # Bind to port
            address = f"[::]:{self.port}"
            self.server.add_insecure_port(address)

            # Start server
            self.server.start()
            logger.info(f"EPII daemon started on {address}")

            # Perform post-startup validation
            self._validate_startup()

            # Wait for termination
            try:
                while not self._stop_requested:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")

            # Server stop is handled by signal handler
            if self.server:
                self.server.wait_for_termination()

        except Exception as e:
            logger.error(f"Error during daemon startup: {e}", exc_info=True)
            raise
        finally:
            # Cleanup is handled by atexit registration
            logger.info("EPII daemon stopped")

    def _validate_startup(self) -> bool:
        """
        Validate that the daemon started successfully.

        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check if server is actually listening
        import socket
        server_listening = False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', self.port))
                if result == 0:
                    logger.info(f"✓ Server successfully listening on port {self.port}")
                    server_listening = True
                else:
                    logger.warning(f"Server may not be listening on port {self.port}")
        except Exception as e:
            logger.warning(f"Could not validate server startup: {e}")

        # Log initial health status
        health_status = self.health_checker.check_runtime_health()
        logger.info(f"Initial health status: {health_status['status']}")

        return server_listening

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status of the daemon.

        Returns:
            Dictionary with health status information
        """
        return self.health_checker.check_runtime_health()

    def stop(self, grace: float = 30.0) -> None:
        """
        Stop the gRPC server gracefully.

        Args:
            grace: Grace period in seconds for ongoing requests
        """
        if self.server:
            logger.info(f"Stopping server with {grace}s grace period")
            self.server.stop(grace)
            self._stop_requested = True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    # Basic validation for Phase 1
    required_fields = ["setup_type"]

    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required configuration field: {field}")
            return False

    if config["setup_type"] not in ["simulation", "hardware"]:
        logger.error(f"Invalid setup_type: {config['setup_type']}")
        return False

    return True


def setup_logging(log_level: str = "INFO", use_systemd: bool = None):
    """
    Configure logging for the daemon.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_systemd: Whether to use systemd journal format. Auto-detected if None.
    """
    # Auto-detect systemd environment if not specified
    if use_systemd is None:
        use_systemd = (
            os.getenv('JOURNAL_STREAM') is not None or
            os.getenv('INVOCATION_ID') is not None or
            'systemd' in os.getenv('_', '')
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create appropriate handler
    if use_systemd:
        # For systemd, use simple format since journald adds timestamp and metadata
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='[%(levelname)s] %(name)s: %(message)s'
        )
    else:
        # For direct execution, use detailed format
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log the logging configuration
    logger.info(f"Logging configured: level={log_level}, systemd={use_systemd}")


def main() -> None:
    """
    Main entry point for the EPII daemon.
    """
    parser = argparse.ArgumentParser(
        description="LeeQ EPII daemon - gRPC service for quantum experiment execution"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind the gRPC server (default: 50051)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--pid-file",
        type=str,
        help="Path to PID file (default: leeq-epii-{port}.pid)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit"
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check and exit"
    )

    parser.add_argument(
        "--no-health-checks",
        action="store_true",
        help="Skip startup health checks (for debugging)"
    )

    parser.add_argument(
        "--debug-grpc",
        action="store_true",
        help="Enable gRPC debug interceptor"
    )

    parser.add_argument(
        "--diagnostic-report",
        action="store_true",
        help="Generate diagnostic report and exit"
    )

    parser.add_argument(
        "--test-experiment",
        type=str,
        help="Test experiment execution and exit (default: rabi)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Load configuration
        config = load_config(args.config)

        # Add port to config if not present
        if "port" not in config:
            config["port"] = args.port

        # Add debug options to config
        if args.debug_grpc:
            config["debug_grpc"] = True

        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            sys.exit(1)

        if args.validate:
            print("Configuration is valid")  # noqa: T201
            sys.exit(0)

        # Handle health check mode
        if args.health_check:
            # For health check, we just need to perform basic checks without starting the daemon
            health_checker = HealthChecker()

            # Skip startup checks that require the daemon to not be running
            basic_checks = [
                ("Configuration validation", lambda: validate_config(config)),
                ("Disk space", lambda: health_checker._check_disk_space()),
                ("Memory availability", lambda: health_checker._check_memory()),
            ]

            print("Performing health checks...")  # noqa: T201
            all_passed = True

            for check_name, check_func in basic_checks:
                try:
                    if check_func():
                        print(f"✓ {check_name}: PASS")  # noqa: T201
                    else:
                        print(f"✗ {check_name}: FAIL")  # noqa: T201
                        all_passed = False
                except Exception as e:
                    print(f"✗ {check_name}: ERROR - {e}")  # noqa: T201
                    all_passed = False

            if all_passed:
                print("All health checks passed")  # noqa: T201
                sys.exit(0)
            else:
                print("One or more health checks failed")  # noqa: T201
                sys.exit(1)

        # Handle diagnostic report
        if args.diagnostic_report:
            print("Generating diagnostic report...")  # noqa: T201
            try:
                # Create temporary service instance for diagnostics
                from .config import create_setup_from_config
                setup = create_setup_from_config(config)
                service = ExperimentPlatformService(setup=setup, config=config)

                report = format_diagnostic_report(service, config)
                print(report)  # noqa: T201
                sys.exit(0)
            except Exception as e:
                logger.error(f"Failed to generate diagnostic report: {e}", exc_info=True)
                sys.exit(1)

        # Handle test experiment
        if args.test_experiment is not None:
            experiment_name = args.test_experiment if args.test_experiment else "rabi"
            print(f"Testing experiment: {experiment_name}")  # noqa: T201
            try:
                # Create temporary service instance for testing
                from .config import create_setup_from_config
                setup = create_setup_from_config(config)
                service = ExperimentPlatformService(setup=setup, config=config)

                diagnostic = DiagnosticTool(service)
                result = diagnostic.test_experiment_execution(experiment_name)

                print(f"Test status: {result['status'].upper()}")  # noqa: T201
                print(f"Message: {result['message']}")  # noqa: T201
                if 'duration_ms' in result:
                    print(f"Duration: {result['duration_ms']:.1f}ms")  # noqa: T201

                sys.exit(0 if result['status'] == 'ok' else 1)
            except Exception as e:
                logger.error(f"Failed to test experiment: {e}", exc_info=True)
                sys.exit(1)

        # Override health checks if requested
        if args.no_health_checks:
            logger.warning("Startup health checks disabled via --no-health-checks")
            config["skip_health_checks"] = True

        # Create and start daemon
        daemon = EPIIDaemon(config=config, port=args.port, pid_file=args.pid_file)
        daemon.start()

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
