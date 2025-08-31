"""
Debugging utilities and troubleshooting tools for the EPII service.

This module provides tools for monitoring, debugging, and troubleshooting
the EPII gRPC service in production environments.
"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import grpc
import numpy as np

from .proto import epii_pb2

logger = logging.getLogger(__name__)


class RequestResponseLogger:
    """
    Logger for gRPC request/response pairs with detailed timing and error tracking.
    """

    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the request/response logger.

        Args:
            log_level: Logging level for request/response logs
            log_file: Optional file path for dedicated request/response logs
        """
        self.log_level = log_level
        self.log_file = log_file

        # Create dedicated logger for request/response
        self.rr_logger = logging.getLogger("epii.request_response")
        self.rr_logger.setLevel(getattr(logging, log_level.upper()))

        # Add file handler if specified
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.rr_logger.addHandler(handler)

    def log_request(self, method_name: str, request: Any, context: grpc.ServicerContext) -> str:
        """
        Log incoming gRPC request with metadata.

        Args:
            method_name: Name of the gRPC method being called
            request: The request message
            context: gRPC context

        Returns:
            Request ID for correlation with response
        """
        request_id = f"{method_name}_{int(time.time() * 1000000)}"

        # Extract client info from context
        peer_info = context.peer() if context else "unknown"

        # Serialize request for logging (truncate large data)
        request_data = self._serialize_message(request, truncate_arrays=True)

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method_name,
            "peer": peer_info,
            "request_data": request_data
        }

        try:
            self.rr_logger.info(f"REQUEST {request_id}: {json.dumps(log_entry, indent=2)}")
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            self.rr_logger.info(f"REQUEST {request_id}: {log_entry} (JSON serialization failed: {e})")
        return request_id

    def log_response(self, request_id: str, response: Any,
                    start_time: float, error: Optional[Exception] = None):
        """
        Log gRPC response with timing and error information.

        Args:
            request_id: Request ID from log_request
            response: The response message (if successful)
            start_time: Start time from time.time()
            error: Exception if the request failed
        """
        duration_ms = (time.time() - start_time) * 1000

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration_ms, 2),
            "success": error is None
        }

        if error:
            log_entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exception(type(error), error, error.__traceback__)
            }
        else:
            # Serialize response for logging (truncate large data)
            log_entry["response_data"] = self._serialize_message(response, truncate_arrays=True)

        level = logging.ERROR if error else logging.INFO
        try:
            self.rr_logger.log(level, f"RESPONSE {request_id}: {json.dumps(log_entry, indent=2)}")
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            self.rr_logger.log(level, f"RESPONSE {request_id}: {log_entry} (JSON serialization failed: {e})")

    def _serialize_message(self, message: Any, truncate_arrays: bool = True) -> Dict[str, Any]:
        """
        Serialize a protobuf message for logging.

        Args:
            message: Protobuf message to serialize
            truncate_arrays: Whether to truncate large numpy arrays

        Returns:
            Dictionary representation suitable for JSON logging
        """
        if message is None:
            return None

        try:
            # Handle different message types
            if hasattr(message, 'DESCRIPTOR'):
                # Protobuf message
                result = {}
                for field, value in message.ListFields():
                    if field.name == 'measurement_data' and truncate_arrays:
                        # Truncate large measurement data for logging
                        result[field.name] = f"<{len(value)} measurement arrays>"
                    elif field.name == 'plots' and truncate_arrays:
                        # Truncate plot data
                        result[field.name] = f"<{len(value)} plots>"
                    elif isinstance(value, bytes) and len(value) > 100:
                        # Truncate large byte fields
                        result[field.name] = f"<{len(value)} bytes>"
                    else:
                        result[field.name] = self._convert_field_value(value)
                return result
            else:
                # Handle other types
                return str(message)

        except Exception as e:
            logger.warning(f"Failed to serialize message for logging: {e}")
            return f"<serialization failed: {type(message).__name__}>"

    def _convert_field_value(self, value: Any) -> Any:
        """Convert protobuf field values to JSON-serializable types."""
        if isinstance(value, (list, tuple)):
            return [self._convert_field_value(v) for v in value]
        elif hasattr(value, 'DESCRIPTOR'):
            return self._serialize_message(value)
        elif isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        elif hasattr(value, '__dict__'):
            # Handle objects with dict-like structure (like ScalarMapContainer)
            try:
                return str(value)
            except Exception:
                return f"<{type(value).__name__}>"
        else:
            # Try to convert to string for unknown types
            try:
                # Test if it's JSON serializable
                import json
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)


class PerformanceMonitor:
    """
    Monitor performance metrics for the EPII service.
    """

    def __init__(self):
        """Initialize performance monitoring."""
        self.experiment_metrics = {}
        self.parameter_metrics = {}
        self.start_time = time.time()

    def record_experiment(self, experiment_name: str, duration_ms: float,
                         success: bool, data_size: Optional[int] = None):
        """
        Record experiment execution metrics.

        Args:
            experiment_name: Name of the experiment
            duration_ms: Execution duration in milliseconds
            success: Whether the experiment succeeded
            data_size: Size of result data in bytes
        """
        if experiment_name not in self.experiment_metrics:
            self.experiment_metrics[experiment_name] = {
                "count": 0,
                "success_count": 0,
                "total_duration_ms": 0,
                "max_duration_ms": 0,
                "min_duration_ms": float('inf'),
                "total_data_size": 0
            }

        metrics = self.experiment_metrics[experiment_name]
        metrics["count"] += 1
        if success:
            metrics["success_count"] += 1

        metrics["total_duration_ms"] += duration_ms
        metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration_ms)
        metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration_ms)

        if data_size:
            metrics["total_data_size"] += data_size

    def get_experiment_stats(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get experiment execution statistics.

        Args:
            experiment_name: Specific experiment name, or None for all

        Returns:
            Dictionary with experiment statistics
        """
        if experiment_name:
            if experiment_name not in self.experiment_metrics:
                return {}

            metrics = self.experiment_metrics[experiment_name]
            return self._calculate_stats(experiment_name, metrics)
        else:
            return {
                name: self._calculate_stats(name, metrics)
                for name, metrics in self.experiment_metrics.items()
            }

    def _calculate_stats(self, name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived statistics from raw metrics."""
        count = metrics["count"]
        if count == 0:
            return {"experiment": name, "count": 0}

        return {
            "experiment": name,
            "count": count,
            "success_rate": metrics["success_count"] / count,
            "avg_duration_ms": metrics["total_duration_ms"] / count,
            "max_duration_ms": metrics["max_duration_ms"],
            "min_duration_ms": metrics["min_duration_ms"] if metrics["min_duration_ms"] != float('inf') else 0,
            "avg_data_size": metrics["total_data_size"] / count if metrics["total_data_size"] > 0 else 0
        }

    def get_service_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time


class DiagnosticTool:
    """
    Diagnostic tool for troubleshooting EPII service issues.
    """

    def __init__(self, service=None):
        """
        Initialize diagnostic tool.

        Args:
            service: EPII service instance for diagnostics
        """
        self.service = service

    def check_service_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the service.

        Returns:
            Dictionary with health check results
        """
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "unknown",
            "checks": {}
        }

        try:
            # Check service instance
            health["checks"]["service_instance"] = {
                "status": "ok" if self.service else "error",
                "message": "Service instance available" if self.service else "No service instance"
            }

            if self.service:
                # Check setup availability
                health["checks"]["setup"] = {
                    "status": "ok" if self.service.setup else "warning",
                    "message": "Setup configured" if self.service.setup else "No setup configured (simulation mode)"
                }

                # Check experiment router
                try:
                    experiments = list(self.service.experiment_router.experiment_map.keys())
                    health["checks"]["experiments"] = {
                        "status": "ok" if experiments else "error",
                        "message": f"{len(experiments)} experiments available: {experiments}"
                    }
                except Exception as e:
                    health["checks"]["experiments"] = {
                        "status": "error",
                        "message": f"Experiment router error: {e}"
                    }

                # Check parameter manager
                try:
                    param_count = len(self.service.parameter_manager.get_all_parameters())
                    health["checks"]["parameters"] = {
                        "status": "ok",
                        "message": f"{param_count} parameters available"
                    }
                except Exception as e:
                    health["checks"]["parameters"] = {
                        "status": "error",
                        "message": f"Parameter manager error: {e}"
                    }

                # Check running experiments
                try:
                    running = self.service.get_running_experiments()
                    health["checks"]["running_experiments"] = {
                        "status": "ok",
                        "message": f"{len(running)} experiments currently running"
                    }
                except Exception as e:
                    health["checks"]["running_experiments"] = {
                        "status": "error",
                        "message": f"Error checking running experiments: {e}"
                    }

            # Determine overall status
            statuses = [check["status"] for check in health["checks"].values()]
            if "error" in statuses:
                health["status"] = "error"
            elif "warning" in statuses:
                health["status"] = "warning"
            else:
                health["status"] = "ok"

        except Exception as e:
            health["status"] = "error"
            health["error"] = f"Health check failed: {e}"

        return health

    def test_experiment_execution(self, experiment_name: str = "rabi") -> Dict[str, Any]:
        """
        Test experiment execution with minimal parameters.

        Args:
            experiment_name: Name of experiment to test

        Returns:
            Test results
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "status": "unknown"
        }

        if not self.service:
            result["status"] = "error"
            result["message"] = "No service instance available"
            return result

        try:
            # Create minimal test request
            request = epii_pb2.ExperimentRequest()
            request.experiment_type = experiment_name

            # Add minimal parameters for the experiment
            if experiment_name == "rabi":
                request.parameters["qubit"] = "0"
                request.parameters["amplitude"] = "0.1"
                request.parameters["start_width"] = "0.0"
                request.parameters["stop_width"] = "1.0"
                request.parameters["width_step"] = "0.1"
            elif experiment_name == "t1":
                request.parameters["qubit"] = "0"
                request.parameters["time_max"] = "50e-6"
                request.parameters["time_step"] = "5e-6"
            elif experiment_name == "ramsey":
                request.parameters["qubit"] = "0"
                request.parameters["evolution_time"] = "10e-6"
                request.parameters["time_step"] = "1e-6"

            start_time = time.time()

            # Execute experiment
            response = self.service.RunExperiment(request, None)

            duration_ms = (time.time() - start_time) * 1000

            result["duration_ms"] = round(duration_ms, 2)
            result["success"] = response.success

            if response.success:
                result["status"] = "ok"
                result["message"] = f"Test experiment completed successfully in {duration_ms:.1f}ms"
                result["data_arrays"] = len(response.measurement_data)
                result["fit_params"] = len(response.calibration_results)
                result["plots"] = len(response.plots)
            else:
                result["status"] = "error"
                result["message"] = f"Test experiment failed: {response.error_message}"

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Test execution failed: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    def check_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate service configuration.

        Args:
            config: Configuration dictionary to check

        Returns:
            Configuration validation results
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "unknown",
            "issues": []
        }

        try:
            # Check required fields
            required_fields = ["setup_type"]
            for field in required_fields:
                if field not in config:
                    result["issues"].append(f"Missing required field: {field}")

            # Check setup type
            if "setup_type" in config:
                valid_types = ["simulation", "hardware"]
                if config["setup_type"] not in valid_types:
                    result["issues"].append(f"Invalid setup_type: {config['setup_type']}, must be one of {valid_types}")

            # Check numeric fields
            numeric_fields = ["max_workers", "experiment_timeout", "num_qubits"]
            for field in numeric_fields:
                if field in config and not isinstance(config[field], (int, float)):
                    try:
                        float(config[field])
                    except (ValueError, TypeError):
                        result["issues"].append(f"Field {field} must be numeric, got: {config[field]}")

            # Check reasonable values
            if "max_workers" in config and config["max_workers"] < 1:
                result["issues"].append("max_workers must be at least 1")

            if "experiment_timeout" in config and config["experiment_timeout"] < 1:
                result["issues"].append("experiment_timeout must be at least 1 second")

            result["status"] = "error" if result["issues"] else "ok"
            if result["issues"]:
                result["message"] = f"Found {len(result['issues'])} configuration issues"
            else:
                result["message"] = "Configuration is valid"

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Configuration check failed: {e}"

        return result


def format_diagnostic_report(service, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a comprehensive diagnostic report for troubleshooting.

    Args:
        service: EPII service instance
        config: Optional configuration dictionary

    Returns:
        Formatted diagnostic report as string
    """
    diagnostic = DiagnosticTool(service)

    report_lines = [
        "=" * 80,
        "EPII Service Diagnostic Report",
        "=" * 80,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        ""
    ]

    # Service health check
    health = diagnostic.check_service_health()
    report_lines.extend([
        "Service Health Check:",
        "-" * 40,
        f"Overall Status: {health['status'].upper()}",
        ""
    ])

    for check_name, check_result in health.get("checks", {}).items():
        report_lines.append(f"  {check_name}: {check_result['status'].upper()} - {check_result['message']}")

    report_lines.append("")

    # Configuration check
    if config:
        config_check = diagnostic.check_configuration(config)
        report_lines.extend([
            "Configuration Check:",
            "-" * 40,
            f"Status: {config_check['status'].upper()}",
            f"Message: {config_check['message']}",
            ""
        ])

        if config_check.get("issues"):
            report_lines.append("Issues found:")
            for issue in config_check["issues"]:
                report_lines.append(f"  - {issue}")
            report_lines.append("")

    # Performance metrics
    if hasattr(service, 'performance_monitor'):
        monitor = service.performance_monitor
        uptime = monitor.get_service_uptime()
        stats = monitor.get_experiment_stats()

        report_lines.extend([
            "Performance Metrics:",
            "-" * 40,
            f"Service Uptime: {uptime:.1f} seconds",
            ""
        ])

        if stats:
            report_lines.append("Experiment Statistics:")
            for exp_name, exp_stats in stats.items():
                report_lines.extend([
                    f"  {exp_name}:",
                    f"    Executions: {exp_stats['count']}",
                    f"    Success Rate: {exp_stats['success_rate']:.1%}",
                    f"    Avg Duration: {exp_stats['avg_duration_ms']:.1f}ms",
                    f"    Max Duration: {exp_stats['max_duration_ms']:.1f}ms",
                    ""
                ])
        else:
            report_lines.append("  No experiment executions recorded")
            report_lines.append("")

    # System information
    report_lines.extend([
        "System Information:",
        "-" * 40,
        f"Python Version: {sys.version}",
        f"NumPy Version: {np.__version__}",
        f"gRPC Version: {grpc.__version__}",
        ""
    ])

    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def create_debug_interceptor():
    """
    Create a gRPC interceptor for debugging request/response flow.

    Returns:
        gRPC server interceptor for debugging
    """
    class DebugInterceptor(grpc.ServerInterceptor):
        def __init__(self):
            self.logger = logging.getLogger("epii.debug")

        def intercept_service(self, continuation, handler_call_details):
            # Log method invocation
            self.logger.debug(f"Method called: {handler_call_details.method}")

            # Continue with normal processing
            return continuation(handler_call_details)

    return DebugInterceptor()
