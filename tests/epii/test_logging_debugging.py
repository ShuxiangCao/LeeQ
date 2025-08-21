"""
Tests for EPII logging and debugging utilities.
"""

import json
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from leeq.epii.utils import (
    RequestResponseLogger,
    PerformanceMonitor,
    DiagnosticTool,
    format_diagnostic_report
)
from leeq.epii.proto import epii_pb2
from leeq.epii.service import ExperimentPlatformService


class TestRequestResponseLogger:
    """Test request/response logging functionality."""

    def test_logger_initialization(self):
        """Test logger initialization with different configurations."""
        # Test with default settings
        logger = RequestResponseLogger()
        assert logger.log_level == "INFO"
        assert logger.log_file is None

        # Test with custom settings
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            logger = RequestResponseLogger(log_level="DEBUG", log_file=f.name)
            assert logger.log_level == "DEBUG"
            assert logger.log_file == f.name

    def test_request_logging(self):
        """Test request logging functionality."""
        logger = RequestResponseLogger()

        # Create mock request
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "rabi"
        request.parameters["amplitude"] = "0.1"

        # Create mock context
        context = MagicMock()
        context.peer.return_value = "test-client"

        # Log request
        request_id = logger.log_request("RunExperiment", request, context)

        # Verify request ID format
        assert request_id.startswith("RunExperiment_")
        assert len(request_id) > len("RunExperiment_")

    def test_response_logging(self):
        """Test response logging functionality."""
        logger = RequestResponseLogger()

        # Create mock response
        response = epii_pb2.ExperimentResponse()
        response.success = True

        # Log response
        import time
        start_time = time.time()
        logger.log_response("test_request_123", response, start_time)

        # Should not raise exceptions
        assert True

    def test_message_serialization(self):
        """Test protobuf message serialization."""
        logger = RequestResponseLogger()

        # Test with experiment request
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "rabi"
        request.parameters["test_param"] = "test_value"

        serialized = logger._serialize_message(request)

        assert isinstance(serialized, dict)
        assert serialized["experiment_type"] == "rabi"

        # Test with None
        assert logger._serialize_message(None) is None


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()

        assert monitor.experiment_metrics == {}
        assert monitor.parameter_metrics == {}
        assert monitor.start_time > 0

    def test_experiment_recording(self):
        """Test experiment performance recording."""
        monitor = PerformanceMonitor()

        # Record successful experiment
        monitor.record_experiment("rabi", 100.5, True, 1024)

        assert "rabi" in monitor.experiment_metrics
        metrics = monitor.experiment_metrics["rabi"]
        assert metrics["count"] == 1
        assert metrics["success_count"] == 1
        assert metrics["total_duration_ms"] == 100.5
        assert metrics["total_data_size"] == 1024

        # Record failed experiment
        monitor.record_experiment("rabi", 50.0, False)

        metrics = monitor.experiment_metrics["rabi"]
        assert metrics["count"] == 2
        assert metrics["success_count"] == 1  # Still 1 success

    def test_experiment_statistics(self):
        """Test experiment statistics calculation."""
        monitor = PerformanceMonitor()

        # Record multiple experiments
        monitor.record_experiment("rabi", 100.0, True)
        monitor.record_experiment("rabi", 200.0, True)
        monitor.record_experiment("t1", 150.0, False)

        # Get all stats
        all_stats = monitor.get_experiment_stats()
        assert "rabi" in all_stats
        assert "t1" in all_stats

        # Check rabi stats
        rabi_stats = all_stats["rabi"]
        assert rabi_stats["count"] == 2
        assert rabi_stats["success_rate"] == 1.0
        assert rabi_stats["avg_duration_ms"] == 150.0
        assert rabi_stats["max_duration_ms"] == 200.0
        assert rabi_stats["min_duration_ms"] == 100.0

        # Check t1 stats
        t1_stats = all_stats["t1"]
        assert t1_stats["count"] == 1
        assert t1_stats["success_rate"] == 0.0

        # Get specific experiment stats
        specific_stats = monitor.get_experiment_stats("rabi")
        assert specific_stats["experiment"] == "rabi"
        assert specific_stats["count"] == 2

    def test_service_uptime(self):
        """Test service uptime calculation."""
        monitor = PerformanceMonitor()

        import time
        time.sleep(0.1)  # Sleep briefly

        uptime = monitor.get_service_uptime()
        assert uptime >= 0.1
        assert uptime < 1.0  # Should be well under 1 second


class TestDiagnosticTool:
    """Test diagnostic tool functionality."""

    def test_tool_initialization(self):
        """Test diagnostic tool initialization."""
        tool = DiagnosticTool()
        assert tool.service is None

        # Test with mock service
        mock_service = MagicMock()
        tool = DiagnosticTool(mock_service)
        assert tool.service == mock_service

    def test_health_check_no_service(self):
        """Test health check without service."""
        tool = DiagnosticTool()

        health = tool.check_service_health()

        assert health["status"] in ["error", "warning"]
        assert "service_instance" in health["checks"]
        assert health["checks"]["service_instance"]["status"] == "error"

    def test_health_check_with_service(self):
        """Test health check with mock service."""
        mock_service = MagicMock()
        mock_service.setup = MagicMock()
        mock_service.experiment_router.experiment_map.keys.return_value = ["rabi", "t1"]
        mock_service.parameter_manager.list_parameters.return_value = [{"name": "test"}]
        mock_service.get_running_experiments.return_value = []

        tool = DiagnosticTool(mock_service)
        health = tool.check_service_health()

        assert health["status"] == "ok"
        assert health["checks"]["service_instance"]["status"] == "ok"
        assert health["checks"]["setup"]["status"] == "ok"
        assert health["checks"]["experiments"]["status"] == "ok"
        assert health["checks"]["parameters"]["status"] == "ok"
        assert health["checks"]["running_experiments"]["status"] == "ok"

    def test_configuration_check(self):
        """Test configuration validation."""
        tool = DiagnosticTool()

        # Test valid configuration
        valid_config = {
            "setup_type": "simulation",
            "max_workers": 10,
            "experiment_timeout": 300
        }

        result = tool.check_configuration(valid_config)
        assert result["status"] == "ok"
        assert len(result["issues"]) == 0

        # Test invalid configuration
        invalid_config = {
            "setup_type": "invalid_type",
            "max_workers": "not_a_number",
            "experiment_timeout": -1
        }

        result = tool.check_configuration(invalid_config)
        assert result["status"] == "error"
        assert len(result["issues"]) > 0

    def test_experiment_testing_no_service(self):
        """Test experiment testing without service."""
        tool = DiagnosticTool()

        result = tool.test_experiment_execution("rabi")

        assert result["status"] == "error"
        assert "No service instance" in result["message"]


class TestDiagnosticReport:
    """Test diagnostic report generation."""

    def test_report_generation(self):
        """Test diagnostic report generation."""
        # Create mock service with all required attributes
        mock_service = MagicMock()
        mock_service.setup = MagicMock()
        mock_service.experiment_router.experiment_map.keys.return_value = ["rabi"]
        mock_service.parameter_manager.list_parameters.return_value = []
        mock_service.get_running_experiments.return_value = []
        mock_service.performance_monitor.get_service_uptime.return_value = 123.45
        mock_service.performance_monitor.get_experiment_stats.return_value = {}

        config = {"setup_type": "simulation"}

        report = format_diagnostic_report(mock_service, config)

        assert isinstance(report, str)
        assert "EPII Service Diagnostic Report" in report
        assert "Service Health Check" in report
        assert "Configuration Check" in report
        assert "Performance Metrics" in report
        assert "System Information" in report
        assert "123." in report and "seconds" in report  # Check uptime formatting

    def test_report_without_performance_monitor(self):
        """Test diagnostic report when service lacks performance monitor."""
        mock_service = MagicMock()
        mock_service.setup = MagicMock()
        mock_service.experiment_router.experiment_map.keys.return_value = ["rabi"]
        mock_service.parameter_manager.list_parameters.return_value = []
        mock_service.get_running_experiments.return_value = []
        # No performance_monitor attribute
        del mock_service.performance_monitor

        config = {"setup_type": "simulation"}

        # Should not raise exception
        report = format_diagnostic_report(mock_service, config)
        assert isinstance(report, str)
        assert "EPII Service Diagnostic Report" in report


if __name__ == "__main__":
    pytest.main([__file__])
