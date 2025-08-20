"""
gRPC service implementation for EPII (Experiment Platform Intelligence Interface).

This module implements the ExperimentPlatformService defined in the EPII v1.0 standard.
"""

import time
import logging
import traceback
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import grpc
import numpy as np

from .proto import epii_pb2
from .proto import epii_pb2_grpc
from .experiments import ExperimentRouter
from .parameters import ParameterManager
from .serialization import (
    numpy_array_to_protobuf,
    plotly_figure_to_protobuf,
    deserialize_value
)
from .utils import RequestResponseLogger, PerformanceMonitor

logger = logging.getLogger(__name__)


class ExperimentPlatformService(epii_pb2_grpc.ExperimentPlatformServiceServicer):
    """
    Implementation of the EPII ExperimentPlatformService.

    This service provides a gRPC interface for external orchestrators to:
    - Execute quantum experiments on LeeQ
    - Manage experiment parameters
    - Query platform capabilities
    """

    def __init__(self, setup=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EPII service.

        Args:
            setup: LeeQ experimental setup instance
            config: Optional configuration dictionary
        """
        self.setup = setup
        self.config = config or {}
        self.platform_name = "LeeQ"
        self.platform_version = "0.1.0"

        # Initialize core components (Phase 2 integration)
        self.experiment_router = ExperimentRouter()
        self.parameter_manager = ParameterManager(setup)

        # Register setup with LeeQ ExperimentManager if provided
        if setup:
            from leeq.experiments.experiments import ExperimentManager
            manager = ExperimentManager()
            manager.clear_setups()  # Clear any existing setups
            manager.register_setup(setup)
            logger.info(f"Registered setup '{setup.name}' with LeeQ ExperimentManager")

        # Get supported experiments from router
        self.supported_experiments = list(self.experiment_router.experiment_map.keys())

        # Parameter types supported
        self.parameter_types = [
            "frequency",
            "amplitude",
            "phase",
            "duration",
            "delay"
        ]

        # Experiment execution configuration
        self.experiment_timeout = self.config.get("experiment_timeout", 300.0)  # 5 minutes default
        self.max_experiment_workers = self.config.get("max_experiment_workers", 2)
        self._experiment_executor = ThreadPoolExecutor(max_workers=self.max_experiment_workers)
        self._running_experiments = {}  # Track running experiments for cancellation

        # Initialize logging and monitoring components
        self.request_logger = RequestResponseLogger(
            log_level=self.config.get("log_level", "INFO"),
            log_file=self.config.get("request_log_file")
        )
        self.performance_monitor = PerformanceMonitor()

        logger.info(f"Initialized EPII service for {self.platform_name} v{self.platform_version}")
        logger.info(f"Available experiments: {self.supported_experiments}")
        logger.info(f"Experiment timeout: {self.experiment_timeout}s, max workers: {self.max_experiment_workers}")
        logger.info(f"Request/response logging enabled, log level: {self.config.get('log_level', 'INFO')}")

    def Ping(self, request: epii_pb2.Empty, context: grpc.ServicerContext) -> epii_pb2.PingResponse:
        """
        Basic connectivity test.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            PingResponse with echo message and timestamp
        """
        start_time = time.time()
        request_id = self.request_logger.log_request("Ping", request, context)

        try:
            response = epii_pb2.PingResponse()
            response.message = "Pong from LeeQ EPII service"
            response.timestamp = int(time.time() * 1000)  # milliseconds since epoch

            logger.debug("Ping received")
            self.request_logger.log_response(request_id, response, start_time)
            return response
        except Exception as e:
            self.request_logger.log_response(request_id, None, start_time, error=e)
            raise

    def GetCapabilities(self, request: epii_pb2.Empty,
                       context: grpc.ServicerContext) -> epii_pb2.CapabilitiesResponse:
        """
        Get platform capabilities and supported features.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            CapabilitiesResponse with platform information
        """
        response = epii_pb2.CapabilitiesResponse()

        # Platform information
        response.framework_name = self.platform_name
        response.framework_version = self.platform_version
        response.epii_version = "1.0.0"

        # Supported backends
        response.supported_backends.extend(["simulation", "hardware"])

        # Data formats
        response.data_formats.extend(["numpy", "json"])

        # Add experiment specifications from router
        experiment_descriptions = self.experiment_router.list_experiments()
        for exp_name in self.supported_experiments:
            exp_spec = epii_pb2.ExperimentSpec()
            exp_spec.name = exp_name
            exp_spec.description = experiment_descriptions.get(exp_name, f"{exp_name.title()} experiment")

            # Add parameter specifications from router
            param_schema = self.experiment_router.get_experiment_parameters(exp_name)
            for param_name, param_info in param_schema.items():
                param_spec = epii_pb2.ParameterSpec()
                param_spec.name = param_name
                param_spec.type = param_info.get("type", "string")
                param_spec.required = param_info.get("required", False)
                param_spec.description = f"Parameter {param_name}"
                if param_info.get("default") is not None:
                    param_spec.default_value = str(param_info["default"])
                exp_spec.parameters.append(param_spec)

            response.experiment_types.append(exp_spec)

        # Extensions (framework-specific capabilities)
        response.extensions["setup_type"] = self.config.get("setup_type", "simulation")
        response.extensions["num_qubits"] = str(self.config.get("num_qubits", 2))
        response.extensions["max_workers"] = str(self.config.get("max_workers", 10))

        logger.info("Capabilities requested and sent")
        return response

    def RunExperiment(self, request: epii_pb2.ExperimentRequest,
                     context: grpc.ServicerContext) -> epii_pb2.ExperimentResponse:
        """
        Execute a quantum experiment.

        Args:
            request: ExperimentRequest with experiment name and parameters
            context: gRPC context

        Returns:
            ExperimentResponse with results or error
        """
        start_time = time.time()
        request_id = self.request_logger.log_request("RunExperiment", request, context)
        response = epii_pb2.ExperimentResponse()
        experiment_name = request.experiment_type

        try:
            logger.info(f"Running experiment: {experiment_name}")

            # Validate experiment exists
            experiment_class = self.experiment_router.get_experiment(experiment_name)
            if not experiment_class:
                if context:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(f"Experiment '{experiment_name}' not found")
                response.success = False
                response.error_message = f"Unknown experiment: {experiment_name}"
                return response

            # Parse and validate parameters
            parameters = {}
            for key, value in request.parameters.items():
                try:
                    # Deserialize parameter value
                    param_data = {
                        "type": "string",  # Default type
                        "value": value
                    }
                    parameters[key] = deserialize_value(param_data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize parameter {key}: {e}")
                    parameters[key] = value

            # Map EPII parameters to LeeQ parameters
            leeq_parameters = self.experiment_router.map_parameters(experiment_name, parameters)

            # Validate parameters
            is_valid, errors = self.experiment_router.validate_parameters(experiment_name, leeq_parameters)
            if not is_valid:
                if context:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Parameter validation failed: {'; '.join(errors)}")
                response.success = False
                response.error_message = f"Invalid parameters: {'; '.join(errors)}"
                return response

            # Check setup availability
            if not self.setup:
                if context:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details("No experimental setup available")
                response.success = False
                response.error_message = "No experimental setup configured"
                return response

            # Generate experiment ID for tracking
            experiment_id = f"{experiment_name}_{int(time.time() * 1000)}"

            # Execute the experiment with timeout
            try:
                future = self._experiment_executor.submit(self._run_experiment_with_tracking,
                                                        experiment_class, experiment_id, leeq_parameters)
                self._running_experiments[experiment_id] = future

                # Wait for experiment to complete with timeout
                experiment = future.result(timeout=self.experiment_timeout)

            except FutureTimeoutError:
                # Cancel the experiment if it times out
                logger.warning(f"Experiment {experiment_name} timed out after {self.experiment_timeout}s")
                if experiment_id in self._running_experiments:
                    self._running_experiments[experiment_id].cancel()
                    del self._running_experiments[experiment_id]

                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details(f"Experiment timed out after {self.experiment_timeout} seconds")
                response.success = False
                response.error_message = f"Experiment timed out after {self.experiment_timeout} seconds"
                return response

            finally:
                # Clean up tracking
                if experiment_id in self._running_experiments:
                    del self._running_experiments[experiment_id]

            # Collect results
            response.success = True

            # Serialize measurement data
            if hasattr(experiment, 'data') and experiment.data is not None:
                if isinstance(experiment.data, np.ndarray):
                    data_msg = numpy_array_to_protobuf(
                        experiment.data,
                        name="measurement_data",
                        metadata={"experiment": experiment_name}
                    )
                    response.measurement_data.append(data_msg)
                else:
                    logger.warning(f"Experiment data is not a numpy array: {type(experiment.data)}")

            # Serialize fit parameters (use calibration_results field)
            if hasattr(experiment, 'fit_params') and experiment.fit_params:
                for key, value in experiment.fit_params.items():
                    if isinstance(value, (int, float)):
                        response.calibration_results[key] = float(value)
                    else:
                        logger.debug(f"Skipping non-numeric fit parameter {key}: {type(value)}")

            # Serialize plot data if available
            if hasattr(experiment, 'plot') and callable(experiment.plot):
                try:
                    figure = experiment.plot()
                    if figure:
                        plot_msg = plotly_figure_to_protobuf(figure)
                        response.plots.append(plot_msg)
                except Exception as e:
                    logger.warning(f"Failed to generate plot for {experiment_name}: {e}")

            logger.info(f"Experiment {experiment_name} completed successfully")

            # Record performance metrics
            duration_ms = (time.time() - start_time) * 1000
            data_size = sum(len(data.data) for data in response.measurement_data) if response.measurement_data else 0
            self.performance_monitor.record_experiment(experiment_name, duration_ms, True, data_size)

            self.request_logger.log_response(request_id, response, start_time)
            return response

        except Exception as e:
            # Log full traceback for debugging
            error_msg = f"Experiment execution failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            # Record performance metrics for failed experiment
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_experiment(experiment_name, duration_ms, False)

            # Set gRPC error status
            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)

            response.success = False
            response.error_message = error_msg

            self.request_logger.log_response(request_id, response, start_time, error=e)
            return response

    def ListAvailableExperiments(self, request: epii_pb2.Empty,
                                context: grpc.ServicerContext) -> epii_pb2.ExperimentsResponse:
        """
        List all available experiments with their specifications.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            ExperimentsResponse with experiment specifications
        """
        response = epii_pb2.ExperimentsResponse()

        # Add experiment specifications from router
        experiment_descriptions = self.experiment_router.list_experiments()
        for exp_name in self.supported_experiments:
            exp_spec = epii_pb2.ExperimentSpec()
            exp_spec.name = exp_name
            default_desc = f"{exp_name.replace('_', ' ').title()} experiment for qubit calibration"
            exp_spec.description = experiment_descriptions.get(exp_name, default_desc)

            # Add parameter specifications from router
            param_schema = self.experiment_router.get_experiment_parameters(exp_name)
            for param_name, param_info in param_schema.items():
                param_spec = epii_pb2.ParameterSpec()
                param_spec.name = param_name
                param_spec.type = param_info.get("type", "string")
                param_spec.required = param_info.get("required", False)
                param_spec.description = f"Parameter {param_name}"
                if param_info.get("default") is not None:
                    param_spec.default_value = str(param_info["default"])
                exp_spec.parameters.append(param_spec)

            response.experiments.append(exp_spec)

        logger.debug("ListAvailableExperiments called")
        return response

    def GetParameters(self, request: epii_pb2.ParameterRequest,
                     context: grpc.ServicerContext) -> epii_pb2.ParametersResponse:
        """
        Get parameter values from the setup.

        Args:
            request: ParameterRequest with list of parameter names
            context: gRPC context

        Returns:
            ParametersResponse with current values
        """
        response = epii_pb2.ParametersResponse()

        try:
            logger.debug(f"GetParameters called for {len(request.parameter_names)} parameters")

            if not request.parameter_names:
                # Return empty response if no parameters requested
                return response

            for param_name in request.parameter_names:
                try:
                    value = self.parameter_manager.get_parameter(param_name)
                    if value is not None:
                        # Serialize parameter value
                        if isinstance(value, (int, float, str, bool)):
                            response.parameters[param_name] = str(value)
                        else:
                            response.parameters[param_name] = str(value)
                    else:
                        logger.warning(f"Parameter '{param_name}' not found")
                        response.parameters[param_name] = "null"

                except Exception as e:
                    logger.error(f"Error getting parameter '{param_name}': {e}")
                    response.parameters[param_name] = f"error: {str(e)}"

            return response

        except Exception as e:
            error_msg = f"Failed to get parameters: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
            return response

    def SetParameters(self, request: epii_pb2.SetParametersRequest,
                     context: grpc.ServicerContext) -> epii_pb2.StatusResponse:
        """
        Set parameter values in the setup.

        Args:
            request: SetParametersRequest with parameter name-value pairs
            context: gRPC context

        Returns:
            StatusResponse indicating success or failure
        """
        response = epii_pb2.StatusResponse()

        try:
            logger.debug(f"SetParameters called for {len(request.parameters)} parameters")

            if not request.parameters:
                response.success = True
                response.message = "No parameters to set"
                return response

            failed_params = []
            success_count = 0

            for param_name, param_value in request.parameters.items():
                try:
                    # Validate parameter before setting
                    if not self.parameter_manager.validate_parameter(param_name, param_value):
                        failed_params.append(f"{param_name}: validation failed")
                        continue

                    # Set the parameter
                    success = self.parameter_manager.set_parameter(param_name, param_value)
                    if success:
                        success_count += 1
                        logger.debug(f"Set parameter {param_name} = {param_value}")
                    else:
                        failed_params.append(f"{param_name}: failed to set")

                except Exception as e:
                    failed_params.append(f"{param_name}: {str(e)}")
                    logger.error(f"Error setting parameter '{param_name}': {e}")

            # Determine overall success
            if failed_params:
                if success_count == 0:
                    # All parameters failed
                    response.success = False
                    response.error_message = f"Failed to set parameters: {'; '.join(failed_params)}"
                    if context:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details(response.error_message)
                else:
                    # Partial success - treat as success but note failures in error_message
                    response.success = True
                    response.error_message = f"Set {success_count} parameters, failed: {'; '.join(failed_params)}"
            else:
                # All parameters succeeded
                response.success = True

            return response

        except Exception as e:
            error_msg = f"Failed to set parameters: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)

            response.success = False
            response.error_message = error_msg
            return response

    def ListParameters(self, request: epii_pb2.Empty,
                      context: grpc.ServicerContext) -> epii_pb2.ParametersListResponse:
        """
        List all available parameters.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            ParametersListResponse with parameter information
        """
        response = epii_pb2.ParametersListResponse()

        try:
            logger.debug("ListParameters called")

            # Get all available parameters from parameter manager
            parameter_list = self.parameter_manager.list_parameters()

            for param_dict in parameter_list:
                param_info = epii_pb2.ParameterInfo()
                param_info.name = param_dict["name"]
                param_info.type = param_dict["type"]
                param_info.current_value = param_dict["current_value"]
                param_info.description = param_dict["description"]
                param_info.read_only = param_dict["read_only"]
                response.parameters.append(param_info)

            logger.debug(f"Listed {len(parameter_list)} parameters")
            return response

        except Exception as e:
            error_msg = f"Failed to list parameters: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
            return response

    def _run_experiment_with_tracking(self, experiment_class: type, experiment_id: str,
                                      leeq_parameters: Dict[str, Any]) -> Any:
        """
        Run an experiment with tracking support for cancellation.

        Args:
            experiment_class: LeeQ experiment class
            experiment_id: Unique identifier for the experiment
            leeq_parameters: Parameters for the experiment
        """
        logger.debug(f"Starting experiment execution: {experiment_id}")
        try:
            # Create and run experiment (run() is called automatically in __init__)
            experiment = experiment_class(**leeq_parameters)
            logger.debug(f"Experiment completed successfully: {experiment_id}")
            return experiment
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            raise

    def cancel_experiment(self, experiment_id: str) -> bool:
        """
        Cancel a running experiment by ID.

        Args:
            experiment_id: ID of the experiment to cancel

        Returns:
            True if experiment was found and cancelled, False otherwise
        """
        if experiment_id in self._running_experiments:
            future = self._running_experiments[experiment_id]
            if not future.done():
                success = future.cancel()
                logger.info(f"Cancelled experiment {experiment_id}: {success}")
                return success
        return False

    def get_running_experiments(self) -> List[str]:
        """
        Get list of currently running experiment IDs.

        Returns:
            List of experiment IDs currently running
        """
        running = []
        for exp_id, future in list(self._running_experiments.items()):
            if not future.done():
                running.append(exp_id)
            else:
                # Clean up completed experiments
                del self._running_experiments[exp_id]
        return running

    def shutdown(self):
        """
        Gracefully shutdown the service, cancelling any running experiments.
        """
        logger.info("Shutting down EPII service...")

        # Cancel all running experiments
        for exp_id in list(self._running_experiments.keys()):
            self.cancel_experiment(exp_id)

        # Shutdown executor
        self._experiment_executor.shutdown(wait=True, timeout=30.0)
        logger.info("EPII service shutdown complete")
