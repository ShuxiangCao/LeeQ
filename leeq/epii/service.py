"""
gRPC service implementation for EPII (Experiment Platform Intelligence Interface).

This module implements the ExperimentPlatformService defined in the EPII v1.0 standard.
"""

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional

import grpc
import numpy as np

from .experiments import ExperimentRouter
from .parameters import ParameterManager
from .proto import epii_pb2, epii_pb2_grpc
from .serialization import deserialize_value, numpy_array_to_protobuf, browser_function_to_plot_component
from .utils import PerformanceMonitor, RequestResponseLogger
from pprint import pprint

logger = logging.getLogger(__name__)


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dictionary using dot notation.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion  
        sep: Separator to use (default: '.')
    
    Returns:
        Flattened dictionary with dot notation keys
    
    Example:
        {'fit_params': {'Frequency': 1.23, 'Amplitude': 0.45}}
        -> {'fit_params.Frequency': 1.23, 'fit_params.Amplitude': 0.45}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_flattened_description(flat_key, epii_info):
    """
    Get proper description for a flattened attribute key from EPII_INFO.
    
    Args:
        flat_key: Flattened key like 'fit_params.Frequency'
        epii_info: EPII_INFO dictionary
    
    Returns:
        Proper description string or generic fallback
    
    Example:
        flat_key='fit_params.Frequency' -> 'float - Rabi oscillation frequency'
    """
    if not epii_info or 'attributes' not in epii_info:
        return f"Experiment attribute: {flat_key}"
    
    attributes = epii_info['attributes']
    key_parts = flat_key.split('.')
    
    # Navigate through nested EPII_INFO structure
    current = attributes
    for i, part in enumerate(key_parts):
        if isinstance(current, dict) and part in current:
            current = current[part]
            
            # If we've reached the end and have a description
            if i == len(key_parts) - 1:
                if isinstance(current, str):
                    return current
                elif isinstance(current, dict) and 'description' in current:
                    return current['description']
            
            # Navigate deeper for nested structures like fit_params.keys.Frequency
            if isinstance(current, dict) and 'keys' in current and i < len(key_parts) - 1:
                current = current['keys']
        else:
            break
    
    # Fallback to generic description
    return f"Experiment attribute: {flat_key}"


def get_attribute_description(key, epii_info):
    """
    Get proper description for a non-flattened attribute key from EPII_INFO.
    
    Args:
        key: Attribute key like 'data' or 'guess_amp'
        epii_info: EPII_INFO dictionary
    
    Returns:
        Proper description string or generic fallback
    """
    if not epii_info or 'attributes' not in epii_info:
        return f"Experiment attribute: {key}"
    
    if key in epii_info['attributes']:
        attr_info = epii_info['attributes'][key]
        # Handle case where attribute info is a dict with 'description' field
        if isinstance(attr_info, dict):
            return attr_info.get('description', f"Experiment attribute: {key}")
        else:
            return str(attr_info)
    else:
        return f"Experiment attribute: {key}"


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
        self.experiment_router = ExperimentRouter(setup)  # Pass setup for backend-aware filtering
        self.parameter_manager = ParameterManager(setup)

        # Register setup with LeeQ ExperimentManager if provided
        if setup:
            from leeq.experiments.experiments import ExperimentManager
            manager = ExperimentManager()
            manager.clear_setups()  # Clear any existing setups
            manager.register_setup(setup)
            logger.info(f"Registered setup '{setup.name}' with LeeQ ExperimentManager")

        # Initialize Chronicle if enabled in config
        self._chronicle = None
        chronicle_config = self.config.get("chronicle", {})
        if chronicle_config.get("enabled", True):  # Default enabled for EPII
            try:
                from leeq.chronicle import Chronicle
                self._chronicle = Chronicle()
                session_name = chronicle_config.get("session_name", "epii_session")
                self._chronicle.start_log(session_name)
                logger.info(f"Chronicle session started: {session_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Chronicle: {e}")
                self._chronicle = None

        # Get supported experiments from dynamically discovered experiments
        self.supported_experiments = list(self.experiment_router.experiment_map.keys())

        # Log discovery statistics
        logger.info(f"Dynamically discovered {len(self.supported_experiments)} experiments")
        logger.info(f"Available experiments (first 10): {sorted(self.supported_experiments)[:10]}...")

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
                    # Try to convert to appropriate type
                    if value.lower() in ['true', 'false']:
                        parameters[key] = value.lower() == 'true'
                    elif '.' in value or 'e' in value.lower():
                        # Try float first for decimal or scientific notation
                        try:
                            parameters[key] = float(value)
                        except ValueError:
                            parameters[key] = value
                    else:
                        # Try integer
                        try:
                            parameters[key] = int(value)
                        except ValueError:
                            # Keep as string if can't convert
                            parameters[key] = value
                except Exception as e:
                    logger.warning(f"Failed to convert parameter {key}: {e}")
                    parameters[key] = value

            # Map EPII parameters to LeeQ parameters and resolve qubit references
            leeq_parameters = self.experiment_router.map_parameters(experiment_name, parameters, self.setup)

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

            # Try to get fit parameters - call fitting() if needed
            if hasattr(experiment, 'fitting') and callable(experiment.fitting):
                if not hasattr(experiment, 'fit_params') or not experiment.fit_params:
                    try:
                        logger.debug(f"Calling fitting() for {experiment_name}")
                        experiment.fitting()
                    except Exception as e:
                        logger.debug(f"fitting() failed for {experiment_name}: {e}")

            # Enhanced plot generation with detailed logging
            plot_count = 0
            if hasattr(experiment, 'get_browser_functions'):
                try:
                    browser_functions = experiment.get_browser_functions()
                    logger.info(f"Found {len(browser_functions)} browser functions: {[name for name, _ in browser_functions]}")
                    
                    for func_name, func_method in browser_functions:
                        logger.debug(f"Processing browser function: {func_name}")
                        try:
                            result = func_method()
                            logger.debug(f"Browser function {func_name} returned: {type(result)}")
                            plot_figure = None
                            

                            # Handle matplotlib and plotly returns
                            if result is None:
                                logger.debug(f"Browser function {func_name} returned None")
                                plot_figure = None
                                    
                            elif hasattr(result, 'to_json') or isinstance(result, dict):
                                logger.debug(f"Browser function {func_name} returned plotly figure")
                                plot_figure = result
                                
                            elif hasattr(result, 'savefig') or hasattr(result, 'get_axes'):
                                # Matplotlib figure case - convert to plotly
                                logger.debug(f"Browser function {func_name} returned matplotlib figure")
                                try:
                                    import plotly.tools
                                    if hasattr(plotly.tools, 'mpl_to_plotly'):
                                        plot_figure = plotly.tools.mpl_to_plotly(result)
                                        logger.info(f"Successfully converted matplotlib figure from {func_name}")
                                        # Close the matplotlib figure after conversion
                                        import matplotlib.pyplot as plt
                                        plt.close(result)
                                    else:
                                        logger.warning(f"plotly.tools.mpl_to_plotly not available")
                                        plot_figure = None
                                except Exception as e:
                                    logger.warning(f"Failed to convert matplotlib figure from {func_name}: {e}")
                                    plot_figure = None
                            else:
                                logger.debug(f"Browser function {func_name} returned unrecognized type: {type(result)}")
                                plot_figure = None
                            
                            logger.debug(f"Result: {result}")

                            # Create component if we have a figure
                            if plot_figure:
                                logger.debug(f"Creating plot component for {func_name}")
                                try:
                                    component = browser_function_to_plot_component(func_name, plot_figure)
                                    response.plots.append(component)
                                    plot_count += 1
                                    logger.info(f"Successfully created plot component for {func_name}: description='{component.description}', json_len={len(component.plotly_json)}, png_len={len(component.image_png)}")
                                except Exception as e:
                                    logger.error(f"Failed to create plot component for {func_name}: {e}")
                            else:
                                logger.warning(f"No valid plot figure found for browser function {func_name}")
                                
                        except Exception as e:
                            logger.error(f"Failed to call browser function {func_name}: {e}")
                    
                    logger.info(f"Generated {plot_count} plot components from {len(browser_functions)} browser functions")
                    
                except Exception as e:
                    logger.error(f"Failed to get browser functions: {e}")
            else:
                logger.info(f"Experiment {experiment_name} has no get_browser_functions method")

            # 1. Create Documentation message
            import json
            exp_info = self.experiment_router.get_experiment_info(experiment_name)
            response.docs.CopyFrom(epii_pb2.Documentation())  # Initialize the docs message

            if exp_info and exp_info.get('run_docstring'):
                response.docs.run = exp_info['run_docstring']

            # 2. Add EPII data documentation and metadata
            epii_info = {}
            if exp_info and exp_info.get('epii_info'):
                epii_info = exp_info['epii_info']

                # Extract data documentation (description of what the data means)
                if 'description' in epii_info:
                    response.docs.data = epii_info['description']

                # Store other EPII_INFO as metadata
                for key, value in epii_info.items():
                    if key != 'description':  # Description goes in docs.data
                        if isinstance(value, str):
                            response.metadata[key] = value
                        else:
                            response.metadata[key] = json.dumps(value)

            # Get ALL attributes from Chronicle and convert to protobuf
            if hasattr(experiment, '__chronicle_record_entry__'):
                try:
                    record_entry = experiment.__chronicle_record_entry__
                    all_attrs = record_entry.load_all_attributes()
                    logger.info(f"Loaded {len(all_attrs)} Chronicle attributes")
                    
                    # Flatten nested dictionaries and convert ALL attributes
                    for key, value in all_attrs.items():
                        if key.startswith('__') or key == '__object__' or key == 'EPII_INFO':
                            continue
                            
                        if isinstance(value, dict):
                            # Flatten nested dictionaries  
                            flat_dict = flatten_dict(value, parent_key=key, sep='.')
                            for flat_key, flat_value in flat_dict.items():
                                item = epii_pb2.DataItem()
                                item.name = flat_key
                                item.description = get_flattened_description(flat_key, epii_info)
                                
                                # Simple type-based serialization
                                if isinstance(flat_value, bool):
                                    item.boolean = flat_value
                                elif isinstance(flat_value, (int, float, np.integer, np.floating)):
                                    item.number = float(flat_value)
                                elif isinstance(flat_value, str):
                                    item.text = flat_value
                                elif isinstance(flat_value, np.ndarray):
                                    array = epii_pb2.NumpyArray()
                                    array.data = flat_value.tobytes()
                                    array.shape.extend(flat_value.shape)
                                    array.dtype = str(flat_value.dtype)
                                    item.array.CopyFrom(array)
                                else:
                                    item.text = str(flat_value)
                                
                                response.data.append(item)
                                logger.debug(f"Added flattened DataItem: {flat_key}")
                        else:
                            # Handle non-dict values directly
                            item = epii_pb2.DataItem()
                            item.name = key
                            item.description = get_attribute_description(key, epii_info)
                            
                            # Simple type-based serialization
                            if isinstance(value, bool):
                                item.boolean = value
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                item.number = float(value)
                            elif isinstance(value, str):
                                item.text = value
                            elif isinstance(value, np.ndarray):
                                array = epii_pb2.NumpyArray()
                                array.data = value.tobytes()
                                array.shape.extend(value.shape)
                                array.dtype = str(value.dtype)
                                item.array.CopyFrom(array)
                            else:
                                item.text = str(value)
                            
                            response.data.append(item)
                            logger.debug(f"Added DataItem: {key}")
                            
                    logger.info(f"Added {len(response.data)} DataItems from Chronicle to EPII response")
                except Exception as e:
                    logger.error(f"Could not extract Chronicle data: {e}", exc_info=True)
            else:
                # Fallback: Get attributes directly from experiment object when no Chronicle record
                logger.info("No Chronicle record, extracting attributes directly from experiment")
                try:
                    # Collect all experiment attributes first
                    experiment_attrs = {}
                    for attr_name in dir(experiment):
                        if not attr_name.startswith('_') and not attr_name.startswith('chronicle') and attr_name != 'EPII_INFO':
                            try:
                                value = getattr(experiment, attr_name)
                                if not callable(value) and value is not None:
                                    experiment_attrs[attr_name] = value
                                    logger.debug(f"Collected experiment attribute {attr_name}")
                            except Exception as e:
                                logger.debug(f"Could not collect {attr_name}: {e}")
                    pprint(experiment_attrs)
                    # Process attributes (ignore EPII_INFO since it's already in docs section)
                    for key, value in experiment_attrs.items():
                        if isinstance(value, dict):
                            # Flatten nested dictionaries  
                            flat_dict = flatten_dict(value, parent_key=key, sep='.')
                            for flat_key, flat_value in flat_dict.items():
                                item = epii_pb2.DataItem()
                                item.name = flat_key
                                item.description = f"Experiment attribute: {flat_key}"
                                
                                # Simple type-based serialization
                                if isinstance(flat_value, bool):
                                    item.boolean = flat_value
                                elif isinstance(flat_value, (int, float, np.integer, np.floating)):
                                    item.number = float(flat_value)
                                elif isinstance(flat_value, str):
                                    item.text = flat_value
                                elif isinstance(flat_value, np.ndarray):
                                    array = epii_pb2.NumpyArray()
                                    array.data = flat_value.tobytes()
                                    array.shape.extend(flat_value.shape)
                                    array.dtype = str(flat_value.dtype)
                                    item.array.CopyFrom(array)
                                else:
                                    item.text = str(flat_value)
                                
                                response.data.append(item)
                                logger.debug(f"Added flattened DataItem: {flat_key}")
                        else:
                            # Handle non-dict values directly
                            item = epii_pb2.DataItem()
                            item.name = key
                            item.description = f"Experiment attribute: {key}"
                            
                            # Simple type-based serialization
                            if isinstance(value, bool):
                                item.boolean = value
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                item.number = float(value)
                            elif isinstance(value, str):
                                item.text = value
                            elif isinstance(value, np.ndarray):
                                array = epii_pb2.NumpyArray()
                                array.data = value.tobytes()
                                array.shape.extend(value.shape)
                                array.dtype = str(value.dtype)
                                item.array.CopyFrom(array)
                            else:
                                item.text = str(value)
                            
                            response.data.append(item)
                            logger.debug(f"Added DataItem: {key}")
                    
                    logger.info(f"Added {len(response.data)} DataItems from experiment object")
                except Exception as e:
                    logger.error(f"Could not extract experiment attributes: {e}")

            logger.info(f"Experiment {experiment_name} completed successfully")

            # Record performance metrics
            duration_ms = (time.time() - start_time) * 1000
            # Calculate data size from DataItem messages with arrays
            data_size = 0
            for item in response.data:
                if item.HasField('array'):
                    data_size += len(item.array.data)
            self.performance_monitor.record_experiment(experiment_name, duration_ms, True, data_size)

            # Log final response structure before sending
            logger.info(f"Final response structure: success={response.success}, data_items={len(response.data)}, plots={len(response.plots)}")
            if response.plots:
                for i, plot in enumerate(response.plots):
                    logger.info(f"Plot {i}: description='{plot.description}', has_json={len(plot.plotly_json) > 0}, has_png={len(plot.image_png) > 0}")

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
        Enhanced to include EPII_INFO and run docstrings.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            ExperimentsResponse with experiment specifications
        """
        response = epii_pb2.ExperimentsResponse()

        # Get all discovered experiments
        for exp_name in self.experiment_router.experiment_map.keys():
            exp_spec = epii_pb2.ExperimentSpec()
            exp_spec.name = exp_name

            # Get EPII_INFO and docstring
            exp_info = self.experiment_router.get_experiment_info(exp_name)
            epii_info = exp_info.get('epii_info', {})

            # Use EPII_INFO description
            exp_spec.description = epii_info.get('description', f'{exp_name} experiment')

            # Add purpose to description if available
            if epii_info.get('purpose'):
                exp_spec.description += f". {epii_info['purpose']}"

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

            # Add attribute specifications from EPII_INFO
            if epii_info.get('attributes'):
                for attr_name, attr_info in epii_info['attributes'].items():
                    attr_spec = epii_pb2.AttributeSpec()
                    attr_spec.name = attr_name
                    attr_spec.type = attr_info.get('type', 'unknown')
                    attr_spec.description = attr_info.get('description', '')
                    if 'shape' in attr_info:
                        attr_spec.shape = str(attr_info['shape'])
                    exp_spec.attributes.append(attr_spec)

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
            # If no specific parameters requested, return all parameters
            if not request.parameter_names:
                logger.debug("GetParameters called - returning all parameters")
                all_params = self.parameter_manager.get_all_parameters()

                # Convert all parameters to protobuf format
                for name, value in all_params.items():
                    serialized = self.parameter_manager.serialize_value(value)
                    response.parameters[name] = str(serialized)

                logger.debug(f"Returned {len(all_params)} parameters")
            else:
                # Return only requested parameters
                logger.debug(f"GetParameters called for {len(request.parameter_names)} specific parameters")

                for param_name in request.parameter_names:
                    value = self.parameter_manager.get_parameter(param_name)
                    if value is not None:
                        serialized = self.parameter_manager.serialize_value(value)
                        response.parameters[param_name] = str(serialized)
                    else:
                        # Parameter not found, return null
                        response.parameters[param_name] = "null"

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
                # StatusResponse doesn't have a message field, use error_message for info
                return response

            # Track successes and failures
            success_count = 0
            failed_params = []

            for param_name, param_value in request.parameters.items():
                try:
                    # Directly set the parameter without validation
                    success = self.parameter_manager.set_parameter(param_name, param_value)
                    if success:
                        success_count += 1
                        logger.debug(f"Set parameter {param_name} = {param_value}")
                    else:
                        failed_params.append(f"{param_name}: unable to set")

                except Exception as e:
                    failed_params.append(f"{param_name}: {str(e)}")
                    logger.warning(f"Error setting parameter '{param_name}': {e}")

            # Build response based on results
            if failed_params:
                if success_count == 0:
                    # All parameters failed
                    response.success = False
                    response.error_message = f"Failed to set all parameters: {'; '.join(failed_params)}"
                else:
                    # Partial success
                    response.success = True
                    response.error_message = f"Set {success_count}/{len(request.parameters)} parameters. Failed: {'; '.join(failed_params)}"
            else:
                # All parameters succeeded
                response.success = True
                # Note: StatusResponse only has success and error_message fields

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

            # Get all parameters as a flat dictionary
            all_params = self.parameter_manager.get_all_parameters()

            # Convert each parameter to ParameterInfo
            for param_name, param_value in all_params.items():
                param_info = epii_pb2.ParameterInfo()
                param_info.name = param_name

                # Determine type from value
                if isinstance(param_value, bool):
                    param_info.type = "bool"
                elif isinstance(param_value, int):
                    param_info.type = "int"
                elif isinstance(param_value, float):
                    param_info.type = "float"
                elif isinstance(param_value, str):
                    param_info.type = "string"
                elif isinstance(param_value, (list, np.ndarray)):
                    param_info.type = "array"
                elif isinstance(param_value, dict):
                    param_info.type = "dict"
                else:
                    param_info.type = "unknown"

                # Serialize current value
                param_info.current_value = str(self.parameter_manager.serialize_value(param_value))

                # Simple description
                param_info.description = f"Parameter {param_name}"

                # No validation means nothing is read-only
                param_info.read_only = False

                response.parameters.append(param_info)

            logger.debug(f"Listed {len(all_params)} parameters")
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
        
        # End Chronicle logging if active
        if hasattr(self, '_chronicle') and self._chronicle:
            try:
                if hasattr(self._chronicle, 'is_recording') and self._chronicle.is_recording():
                    self._chronicle.end_log()
                    logger.info("Chronicle session ended")
            except Exception as e:
                logger.warning(f"Error ending Chronicle session: {e}")
        
        logger.info("EPII service shutdown complete")
