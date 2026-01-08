"""LeeQ EPII Backend - bridges declarative service to EPIIBackend interface.

This module provides a backend that loads the LeeQ declarative service file
and exposes it via the EPIIBackend interface for in-process execution.

Usage:
    1. Register the backend with the factory at startup:
       from leeq.epii.backend import register_leeq_backend
       register_leeq_backend()

    2. Configure epii_config.yaml:
       epii:
         enabled: true
         backend_type: leeq
         backend_config:
           service_file: /path/to/leeq_epii_service.py
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def create_leeq_backend_class():
    """Create the LeeQBackend class dynamically to avoid import issues."""
    from quantum_calibration_agent.epii.backends.base import EPIIBackend
    from quantum_calibration_agent.epii.server.declarative import load_service_from_file

    class LeeQBackend(EPIIBackend):
        """EPII backend that uses LeeQ declarative service directly in-process."""

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            super().__init__(config)
            self._service = None

            # Get service file path from config or use default
            self.service_file = self.config.get(
                "service_file",
                os.path.join(os.path.dirname(__file__), "leeq_epii_service.py")
            )

            # Set LEEQ_EPII_CONFIG if specified
            leeq_config = self.config.get("leeq_config")
            if leeq_config:
                os.environ["LEEQ_EPII_CONFIG"] = leeq_config

        async def initialize(self) -> None:
            """Load the LeeQ declarative service."""
            logger.info(f"Loading LeeQ service from: {self.service_file}")
            self._service = load_service_from_file(self.service_file)
            logger.info(f"LeeQ service loaded: {self._service.get_framework_info()}")
            await super().initialize()

        async def ping(self) -> Dict[str, Any]:
            """Health check."""
            if self._service is None:
                return {
                    "status": "error",
                    "message": "LeeQ service not initialized",
                    "timestamp": time.time()
                }

            info = self._service.get_framework_info()
            return {
                "status": "healthy",
                "message": "LeeQ backend is operational",
                "timestamp": time.time(),
                "backend_type": "leeq",
                "framework_name": info.get("name", "LeeQ"),
                "framework_version": info.get("version", "unknown"),
                "experiments_available": len(self._service.list_experiments())
            }

        async def get_capabilities(self) -> Dict[str, Any]:
            """Get LeeQ platform capabilities."""
            if self._service is None:
                await self.initialize()

            info = self._service.get_framework_info()
            experiments = self._service.list_experiments()

            return {
                "framework_name": info.get("name", "LeeQ"),
                "framework_version": info.get("version", "1.0.0"),
                "epii_version": "1.0.0",
                "supported_backends": info.get("supported_backends", ["simulation"]),
                "data_formats": ["numpy", "json"],
                "experiment_types": [
                    {
                        "name": exp.name,
                        "description": exp.description,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type,
                                "required": p.required,
                                "default_value": p.default_value,
                                "description": p.description,
                                "allowed_values": p.allowed_values
                            }
                            for p in exp.parameters
                        ],
                        "output_parameters": exp.output_parameters
                    }
                    for exp in experiments
                ]
            }

        async def list_experiments(self) -> List[Dict[str, Any]]:
            """List available experiments."""
            if self._service is None:
                await self.initialize()

            experiments = self._service.list_experiments()
            return [
                {
                    "name": exp.name,
                    "description": exp.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "required": p.required,
                            "default_value": p.default_value,
                            "description": p.description,
                            "allowed_values": p.allowed_values
                        }
                        for p in exp.parameters
                    ],
                    "output_parameters": exp.output_parameters
                }
                for exp in experiments
            ]

        async def run_experiment(self,
                                experiment_type: str,
                                parameters: Dict[str, Any],
                                return_raw_data: bool = True,
                                return_plots: bool = True,
                                **kwargs) -> Dict[str, Any]:
            """Run experiment via LeeQ declarative service."""
            if self._service is None:
                await self.initialize()

            try:
                result = self._service.run_experiment(experiment_type, parameters)
                return result
            except Exception as e:
                logger.error(f"Experiment {experiment_type} failed: {e}")
                return self.format_error_response(e, experiment_type)

        async def get_parameters(self,
                                parameter_names: Optional[List[str]] = None) -> Dict[str, str]:
            """Get parameter values."""
            if self._service is None:
                await self.initialize()

            all_params = {p.name: p.current_value for p in self._service.list_parameters()}

            if parameter_names is None:
                return all_params

            return {
                name: all_params.get(name, "unknown")
                for name in parameter_names
            }

        async def list_parameters(self) -> List[Dict[str, Any]]:
            """List parameters with metadata."""
            if self._service is None:
                await self.initialize()

            return [
                {
                    "name": p.name,
                    "type": p.type,
                    "current_value": p.current_value,
                    "description": p.description,
                    "read_only": p.read_only
                }
                for p in self._service.list_parameters()
            ]

        async def set_parameters(self, parameters: Dict[str, str]) -> Dict[str, Any]:
            """Set parameter values."""
            if self._service is None:
                await self.initialize()

            updated, failed = [], []
            for name, value in parameters.items():
                try:
                    success = self._service.set_parameter(name, value)
                    if success:
                        updated.append(name)
                    else:
                        failed.append(name)
                except Exception:
                    failed.append(name)

            return {
                "success": len(failed) == 0,
                "updated": updated,
                "failed": failed,
                "message": f"Updated {len(updated)} parameters"
            }

        async def cleanup(self) -> None:
            """Clean up resources."""
            self._service = None
            await super().cleanup()

        def __repr__(self) -> str:
            return f"LeeQBackend(service_file={self.service_file}, initialized={self._initialized})"

    return LeeQBackend


def register_leeq_backend():
    """Register the LeeQ backend with the global factory.

    Call this at startup before using the EPII client.
    """
    try:
        from quantum_calibration_agent.epii.backends import register_backend

        LeeQBackend = create_leeq_backend_class()
        register_backend("leeq")(LeeQBackend)
        logger.info("Registered 'leeq' backend with EPII factory")
    except ImportError as e:
        logger.warning(f"Could not register LeeQ backend: {e}")
