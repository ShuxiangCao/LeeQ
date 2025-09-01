"""
EPII Configuration Management

This module handles loading and validation of JSON configuration files
for EPII daemon setup and initialization.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from leeq.setups.setup_base import ExperimentalSetup
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon

logger = logging.getLogger(__name__)


class EPIIConfig:
    """
    Configuration manager for EPII daemon.

    Handles loading JSON configuration files and environment variables
    for setting up the EPII service with appropriate LeeQ backends.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or environment.

        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_defaults()

        if config_path:
            self.load_from_file(config_path)

        self._load_environment()

    def _load_defaults(self):
        """Load default configuration values."""
        self.config = {
            "setup_type": "simulation",  # Changed from "type" to match test fixtures
            "setup_name": "default_setup",  # Changed from "name" to match test fixtures
            "port": 50051,
            "log_level": "INFO",
            "max_workers": 10,
            "num_qubits": 2,  # Default number of qubits for simulation
            "parameters": {},
            "simulation_backend": "numpy"  # Default simulation backend
        }

    def load_from_file(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file

        Returns:
            Loaded configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        config_file = Path(path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
                logger.info("Loaded configuration from %s", path)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in configuration file %s: %s", path, e)
            raise

        return self.config

    def _load_environment(self):
        """Load configuration from environment variables."""
        # Override with environment variables if present
        if "LEEQ_EPII_PORT" in os.environ:
            self.config["port"] = int(os.environ["LEEQ_EPII_PORT"])

        if "LEEQ_EPII_LOG_LEVEL" in os.environ:
            self.config["log_level"] = os.environ["LEEQ_EPII_LOG_LEVEL"]

        if "LEEQ_EPII_CONFIG_DIR" in os.environ:
            self.config["config_dir"] = os.environ["LEEQ_EPII_CONFIG_DIR"]

    def validate(self) -> bool:
        """
        Validate the configuration for required fields.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Support both old and new field names for backward compatibility
        type_field = "setup_type" if "setup_type" in self.config else "type"
        name_field = "setup_name" if "setup_name" in self.config else "name"

        required_fields = [type_field, name_field, "port"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate port range
        port = self.config["port"]
        if not (50051 <= port <= 50099):
            raise ValueError(f"Port must be between 50051 and 50099, got {port}")

        # Validate setup type
        valid_types = ["simulation", "hardware"]
        setup_type = self.config.get(type_field)
        if setup_type not in valid_types:
            raise ValueError(f"Invalid setup type: {setup_type}")

        # Validate simulation-specific parameters
        if setup_type == "simulation":
            if "num_qubits" in self.config:
                num_qubits = self.config["num_qubits"]
                if not isinstance(num_qubits, int) or num_qubits < 1:
                    raise ValueError(f"num_qubits must be a positive integer, got {num_qubits}")

            if "simulation_backend" in self.config:
                valid_backends = ["numpy", "qutip", "high_level"]
                backend = self.config.get("simulation_backend", "numpy")
                if backend not in valid_backends:
                    raise ValueError(f"Invalid simulation backend: {backend}")

        logger.info("Configuration validated successfully")
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def create_setup(self) -> ExperimentalSetup:
        """
        Create a LeeQ setup based on the configuration.

        Returns:
            LeeQ ExperimentalSetup instance

        Raises:
            ValueError: If setup type is not supported
            ImportError: If required modules are not available
        """
        # Support both old and new field names
        setup_type = self.config.get("setup_type", self.config.get("type", "simulation"))
        setup_name = self.config.get("setup_name", self.config.get("name", "default_setup"))

        logger.info("Creating %s setup: %s", setup_type, setup_name)

        if setup_type == "simulation":
            return self._create_simulation_setup(setup_name)
        elif setup_type == "hardware":
            return self._create_hardware_setup(setup_name)
        else:
            raise ValueError(f"Unsupported setup type: {setup_type}")

    def _create_simulation_setup(self, name: str) -> ExperimentalSetup:
        """
        Create a simulation setup based on configuration.

        Args:
            name: Name for the setup

        Returns:
            Simulation setup instance
        """
        backend = self.config.get("simulation_backend",
                                 self.config.get("parameters", {}).get("simulation_backend", "numpy"))
        num_qubits = self.config.get("num_qubits", 2)

        logger.info(f"Creating {backend} simulation with {num_qubits} qubits")

        if backend == "high_level":
            return self._create_high_level_simulation(name, num_qubits)
        elif backend == "numpy":
            return self._create_numpy_simulation(name, num_qubits)
        elif backend == "qutip":
            return self._create_qutip_simulation(name, num_qubits)
        else:
            raise ValueError(f"Unsupported simulation backend: {backend}")

    def _create_high_level_simulation(self, name: str, num_qubits: int) -> ExperimentalSetup:
        """
        Create a high-level simulation setup with proper qubit elements.

        Args:
            name: Setup name
            num_qubits: Number of qubits

        Returns:
            HighLevelSimulationSetup instance with TransmonElement qubits
        """
        from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
        from leeq.core.elements.built_in.qudit_transmon import TransmonElement

        # Create virtual qubits from configuration
        virtual_qubits = {}
        qubits_config = self.config.get("parameters", {}).get("qubits", {})

        for i in range(num_qubits):
            qubit_name = f"q{i}"
            qubit_params = qubits_config.get(qubit_name, {})

            # Use defaults if not specified
            virtual_qubit = VirtualTransmon(
                name=qubit_name,
                qubit_frequency=qubit_params.get("f01", 5000 + i * 100) / 1e6,  # Convert Hz to MHz
                anharmonicity=qubit_params.get("anharmonicity", -330) / 1e6,  # Convert Hz to MHz
                t1=qubit_params.get("characterizations.SimpleT1", 20e-6) * 1e6,  # Convert seconds to microseconds
                t2=qubit_params.get("t2", 15e-6) * 1e6,  # Convert seconds to microseconds
                readout_frequency=qubit_params.get("readout_frequency", 8800 + i * 200),
                readout_linewith=1,
                readout_dipsersive_shift=0.3,
                truncate_level=4,
                quiescent_state_distribution=[0.9, 0.08, 0.02, 0]
            )
            # Map to channel index (drive channels are even, readout channels are odd)
            virtual_qubits[i * 2] = virtual_qubit

        # Create coupling map if specified
        coupling_map = {}
        couplings_config = self.config.get("parameters", {}).get("couplings", {})
        for coupling_name, coupling_params in couplings_config.items():
            # Parse coupling name like "q0-q1"
            if "-" in coupling_name:
                q1, q2 = coupling_name.split("-")
                q1_idx = int(q1[1:]) if q1.startswith("q") else int(q1)
                q2_idx = int(q2[1:]) if q2.startswith("q") else int(q2)
                strength = coupling_params.get("strength", 5e6) / 1e6  # Convert Hz to MHz
                coupling_map[frozenset([q1_idx * 2, q2_idx * 2])] = strength

        # Create the setup
        setup = HighLevelSimulationSetup(
            name=name,
            virtual_qubits=virtual_qubits,
            coupling_strength_map=coupling_map if coupling_map else None
        )

        # Create TransmonElement objects and add them to the setup
        setup.qubits = []
        for i in range(num_qubits):
            qubit_name = f"q{i}"
            qubit_params = qubits_config.get(qubit_name, {})

            # Get frequencies from config or use defaults
            f01_freq = qubit_params.get("f01", 5000e6 + i * 100e6) / 1e6  # Convert Hz to MHz
            readout_freq = qubit_params.get("readout_frequency", 9645e6 + i * 10e6) / 1e6  # Convert Hz to MHz
            anharmonicity = qubit_params.get("anharmonicity", -330e6) / 1e6  # Convert Hz to MHz

            # Create TransmonElement with proper parameter structure
            element_params = {
                'hrid': qubit_name.upper(),  # Use uppercase for consistency
                'lpb_collections': {
                    'f01': {
                        'type': 'SimpleDriveCollection',
                        'freq': f01_freq,
                        'channel': i * 2,  # Drive channel
                        'shape': 'blackman_drag',
                        'amp': 0.5,  # Default amplitude
                        'phase': 0.,
                        'width': 0.05,  # 50ns pulse width
                        'alpha': 500,
                        'trunc': 1.2
                    },
                    'f12': {
                        'type': 'SimpleDriveCollection',
                        'freq': f01_freq + anharmonicity,
                        'channel': i * 2,  # Same drive channel
                        'shape': 'blackman_drag',
                        'amp': 0.1 / np.sqrt(2),
                        'phase': 0.,
                        'width': 0.025,
                        'alpha': 425,
                        'trunc': 1.2
                    }
                },
                'measurement_primitives': {
                    '0': {
                        'type': 'SimpleDispersiveMeasurement',
                        'freq': readout_freq,
                        'channel': i * 2 + 1,  # Readout channel
                        'shape': 'square',
                        'amp': 0.15,
                        'phase': 0.,
                        'width': 1,  # 1us readout
                        'trunc': 1.2,
                        'distinguishable_states': [0, 1]
                    }
                }
            }

            # Create the TransmonElement
            transmon = TransmonElement(name=qubit_name.upper(), parameters=element_params)

            # Add to setup's qubit list
            setup.qubits.append(transmon)
            # Also set as attribute for direct access
            setattr(setup, qubit_name, transmon)

        return setup

    def _create_numpy_simulation(self, name: str, num_qubits: int) -> ExperimentalSetup:
        """
        Create a numpy-based fast simulation setup.

        Args:
            name: Setup name
            num_qubits: Number of qubits

        Returns:
            Numpy2QVirtualDeviceSetup instance
        """
        from leeq.setups.built_in.setup_numpy_2q_virtual_device import Numpy2QVirtualDeviceSetup

        # For now, use the default 2-qubit setup
        # In the future, this can be extended to support configurable parameters
        sampling_rate = self.config.get("parameters", {}).get("sampling_rate", 1e6)

        return Numpy2QVirtualDeviceSetup(sampling_rate=sampling_rate)

    def _create_qutip_simulation(self, name: str, num_qubits: int) -> ExperimentalSetup:
        """
        Create a QuTiP-based simulation setup.

        Args:
            name: Setup name
            num_qubits: Number of qubits

        Returns:
            QuTiP setup instance
        """
        from leeq.setups.built_in.setup_qutip_2q_local import QuTipLocalSetup

        # Create QuTiP setup with configured parameters
        # This is a simplified version - extend as needed
        return QuTipLocalSetup()

    def _create_hardware_setup(self, name: str) -> ExperimentalSetup:
        """
        Create a hardware setup based on configuration.

        Args:
            name: Name for the setup

        Returns:
            Hardware setup instance

        Raises:
            NotImplementedError: Hardware setups require specific lab configurations
        """
        hardware_type = self.config.get("parameters", {}).get("hardware_type", "qubic")

        if hardware_type == "qubic":
            # Hardware setups require specific lab configurations
            # For now, raise an informative error
            raise NotImplementedError(
                "Hardware setup creation requires specific lab configuration. "
                "Please provide QUBIC hardware parameters in the configuration."
            )
        else:
            raise ValueError(f"Unsupported hardware type: {hardware_type}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def save_to_file(self, path: str):
        """
        Save the current configuration to a JSON file.

        Args:
            path: Path to save the configuration file
        """
        config_file = Path(path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info("Configuration saved to %s", path)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EPIIConfig":
        """
        Create an EPIIConfig instance from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            EPIIConfig instance
        """
        instance = cls()
        instance.config = config_dict.copy()
        return instance


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file path.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configuration dictionary
    """
    config_manager = EPIIConfig(config_path)
    config_manager.validate()
    return config_manager.to_dict()


def create_setup_from_config(config: Dict[str, Any]) -> ExperimentalSetup:
    """
    Create a LeeQ setup from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        LeeQ ExperimentalSetup instance
    """
    config_manager = EPIIConfig.from_dict(config)
    config_manager.validate()
    return config_manager.create_setup()


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        config_manager = EPIIConfig.from_dict(config)
        config_manager.validate()
        return True
    except (ValueError, ImportError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
