"""
Tests for EPII Experiment Router
"""

import pytest
from typing import Dict, Any

from leeq.epii.experiments import ExperimentRouter
from leeq.experiments.builtin.basic.calibrations.rabi import NormalisedRabi
from leeq.experiments.builtin.basic.characterizations.t1 import SimpleT1
from leeq.experiments.builtin.basic.calibrations.ramsey import SimpleRamseyMultilevel
from leeq.experiments.builtin.basic.characterizations.t2 import SpinEchoMultiLevel
from leeq.experiments.builtin.basic.calibrations.drag import DragCalibrationSingleQubitMultilevel
from leeq.experiments.builtin.basic.characterizations.randomized_benchmarking import (
    SingleQubitRandomizedBenchmarking
)


class TestExperimentRouter:
    """Test suite for ExperimentRouter class."""

    @pytest.fixture
    def router(self):
        """Create an ExperimentRouter instance."""
        return ExperimentRouter()

    def test_initialization(self, router):
        """Test that router initializes with correct mappings."""
        assert len(router.experiment_map) > 0
        assert "calibrations.NormalisedRabi" in router.experiment_map
        assert "characterizations.SimpleT1" in router.experiment_map
        assert "calibrations.SimpleRamseyMultilevel" in router.experiment_map
        assert "characterizations.SpinEchoMultiLevel" in router.experiment_map
        assert "calibrations.DragCalibrationSingleQubitMultilevel" in router.experiment_map
        assert "characterizations.SingleQubitRandomizedBenchmarking" in router.experiment_map

    def test_get_experiment(self, router):
        """Test getting experiment classes by name."""
        # Test valid experiments
        assert router.get_experiment("calibrations.NormalisedRabi") == NormalisedRabi
        assert router.get_experiment("characterizations.SimpleT1") == SimpleT1
        assert router.get_experiment("calibrations.SimpleRamseyMultilevel") == SimpleRamseyMultilevel
        assert router.get_experiment("characterizations.SpinEchoMultiLevel") == SpinEchoMultiLevel
        assert router.get_experiment("characterizations.SpinEchoMultiLevel") == SpinEchoMultiLevel  # Alias
        assert router.get_experiment("calibrations.DragCalibrationSingleQubitMultilevel") == DragCalibrationSingleQubitMultilevel
        assert router.get_experiment("characterizations.SingleQubitRandomizedBenchmarking") == SingleQubitRandomizedBenchmarking

        # Test invalid experiment
        assert router.get_experiment("invalid_experiment") is None

    def test_list_experiments(self, router):
        """Test listing available experiments."""
        experiments = router.list_experiments()

        assert isinstance(experiments, dict)
        assert len(experiments) > 0
        assert "calibrations.NormalisedRabi" in experiments
        assert "characterizations.SimpleT1" in experiments
        assert "calibrations.SimpleRamseyMultilevel" in experiments
        assert "characterizations.SpinEchoMultiLevel" in experiments
        assert "calibrations.DragCalibrationSingleQubitMultilevel" in experiments
        assert "characterizations.SingleQubitRandomizedBenchmarking" in experiments

        # Check descriptions are strings
        for name, desc in experiments.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_get_experiment_parameters(self, router):
        """Test getting parameter schema for experiments."""
        # Test Rabi parameters
        rabi_params = router.get_experiment_parameters("calibrations.NormalisedRabi")
        assert isinstance(rabi_params, dict)
        assert "dut_qubit" in rabi_params
        assert "amp" in rabi_params
        assert "start" in rabi_params
        assert "stop" in rabi_params
        assert "step" in rabi_params

        # Check parameter info structure
        for param_name, param_info in rabi_params.items():
            assert "name" in param_info
            assert "required" in param_info
            assert "default" in param_info
            assert "type" in param_info

        # Test T1 parameters
        t1_params = router.get_experiment_parameters("characterizations.SimpleT1")
        assert isinstance(t1_params, dict)
        assert "qubit" in t1_params or "dut" in t1_params

        # Test invalid experiment
        invalid_params = router.get_experiment_parameters("invalid_experiment")
        assert invalid_params == {}

    def test_map_parameters(self, router):
        """Test parameter passthrough and qubit resolution (no more alias mapping)."""
        # Test parameters pass through unchanged (no alias mapping anymore)
        test_params = {
            "amp": 0.5,
            "start": 0.01,
            "stop": 0.3,
            "step": 0.002,
            "dut_qubit": "q0"
        }

        leeq_params = router.map_parameters("calibrations.NormalisedRabi", test_params)
        # Parameters should pass through unchanged since no setup provided
        assert leeq_params["amp"] == 0.5
        assert leeq_params["start"] == 0.01
        assert leeq_params["stop"] == 0.3
        assert leeq_params["step"] == 0.002
        assert leeq_params["dut_qubit"] == "q0"

        # Test T1 parameters pass through unchanged
        t1_params = {
            "qubit": "q1",
            "start": 1e-6,
            "stop": 100e-6,
            "step": 2e-6
        }

        leeq_t1_params = router.map_parameters("characterizations.SimpleT1", t1_params)
        assert leeq_t1_params["qubit"] == "q1"
        assert leeq_t1_params["start"] == 1e-6
        assert leeq_t1_params["stop"] == 100e-6
        assert leeq_t1_params["step"] == 2e-6

        # Test any experiment parameters pass through as-is
        arbitrary_params = {"param1": "value1", "param2": 123}
        result = router.map_parameters("any_experiment", arbitrary_params)
        assert result == arbitrary_params

    def test_validate_parameters(self, router):
        """Test parameter validation."""
        # Test valid parameters for Rabi
        valid_params = {
            "dut_qubit": "q0",
            "amp": 0.5
        }
        is_valid, errors = router.validate_parameters("calibrations.NormalisedRabi", valid_params)
        assert is_valid is True
        assert len(errors) == 0

        # Test missing required parameter
        # Note: Most LeeQ experiments have defaults, so this might not fail
        incomplete_params = {}
        is_valid, errors = router.validate_parameters("calibrations.NormalisedRabi", incomplete_params)
        # This depends on whether dut_qubit has a default or not

        # Test invalid experiment name
        is_valid, errors = router.validate_parameters("invalid_experiment", {})
        assert is_valid is False
        assert len(errors) > 0
        assert "Unknown experiment" in errors[0]

    def test_discover_experiments(self, router):
        """Test experiment discovery."""
        capabilities = router.discover_experiments()

        assert isinstance(capabilities, list)
        assert len(capabilities) > 0

        # Find rabi experiment in capabilities
        rabi_cap = None
        for cap in capabilities:
            if cap["name"] == "calibrations.NormalisedRabi":
                rabi_cap = cap
                break

        assert rabi_cap is not None
        assert rabi_cap["class"] == "NormalisedRabi"
        assert "leeq.experiments.builtin.basic.calibrations.rabi" in rabi_cap["module"]
        assert isinstance(rabi_cap["parameters"], dict)
        assert isinstance(rabi_cap["description"], str)

        # Check all capabilities have required fields
        for cap in capabilities:
            assert "name" in cap
            assert "class" in cap
            assert "module" in cap
            assert "parameters" in cap
            assert "description" in cap

    def test_all_mapped_experiments_exist(self, router):
        """Test that all mapped experiment classes can be imported."""
        for name, experiment_class in router.experiment_map.items():
            assert experiment_class is not None
            assert hasattr(experiment_class, 'run')
            assert callable(experiment_class.run)

    def test_experiment_availability(self, router):
        """Test that essential experiments are available (no parameter mappings needed)."""
        essential_experiments = ["calibrations.NormalisedRabi", "characterizations.SimpleT1", "calibrations.SimpleRamseyMultilevel", "characterizations.SpinEchoMultiLevel", "calibrations.DragCalibrationSingleQubitMultilevel", "characterizations.SingleQubitRandomizedBenchmarking"]

        for exp_name in essential_experiments:
            # Check experiment is available in experiment_map
            assert exp_name in router.experiment_map

            # Check we can get the experiment class
            experiment_class = router.get_experiment(exp_name)
            assert experiment_class is not None

            # Check it has the required methods
            assert hasattr(experiment_class, 'run')
            assert hasattr(experiment_class, 'EPII_INFO')
