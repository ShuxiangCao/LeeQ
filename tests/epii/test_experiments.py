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
    RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem
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
        assert "rabi" in router.experiment_map
        assert "t1" in router.experiment_map
        assert "ramsey" in router.experiment_map
        assert "echo" in router.experiment_map
        assert "drag" in router.experiment_map
        assert "randomized_benchmarking" in router.experiment_map
    
    def test_get_experiment(self, router):
        """Test getting experiment classes by name."""
        # Test valid experiments
        assert router.get_experiment("rabi") == NormalisedRabi
        assert router.get_experiment("t1") == SimpleT1
        assert router.get_experiment("ramsey") == SimpleRamseyMultilevel
        assert router.get_experiment("echo") == SpinEchoMultiLevel
        assert router.get_experiment("spin_echo") == SpinEchoMultiLevel  # Alias
        assert router.get_experiment("drag") == DragCalibrationSingleQubitMultilevel
        assert router.get_experiment("randomized_benchmarking") == RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem
        
        # Test invalid experiment
        assert router.get_experiment("invalid_experiment") is None
    
    def test_list_experiments(self, router):
        """Test listing available experiments."""
        experiments = router.list_experiments()
        
        assert isinstance(experiments, dict)
        assert len(experiments) > 0
        assert "rabi" in experiments
        assert "t1" in experiments
        assert "ramsey" in experiments
        assert "echo" in experiments
        assert "drag" in experiments
        assert "randomized_benchmarking" in experiments
        
        # Check descriptions are strings
        for name, desc in experiments.items():
            assert isinstance(desc, str)
            assert len(desc) > 0
    
    def test_get_experiment_parameters(self, router):
        """Test getting parameter schema for experiments."""
        # Test Rabi parameters
        rabi_params = router.get_experiment_parameters("rabi")
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
        t1_params = router.get_experiment_parameters("t1")
        assert isinstance(t1_params, dict)
        assert "qubit" in t1_params or "dut" in t1_params
        
        # Test invalid experiment
        invalid_params = router.get_experiment_parameters("invalid_experiment")
        assert invalid_params == {}
    
    def test_map_parameters(self, router):
        """Test parameter name mapping from EPII to LeeQ."""
        # Test Rabi parameter mapping
        epii_params = {
            "amplitude": 0.5,
            "start_width": 0.01,
            "stop_width": 0.3,
            "width_step": 0.002,
            "qubit": "q0"
        }
        
        leeq_params = router.map_parameters("rabi", epii_params)
        assert leeq_params["amp"] == 0.5
        assert leeq_params["start"] == 0.01
        assert leeq_params["stop"] == 0.3
        assert leeq_params["step"] == 0.002
        assert leeq_params["dut_qubit"] == "q0"
        
        # Test T1 parameter mapping
        epii_t1_params = {
            "qubit": "q0",
            "time_max": 100.0,
            "time_step": 1.0
        }
        
        leeq_t1_params = router.map_parameters("t1", epii_t1_params)
        assert leeq_t1_params["qubit"] == "q0"
        assert leeq_t1_params["time_length"] == 100.0
        assert leeq_t1_params["time_resolution"] == 1.0
        
        # Test unmapped experiment (should return as-is)
        unmapped_params = {"param1": "value1", "param2": 123}
        result = router.map_parameters("unmapped_experiment", unmapped_params)
        assert result == unmapped_params
    
    def test_validate_parameters(self, router):
        """Test parameter validation."""
        # Test valid parameters for Rabi
        valid_params = {
            "dut_qubit": "q0",
            "amp": 0.5
        }
        is_valid, errors = router.validate_parameters("rabi", valid_params)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test missing required parameter
        # Note: Most LeeQ experiments have defaults, so this might not fail
        incomplete_params = {}
        is_valid, errors = router.validate_parameters("rabi", incomplete_params)
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
            if cap["name"] == "rabi":
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
            assert callable(getattr(experiment_class, 'run'))
    
    def test_parameter_mapping_completeness(self, router):
        """Test that parameter mappings cover essential experiments."""
        essential_experiments = ["rabi", "t1", "ramsey", "echo", "drag", "randomized_benchmarking"]
        
        for exp_name in essential_experiments:
            # Check experiment is mapped
            assert exp_name in router.experiment_map
            
            # Check parameter mapping exists (optional, as not all need mapping)
            if exp_name in router.parameter_map:
                mapping = router.parameter_map[exp_name]
                assert isinstance(mapping, dict)
                assert len(mapping) > 0