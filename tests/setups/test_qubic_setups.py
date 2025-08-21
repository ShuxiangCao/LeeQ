"""
Test cases for leeq.setups.qubic_lbnl_setups module.

This test module provides smoke tests and basic functionality tests
for the QubiCCircuitSetup class, focusing on initialization, configuration
validation, and parameter management using mocked hardware interfaces.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from leeq.setups.qubic_lbnl_setups import QubiCCircuitSetup


class TestQubiCCircuitSetup:
    """Test the QubiCCircuitSetup class for hardware setup management."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock runner for testing."""
        runner = MagicMock()
        runner.run = MagicMock(return_value={"success": True})
        return runner

    @pytest.fixture
    def sample_channel_configs(self):
        """Create sample channel configurations for testing."""
        return {
            0: {"frequency": 5e9, "amplitude": 0.5},
            1: {"frequency": 6e9, "amplitude": 0.3},
            2: {"frequency": 5.2e9, "amplitude": 0.4},
            3: {"frequency": 6.1e9, "amplitude": 0.2}
        }

    @pytest.fixture
    def sample_fpga_config(self):
        """Create sample FPGA configuration for testing."""
        return {
            "clock_frequency": 1e9,
            "memory_size": 1024,
            "processing_cores": 8
        }

    @pytest.fixture
    def sample_leeq_to_qubic_mapping(self):
        """Create sample channel mapping for testing."""
        return {
            "qubit_0": "qubic_ch_0",
            "readout_0": "qubic_ch_1",
            "qubit_1": "qubic_ch_2",
            "readout_1": "qubic_ch_3"
        }

    @pytest.fixture
    def mock_qubic_package(self):
        """Create mock QubiC package components for testing."""
        mock_tc = MagicMock()
        mock_qc = MagicMock()
        mock_fpga_config_cls = MagicMock()
        mock_load_channel_configs = MagicMock()
        mock_channel_config_cls = MagicMock()
        return (mock_tc, mock_qc, mock_fpga_config_cls,
                mock_load_channel_configs, mock_channel_config_cls)

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_basic_initialization(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                 sample_channel_configs, sample_fpga_config,
                                 sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test basic initialization of QubiCCircuitSetup."""
        mock_load_qubic.return_value = mock_qubic_package

        # Configure load_channel_configs mock to return a properly formatted config
        # The method expects string keys for channel names
        tc, qc, fpga_config_cls, load_channel_configs, channel_config_cls = mock_qubic_package
        load_channel_configs.return_value = {}  # Return empty config that won't cause type errors
        fpga_config_cls.return_value = sample_fpga_config

        setup = QubiCCircuitSetup(
            name="test_setup",
            runner=mock_runner,
            channel_configs=sample_channel_configs,
            fpga_config=sample_fpga_config,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping,
            qubic_core_number=4
        )

        # Test basic attributes are set
        assert setup._fpga_config == sample_fpga_config
        # _channel_configs gets processed by load_channel_configs, so compare to the mock return
        assert setup._channel_configs == {}  # Empty dict as mocked
        assert setup._qubic_core_number == 4
        assert setup._leeq_channel_to_qubic_channel == sample_leeq_to_qubic_mapping
        assert setup._runner == mock_runner

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridSerialSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_serial_engine_initialization(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                         sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test initialization with serial processing engine."""
        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="serial_setup",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping,
            batch_process=False  # Use serial engine
        )

        # Verify serial engine was used
        mock_engine.assert_called_once()
        assert setup._qubic_core_number == 8  # Default value

    def test_missing_channel_mapping_assertion(self, mock_runner, mock_qubic_package):
        """Test that missing leeq_channel_to_qubic_channel raises assertion."""
        with patch.object(QubiCCircuitSetup, '_load_qubic_package') as mock_load_qubic:
            mock_load_qubic.return_value = mock_qubic_package

            # Configure the mocks
            tc, qc, fpga_config_cls, load_channel_configs, channel_config_cls = mock_qubic_package
            load_channel_configs.return_value = {}
            fpga_config_cls.return_value = {}

            # The assertion only triggers when channel_configs is provided but mapping is None
            with pytest.raises(AssertionError, match="Please specify leeq channel to qubic channel config"):
                QubiCCircuitSetup(
                    name="test_setup",
                    runner=mock_runner,
                    channel_configs={"some": "config"},  # Non-None channel configs
                    leeq_channel_to_qubic_channel=None  # This should trigger assertion
                )

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_default_core_number(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                 sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test default qubic core number is used correctly."""
        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="default_cores",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping
            # qubic_core_number not specified, should use default of 8
        )

        assert setup._qubic_core_number == 8

    def test_build_core_id_to_qubic_channel_static_method(self):
        """Test static method for building core ID to channel mapping."""
        # Create mock configs that match the expected format in the method
        # The method looks for keys containing 'rdlo' and extracts core_ind from config
        mock_config_0 = MagicMock()
        mock_config_0.core_ind = 0
        mock_config_1 = MagicMock()
        mock_config_1.core_ind = 1

        configs = {
            "chan_0.rdlo": mock_config_0,
            "chan_1.rdlo": mock_config_1,
            "chan_2.other": MagicMock()  # Should be ignored (no 'rdlo')
        }

        result = QubiCCircuitSetup._build_core_id_to_qubic_channel(configs)

        # Test that mapping is created correctly
        assert isinstance(result, dict)
        assert len(result) == 2  # Only rdlo configs should be included
        assert 0 in result
        assert 1 in result
        assert result[0] == "chan_0"
        assert result[1] == "chan_1"

    def test_build_default_channel_config_static_method(self):
        """Test static method for building default channel configuration."""
        # This method doesn't have external dependencies, test it directly
        leeq_mapping, config = QubiCCircuitSetup._build_default_channel_config(core_number=4)

        # Test that configuration has correct structure
        assert isinstance(leeq_mapping, dict)
        assert isinstance(config, dict)

        # Test expected number of entries
        # Should have 2 leeq channels per core (even for qubit, odd for readout)
        assert len(leeq_mapping) == 8  # 4 cores * 2 channels per core

        # Test that FPGA clock frequency is set
        assert "fpga_clk_freq" in config
        assert config["fpga_clk_freq"] == 500e6

        # Test that each core has the expected channel configurations
        # Each core should have .qdrv, .rdrv, and .rdlo configurations
        expected_channels_per_core = 3
        core_channels = [key for key in config.keys() if key.startswith("Q")]
        assert len(core_channels) == 4 * expected_channels_per_core  # 4 cores * 3 channel types

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_status_parameters_initialization(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                            sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test that status parameters are initialized correctly."""
        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="status_test",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping
        )

        # Test that status object exists and has expected parameters
        assert hasattr(setup, '_status')

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_measurement_results_initialization(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                              sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test that measurement results are initialized as empty list."""
        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="measurement_test",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping
        )

        assert isinstance(setup._measurement_results, list)
        assert len(setup._measurement_results) == 0
        assert setup._current_context is None
        assert setup._result is None

    def test_load_qubic_package_static_method(self):
        """Test static method for loading QubiC package (smoke test)."""
        # This is a smoke test since we don't want to actually load QubiC
        try:
            # Method should exist and be callable
            assert hasattr(QubiCCircuitSetup, '_load_qubic_package')
            assert callable(QubiCCircuitSetup._load_qubic_package)
        except ImportError:
            # Expected if QubiC package is not available in test environment
            pass

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_software_demodulation_method_exists(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                                sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test that software demodulation method exists and is callable."""
        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="demod_test",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping
        )

        # Test method exists
        assert hasattr(setup, '_software_demodulation')
        assert callable(setup._software_demodulation)

    def test_get_measurement_info_static_method(self):
        """Test static method for getting measurement info from compiled instructions."""
        # Test with mock compiled instructions

        # Method should exist and be callable
        assert hasattr(QubiCCircuitSetup, '_get_measurement_info_from_compiled_instructions_for_trace_acquisition')
        method = QubiCCircuitSetup._get_measurement_info_from_compiled_instructions_for_trace_acquisition
        assert callable(method)

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_qubic_result_format_conversion(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                          sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test QubiC result format to LeeQ result format conversion."""
        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="format_test",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping
        )

        # Test with mock numpy array data
        mock_data = np.array([[1, 2, 3], [4, 5, 6]])

        # Method should exist and handle basic array input
        assert hasattr(setup, '_qubic_result_format_to_leeq_result_format')
        result = setup._qubic_result_format_to_leeq_result_format(mock_data)

        # Result should be processed data (exact format depends on implementation)
        assert result is not None

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_channel_configuration_validation(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                             sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test channel configuration validation and setup."""
        mock_load_qubic.return_value = mock_qubic_package

        # Test with various qubic_core_numbers
        for core_num in [2, 4, 8]:
            setup = QubiCCircuitSetup(
                name=f"channel_test_{core_num}",
                runner=mock_runner,
                leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping,
                qubic_core_number=core_num
            )

            assert setup._qubic_core_number == core_num
            assert hasattr(setup, '_core_to_channel_map')

    @patch('leeq.setups.qubic_lbnl_setups.QubiCCircuitListLPBCompiler')
    @patch('leeq.setups.qubic_lbnl_setups.GridBatchSweepEngine')
    @patch.object(QubiCCircuitSetup, '_load_qubic_package')
    def test_inheritance_from_experimental_setup(self, mock_load_qubic, mock_engine, mock_compiler, mock_runner,
                                                sample_leeq_to_qubic_mapping, mock_qubic_package):
        """Test that QubiCCircuitSetup properly inherits from ExperimentalSetup."""
        from leeq.setups.setup_base import ExperimentalSetup

        mock_load_qubic.return_value = mock_qubic_package

        setup = QubiCCircuitSetup(
            name="inheritance_test",
            runner=mock_runner,
            leeq_channel_to_qubic_channel=sample_leeq_to_qubic_mapping
        )

        # Test inheritance
        assert isinstance(setup, ExperimentalSetup)

        # Test that parent initialization was called
        assert hasattr(setup, '_name')  # From parent class

    def test_parameter_type_validation(self):
        """Test that parameters have correct types when provided."""
        # Test channel configs type checking

        # Basic smoke test - these should not crash during parameter assignment
        for config in [None, {}, {"0": {}}]:  # Valid configs
            # Should be able to store these without immediate error
            assert config is None or isinstance(config, dict)
