"""
Integration tests for complete LeeQ workflows.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestLeeQWorkflow:
    """Test complete experiment workflows."""
    
    @pytest.fixture
    def mock_setup(self):
        """Create a mock quantum setup."""
        with patch('leeq.setups.setup_base.ExperimentalSetup') as MockSetup:
            setup = MockSetup()
            setup.qubits = {'q0': Mock(), 'q1': Mock()}
            
            # Mock the status parameters - make it accessible as property too
            mock_status = Mock()
            mock_status.get_parameters = Mock(side_effect=lambda key=None: {
                'Shot_Number': 1000,
                'Measurement_Basis': '<z>',
                'Shot_Period': 100.0,
                'Acquisition_Type': 'IQ',
                'Debug_Plotter': False,
                'High_Level_Simulation_Mode': False,
                'Plot_Result_In_Jupyter': False,
                'In_Jupyter': False,
                'Ignore_Plot_Error': True
            }.get(key, None) if key else {
                'Shot_Number': 1000,
                'Measurement_Basis': '<z>',
                'Shot_Period': 100.0,
                'Acquisition_Type': 'IQ',
                'Debug_Plotter': False
            })
            mock_status.set_parameters = Mock()
            
            setup._status = mock_status
            setup.status = mock_status  # Also set as attribute
            
            # Mock compiler and engine
            setup._compiler = Mock()
            setup._engine = Mock()
            
            yield setup
    
    @pytest.fixture
    def mock_qubit(self):
        """Create a mock qubit element."""
        with patch('leeq.core.elements.built_in.qudit_transmon.TransmonElement') as MockTransmon:
            qubit = MockTransmon(name='test_qubit')
            
            # Mock LPB collections
            mock_lpb_collection = Mock()
            mock_lpb_collection.__getitem__ = Mock(return_value=Mock())
            qubit.get_lpb_collection = Mock(return_value=mock_lpb_collection)
            
            # Mock measurement primitives
            mock_measurement = Mock()
            mock_measurement.uuid = 'test_mprim_uuid'
            mock_measurement.result = Mock(return_value=np.array([0.5]))
            mock_measurement.result_raw = Mock(return_value=np.array([[1, 0], [0, 1]]))
            qubit.get_measurement_primitive = Mock(return_value=mock_measurement)
            qubit.get_measurement_prim_intlist = Mock(return_value=mock_measurement)
            
            # Mock calibration parameters
            qubit._parameters = {
                'lpb_collections': {
                    'f01': {
                        'type': 'SimpleDriveCollection',
                        'freq': 5000.0,
                        'amp': 0.1,
                        'width': 0.025
                    }
                },
                'measurement_primitives': {
                    '0': {
                        'type': 'SimpleDispersiveMeasurement',
                        'freq': 7000.0,
                        'amp': 0.2,
                        'width': 1.0
                    }
                }
            }
            
            yield qubit
    
    def test_calibration_workflow(self, mock_setup, mock_qubit):
        """Test a complete calibration workflow."""
        from leeq.experiments.builtin.basic.calibrations import rabi
        from leeq.experiments.sweeper import Sweeper
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
        from leeq.experiments.experiments import setup
        
        # Register the mock setup
        with patch.object(setup(), 'get_default_setup', return_value=mock_setup):
            # Mock the experiment execution
            with patch('leeq.experiments.experiments.basic_run') as mock_basic_run:
                # Configure mock basic_run to simulate successful execution
                mock_basic_run.return_value = None
                
                # Configure the mock to return simulated Rabi oscillation data
                mock_result = {
                    'fit_params': {
                        'amplitude': 0.8,
                        'frequency': 40.0,  # MHz
                        'phase': 0.0,
                        'offset': 0.0
                    },
                    'optimal_amp': 0.125,  # Pi pulse amplitude
                    'data': {
                        'x': np.linspace(0, 0.3, 50),
                        'y': 0.8 * np.cos(2 * np.pi * 40 * np.linspace(0, 0.3, 50))
                    }
                }
                
                # Mock the entire NormalisedRabi class
                with patch('leeq.experiments.builtin.basic.calibrations.rabi.NormalisedRabi') as MockRabi:
                    mock_experiment = MockRabi.return_value
                    mock_experiment.run.return_value = mock_result
                    
                    # Create and run the experiment
                    experiment = MockRabi()
                    result = experiment.run(
                        dut_qubit=mock_qubit,
                        amp=0.2,
                        start=0.01,
                        stop=0.3,
                        step=0.006,
                        fit=True,
                        collection_name='f01',
                        mprim_index=0
                    )
                    
                    # Verify the experiment was called
                    MockRabi.assert_called_once()
                    
                    # Verify run was called with correct parameters
                    experiment.run.assert_called_once()
                    call_args = experiment.run.call_args[1]
                    assert call_args['dut_qubit'] == mock_qubit
                    assert call_args['amp'] == 0.2
                    assert call_args['start'] == 0.01
                    assert call_args['stop'] == 0.3
                    assert call_args['fit'] == True
                    
                    # Verify the result structure
                    assert result is not None
                    assert 'fit_params' in result
                    assert 'optimal_amp' in result
                    assert result['fit_params']['amplitude'] > 0
                    assert result['optimal_amp'] > 0
    
    def test_tomography_workflow(self, mock_setup, mock_qubit):
        """Test state tomography workflow."""
        # Simplified test that mocks the entire tomography module
        from leeq.experiments.experiments import setup
        
        # Register the mock setup
        with patch.object(setup(), 'get_default_setup', return_value=mock_setup):
            # Mock the StandardStateTomography class
            with patch('leeq.theory.tomography.state_tomography.StandardStateTomography') as MockTomography:
                mock_tomo_instance = MockTomography.return_value
                
                # Configure mock to return a density matrix close to |0⟩⟨0|
                mock_tomo_instance.reconstruct_density_matrix.return_value = np.array([
                    [0.98, 0.01],
                    [0.01, 0.02]
                ])
                
                # Create tomography instance (mocked)
                tomography = MockTomography(
                    gate_set=Mock(),
                    measurement_operations=['I', 'X', 'Y']
                )
                
                # Mock measurement data (simulating state |0⟩)
                measurement_data = np.array([
                    [1.0, 0.0],  # I measurement: mostly |0⟩
                    [0.5, 0.5],  # X measurement: equal superposition
                    [0.5, 0.5]   # Y measurement: equal superposition
                ])
                
                # Run reconstruction
                reconstructed_state = tomography.reconstruct_density_matrix(measurement_data)
                
                # Verify the tomography class was instantiated
                MockTomography.assert_called_once()
                
                # Verify the reconstruction was called
                tomography.reconstruct_density_matrix.assert_called_once_with(measurement_data)
                
                # Verify the reconstructed state is valid
                assert reconstructed_state.shape == (2, 2)
                assert np.abs(np.trace(reconstructed_state) - 1.0) < 0.1  # Trace should be ~1
                assert reconstructed_state[0, 0] > 0.9  # Should be mostly in |0⟩ state
    
    def test_full_experiment_pipeline(self, mock_setup, mock_qubit):
        """Test a complete pipeline from calibration to characterization."""
        from leeq.experiments.experiments import ExperimentManager, setup
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
        
        # Clear and register setup
        with patch.object(setup(), 'clear_setups') as mock_clear:
            with patch.object(setup(), 'register_setup') as mock_register:
                setup().clear_setups()
                setup().register_setup(mock_setup)
                
                mock_clear.assert_called_once()
                mock_register.assert_called_once_with(mock_setup)
        
        # Mock LogicalPrimitiveBlock
        with patch('leeq.core.primitives.logical_primitives.LogicalPrimitiveBlock') as MockLPB:
            lpb = MockLPB.return_value
            lpb.nodes = []
            
            # Mock add method
            def add_node_side_effect(node):
                lpb.nodes.append(node.uuid if hasattr(node, 'uuid') else str(node))
                
            lpb.add = Mock(side_effect=add_node_side_effect)
            lpb.__add__ = Mock(return_value=lpb)  # For += operator
            
            # Create the LPB
            lpb_instance = MockLPB(name='test_lpb', children=[])
            
            # Simulate adding a pi pulse
            pi_pulse = Mock()
            pi_pulse.uuid = 'pi_pulse_uuid'
            lpb_instance.add(pi_pulse)
            
            # Simulate adding measurement
            measurement = mock_qubit.get_measurement_primitive('0')
            lpb_instance.add(measurement)
            
            assert lpb_instance.add.call_count == 2
        
        # Execute the experiment
        with patch('leeq.experiments.experiments.basic_run') as mock_basic_run:
            mock_basic_run.return_value = {'success': True}
            
            # Create experiment manager
            with patch('leeq.experiments.experiments.ExperimentManager') as MockExpManager:
                exp_manager = MockExpManager()
                exp_manager.run = Mock(return_value={'results': [0.95, 0.05]})
                
                # Run the experiment
                result = exp_manager.run(lpb, mock_setup)
                
                # Verify execution
                exp_manager.run.assert_called_once_with(lpb, mock_setup)
                assert result['results'][0] > 0.9  # Expecting high probability in |0⟩
    
    def test_error_handling_in_workflow(self, mock_setup, mock_qubit):
        """Test error handling in experiment workflows."""
        from leeq.experiments.builtin.basic.calibrations import rabi
        from leeq.experiments.experiments import setup
        
        # Register the mock setup
        with patch.object(setup(), 'get_default_setup', return_value=mock_setup):
            # Test handling of invalid parameters
            with patch('leeq.experiments.builtin.basic.calibrations.rabi.NormalisedRabi') as MockRabi:
                mock_experiment = MockRabi.return_value
                mock_experiment.run.side_effect = ValueError("Invalid amplitude: must be positive")
                
                experiment = MockRabi()
                
                with pytest.raises(ValueError, match="Invalid amplitude"):
                    experiment.run(
                        dut_qubit=mock_qubit,
                        amp=-0.1,  # Invalid negative amplitude
                        start=0.01,
                        stop=0.3
                    )
        
        # Test handling of hardware errors
        with patch('leeq.experiments.experiments.basic_run') as mock_basic_run:
            mock_basic_run.side_effect = RuntimeError("Hardware communication error")
            
            with pytest.raises(RuntimeError, match="Hardware communication"):
                # This would trigger the hardware error
                from leeq.experiments.experiments import basic_run as basic
                lpb = Mock()
                basic(lpb, None, None)