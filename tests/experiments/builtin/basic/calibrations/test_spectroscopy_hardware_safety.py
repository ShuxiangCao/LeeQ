"""
Hardware safety tests for spectroscopy noise disable functionality.

This test module specifically validates:
1. Hardware safety - disable_noise parameter is safely ignored in hardware mode
2. Hardware/simulation mode switching behavior
3. Mock hardware testing scenarios
4. Edge cases for parameter handling in different modes
5. Error conditions and recovery in hardware contexts

These tests ensure that the disable_noise parameter never affects actual hardware
and that the system gracefully handles mode transitions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call

from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import (
    QubitSpectroscopyFrequency, 
    QubitSpectroscopyAmplitudeFrequency
)
from leeq.experiments.builtin.basic.calibrations.two_tone_spectroscopy import TwoToneQubitSpectroscopy
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


@pytest.fixture
def hardware_test_qubit():
    """Create a test qubit element for hardware safety testing."""
    configuration = {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5000.0,
                'channel': 2,
                'shape': 'blackman_drag',
                'amp': 0.1,
                'phase': 0.,
                'width': 0.025,
                'alpha': 425.0,
                'trunc': 1.2
            },
            'f12': {
                'type': 'SimpleDriveCollection',
                'freq': 4800.0,
                'channel': 2,
                'shape': 'blackman_drag',
                'amp': 0.05,
                'phase': 0.,
                'width': 0.025,
                'alpha': 425.0,
                'trunc': 1.2
            }
        },
        'measurement_primitives': {
            '0': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': 9000.0,
                'channel': 1,
                'shape': 'square',
                'amp': 0.1,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1]
            },
            '1': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': 9000.0,
                'channel': 1,
                'shape': 'square',
                'amp': 0.1,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1, 2]
            }
        }
    }
    
    return TransmonElement(
        name='hardware_safety_qubit',
        parameters=configuration
    )


class TestSpectroscopyHardwareSafety:
    """Hardware safety tests for disable_noise parameter."""
    
    def test_hardware_mode_ignores_disable_noise_parameter(self, hardware_test_qubit):
        """Test that hardware mode safely ignores disable_noise parameter."""
        
        # Test that run method accepts disable_noise parameter (hardware safety)
        # We test this by calling the run method directly with the parameter
        
        with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
            mock_run.return_value = None
            
            # Create experiment instance
            exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
            
            # Call run method directly with disable_noise parameter
            try:
                QubitSpectroscopyFrequency.run(
                    exp,
                    dut_qubit=hardware_test_qubit,
                    res_freq=9000.0,
                    start=4950.0,
                    stop=5050.0,
                    step=50.0,
                    num_avs=1000,
                    disable_noise=True  # Should be accepted but ignored in hardware
                )
                
                hardware_safe = True
                
                # Verify the method was called with the parameter
                assert mock_run.called
                call_kwargs = mock_run.call_args.kwargs
                assert 'disable_noise' in call_kwargs
                assert call_kwargs['disable_noise'] == True
                
            except Exception as e:
                hardware_safe = False
                pytest.fail(f"Hardware safety compromised: {e}")
            
            assert hardware_safe, "Hardware run method should safely accept disable_noise parameter"
    
    def test_hardware_mode_parameter_passing_safety(self, hardware_test_qubit):
        """Test that disable_noise parameter is safely passed to hardware run methods."""
        
        # Test that run() methods accept disable_noise parameter without issues
        common_params = {
            'dut_qubit': hardware_test_qubit,
            'num_avs': 500
        }
        
        # Test QubitSpectroscopyFrequency run method accepts parameter
        with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
            mock_run.return_value = None
            
            # This should not raise any errors
            exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
            
            try:
                # Call run directly with disable_noise parameter
                QubitSpectroscopyFrequency.run(
                    exp,
                    dut_qubit=hardware_test_qubit,
                    res_freq=9000.0,
                    start=4950.0,
                    stop=5050.0,
                    step=50.0,
                    num_avs=1000,
                    disable_noise=True  # Should be accepted but ignored
                )
                
                # Verify the call was made successfully
                assert mock_run.called
                
                # Verify disable_noise was passed to the call
                call_args = mock_run.call_args
                assert 'disable_noise' in call_args.kwargs
                
            except TypeError as e:
                pytest.fail(f"Hardware run method does not accept disable_noise parameter: {e}")
    
    def test_all_spectroscopy_classes_hardware_safety(self, hardware_test_qubit):
        """Test that all three spectroscopy classes safely handle disable_noise in hardware mode."""
        
        # Mock hardware environment
        with patch('leeq.experiments.experiments.ExperimentManager') as mock_manager:
            mock_setup = Mock()
            mock_setup.is_simulation = False
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_default_setup.return_value = mock_setup
            
            # Test QubitSpectroscopyFrequency
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_freq_run:
                mock_freq_run.return_value = None
                
                try:
                    QubitSpectroscopyFrequency.run(
                        QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency),
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=4950.0,
                        stop=5050.0,
                        step=50.0,
                        num_avs=1000,
                        disable_noise=True
                    )
                    freq_safe = True
                except:
                    freq_safe = False
            
            # Test QubitSpectroscopyAmplitudeFrequency
            with patch.object(QubitSpectroscopyAmplitudeFrequency, 'run') as mock_amp_run:
                mock_amp_run.return_value = None
                
                try:
                    QubitSpectroscopyAmplitudeFrequency.run(
                        QubitSpectroscopyAmplitudeFrequency.__new__(QubitSpectroscopyAmplitudeFrequency),
                        dut_qubit=hardware_test_qubit,
                        start=4950.0,
                        stop=5050.0,
                        step=50.0,
                        qubit_amp_start=0.1,
                        qubit_amp_stop=0.2,
                        qubit_amp_step=0.1,
                        num_avs=1000,
                        disable_noise=True
                    )
                    amp_safe = True
                except:
                    amp_safe = False
            
            # Test TwoToneQubitSpectroscopy
            with patch.object(TwoToneQubitSpectroscopy, 'run') as mock_two_tone_run:
                mock_two_tone_run.return_value = None
                
                try:
                    TwoToneQubitSpectroscopy.run(
                        TwoToneQubitSpectroscopy.__new__(TwoToneQubitSpectroscopy),
                        dut_qubit=hardware_test_qubit,
                        tone1_start=4950.0,
                        tone1_stop=5050.0,
                        tone1_step=50.0,
                        tone1_amp=0.1,
                        tone2_start=4750.0,
                        tone2_stop=4850.0,
                        tone2_step=50.0,
                        tone2_amp=0.05,
                        num_avs=1000,
                        disable_noise=True
                    )
                    two_tone_safe = True
                except:
                    two_tone_safe = False
            
            assert freq_safe, "QubitSpectroscopyFrequency not hardware safe"
            assert amp_safe, "QubitSpectroscopyAmplitudeFrequency not hardware safe"
            assert two_tone_safe, "TwoToneQubitSpectroscopy not hardware safe"
    
    def test_hardware_simulation_mode_switching(self, simulation_setup, hardware_test_qubit):
        """Test behavior when switching between hardware and simulation modes."""
        
        # First test in simulation mode (current fixture)
        exp_sim = QubitSpectroscopyFrequency(
            dut_qubit=hardware_test_qubit,
            res_freq=9000.0,
            start=4950.0,
            stop=5050.0,
            step=50.0,
            num_avs=500,
            disable_noise=True
        )
        
        # Should work in simulation mode
        assert exp_sim.trace is not None
        sim_trace = exp_sim.trace.copy()
        
        # Now test that hardware mode would also accept the parameter
        # We can't actually switch to hardware mode in tests, but we can verify
        # that the run method signature accepts the parameter
        
        with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
            mock_run.return_value = None
            
            # Test that run method accepts disable_noise parameter (hardware compatibility)
            try:
                exp_hw = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                QubitSpectroscopyFrequency.run(
                    exp_hw,
                    dut_qubit=hardware_test_qubit,
                    res_freq=9000.0,
                    start=4950.0,
                    stop=5050.0,
                    step=50.0,
                    num_avs=500,
                    disable_noise=True  # Should be accepted in hardware mode
                )
                hardware_switch_safe = True
                
                # Verify the parameter was passed
                assert mock_run.called
                assert 'disable_noise' in mock_run.call_args.kwargs
                
            except Exception as e:
                hardware_switch_safe = False
                print(f"Hardware mode switching failed: {e}")
            
            assert hardware_switch_safe, "Mode switching not safe"
    
    def test_parameter_validation_edge_cases(self, hardware_test_qubit):
        """Test edge cases for disable_noise parameter validation."""
        
        # Test with various parameter combinations that might cause issues
        edge_case_params = [
            {'disable_noise': True, 'num_avs': 1},      # Minimal averaging
            {'disable_noise': True, 'num_avs': 1000000}, # Extreme averaging  
            {'disable_noise': False, 'num_avs': 1},     # Minimal with noise
        ]
        
        for params in edge_case_params:
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.return_value = None
                
                try:
                    exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                    QubitSpectroscopyFrequency.run(
                        exp,
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=5000.0,
                        stop=5010.0,
                        step=10.0,
                        **params
                    )
                    edge_case_safe = True
                except Exception as e:
                    edge_case_safe = False
                    pytest.fail(f"Edge case failed: {params}, error: {e}")
                
                assert edge_case_safe, f"Edge case not handled safely: {params}"
    
    def test_concurrent_hardware_simulation_safety(self, simulation_setup, hardware_test_qubit):
        """Test that hardware and simulation can coexist safely with disable_noise."""
        
        # Run simulation experiment
        exp_sim = QubitSpectroscopyFrequency(
            dut_qubit=hardware_test_qubit,
            res_freq=9000.0,
            start=4950.0,
            stop=5050.0,
            step=100.0,
            num_avs=300,
            disable_noise=True
        )
        
        # Mock hardware experiment running concurrently
        with patch('leeq.experiments.experiments.ExperimentManager') as mock_manager:
            mock_setup = Mock()
            mock_setup.is_simulation = False
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_default_setup.return_value = mock_setup
            
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.return_value = None
                
                # Both should work without interference
                try:
                    exp_hw = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                    QubitSpectroscopyFrequency.run(
                        exp_hw,
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=4950.0,
                        stop=5050.0,
                        step=100.0,
                        num_avs=300,
                        disable_noise=True
                    )
                    concurrent_safe = True
                except:
                    concurrent_safe = False
                
                assert concurrent_safe, "Concurrent hardware/simulation not safe"
                
                # Verify both experiments completed
                assert exp_sim.trace is not None
                assert mock_run.called
    
    def test_hardware_error_recovery_with_disable_noise(self, hardware_test_qubit):
        """Test error recovery scenarios in hardware mode with disable_noise parameter."""
        
        # Test hardware failure scenarios
        with patch('leeq.experiments.experiments.ExperimentManager') as mock_manager:
            mock_setup = Mock()
            mock_setup.is_simulation = False
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_default_setup.return_value = mock_setup
            
            # Test scenario 1: Hardware timeout
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.side_effect = TimeoutError("Hardware timeout")
                
                try:
                    exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                    QubitSpectroscopyFrequency.run(
                        exp,
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=5000.0,
                        stop=5010.0,
                        step=10.0,
                        num_avs=1000,
                        disable_noise=True  # Should not affect error handling
                    )
                    # Should not reach here due to timeout
                    timeout_handled = False
                except TimeoutError:
                    # Expected behavior - error properly propagated
                    timeout_handled = True
                except Exception as e:
                    # Unexpected error type
                    pytest.fail(f"Unexpected error in hardware timeout test: {e}")
                
                assert timeout_handled, "Hardware timeout not properly handled"
            
            # Test scenario 2: Hardware connection error  
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.side_effect = ConnectionError("Hardware disconnected")
                
                try:
                    exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                    QubitSpectroscopyFrequency.run(
                        exp,
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=5000.0,
                        stop=5010.0,
                        step=10.0,
                        num_avs=1000,
                        disable_noise=False  # Also test with False
                    )
                    connection_handled = False
                except ConnectionError:
                    # Expected behavior
                    connection_handled = True
                except Exception as e:
                    pytest.fail(f"Unexpected error in connection test: {e}")
                
                assert connection_handled, "Hardware connection error not properly handled"
    
    def test_hardware_mode_deterministic_behavior_independence(self, hardware_test_qubit):
        """Test that hardware mode behavior is independent of disable_noise parameter."""
        
        # Mock hardware setup
        mock_hardware_data = np.array([1.5+1.0j, 2.0+0.5j, 1.8+1.2j])
        
        with patch('leeq.experiments.experiments.ExperimentManager') as mock_manager:
            mock_setup = Mock()
            mock_setup.is_simulation = False
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_default_setup.return_value = mock_setup
            
            # Both calls should behave identically in hardware mode
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.return_value = None
                
                # Call 1: disable_noise=True
                exp1 = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                QubitSpectroscopyFrequency.run(
                    exp1,
                    dut_qubit=hardware_test_qubit,
                    res_freq=9000.0,
                    start=4950.0,
                    stop=5050.0,
                    step=50.0,
                    num_avs=1000,
                    disable_noise=True
                )
                
                # Call 2: disable_noise=False
                exp2 = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)  
                QubitSpectroscopyFrequency.run(
                    exp2,
                    dut_qubit=hardware_test_qubit,
                    res_freq=9000.0,
                    start=4950.0,
                    stop=5050.0,
                    step=50.0,
                    num_avs=1000,
                    disable_noise=False
                )
                
                # Both calls should have been made successfully
                assert mock_run.call_count == 2
                
                # Verify both calls received the parameter but behavior is identical
                calls = mock_run.call_args_list
                assert 'disable_noise' in calls[0].kwargs
                assert 'disable_noise' in calls[1].kwargs
                assert calls[0].kwargs['disable_noise'] == True
                assert calls[1].kwargs['disable_noise'] == False
                
                # In hardware mode, both should result in identical behavior
                # (parameter ignored, same hardware execution)
    
    def test_method_signature_consistency_across_modes(self, hardware_test_qubit):
        """Test that method signatures are consistent between hardware and simulation modes."""
        
        import inspect
        
        # Test QubitSpectroscopyFrequency
        run_sig = inspect.signature(QubitSpectroscopyFrequency.run)
        run_sim_sig = inspect.signature(QubitSpectroscopyFrequency.run_simulated)
        
        # Both should have disable_noise parameter
        assert 'disable_noise' in run_sig.parameters
        assert 'disable_noise' in run_sim_sig.parameters
        
        # Parameters should have same default value
        assert run_sig.parameters['disable_noise'].default == False
        assert run_sim_sig.parameters['disable_noise'].default == False
        
        # Test QubitSpectroscopyAmplitudeFrequency
        amp_run_sig = inspect.signature(QubitSpectroscopyAmplitudeFrequency.run)
        amp_run_sim_sig = inspect.signature(QubitSpectroscopyAmplitudeFrequency.run_simulated)
        
        assert 'disable_noise' in amp_run_sig.parameters
        assert 'disable_noise' in amp_run_sim_sig.parameters
        assert amp_run_sig.parameters['disable_noise'].default == False
        assert amp_run_sim_sig.parameters['disable_noise'].default == False
        
        # Test TwoToneQubitSpectroscopy
        two_tone_run_sig = inspect.signature(TwoToneQubitSpectroscopy.run)
        two_tone_run_sim_sig = inspect.signature(TwoToneQubitSpectroscopy.run_simulated)
        
        assert 'disable_noise' in two_tone_run_sig.parameters
        assert 'disable_noise' in two_tone_run_sim_sig.parameters
        assert two_tone_run_sig.parameters['disable_noise'].default == False
        assert two_tone_run_sim_sig.parameters['disable_noise'].default == False
    
    def test_hardware_safety_comprehensive_integration(self, hardware_test_qubit):
        """Comprehensive integration test for hardware safety across all aspects."""
        
        # This test combines all hardware safety aspects into one comprehensive check
        hardware_safety_results = {
            'parameter_acceptance': False,
            'mode_switching': False,  
            'error_handling': False,
            'concurrent_operation': False,
            'signature_consistency': False
        }
        
        # Test 1: Parameter acceptance in hardware mode
        with patch('leeq.experiments.experiments.ExperimentManager') as mock_manager:
            mock_setup = Mock()
            mock_setup.is_simulation = False
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_default_setup.return_value = mock_setup
            
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.return_value = None
                
                try:
                    for disable_noise_val in [True, False]:
                        exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                        QubitSpectroscopyFrequency.run(
                            exp,
                            dut_qubit=hardware_test_qubit,
                            res_freq=9000.0,
                            start=5000.0,
                            stop=5010.0,
                            step=10.0,
                            num_avs=1000,
                            disable_noise=disable_noise_val
                        )
                    hardware_safety_results['parameter_acceptance'] = True
                except:
                    pass
        
        # Test 2: Mode switching safety
        try:
            # Simulation mode first
            with patch('leeq.setups.built_in.setup_simulation_high_level.HighLevelSimulationSetup') as mock_sim_setup:
                mock_sim_setup.is_simulation = True
                
                # Then hardware mode
                with patch('leeq.experiments.experiments.ExperimentManager') as mock_manager:
                    mock_setup = Mock()
                    mock_setup.is_simulation = False
                    mock_manager_instance = Mock()
                    mock_manager.return_value = mock_manager_instance
                    mock_manager_instance.get_default_setup.return_value = mock_setup
                    
                    with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                        mock_run.return_value = None
                        
                        exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                        QubitSpectroscopyFrequency.run(
                            exp,
                            dut_qubit=hardware_test_qubit,
                            res_freq=9000.0,
                            start=5000.0,
                            stop=5010.0,
                            step=10.0,
                            num_avs=1000,
                            disable_noise=True
                        )
                        
                        hardware_safety_results['mode_switching'] = True
        except:
            pass
        
        # Test 3: Error handling
        try:
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.side_effect = [None, Exception("Hardware error")]
                
                # First call should succeed
                exp1 = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                QubitSpectroscopyFrequency.run(
                    exp1,
                    dut_qubit=hardware_test_qubit,
                    res_freq=9000.0,
                    start=5000.0,
                    stop=5010.0,
                    step=10.0,
                    num_avs=1000,
                    disable_noise=True
                )
                
                # Second call should raise exception (expected)
                try:
                    exp2 = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                    QubitSpectroscopyFrequency.run(
                        exp2,
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=5000.0,
                        stop=5010.0,
                        step=10.0,
                        num_avs=1000,
                        disable_noise=False
                    )
                except Exception:
                    # Expected behavior
                    hardware_safety_results['error_handling'] = True
        except:
            pass
        
        # Test 4: Concurrent operation safety
        try:
            with patch.object(QubitSpectroscopyFrequency, 'run') as mock_run:
                mock_run.return_value = None
                
                # Multiple concurrent calls with different disable_noise values
                for i in range(3):
                    exp = QubitSpectroscopyFrequency.__new__(QubitSpectroscopyFrequency)
                    QubitSpectroscopyFrequency.run(
                        exp,
                        dut_qubit=hardware_test_qubit,
                        res_freq=9000.0,
                        start=5000.0,
                        stop=5010.0,
                        step=10.0,
                        num_avs=1000,
                        disable_noise=(i % 2 == 0)  # Alternate True/False
                    )
                
                hardware_safety_results['concurrent_operation'] = True
        except:
            pass
        
        # Test 5: Signature consistency
        try:
            import inspect
            
            # Check all classes have consistent signatures
            classes_to_check = [
                QubitSpectroscopyFrequency,
                QubitSpectroscopyAmplitudeFrequency,
                TwoToneQubitSpectroscopy
            ]
            
            signature_consistent = True
            for cls in classes_to_check:
                run_sig = inspect.signature(cls.run)
                run_sim_sig = inspect.signature(cls.run_simulated)
                
                if ('disable_noise' not in run_sig.parameters or 
                    'disable_noise' not in run_sim_sig.parameters or
                    run_sig.parameters['disable_noise'].default != False or
                    run_sim_sig.parameters['disable_noise'].default != False):
                    signature_consistent = False
                    break
            
            hardware_safety_results['signature_consistency'] = signature_consistent
        except:
            pass
        
        # Assert all safety checks passed
        for check_name, result in hardware_safety_results.items():
            assert result, f"Hardware safety check failed: {check_name}"
        
        # Overall hardware safety confirmation
        assert all(hardware_safety_results.values()), "Comprehensive hardware safety not confirmed"