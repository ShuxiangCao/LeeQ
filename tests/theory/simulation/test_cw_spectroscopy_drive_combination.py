import pytest
import numpy as np
from collections import defaultdict
from leeq.theory.simulation.numpy.cw_spectroscopy import CWSpectroscopySimulator
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager


@pytest.fixture(autouse=True)
def clean_singleton():
    """Clean up ExperimentManager singleton state before and after each test."""
    # Reset singleton to get fresh instance
    ExperimentManager._reset_singleton()
    
    yield
    
    # Clear state and reset singleton after test
    try:
        manager = ExperimentManager()
        manager.clear_setups()
        manager._active_experiment_instance = None
    finally:
        ExperimentManager._reset_singleton()


class TestDriveCombination:
    """Test suite for drive combination functionality."""
    
    @pytest.fixture
    def single_qubit_setup(self):
        """Setup with single qubit for testing."""
        vq = VirtualTransmon(
            name="Q1",
            qubit_frequency=5000.0,
            anharmonicity=-200.0,
            t1=50.0,
            t2=30.0,
            readout_frequency=7000.0,
            truncate_level=3
        )
        return HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq},
            omega_to_amp_map={1: 500.0}
        )
    
    @pytest.fixture
    def two_qubit_setup(self):
        """Setup with two coupled qubits for crosstalk testing."""
        vq1 = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                             anharmonicity=-200.0, t1=50.0, t2=30.0, 
                             readout_frequency=7000.0, truncate_level=3)
        vq2 = VirtualTransmon(name="Q2", qubit_frequency=5200.0, 
                             anharmonicity=-210.0, t1=50.0, t2=30.0, 
                             readout_frequency=7100.0, truncate_level=3)
        return HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq1, 2: vq2},
            omega_to_amp_map={1: 500.0, 2: 520.0},
            coupling_strength_map={frozenset(["Q1", "Q2"]): 10.0}
        )
    
    def test_single_drive_unchanged(self, single_qubit_setup):
        """Test that single drives behave identically to before."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Single drive should pass through unchanged
        single_drive = [(5000.0, 50.0)]
        result = sim._combine_drives(single_drive)
        
        assert result == (5000.0, 50.0)
    
    def test_same_frequency_amplitude_addition(self, single_qubit_setup):
        """Test amplitude addition for same frequency drives."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Two drives at exactly same frequency
        drives = [(5000.0, 30.0), (5000.0, 20.0)]
        result = sim._combine_drives(drives)
        
        assert result[0] == 5000.0  # Frequency unchanged
        assert result[1] == 50.0    # Amplitudes added: 30 + 20 = 50
    
    def test_near_same_frequency_combination(self, single_qubit_setup):
        """Test drives within frequency tolerance are combined."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Two drives within default 1.0 MHz tolerance
        drives = [(5000.0, 40.0), (5000.5, 20.0)]
        result = sim._combine_drives(drives)
        
        # Should combine as same frequency
        expected_freq = (5000.0 * 40.0 + 5000.5 * 20.0) / (40.0 + 20.0)  # ~5000.17
        assert abs(result[0] - expected_freq) < 0.01
        assert result[1] == 60.0  # Total amplitude
    
    def test_different_frequency_weighted_average(self, single_qubit_setup):
        """Test amplitude-weighted frequency for different frequencies."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Two drives with significant frequency difference
        drives = [(5000.0, 40.0), (5020.0, 20.0)]
        result = sim._combine_drives(drives)
        
        # Expected weighted average: (5000*40 + 5020*20)/(40+20) = 5006.67
        expected_freq = (5000.0 * 40.0 + 5020.0 * 20.0) / (40.0 + 20.0)
        assert abs(result[0] - expected_freq) < 0.1
        assert result[1] == 60.0  # Total amplitude
    
    def test_drive_order_symmetry(self, single_qubit_setup):
        """Test that drive order doesn't affect result."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives1 = [(5000.0, 30.0), (5010.0, 50.0)]
        drives2 = [(5010.0, 50.0), (5000.0, 30.0)]
        
        result1 = sim._combine_drives(drives1)
        result2 = sim._combine_drives(drives2)
        
        assert abs(result1[0] - result2[0]) < 1e-10  # Same frequency
        assert abs(result1[1] - result2[1]) < 1e-10  # Same amplitude
    
    def test_zero_amplitude_handling(self, single_qubit_setup):
        """Test handling of zero amplitude drives."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives = [(5000.0, 0.0), (5010.0, 30.0)]
        result = sim._combine_drives(drives)
        
        # Should be dominated by non-zero drive
        assert result[0] == 5010.0  # Frequency of non-zero drive
        assert result[1] == 30.0    # Total amplitude
    
    def test_three_drive_combination(self, single_qubit_setup):
        """Test combination of three drives."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives = [(5000.0, 20.0), (5010.0, 30.0), (5020.0, 10.0)]
        result = sim._combine_drives(drives)
        
        # Weighted average: (5000*20 + 5010*30 + 5020*10)/(20+30+10)
        expected_freq = (5000.0*20 + 5010.0*30 + 5020.0*10) / 60.0
        assert abs(result[0] - expected_freq) < 0.1
        assert result[1] == 60.0
    
    def test_empty_drive_list(self, single_qubit_setup):
        """Test handling of empty drive list."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        result = sim._combine_drives([])
        assert result == (0.0, 0.0)
    
    def test_frequency_tolerance_parameter(self, single_qubit_setup):
        """Test custom frequency tolerance parameter."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives = [(5000.0, 30.0), (5005.0, 20.0)]
        
        # With default tolerance (1.0), should be treated as different frequencies
        result_default = sim._combine_drives(drives)
        expected_freq_different = (5000.0*30 + 5005.0*20) / 50.0
        assert abs(result_default[0] - expected_freq_different) < 0.1
        
        # With larger tolerance (10.0), should be treated as same frequency
        result_large_tol = sim._combine_drives(drives, freq_tolerance=10.0)
        expected_freq_same = (5000.0*30 + 5005.0*20) / 50.0
        assert abs(result_large_tol[0] - expected_freq_same) < 0.1
        assert result_large_tol[1] == 50.0


class TestEffectiveDrivesIntegration:
    """Test integration of _combine_drives with _calculate_effective_drives."""
    
    @pytest.fixture
    def single_qubit_setup(self):
        """Setup with single qubit for testing."""
        vq = VirtualTransmon(
            name="Q1",
            qubit_frequency=5000.0,
            anharmonicity=-200.0,
            t1=50.0,
            t2=30.0,
            readout_frequency=7000.0,
            truncate_level=3
        )
        return HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq},
            omega_to_amp_map={1: 500.0}
        )
    
    @pytest.fixture
    def two_qubit_setup(self):
        """Setup with two coupled qubits for crosstalk testing."""
        vq1 = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                             anharmonicity=-200.0, t1=50.0, t2=30.0, 
                             readout_frequency=7000.0, truncate_level=3)
        vq2 = VirtualTransmon(name="Q2", qubit_frequency=5200.0, 
                             anharmonicity=-210.0, t1=50.0, t2=30.0, 
                             readout_frequency=7100.0, truncate_level=3)
        return HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq1, 2: vq2},
            omega_to_amp_map={1: 500.0, 2: 520.0},
            coupling_strength_map={frozenset(["Q1", "Q2"]): 10.0}
        )
    
    def test_same_channel_multiple_drives(self, single_qubit_setup):
        """Test multiple drives on same channel are combined correctly."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Multiple drives on channel 1
        drives = [(1, 5000.0, 30.0), (1, 5000.0, 20.0)]
        effective = sim._calculate_effective_drives(drives)
        
        assert 1 in effective
        assert effective[1][0] == 5000.0  # Frequency preserved
        assert effective[1][1] == 50.0    # Amplitudes combined: 30 + 20
    
    def test_different_frequency_same_channel(self, single_qubit_setup):
        """Test different frequencies on same channel use weighted average."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives = [(1, 5000.0, 40.0), (1, 5020.0, 20.0)]
        effective = sim._calculate_effective_drives(drives)
        
        expected_freq = (5000.0*40 + 5020.0*20) / 60.0
        assert abs(effective[1][0] - expected_freq) < 0.1
        assert effective[1][1] == 60.0
    
    def test_crosstalk_with_combined_drives(self, two_qubit_setup):
        """Test crosstalk calculation with combined drives on source channel."""
        sim = CWSpectroscopySimulator(two_qubit_setup)
        
        # Multiple drives on channel 1, should create combined crosstalk on channel 2
        drives = [(1, 5000.0, 30.0), (1, 5020.0, 20.0)]
        effective = sim._calculate_effective_drives(drives)
        
        # Should have combined direct drive on channel 1
        assert 1 in effective
        assert effective[1][1] == 50.0  # Combined amplitude
        
        # Should have crosstalk on channel 2 from both drives
        assert 2 in effective  # Crosstalk channel
        # Crosstalk amplitude should reflect contributions from both drives
        assert effective[2][1] > 0  # Some crosstalk amplitude


def test_backward_compatibility():
    """Test that existing single drive behavior is unchanged."""
    vq = VirtualTransmon(
        name="Q1",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,
        t2=30.0,
        readout_frequency=7000.0,
        truncate_level=3
    )
    setup = HighLevelSimulationSetup(
        name="test",
        virtual_qubits={1: vq},
        omega_to_amp_map={1: 500.0}
    )
    sim = CWSpectroscopySimulator(setup)
    
    # Test single drive - should be identical to original behavior
    single_drive = [(1, 5000.0, 50.0)]
    effective = sim._calculate_effective_drives(single_drive)
    
    assert effective[1] == (5000.0, 50.0)


def test_physics_conservation():
    """Test that physics principles are preserved."""
    vq = VirtualTransmon(
        name="Q1",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,
        t2=30.0,
        readout_frequency=7000.0,
        truncate_level=3
    )
    setup = HighLevelSimulationSetup(
        name="test",
        virtual_qubits={1: vq},
        omega_to_amp_map={1: 500.0}
    )
    sim = CWSpectroscopySimulator(setup)
    
    # Test amplitude conservation in combinations
    drives = [(5000.0, 30.0), (5010.0, 40.0)]
    result = sim._combine_drives(drives)
    
    # Total amplitude should be conserved
    input_total = 30.0 + 40.0
    output_total = result[1]
    assert abs(input_total - output_total) < 1e-10
    
    # Test that frequency is reasonable weighted average
    expected_freq = (5000.0*30 + 5010.0*40) / 70.0
    assert abs(result[0] - expected_freq) < 0.1


class TestPhysicsValidation:
    """Test physics principles in drive combinations."""
    
    @pytest.fixture
    def single_qubit_setup(self):
        """Setup with single qubit for testing."""
        vq = VirtualTransmon(
            name="Q1",
            qubit_frequency=5000.0,
            anharmonicity=-200.0,
            t1=50.0,
            t2=30.0,
            readout_frequency=7000.0,
            truncate_level=3
        )
        return HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq},
            omega_to_amp_map={1: 500.0}
        )
    
    @pytest.fixture
    def two_qubit_setup(self):
        """Setup with two coupled qubits for crosstalk testing."""
        vq1 = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                             anharmonicity=-200.0, t1=50.0, t2=30.0, 
                             readout_frequency=7000.0, truncate_level=3)
        vq2 = VirtualTransmon(name="Q2", qubit_frequency=5200.0, 
                             anharmonicity=-210.0, t1=50.0, t2=30.0, 
                             readout_frequency=7100.0, truncate_level=3)
        return HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq1, 2: vq2},
            omega_to_amp_map={1: 500.0, 2: 520.0},
            coupling_strength_map={frozenset(["Q1", "Q2"]): 10.0}
        )
    
    def test_energy_conservation(self, single_qubit_setup):
        """Verify total drive energy is conserved in combinations."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives = [(1, 5000.0, 30.0), (1, 5010.0, 40.0)]
        effective = sim._calculate_effective_drives(drives)
        
        # Total amplitude should be conserved (linear addition)
        input_total = 30.0 + 40.0
        output_total = effective[1][1]
        
        assert abs(input_total - output_total) < 1e-10, "Amplitude not conserved"
    
    def test_frequency_weighting_physics(self, single_qubit_setup):
        """Verify amplitude-weighted frequency follows physics expectations."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Strong drive at 5000, weak at 5100 -> frequency closer to 5000
        drives = [(1, 5000.0, 90.0), (1, 5100.0, 10.0)]
        effective = sim._calculate_effective_drives(drives)
        
        expected_freq = (5000*90 + 5100*10) / (90+10)  # 5009
        assert abs(effective[1][0] - expected_freq) < 0.1
        assert effective[1][1] == 100.0  # Total amplitude
    
    def test_symmetry_preservation(self, single_qubit_setup):
        """Test that drive order doesn't affect final result."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        drives1 = [(1, 5000.0, 30.0), (1, 5010.0, 50.0)]
        drives2 = [(1, 5010.0, 50.0), (1, 5000.0, 30.0)]
        
        eff1 = sim._calculate_effective_drives(drives1)
        eff2 = sim._calculate_effective_drives(drives2)
        
        assert abs(eff1[1][0] - eff2[1][0]) < 1e-10, "Frequency not symmetric"
        assert abs(eff1[1][1] - eff2[1][1]) < 1e-10, "Amplitude not symmetric"
    
    def test_two_tone_spectroscopy_symmetry(self, single_qubit_setup):
        """Test that two-tone spectroscopy shows proper symmetry."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Define readout parameters
        readout_params = {1: {'frequency': 6000.0, 'amplitude': 10.0}}
        
        # Test symmetric drive pairs
        drives1 = [(1, 5000.0, 30.0), (1, 5010.0, 50.0)]
        drives2 = [(1, 5010.0, 50.0), (1, 5000.0, 30.0)]
        
        iq1 = sim.simulate_spectroscopy_iq(drives1, readout_params)
        iq2 = sim.simulate_spectroscopy_iq(drives2, readout_params)
        
        # Should show perfect symmetry
        assert abs(iq1[1] - iq2[1]) < 1e-10, "Two-tone response not symmetric"
    
    def test_crosstalk_with_multiple_drives(self, two_qubit_setup):
        """Test crosstalk calculation with multiple drives per channel."""
        sim = CWSpectroscopySimulator(two_qubit_setup)
        
        # Multiple drives on channel 1, should create proper crosstalk on channel 2
        drives = [(1, 5000.0, 40.0), (1, 5020.0, 20.0)]
        effective = sim._calculate_effective_drives(drives)
        
        assert 1 in effective, "Direct channel missing"
        assert 2 in effective, "Crosstalk channel missing"
        
        # Combined direct drive
        assert effective[1][1] == 60.0, "Direct drives not combined"
        
        # Crosstalk should be from both drives
        # Each drive contributes: coupling / detuning * amplitude
        # Total crosstalk should be sum of individual contributions
        assert effective[2][1] > 0, "No crosstalk detected"
    
    def test_amplitude_scaling_preservation(self, single_qubit_setup):
        """Test that amplitude scaling relationships are preserved."""
        sim = CWSpectroscopySimulator(single_qubit_setup)
        
        # Test linear scaling
        drives_1x = [(1, 5000.0, 10.0), (1, 5010.0, 20.0)]
        drives_2x = [(1, 5000.0, 20.0), (1, 5010.0, 40.0)]
        
        eff_1x = sim._calculate_effective_drives(drives_1x)
        eff_2x = sim._calculate_effective_drives(drives_2x)
        
        # Amplitudes should scale linearly
        assert abs(eff_2x[1][1] / eff_1x[1][1] - 2.0) < 1e-10
        
        # Frequencies should be identical (same weighting ratio)
        assert abs(eff_1x[1][0] - eff_2x[1][0]) < 1e-10