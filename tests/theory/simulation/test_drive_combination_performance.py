import pytest
import time
import numpy as np
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


@pytest.fixture
def single_qubit_setup():
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


def test_performance_many_drives(single_qubit_setup):
    """Test performance with many drives per channel."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # Test with increasing numbers of drives
    for num_drives in [1, 5, 10, 20, 50]:
        # Create many drives on same channel
        drives = [(1, 5000.0 + i, 10.0) for i in range(num_drives)]
        
        start_time = time.time()
        effective = sim._calculate_effective_drives(drives)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete in reasonable time (< 1ms per drive)
        max_time = num_drives * 0.001  # 1ms per drive
        assert duration < max_time, f"Too slow for {num_drives} drives: {duration:.4f}s"
        
        # Should produce correct result
        assert 1 in effective
        assert effective[1][1] == num_drives * 10.0  # Total amplitude


def test_performance_comparison(single_qubit_setup):
    """Compare performance of new vs old implementation."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # Single drive case - should be nearly identical performance
    single_drive = [(1, 5000.0, 50.0)]
    
    # Time multiple runs
    times = []
    for _ in range(100):
        start = time.time()
        effective = sim._calculate_effective_drives(single_drive)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    
    # Should be fast for single drives (< 100 microseconds typical)
    assert avg_time < 0.0001, f"Single drive too slow: {avg_time:.6f}s average"


def test_performance_absolute_timing(single_qubit_setup):
    """Test that drive combination performs within acceptable absolute time limits."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # Use more iterations to get stable measurements
    iterations = 1000
    
    # Test various drive combinations
    test_cases = [
        ("Single drive", [(1, 5000.0, 50.0)]),
        ("Two drives same freq", [(1, 5000.0, 25.0), (1, 5000.0, 25.0)]),
        ("Two drives diff freq", [(1, 5000.0, 25.0), (1, 5010.0, 25.0)]),
        ("Five drives", [(1, 5000.0 + i, 10.0) for i in range(5)]),
    ]
    
    for test_name, drives in test_cases:
        start_time = time.time()
        for _ in range(iterations):
            effective = sim._calculate_effective_drives(drives)
        total_time = time.time() - start_time
        avg_time_ms = total_time * 1000 / iterations
        
        # Require all operations complete within 1ms per call (very generous)
        assert avg_time_ms < 1.0, f"{test_name} too slow: {avg_time_ms:.3f} ms per call"
        
        print(f"{test_name}: {avg_time_ms:.3f} ms per call")
    
    # Verify correctness - equivalent drives should give same result
    single_result = sim._calculate_effective_drives([(1, 5000.0, 50.0)])
    combined_result = sim._calculate_effective_drives([(1, 5000.0, 25.0), (1, 5000.0, 25.0)])
    
    assert single_result[1][1] == combined_result[1][1], "Equivalent amplitudes don't match"
    assert abs(single_result[1][0] - combined_result[1][0]) < 1e-10, "Equivalent frequencies don't match"


def test_performance_large_number_drives(single_qubit_setup):
    """Test performance with unrealistically large number of drives."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # Test with very large number of drives (stress test)
    num_drives = 1000
    drives = [(1, 5000.0 + i * 0.1, 1.0) for i in range(num_drives)]
    
    start_time = time.time()
    effective = sim._calculate_effective_drives(drives)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Should complete within reasonable time even for large numbers
    # Allow 1 second for 1000 drives (generous limit)
    assert duration < 1.0, f"Too slow for {num_drives} drives: {duration:.4f}s"
    
    # Should produce correct total amplitude
    assert 1 in effective
    assert abs(effective[1][1] - num_drives * 1.0) < 1e-10