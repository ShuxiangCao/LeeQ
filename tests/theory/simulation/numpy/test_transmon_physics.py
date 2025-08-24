"""
Unit tests for transmon physics calculations.

Tests energy level calculations, transition frequencies, and coupling matrix elements.
"""

import pytest
import numpy as np
from leeq.theory.simulation.numpy.dispersive_readout.transmon_physics import (
    calculate_transmon_energies,
    calculate_transition_frequencies,
    calculate_coupling_matrix_elements,
    get_level_populations,
    effective_anharmonicity,
    ac_stark_shift
)


class TestTransmonPhysics:
    """Test suite for transmon physics functions."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.f_q = 5000  # MHz
        self.anharmonicity = -250  # MHz
        self.g = 50  # MHz
        self.num_levels = 4
        
    def test_energy_calculation(self):
        """Test transmon energy level calculation."""
        energies = calculate_transmon_energies(
            self.f_q, self.anharmonicity, self.num_levels
        )
        
        # Check ground state
        assert energies[0] == 0
        
        # Check first excited state
        assert energies[1] == self.f_q
        
        # Check second excited state: E_2 = 2*f_q + 1*alpha
        expected_e2 = 2 * self.f_q + self.anharmonicity
        assert abs(energies[2] - expected_e2) < 1e-10
        
        # Check third excited state: E_3 = 3*f_q + 3*alpha
        expected_e3 = 3 * self.f_q + 3 * self.anharmonicity
        assert abs(energies[3] - expected_e3) < 1e-10
        
    def test_transition_frequencies(self):
        """Test transition frequency calculation."""
        transitions = calculate_transition_frequencies(
            self.f_q, self.anharmonicity, self.num_levels
        )
        
        # ω_01 = f_q + 0*alpha = f_q
        assert abs(transitions[0] - self.f_q) < 1e-10
        
        # ω_12 = f_q + 1*alpha
        expected_12 = self.f_q + self.anharmonicity
        assert abs(transitions[1] - expected_12) < 1e-10
        
        # ω_23 = f_q + 2*alpha  
        expected_23 = self.f_q + 2 * self.anharmonicity
        assert abs(transitions[2] - expected_23) < 1e-10
        
        # Check that we get num_levels-1 transitions
        assert len(transitions) == self.num_levels - 1
        
    def test_coupling_matrix_symmetry(self):
        """Test that coupling matrix is Hermitian."""
        coupling_matrix = calculate_coupling_matrix_elements(self.g, self.num_levels)
        
        # Check Hermitian property
        assert np.allclose(coupling_matrix, coupling_matrix.T)
        
        # Check diagonal elements are zero (no self-coupling)
        assert np.allclose(np.diag(coupling_matrix), 0)
        
    def test_coupling_matrix_values(self):
        """Test specific coupling matrix element values."""
        coupling_matrix = calculate_coupling_matrix_elements(self.g, self.num_levels)
        
        # Check specific elements
        # g_01 = g * sqrt(1)
        assert abs(coupling_matrix[0, 1] - self.g * np.sqrt(1)) < 1e-10
        assert abs(coupling_matrix[1, 0] - self.g * np.sqrt(1)) < 1e-10
        
        # g_12 = g * sqrt(2)
        assert abs(coupling_matrix[1, 2] - self.g * np.sqrt(2)) < 1e-10
        assert abs(coupling_matrix[2, 1] - self.g * np.sqrt(2)) < 1e-10
        
        # g_23 = g * sqrt(3)
        assert abs(coupling_matrix[2, 3] - self.g * np.sqrt(3)) < 1e-10
        assert abs(coupling_matrix[3, 2] - self.g * np.sqrt(3)) < 1e-10
        
        # Non-adjacent elements should be zero
        assert coupling_matrix[0, 2] == 0
        assert coupling_matrix[0, 3] == 0
        assert coupling_matrix[1, 3] == 0
        
    def test_level_populations_zero_temperature(self):
        """Test level populations at zero temperature."""
        populations = get_level_populations(
            temperature=0.0,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            num_levels=self.num_levels
        )
        
        # At zero temperature, only ground state should be populated
        assert populations[0] == 1.0
        assert np.all(populations[1:] == 0.0)
        
    def test_level_populations_high_temperature(self):
        """Test level populations at high temperature."""
        # High temperature (classical limit)
        high_temp = 1.0  # K (very high for superconducting qubits)
        
        populations = get_level_populations(
            temperature=high_temp,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            num_levels=self.num_levels
        )
        
        # At high temperature, populations should be more evenly distributed
        assert populations[0] > populations[1] > populations[2] > populations[3]
        assert np.sum(populations) == pytest.approx(1.0)
        
    def test_level_populations_normalization(self):
        """Test that level populations are properly normalized."""
        for temp in [0.01, 0.05, 0.1, 0.2]:  # K
            populations = get_level_populations(
                temperature=temp,
                f_q=self.f_q,
                anharmonicity=self.anharmonicity,
                num_levels=self.num_levels
            )
            assert np.sum(populations) == pytest.approx(1.0)
            assert np.all(populations >= 0)
            
    def test_effective_anharmonicity(self):
        """Test effective anharmonicity calculation."""
        for level in range(4):
            alpha_eff = effective_anharmonicity(
                self.f_q, self.anharmonicity, level
            )
            # For transmon, effective anharmonicity should equal the input
            assert alpha_eff == self.anharmonicity
            
    def test_ac_stark_shift_dimensions(self):
        """Test AC Stark shift calculation dimensions."""
        f_drive = 5200  # MHz
        rabi_freq = 10  # MHz
        
        stark_shifts = ac_stark_shift(
            f_drive, self.f_q, self.anharmonicity, rabi_freq, self.num_levels
        )
        
        assert len(stark_shifts) == self.num_levels
        assert np.all(np.isfinite(stark_shifts))
        
    def test_ac_stark_shift_scaling(self):
        """Test that AC Stark shifts scale with Rabi frequency squared."""
        f_drive = 5200  # MHz
        rabi1 = 10  # MHz
        rabi2 = 20  # MHz
        
        stark1 = ac_stark_shift(
            f_drive, self.f_q, self.anharmonicity, rabi1, self.num_levels
        )
        stark2 = ac_stark_shift(
            f_drive, self.f_q, self.anharmonicity, rabi2, self.num_levels
        )
        
        # Stark shifts should scale as Rabi frequency squared
        for i in range(self.num_levels):
            if abs(stark1[i]) > 1e-10:  # Avoid division by very small numbers
                ratio = stark2[i] / stark1[i]
                expected_ratio = (rabi2 / rabi1) ** 2
                assert abs(ratio - expected_ratio) < 0.1
                
    def test_energy_level_ordering(self):
        """Test that energy levels are in ascending order."""
        energies = calculate_transmon_energies(
            self.f_q, self.anharmonicity, self.num_levels
        )
        
        # Energy levels should be in ascending order for negative anharmonicity
        for i in range(len(energies) - 1):
            assert energies[i+1] > energies[i]
            
    def test_transition_frequency_anharmonicity_effect(self):
        """Test that anharmonicity reduces transition frequencies."""
        transitions = calculate_transition_frequencies(
            self.f_q, self.anharmonicity, self.num_levels
        )
        
        # With negative anharmonicity, higher transitions should be red-detuned
        assert transitions[0] > transitions[1]  # ω_01 > ω_12
        assert transitions[1] > transitions[2]  # ω_12 > ω_23
        
        # Each transition should be reduced by |alpha|
        for i in range(len(transitions) - 1):
            diff = transitions[i] - transitions[i+1]
            assert abs(diff - abs(self.anharmonicity)) < 1e-10
            
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single level system
        energies_1 = calculate_transmon_energies(self.f_q, self.anharmonicity, 1)
        assert len(energies_1) == 1
        assert energies_1[0] == 0
        
        # Two level system
        energies_2 = calculate_transmon_energies(self.f_q, self.anharmonicity, 2)
        assert len(energies_2) == 2
        assert energies_2[1] == self.f_q
        
        # Zero anharmonicity
        energies_harmonic = calculate_transmon_energies(self.f_q, 0, self.num_levels)
        expected_harmonic = np.arange(self.num_levels) * self.f_q
        assert np.allclose(energies_harmonic, expected_harmonic)
        
    def test_coupling_matrix_scaling(self):
        """Test coupling matrix scaling with different coupling strengths."""
        g1 = 25  # MHz
        g2 = 50  # MHz
        
        matrix1 = calculate_coupling_matrix_elements(g1, self.num_levels)
        matrix2 = calculate_coupling_matrix_elements(g2, self.num_levels)
        
        # Matrix should scale linearly with coupling strength
        scaling_factor = g2 / g1
        assert np.allclose(matrix2, matrix1 * scaling_factor)
        
    def test_physical_parameter_ranges(self):
        """Test with realistic experimental parameter ranges."""
        # Test with various realistic parameters
        test_params = [
            (4500, -200),  # Typical transmon
            (5500, -300),  # Higher frequency transmon
            (3000, -150),  # Lower frequency transmon
        ]
        
        for f_q, alpha in test_params:
            energies = calculate_transmon_energies(f_q, alpha, 3)
            transitions = calculate_transition_frequencies(f_q, alpha, 3)
            
            # Basic sanity checks
            assert energies[0] == 0
            assert energies[1] == f_q
            assert transitions[0] == f_q
            assert transitions[1] == f_q + alpha
            
            # Energies should be increasing
            assert np.all(np.diff(energies) > 0)