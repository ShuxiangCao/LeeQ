"""
Unit tests for chi shift physics calculations.

Tests the ChiShiftCalculator class for accuracy against analytical formulas
and known physics limits.
"""

import pytest
import numpy as np
from leeq.theory.simulation.numpy.dispersive_readout.physics import ChiShiftCalculator


class TestChiShiftCalculator:
    """Test suite for ChiShiftCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calc = ChiShiftCalculator()
        
        # Standard test parameters (typical experimental values)
        self.f_r = 7000  # MHz
        self.f_q = 5000  # MHz  
        self.anharmonicity = -250  # MHz
        self.g = 50  # MHz
        
    def test_initialization(self):
        """Test that calculator initializes correctly."""
        calc = ChiShiftCalculator()
        assert calc is not None
        
    def test_two_level_limit(self):
        """Test that two-level limit gives correct chi shifts."""
        g = 50  # MHz
        f_r = 7000  # MHz
        f_q = 5000  # MHz
        
        chi = self.calc.calculate_chi_shifts(
            f_r=f_r,
            f_q=f_q,
            anharmonicity=-250,
            g=g,
            num_levels=2,
            relative=False  # Use absolute chi shifts for physics validation
        )
        
        # In two-level limit:
        # χ_0 has transition 0->1 with E_1 = f_q, so χ_0 = g² / (f_r - f_q) 
        # χ_1 has transition 1->0 with E_0 = 0, so χ_1 = g² / f_r
        expected_chi_0 = g**2 / (f_r - f_q)
        expected_chi_1 = g**2 / f_r
        
        # Allow 1% tolerance
        assert abs(chi[0] - expected_chi_0) / abs(expected_chi_0) < 0.01
        assert abs(chi[1] - expected_chi_1) / abs(expected_chi_1) < 0.01
        
    def test_level_scaling(self):
        """Test that chi shifts show reasonable level dependence."""
        chis = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            g=self.g,
            num_levels=4,
            relative=False
        )
        
        # Check that chi values have reasonable magnitudes
        # Chi shifts can be positive or negative depending on transition energies
        
        # Check that all chi values are within reasonable range (can be positive or negative)
        for chi in chis:
            assert abs(chi) < 50  # Reasonable magnitude in MHz for these parameters
        
        # Check that level differences exist (not all the same)
        chi_values_unique = len(set(np.round(chis, 6))) > 1
        assert chi_values_unique, "Chi values should be different for different levels"
        
    def test_anharmonicity_effect(self):
        """Test that anharmonicity affects chi shifts in a measurable way."""
        chi_small_alpha = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=-100,  # Small anharmonicity
            g=self.g,
            num_levels=4,
            relative=False
        )
        
        chi_large_alpha = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=-300,  # Large anharmonicity
            g=self.g,
            num_levels=4,
            relative=False
        )
        
        # Different anharmonicity should produce different chi shifts for higher levels
        # Level 0 is unaffected by anharmonicity since it only involves 0->1 transition
        # Check that chi values for levels 1 and above are noticeably different
        for i in range(1, 4):
            diff = abs(chi_small_alpha[i] - chi_large_alpha[i])
            assert diff > 1e-6, f"Chi values should differ significantly for level {i}"
        
        # Level 0 should be the same since it only depends on the 0->1 transition
        assert abs(chi_small_alpha[0] - chi_large_alpha[0]) < 1e-10
        
        # Check that all values are reasonable
        for chi_set in [chi_small_alpha, chi_large_alpha]:
            assert all(np.isfinite(chi) for chi in chi_set), "All chi values should be finite"
        
    def test_coupling_strength_scaling(self):
        """Test that chi shifts scale as g².""" 
        g1 = 50  # MHz
        g2 = 100  # MHz (2x stronger)
        
        chi1 = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            g=g1,
            num_levels=3,
            relative=False
        )
        
        chi2 = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            g=g2,
            num_levels=3,
            relative=False
        )
        
        # Chi should scale as g², so chi2 should be 4x chi1
        for i in range(3):
            ratio = chi2[i] / chi1[i] if chi1[i] != 0 else np.inf
            assert abs(ratio - 4.0) < 0.1  # 4 = (g2/g1)²
            
    def test_energy_calculation(self):
        """Test energy level calculation with anharmonicity."""
        f_q = 5000  # MHz
        alpha = -250  # MHz
        
        # Test a few energy levels
        E0 = self.calc._calculate_energy(0, f_q, alpha)
        E1 = self.calc._calculate_energy(1, f_q, alpha)
        E2 = self.calc._calculate_energy(2, f_q, alpha)
        
        assert E0 == 0  # Ground state energy is zero
        assert E1 == f_q  # First excited state
        assert E2 == 2 * f_q + alpha  # Second excited state with anharmonicity
        
    def test_coupling_elements(self):
        """Test coupling matrix element calculations."""
        g = 50  # MHz
        
        # Test adjacent level couplings
        g_01 = self.calc._coupling_element(0, 1, g)
        g_12 = self.calc._coupling_element(1, 2, g)
        g_23 = self.calc._coupling_element(2, 3, g)
        
        assert g_01 == g * np.sqrt(1)  # sqrt(max(0,1)) = sqrt(1)
        assert g_12 == g * np.sqrt(2)  # sqrt(max(1,2)) = sqrt(2)
        assert g_23 == g * np.sqrt(3)  # sqrt(max(2,3)) = sqrt(3)
        
        # Test non-adjacent couplings should be zero
        g_02 = self.calc._coupling_element(0, 2, g)
        g_13 = self.calc._coupling_element(1, 3, g)
        
        assert g_02 == 0
        assert g_13 == 0
        
    def test_two_level_analytical_function(self):
        """Test the dedicated two-level chi calculation."""
        g = 75  # MHz
        f_r = 8000  # MHz
        f_q = 6000  # MHz
        
        chi_analytical = self.calc.calculate_two_level_chi(f_r, f_q, g)
        expected = g**2 / (f_r - f_q)
        
        assert abs(chi_analytical - expected) < 1e-10
        
    def test_dispersive_regime_validation(self):
        """Test dispersive regime validation."""
        # Good dispersive regime
        assert self.calc.validate_dispersive_regime(
            f_r=7000, f_q=5000, g=50  # g/|Δ| = 50/2000 = 0.025 < 0.1
        )
        
        # Bad dispersive regime  
        assert not self.calc.validate_dispersive_regime(
            f_r=7000, f_q=5000, g=500  # g/|Δ| = 500/2000 = 0.25 > 0.1
        )
        
        # Edge case: zero detuning
        assert not self.calc.validate_dispersive_regime(
            f_r=5000, f_q=5000, g=50  # Zero detuning
        )
        
    def test_dispersive_regime_warning(self):
        """Test that warning is issued when not in dispersive regime."""
        with pytest.warns(UserWarning, match="Not in dispersive regime"):
            self.calc.calculate_chi_shifts(
                f_r=5100,  # Close to qubit frequency
                f_q=5000,
                anharmonicity=-250,
                g=200,  # Large coupling
                num_levels=3,
                relative=False
            )
            
    def test_num_levels_parameter(self):
        """Test that num_levels parameter works correctly."""
        for num_levels in [2, 3, 4, 5]:
            chis = self.calc.calculate_chi_shifts(
                f_r=self.f_r,
                f_q=self.f_q,
                anharmonicity=self.anharmonicity,
                g=self.g,
                num_levels=num_levels,
                relative=False
            )
            assert len(chis) == num_levels
            
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small coupling
        chis_small = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            g=0.1,  # Very small
            num_levels=3,
            relative=False
        )
        assert np.all(np.isfinite(chis_small))
        
        # Very large detuning
        chis_large_det = self.calc.calculate_chi_shifts(
            f_r=15000,  # Large detuning
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            g=self.g,
            num_levels=3,
            relative=False
        )
        assert np.all(np.isfinite(chis_large_det))
        
    def test_chi_ordering(self):
        """Test that chi shifts have reasonable ordering."""
        chis = self.calc.calculate_chi_shifts(
            f_r=self.f_r,
            f_q=self.f_q,
            anharmonicity=self.anharmonicity,
            g=self.g,
            num_levels=4,
            relative=False
        )
        
        # Check that adjacent chi shifts have reasonable differences
        for i in range(len(chis) - 1):
            # Check that the difference between adjacent levels is reasonable
            diff = abs(chis[i+1] - chis[i])
            assert diff > 1e-6  # Should have measurable differences
            assert diff < 100  # But not unreasonably large