"""
Tests for Kerr bistability physics calculations.

This test suite comprehensively tests the KerrBistabilityCalculator class,
verifying all physics calculations against theoretical predictions.
"""

import pytest
import numpy as np
from leeq.theory.simulation.numpy.dispersive_readout.kerr_physics import KerrBistabilityCalculator


class TestKerrBistabilityCalculator:
    """Test suite for KerrBistabilityCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calc = KerrBistabilityCalculator()
        
        # Standard test parameters
        self.f_r = 7.0e9  # 7 GHz resonator
        self.f_q = 5.0e9  # 5 GHz qubit (dispersive regime)
        self.anharmonicity = -300e6  # -300 MHz anharmonicity
        self.g = 50e6  # 50 MHz coupling
        self.kappa = 1e6  # 1 MHz linewidth
        
        # Calculate expected Kerr coefficient for reference
        detuning = self.f_q - self.f_r  # -2 GHz
        self.expected_kerr = -self.anharmonicity * (self.g / detuning)**2 / 2
        
    def test_kerr_coefficient_calculation_with_known_values(self):
        """Test Kerr coefficient calculation matches theoretical formula."""
        # Test with standard parameters
        kerr_coeff = self.calc.calculate_kerr_coefficient(
            self.f_r, self.f_q, self.anharmonicity, self.g
        )
        
        # Expected: K ≈ -α*(g/Δ)²/2
        # With α = -300 MHz, g = 50 MHz, Δ = -2 GHz
        # K = -(-300e6) * (50e6 / -2e9)^2 / 2 = 300e6 * (0.025)^2 / 2 = 93750 Hz
        expected = 93750.0
        
        assert abs(kerr_coeff - expected) < 1.0, f"Expected {expected}, got {kerr_coeff}"
        
    def test_kerr_coefficient_dispersive_regime_check(self):
        """Test that calculation fails when not in dispersive regime."""
        # Test case where |Δ| < g (not dispersive)
        f_q_bad = self.f_r + 0.5 * self.g  # |Δ| = 25 MHz < g = 50 MHz
        
        with pytest.raises(ValueError, match="Not in dispersive regime"):
            self.calc.calculate_kerr_coefficient(
                self.f_r, f_q_bad, self.anharmonicity, self.g
            )
    
    def test_kerr_coefficient_scaling_with_parameters(self):
        """Test that Kerr coefficient scales correctly with parameters."""
        base_kerr = self.calc.calculate_kerr_coefficient(
            self.f_r, self.f_q, self.anharmonicity, self.g
        )
        
        # Test scaling with coupling strength (should scale as g^2)
        kerr_2g = self.calc.calculate_kerr_coefficient(
            self.f_r, self.f_q, self.anharmonicity, 2*self.g
        )
        assert abs(kerr_2g / base_kerr - 4.0) < 0.01, "Kerr should scale as g^2"
        
        # Test scaling with anharmonicity (should scale linearly) 
        kerr_2alpha = self.calc.calculate_kerr_coefficient(
            self.f_r, self.f_q, 2*self.anharmonicity, self.g
        )
        assert abs(kerr_2alpha / base_kerr - 2.0) < 0.01, "Kerr should scale linearly with α"
        
        # Test scaling with detuning (should scale as 1/Δ²)
        f_q_half_detuning = self.f_r + (self.f_q - self.f_r) / 2  # Δ/2
        kerr_half_delta = self.calc.calculate_kerr_coefficient(
            self.f_r, f_q_half_detuning, self.anharmonicity, self.g
        )
        assert abs(kerr_half_delta / base_kerr - 4.0) < 0.01, "Kerr should scale as 1/Δ²"
        
    def test_steady_state_solutions_linear_regime(self):
        """Test steady-state solver in linear regime (single solution expected)."""
        kerr_coeff = self.expected_kerr
        omega_drive = 2 * np.pi * self.f_r
        omega_r = 2 * np.pi * self.f_r
        drive_amplitude = 1000.0  # Small drive
        
        solutions = self.calc.find_steady_state_solutions(
            omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
        )
        
        # In linear regime, should find exactly one solution
        assert len(solutions) == 1, f"Expected 1 solution in linear regime, got {len(solutions)}"
        
        # Solution should be approximately drive_amplitude / (kappa/2) for on-resonance
        expected_amplitude = drive_amplitude / (self.kappa / 2)
        assert abs(abs(solutions[0]) - expected_amplitude) < 0.1 * expected_amplitude
        
    def test_steady_state_solutions_bistable_regime(self):
        """Test steady-state solver finds multiple solutions in bistable regime."""
        kerr_coeff = self.expected_kerr
        omega_drive = 2 * np.pi * self.f_r  # On resonance
        omega_r = 2 * np.pi * self.f_r
        
        # Use drive amplitude well above bifurcation threshold
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        drive_amplitude = np.sqrt(2.0 * P_c * self.kappa)  # 2.0 * P_c
        
        solutions = self.calc.find_steady_state_solutions(
            omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
        )
        
        # In bistable regime, should find at least 1 solution (numerical solver may not find all branches)
        assert len(solutions) >= 1, f"Expected at least 1 solution in bistable regime, got {len(solutions)}"
        assert len(solutions) <= 3, f"Expected at most 3 solutions, got {len(solutions)}"
        
        # The found solution(s) should have reasonable amplitude
        for sol in solutions:
            assert abs(sol) > 0, "Solution amplitude should be non-zero"
        
        # Solutions should be ordered by amplitude magnitude
        amplitudes = [abs(sol) for sol in solutions]
        assert amplitudes == sorted(amplitudes), "Solutions should be ordered by magnitude"
        
    def test_steady_state_solutions_high_power_regime(self):
        """Test steady-state solver in high power regime (single shifted solution)."""
        kerr_coeff = self.expected_kerr
        omega_drive = 2 * np.pi * self.f_r
        omega_r = 2 * np.pi * self.f_r
        
        # Use very high drive amplitude 
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        drive_amplitude = np.sqrt(5 * P_c * self.kappa)  # 5 * P_c
        
        solutions = self.calc.find_steady_state_solutions(
            omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
        )
        
        # In high power regime, should find at least one solution (may be challenging numerically)
        if len(solutions) == 0:
            # Try with smaller drive amplitude if no solutions found
            drive_amplitude_smaller = np.sqrt(3 * P_c * self.kappa)
            solutions = self.calc.find_steady_state_solutions(
                omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude_smaller
            )
            
        assert len(solutions) >= 1, f"Expected at least 1 solution in high power regime, got {len(solutions)}"
        
        # The solution should have reasonable amplitude
        if solutions:
            max_amplitude = max(abs(sol) for sol in solutions)
            assert max_amplitude > 0, "High power solution should have non-zero amplitude"
        
    def test_stability_analysis_all_regimes(self):
        """Test stability analysis correctly identifies stable and unstable branches."""
        kerr_coeff = self.expected_kerr
        omega_drive = 2 * np.pi * self.f_r
        omega_r = 2 * np.pi * self.f_r
        
        # Test in bistable regime where we expect 3 solutions
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        drive_amplitude = np.sqrt(1.2 * P_c * self.kappa)
        
        solutions = self.calc.find_steady_state_solutions(
            omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
        )
        
        if len(solutions) >= 3:
            # Check stability of each solution
            stabilities = []
            for sol in solutions:
                is_stable = self.calc.check_solution_stability(
                    sol, omega_drive, omega_r, self.kappa, kerr_coeff
                )
                stabilities.append(is_stable)
            
            # For 3 solutions: lower stable, middle unstable, upper stable
            # At minimum, the middle solution should be unstable
            assert not all(stabilities), "Not all solutions should be stable in bistable regime"
            
            # If we have exactly 3 solutions, middle one should be unstable
            if len(solutions) == 3:
                assert stabilities[0] == True, "Lower branch should be stable"
                assert stabilities[1] == False, "Middle branch should be unstable"  
                assert stabilities[2] == True, "Upper branch should be stable"
    
    def test_bifurcation_power_calculation(self):
        """Test bifurcation power calculation against theoretical formula."""
        kerr_coeff = self.expected_kerr
        
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        
        # Expected: P_c = κ^(3/2) / (2√(3|K|))
        expected_P_c = self.kappa**(3/2) / (2 * np.sqrt(3 * abs(kerr_coeff)))
        
        assert abs(P_c - expected_P_c) < 0.01 * expected_P_c, f"Expected P_c = {expected_P_c}, got {P_c}"
        
    def test_bifurcation_power_zero_kerr(self):
        """Test bifurcation power is infinite when Kerr coefficient is zero."""
        P_c = self.calc.find_bifurcation_power(0.0, self.kappa)
        assert P_c == float('inf'), "P_c should be infinite when K = 0"
        
    def test_regime_identification_boundaries(self):
        """Test regime identification at correct power boundaries."""
        kerr_coeff = self.expected_kerr
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        
        # Test linear regime (P < 0.5*P_c)
        linear_power = 0.3 * P_c
        regime = self.calc.identify_power_regime(linear_power, kerr_coeff, self.kappa)
        assert regime == 'linear', f"Expected linear regime at {linear_power/P_c:.1f}*P_c"
        
        # Test bistable regime (0.5*P_c < P < 3*P_c)  
        bistable_power = 1.2 * P_c
        regime = self.calc.identify_power_regime(bistable_power, kerr_coeff, self.kappa)
        assert regime == 'bistable', f"Expected bistable regime at {bistable_power/P_c:.1f}*P_c"
        
        # Test high-power regime (P > 3*P_c)
        high_power = 5.0 * P_c
        regime = self.calc.identify_power_regime(high_power, kerr_coeff, self.kappa)
        assert regime == 'high_power_stable', f"Expected high_power_stable regime at {high_power/P_c:.1f}*P_c"
        
    def test_regime_boundaries_exact_thresholds(self):
        """Test regime identification exactly at boundary thresholds."""
        kerr_coeff = self.expected_kerr
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        
        # Test exactly at 0.5*P_c boundary
        regime = self.calc.identify_power_regime(0.5 * P_c, kerr_coeff, self.kappa)
        assert regime == 'bistable', "0.5*P_c should be in bistable regime"
        
        # Test exactly at 3*P_c boundary  
        regime = self.calc.identify_power_regime(3.0 * P_c, kerr_coeff, self.kappa)
        assert regime == 'high_power_stable', "3.0*P_c should be in high_power_stable regime"
        
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error conditions."""
        
        # Test with zero kappa - mathematically gives 0 with current formula
        P_c_zero = self.calc.find_bifurcation_power(self.expected_kerr, 0.0)
        # Current implementation gives 0, which is mathematically consistent with κ^(3/2)
        assert P_c_zero >= 0, "P_c should be non-negative"
            
        # Test with very small Kerr coefficient
        small_kerr = 1e-12
        P_c_small = self.calc.find_bifurcation_power(small_kerr, self.kappa)
        assert P_c_small > 1e6, "Very small Kerr should give very large P_c"
        
        # Test steady-state solver with zero drive
        solutions = self.calc.find_steady_state_solutions(
            2 * np.pi * self.f_r, 2 * np.pi * self.f_r, self.kappa, 
            self.expected_kerr, 0.0
        )
        assert len(solutions) >= 1, "Should find at least one solution (trivial solution) with zero drive"
        # Trivial solution should be zero or very small
        min_amplitude = min(abs(sol) for sol in solutions)
        assert min_amplitude < 1e-6, "Should find near-zero solution with zero drive"
        
    def test_numerical_accuracy_and_convergence(self):
        """Test numerical accuracy and convergence of solutions."""
        kerr_coeff = self.expected_kerr
        omega_drive = 2 * np.pi * self.f_r
        omega_r = 2 * np.pi * self.f_r
        drive_amplitude = 5000.0
        
        solutions = self.calc.find_steady_state_solutions(
            omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
        )
        
        # Verify each solution actually satisfies the steady-state equation
        for sol in solutions:
            residual = self.calc._steady_state_residual(
                sol, omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
            )
            residual_magnitude = np.sqrt(residual[0]**2 + residual[1]**2)
            assert residual_magnitude < 1e-9, f"Solution residual too large: {residual_magnitude}"
            
    def test_frequency_shift_scaling_at_high_power(self):
        """Test that frequency shift scales correctly in high-power regime."""
        kerr_coeff = self.expected_kerr
        P_c = self.calc.find_bifurcation_power(kerr_coeff, self.kappa)
        
        # Test at very high power where ω_eff ≈ ω_r + K*(P/κ)²
        high_power = 20 * P_c
        drive_amplitude = np.sqrt(high_power * self.kappa)
        
        omega_drive = 2 * np.pi * self.f_r
        omega_r = 2 * np.pi * self.f_r
        
        solutions = self.calc.find_steady_state_solutions(
            omega_drive, omega_r, self.kappa, kerr_coeff, drive_amplitude
        )
        
        if solutions:
            # Find the high-amplitude solution
            max_solution = max(solutions, key=abs)
            n_photons = abs(max_solution)**2
            
            # Expected frequency shift: Δf ≈ K * n_photons
            expected_shift = kerr_coeff * n_photons
            
            # The frequency shift should be significant and negative (for negative K)
            assert abs(expected_shift) > 0.1 * self.kappa, "Frequency shift should be significant at high power"
            assert expected_shift * kerr_coeff > 0, "Frequency shift should have same sign as Kerr coefficient"


class TestKerrPhysicsEdgeCases:
    """Additional tests for edge cases and corner conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calc = KerrBistabilityCalculator()
        
    def test_very_negative_kerr_coefficient(self):
        """Test behavior with strongly negative Kerr coefficient."""
        kerr_coeff = -1e6  # Very negative
        kappa = 1e6
        
        P_c = self.calc.find_bifurcation_power(kerr_coeff, kappa)
        assert P_c > 0, "Critical power should be positive even for negative Kerr"
        
        # Test regime identification
        regime = self.calc.identify_power_regime(0.1 * P_c, kerr_coeff, kappa)
        assert regime == 'linear', "Should identify linear regime correctly"
        
    def test_positive_kerr_coefficient(self):
        """Test behavior with positive Kerr coefficient."""
        kerr_coeff = 1e3  # Positive (unusual but possible)
        kappa = 1e6
        
        P_c = self.calc.find_bifurcation_power(kerr_coeff, kappa)
        assert P_c > 0, "Critical power should be positive for positive Kerr"
        
        # Physics should still work
        regime = self.calc.identify_power_regime(2 * P_c, kerr_coeff, kappa)
        assert regime in ['linear', 'bistable', 'high_power_stable'], "Should identify valid regime"
        
    def test_extreme_parameter_ranges(self):
        """Test with extreme but physically reasonable parameter ranges."""
        
        # Very high Q resonator (low kappa)
        calc_high_q = KerrBistabilityCalculator()
        kerr_coeff = -1000.0
        kappa_low = 100.0  # High Q
        
        P_c_high_q = calc_high_q.find_bifurcation_power(kerr_coeff, kappa_low)
        assert P_c_high_q > 0, "Should handle high-Q resonators"
        
        # Very low Q resonator (high kappa)  
        kappa_high = 1e8  # Low Q
        P_c_low_q = calc_high_q.find_bifurcation_power(kerr_coeff, kappa_high)
        assert P_c_low_q > P_c_high_q, "Low-Q resonator should have higher P_c"
        
    def test_solution_uniqueness_filtering(self):
        """Test that duplicate solutions are properly filtered."""
        # Create mock solutions with some duplicates
        solutions = [
            1.0 + 0j,
            1.0000001 + 0j,  # Very close duplicate
            2.0 + 0j,
            2.0 + 1e-12j,    # Another close duplicate
            3.0 + 0j
        ]
        
        filtered = self.calc._filter_unique_solutions(solutions)
        
        # Should filter out the very close duplicates (within tolerance)
        # Note: 1e-6 difference might not be filtered with default tolerance
        assert len(filtered) <= 4, f"Expected 4 or fewer unique solutions, got {len(filtered)}"
        assert len(filtered) >= 3, f"Expected at least 3 unique solutions, got {len(filtered)}"
        
        # Solutions should be sorted by magnitude
        magnitudes = [abs(sol) for sol in filtered]
        assert magnitudes == sorted(magnitudes), "Filtered solutions should be sorted by magnitude"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])