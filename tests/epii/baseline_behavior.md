# EPII Parameter Management - Current Baseline Behavior

## Test Results Baseline
Date: 2025-08-31
Total Tests: 27
- Passing: 24 (88.9%)
- Failing: 3 (11.1%)

### Failing Tests:
1. `test_get_element_parameter` - TypeError: object of type 'Mock' has no len()
2. `test_get_nonexistent_parameter` - TypeError: object of type 'Mock' has no len()
3. `test_list_parameters` - TypeError: 'Mock' object is not iterable

These failures are due to mock setup issues where the qubits list wasn't properly configured.

## Current Parameter Structure

### Status Parameters (from setup.status._internal_dict):
- shot_number: int
- shot_period: float
- acquisition_type: string
- debug_plotter: bool
- measurement_basis: string

### Element Parameters (from element._parameters):
- Qubits (q0, q1):
  - f01: float (frequency)
  - anharmonicity: float
  - t1, t2: float (coherence times)
  - pi_amp: float (pulse amplitude)
  - pi_len: int (pulse length)

### Current Access Pattern:
- Status: `pm.get_parameter("status.parameter_name")`
- Elements: `pm.get_parameter("element_name.parameter_name")`
- Qubits by index: `pm.get_parameter("q0.f01")`

## Current Implementation Complexity:
- File: leeq/epii/parameters.py
- Lines of code: 500+ lines
- Features:
  - Complex validation logic
  - Hardcoded parameter paths
  - Type conversion and validation
  - Readonly parameter enforcement
  - Hierarchical categorization

## Notes for Simplification:
1. Remove all validation logic
2. Direct dictionary access pattern
3. Flatten all nested dictionaries with dot notation
4. Support basic types: int, float, bool, string, numpy arrays
5. Target: < 150 lines of code