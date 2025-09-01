#!/usr/bin/env python3
"""Test qubit resolution with mock setup."""

import logging
from leeq.epii.experiments import ExperimentRouter

# Set up logging to see warnings
logging.basicConfig(level=logging.WARNING)

# Create a mock setup that mimics the structure
class MockQubit:
    def __init__(self, name):
        self.name = name
        self._parameters = {"f01": {"freq": 5000 + int(name[1:]) * 100}}
    
    def __repr__(self):
        return f"MockQubit({self.name})"
    
    def get_default_c1(self):
        """Mock method for compatibility."""
        class MockC1:
            channel = 1
        return MockC1()

class MockSetup:
    def __init__(self):
        self.name = "mock_setup"
        # Create qubit list
        self.qubits = [MockQubit(f"q{i}") for i in range(3)]
        # Also add as attributes
        for i, q in enumerate(self.qubits):
            setattr(self, f"q{i}", q)

# Create setup
setup = MockSetup()

print("Mock Setup Info:")
print(f"  Has qubits list: {hasattr(setup, 'qubits')}")
print(f"  Number of qubits: {len(setup.qubits)}")
print(f"  Qubits: {setup.qubits}")
print(f"  Has q0 attribute: {hasattr(setup, 'q0')}")
print(f"  q0 value: {setup.q0}")

# Test experiment router
print("\n=== Testing ExperimentRouter qubit resolution ===")
router = ExperimentRouter()

# Test different parameter formats
test_cases = [
    ("Pure number '0'", {"dut_qubit": "0"}),
    ("Pure number '1'", {"dut_qubit": "1"}),
    ("q-prefixed 'q0'", {"dut_qubit": "q0"}),
    ("Uppercase 'Q1'", {"dut_qubit": "Q1"}),
    ("Different param name", {"qubit": "2"}),
    ("Another param name", {"dut": "q0"}),
    ("List of qubits", {"dut_list": ["0", "1", "q2"]}),
    ("Invalid qubit '5'", {"dut_qubit": "5"}),  # Should fail
]

for description, params in test_cases:
    print(f"\n{description}: {params}")
    try:
        resolved = router._resolve_qubit_references(params, setup)
        for key, value in resolved.items():
            if isinstance(value, list):
                print(f"  {key}: {value} (types: {[type(v).__name__ for v in value]})")
            else:
                print(f"  {key}: {value} (type: {type(value).__name__})")
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== Testing full parameter mapping for 'calibrations.NormalisedRabi' experiment ===")
epii_params = {
    "qubit": "0",  # This should map to dut_qubit and resolve to MockQubit(q0)
    "amplitude": "0.5",  # Should map to 'amp'
    "start_width": "100",  # Should map to 'start'
    "stop_width": "200",  # Should map to 'stop'
    "width_step": "10",  # Should map to 'step'
}

print(f"Input EPII params: {epii_params}")
leeq_params = router.map_parameters("calibrations.NormalisedRabi", epii_params, setup)
print(f"Mapped LeeQ params:")
for key, value in leeq_params.items():
    print(f"  {key}: {value} (type: {type(value).__name__})")

# Test that the qubit was properly resolved
if "dut_qubit" in leeq_params:
    qubit = leeq_params["dut_qubit"]
    if hasattr(qubit, 'get_default_c1'):
        print(f"\n✓ Qubit properly resolved and has get_default_c1 method")
    else:
        print(f"\n✗ Qubit resolved but missing get_default_c1 method")
else:
    print(f"\n✗ dut_qubit not found in mapped parameters")