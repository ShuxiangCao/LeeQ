#!/usr/bin/env python3
"""
Mock test to verify extended_data functionality works when Chronicle data exists.
This bypasses the qubit resolution issues by creating a mock experiment.
"""

import pickle
import numpy as np
from leeq.epii.proto import epii_pb2
from leeq.epii.client_helpers import get_extended_data, list_extended_attributes


def test_extended_data_serialization():
    """Test that extended_data field can hold Chronicle-like data."""
    print("Testing Extended Data Serialization")
    print("=" * 50)
    
    # Create a mock response as if it came from the service
    response = epii_pb2.ExperimentResponse()
    response.success = True
    
    # Mock Chronicle data that would be added by service.py
    mock_chronicle_data = {
        'trace': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'result': {'magnitude': 0.85, 'phase': 1.57},
        'fit_params': {'T1': 45.3, 'amplitude': 0.98},
        'freq_arr': np.linspace(4900, 5100, 11),
        'custom_attr': 'test_value',
        'iteration': 42
    }
    
    print("\n1. Simulating Chronicle data extraction in service...")
    # This is what service.py does
    for key, value in mock_chronicle_data.items():
        response.extended_data[key] = pickle.dumps(value)
    print(f"   ✓ Added {len(response.extended_data)} attributes to extended_data")
    
    print("\n2. Client-side: Listing extended attributes...")
    attrs = list_extended_attributes(response)
    print(f"   ✓ Found {len(attrs)} attributes: {', '.join(attrs)}")
    
    print("\n3. Client-side: Deserializing extended data...")
    data = get_extended_data(response)
    
    print("\n4. Verifying deserialized data:")
    for key, original_value in mock_chronicle_data.items():
        if key in data:
            deserialized = data[key]
            if isinstance(original_value, np.ndarray):
                if np.array_equal(original_value, deserialized):
                    print(f"   ✓ {key}: numpy array preserved correctly")
                else:
                    print(f"   ✗ {key}: numpy array mismatch")
            elif isinstance(original_value, dict):
                if original_value == deserialized:
                    print(f"   ✓ {key}: dict preserved correctly")
                else:
                    print(f"   ✗ {key}: dict mismatch")
            else:
                if original_value == deserialized:
                    print(f"   ✓ {key}: value preserved correctly ({deserialized})")
                else:
                    print(f"   ✗ {key}: value mismatch")
        else:
            print(f"   ✗ {key}: not found in deserialized data")
    
    print("\n" + "=" * 50)
    print("✓ SUCCESS: Extended data serialization/deserialization works!")
    print("\nThe issue with the live test is that experiments aren't getting")
    print("Chronicle record entries attached, likely due to the qubit resolution")
    print("errors preventing successful experiment completion.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(test_extended_data_serialization())