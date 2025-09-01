#!/usr/bin/env python3
"""
Test script to verify Chronicle data is exposed through EPII extended_data field.
"""

import sys
import grpc
import numpy as np
from pathlib import Path

# Add notebooks path for simulated setup
notebook_path = Path(__file__).parent / "notebooks" / "SimulatedSystem"
sys.path.insert(0, str(notebook_path))

from leeq.epii.proto import epii_pb2, epii_pb2_grpc
from leeq.epii.client_helpers import get_extended_data, list_extended_attributes

def main():
    """Test extended_data functionality."""
    print("Testing EPII Extended Data Feature")
    print("=" * 50)
    
    # Connect to EPII service
    channel = grpc.insecure_channel('localhost:50051')
    stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
    
    try:
        # Test connection
        print("\n1. Testing connection...")
        response = stub.Ping(epii_pb2.Empty())
        print(f"   ✓ Service online: {response.message}")
        
        # Run a simple T1 experiment  
        print("\n2. Running T1 experiment to generate Chronicle data...")
        request = epii_pb2.ExperimentRequest(
            experiment_type="t1",
            parameters={
                "qubit": "q0",  # Use proper qubit reference
                "time_length": "100",
                "time_resolution": "10"
            },
            return_raw_data=True
        )
        
        response = stub.RunExperiment(request)
        
        if not response.success:
            print(f"   ✗ Experiment failed: {response.error_message}")
            return 1
        
        print(f"   ✓ Experiment completed successfully")
        
        # Check standard fields
        print("\n3. Standard EPII response fields:")
        print(f"   - Calibration results: {len(response.calibration_results)} parameters")
        print(f"   - Measurement data: {len(response.measurement_data)} arrays")
        
        # Check extended_data field
        print("\n4. Extended data (Chronicle attributes):")
        
        # List all extended attributes
        extended_attrs = list_extended_attributes(response)
        print(f"   - Total extended attributes: {len(extended_attrs)}")
        
        if extended_attrs:
            print(f"   - Available attributes: {', '.join(extended_attrs[:10])}")
            if len(extended_attrs) > 10:
                print(f"     ... and {len(extended_attrs) - 10} more")
        
        # Get the full extended data
        extended_data = get_extended_data(response)
        
        # Check for key Chronicle attributes
        expected_attrs = ['trace', 'result', 'freq_arr', 'fit_params']
        found_attrs = [attr for attr in expected_attrs if attr in extended_data]
        
        print(f"\n5. Verifying key Chronicle attributes:")
        for attr in expected_attrs:
            if attr in extended_data:
                value = extended_data[attr]
                if isinstance(value, np.ndarray):
                    print(f"   ✓ {attr}: numpy array with shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"   ✓ {attr}: dict with {len(value)} keys")
                else:
                    print(f"   ✓ {attr}: {type(value).__name__}")
            else:
                print(f"   ✗ {attr}: not found")
        
        # Summary
        print("\n" + "=" * 50)
        if len(extended_attrs) > 0:
            print("✓ SUCCESS: Chronicle data is accessible through extended_data!")
            print(f"  - {len(extended_attrs)} attributes available")
            print(f"  - {len(found_attrs)}/{len(expected_attrs)} key attributes found")
        else:
            print("✗ FAILURE: No extended_data found in response")
            print("  Check that Chronicle is enabled and experiment was logged")
        
        return 0 if extended_attrs else 1
        
    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()} - {e.details()}")
        print("\nMake sure the EPII service is running:")
        print("  python -m leeq.epii.server")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        channel.close()

if __name__ == "__main__":
    sys.exit(main())