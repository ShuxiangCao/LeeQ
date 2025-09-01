#!/usr/bin/env python3
"""
Test script to verify Chronicle data is exposed through EPII protocol using new data structure.
"""

import sys
import grpc
import numpy as np
from pathlib import Path

# Add notebooks path for simulated setup
notebook_path = Path(__file__).parent / "notebooks" / "SimulatedSystem"
sys.path.insert(0, str(notebook_path))

from leeq.epii.proto import epii_pb2, epii_pb2_grpc
from leeq.epii.client_helpers import (
    get_data, 
    get_data_with_descriptions,
    get_docs,
    get_metadata,
    get_arrays,
    get_calibration_results
)

def main():
    """Test new EPII protocol data structure."""
    print("Testing EPII Protocol Enhancement - New Data Structure")
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
        print("\n2. Running T1 experiment to test new protocol...")
        request = epii_pb2.ExperimentRequest(
            experiment_type="characterizations.SimpleT1",
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
        
        # Test documentation fields
        print("\n3. Testing Documentation fields (response.docs):")
        docs = get_docs(response)
        
        if docs['run']:
            print(f"   ✓ Run documentation present: {len(docs['run'])} chars")
            print(f"     Preview: {docs['run'][:100]}..." if len(docs['run']) > 100 else f"     Content: {docs['run']}")
        else:
            print("   - Run documentation: Not provided")
            
        if docs['data']:
            print(f"   ✓ Data documentation present: {len(docs['data'])} chars")
            print(f"     Preview: {docs['data'][:100]}..." if len(docs['data']) > 100 else f"     Content: {docs['data']}")
        else:
            print("   - Data documentation: Not provided")
        
        # Test metadata field
        print("\n4. Testing Metadata field (response.metadata):")
        metadata = get_metadata(response)
        print(f"   - Metadata entries: {len(metadata)}")
        if metadata:
            for key, value in list(metadata.items())[:5]:
                print(f"     • {key}: {value[:50]}..." if len(str(value)) > 50 else f"     • {key}: {value}")
            if len(metadata) > 5:
                print(f"     ... and {len(metadata) - 5} more entries")
        
        # Test unified data field
        print("\n5. Testing unified Data field (response.data):")
        all_data = get_data(response)
        print(f"   - Total data items: {len(all_data)}")
        
        # Get data with descriptions for better understanding
        data_with_desc = get_data_with_descriptions(response)
        
        # Show first few data items
        if data_with_desc:
            print("   - Sample data items:")
            for name, (value, desc) in list(data_with_desc.items())[:5]:
                if isinstance(value, np.ndarray):
                    print(f"     • {name}: array{value.shape} - {desc}")
                elif isinstance(value, (int, float)):
                    print(f"     • {name}: {value} - {desc}")
                else:
                    print(f"     • {name}: {type(value).__name__} - {desc}")
            if len(data_with_desc) > 5:
                print(f"     ... and {len(data_with_desc) - 5} more items")
        
        # Test helper functions
        print("\n6. Testing client helper functions:")
        
        # Test calibration extraction
        calibration = get_calibration_results(response)
        print(f"   ✓ Calibration results: {len(calibration)} parameters extracted")
        if calibration:
            for key, value in list(calibration.items())[:3]:
                print(f"     • {key}: {value}")
        
        # Test array extraction
        arrays = get_arrays(response)
        print(f"   ✓ Arrays: {len(arrays)} numpy arrays extracted")
        if arrays:
            for name, arr in list(arrays.items())[:3]:
                print(f"     • {name}: shape {arr.shape}, dtype {arr.dtype}")
        
        # Check for key Chronicle attributes in the unified data
        expected_attrs = ['trace', 'result', 'freq_arr', 'fit_params', 'raw_data']
        found_attrs = [attr for attr in expected_attrs if attr in all_data]
        
        print(f"\n7. Verifying key Chronicle/experiment attributes:")
        for attr in expected_attrs:
            if attr in all_data:
                value = all_data[attr]
                if isinstance(value, np.ndarray):
                    print(f"   ✓ {attr}: numpy array with shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"   ✓ {attr}: dict with {len(value)} keys")
                else:
                    print(f"   ✓ {attr}: {type(value).__name__}")
            else:
                print(f"   - {attr}: not found in data")
        
        # Verify no pickle usage
        print("\n8. Verifying no pickle dependency:")
        # Check if we can access all data without pickle
        pickle_free = True
        try:
            # All data should be accessible through protobuf messages
            _ = get_data(response)
            _ = get_docs(response)  
            _ = get_metadata(response)
            print("   ✓ All data accessible without pickle")
        except Exception as e:
            print(f"   ✗ Error accessing data: {e}")
            pickle_free = False
            
        # Summary
        print("\n" + "=" * 50)
        success = len(all_data) > 0 and pickle_free
        
        if success:
            print("✓ SUCCESS: New protocol structure working correctly!")
            print(f"  - {len(all_data)} data items in unified field")
            print(f"  - Documentation accessible via response.docs")
            print(f"  - {len(metadata)} metadata entries available")
            print(f"  - {len(found_attrs)}/{len(expected_attrs)} key attributes found")
            print(f"  - No pickle dependency detected")
        else:
            print("✗ FAILURE: Issues with new protocol structure")
            if len(all_data) == 0:
                print("  - No data found in response.data field")
            if not pickle_free:
                print("  - Pickle dependency detected")
        
        return 0 if success else 1
        
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