#!/usr/bin/env python3
"""
Simple EPII client example demonstrating basic usage patterns.
"""

import grpc
import numpy as np
from leeq.epii.proto import epii_pb2, epii_pb2_grpc

def serialize_array(array):
    """Convert NumPy array to protobuf bytes."""
    return array.astype(np.float64).tobytes()

def deserialize_array(data):
    """Convert protobuf bytes back to NumPy array."""
    return np.frombuffer(data, dtype=np.float64)

def main():
    """Main client function."""
    # Connect to EPII service
    channel = grpc.insecure_channel('localhost:50051')
    stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
    
    try:
        # Test connection
        print("Testing connection...")
        response = stub.Ping(epii_pb2.PingRequest())
        print(f"✓ Service online: {response.message}")
        
        # Get service capabilities
        print("\nGetting capabilities...")
        response = stub.GetCapabilities(epii_pb2.Empty())
        print(f"✓ Available experiments: {', '.join(response.experiments)}")
        
        # List parameters
        print("\nListing parameters...")
        response = stub.ListParameters(epii_pb2.Empty())
        print(f"✓ Found {len(response.parameters)} parameters")
        for param in response.parameters[:5]:  # Show first 5
            print(f"  - {param.name}: {param.value} ({param.type})")
        
        # Run a simple Rabi experiment
        print("\nRunning Rabi experiment...")
        amplitudes = np.linspace(0, 1, 11)
        request = epii_pb2.ExperimentRequest(
            experiment_name="calibrations.NormalisedRabi",
            parameters={
                "qubit": "q0",
                "amplitudes": serialize_array(amplitudes),
                "num_shots": "1000"
            }
        )
        
        response = stub.RunExperiment(request)
        populations = deserialize_array(response.data)
        
        print(f"✓ Experiment completed")
        print(f"  - Data points: {len(populations)}")
        print(f"  - Population range: {populations.min():.3f} - {populations.max():.3f}")
        
        # Show fit parameters if available
        if response.fit_params:
            print("  - Fit parameters:")
            for key, value in response.fit_params.items():
                print(f"    {key}: {value}")
        
        print("\n✓ All tests passed!")
        
    except grpc.RpcError as e:
        print(f"✗ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        channel.close()

if __name__ == "__main__":
    main()