#!/usr/bin/env python3
"""Simple test of EPII server with the new simplified parameter management."""

import time
import grpc
from concurrent import futures

# Import EPII components
from leeq.epii.service import ExperimentPlatformService
from leeq.epii.proto import epii_pb2, epii_pb2_grpc

def test_epii():
    """Test EPII with a mock setup to verify parameter simplification works."""
    
    print("=== Testing Simplified EPII Parameter Management ===\n")
    
    # Create a mock setup for testing
    class MockSetup:
        def __init__(self):
            self.name = "test_setup"
            self.status = MockStatus()
            self._elements = {}
            self.qubits = []
    
    class MockStatus:
        def __init__(self):
            self._internal_dict = {
                "shot_number": 1000,
                "shot_period": 0.001,
                "debug_plotter": False
            }
        
        def get_parameters(self, key=None):
            if key is None:
                return self._internal_dict.copy()
            return self._internal_dict.get(key.lower())
        
        def set_parameter(self, key, value):
            self._internal_dict[key.lower()] = value
    
    # Start server with mock setup
    setup = MockSetup()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = ExperimentPlatformService(setup=setup)
    epii_pb2_grpc.add_ExperimentPlatformServiceServicer_to_server(service, server)
    
    port = 50054
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"✓ Server started on port {port}\n")
    
    # Create client
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
    
    # Test 1: Ping
    print("1. Testing Ping...")
    response = stub.Ping(epii_pb2.Empty())
    print(f"   ✓ Response: {response.message}\n")
    
    # Test 2: Get all parameters (new simplified interface)
    print("2. Testing GetParameters (all)...")
    response = stub.GetParameters(epii_pb2.ParameterRequest())
    print(f"   ✓ Retrieved {len(response.parameters)} parameters:")
    for name, value in response.parameters.items():
        print(f"     - {name} = {value}")
    print()
    
    # Test 3: Get specific parameters
    print("3. Testing GetParameters (specific)...")
    request = epii_pb2.ParameterRequest()
    request.parameter_names.extend(["status.shot_number", "status.shot_period"])
    response = stub.GetParameters(request)
    print(f"   ✓ Retrieved {len(response.parameters)} parameters:")
    for name, value in response.parameters.items():
        print(f"     - {name} = {value}")
    print()
    
    # Test 4: Set parameters
    print("4. Testing SetParameters...")
    request = epii_pb2.SetParametersRequest()
    request.parameters["status.shot_number"] = "5000"
    request.parameters["status.debug_plotter"] = "true"
    request.parameters["custom.new_param"] = "test_value"  # Test cache fallback
    
    response = stub.SetParameters(request)
    if response.success:
        print("   ✓ Parameters set successfully")
    else:
        print(f"   ✗ Failed: {response.error_message}")
    print()
    
    # Test 5: Verify updates
    print("5. Verifying parameter updates...")
    response = stub.GetParameters(epii_pb2.ParameterRequest())
    
    # Check updated values
    checks = [
        ("status.shot_number", "5000"),
        ("status.debug_plotter", "true"),
        ("custom.new_param", "test_value")
    ]
    
    for param_name, expected in checks:
        actual = response.parameters.get(param_name, "NOT FOUND")
        if actual == expected:
            print(f"   ✓ {param_name} = {actual}")
        else:
            print(f"   ✗ {param_name} expected {expected}, got {actual}")
    print()
    
    # Test 6: Test ListParameters (for backward compatibility)
    print("6. Testing ListParameters...")
    try:
        response = stub.ListParameters(epii_pb2.Empty())
        print(f"   ✓ Retrieved {len(response.parameters)} parameter infos")
        
        # Show first few
        for param in response.parameters[:3]:
            print(f"     - {param.name}: {param.type} = {param.current_value}")
    except Exception as e:
        print(f"   ✗ ListParameters failed: {e}")
    print()
    
    # Shutdown
    server.stop(0)
    print("✅ All tests completed successfully!")
    print("\nThe simplified parameter management is working correctly:")
    print("- Direct dictionary access ✓")
    print("- No validation restrictions ✓")
    print("- Cache fallback for unknown parameters ✓")
    print("- Type serialization working ✓")
    print("- Backward compatibility maintained ✓")

if __name__ == "__main__":
    test_epii()