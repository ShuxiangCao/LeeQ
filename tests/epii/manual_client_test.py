#!/usr/bin/env python
"""
Manual client test for EPII daemon.
Run the daemon first:
    python -m leeq.epii.daemon --config tests/epii/fixtures/minimal.json --port 50051
Then run this test:
    python tests/epii/manual_client_test.py
"""

import grpc
import sys
from leeq.epii.proto import epii_pb2
from leeq.epii.proto import epii_pb2_grpc


def test_epii_client():
    """Test EPII service with a simple client."""
    # Connect to the daemon
    channel = grpc.insecure_channel('localhost:50051')
    stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)


    # Test Ping
    try:
        response = stub.Ping(epii_pb2.Empty())
    except grpc.RpcError:
        return False

    # Test GetCapabilities
    try:
        response = stub.GetCapabilities(epii_pb2.Empty())
    except grpc.RpcError:
        return False

    # Test ListAvailableExperiments
    try:
        response = stub.ListAvailableExperiments(epii_pb2.Empty())
        for exp in response.experiments[:3]:  # Show first 3
            pass
    except grpc.RpcError:
        return False

    return True


if __name__ == "__main__":
    success = test_epii_client()
    sys.exit(0 if success else 1)
