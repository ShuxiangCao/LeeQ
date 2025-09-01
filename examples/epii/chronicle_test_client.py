#!/usr/bin/env python3
"""
EPII Test Client with Chronicle Integration

This client demonstrates executing quantum experiments via the EPII protocol.
It runs experiments similar to those in TuneUpDemo.ipynb and shows how results
appear in the Chronicle dashboard.

Usage:
    python chronicle_test_client.py [--host HOST] [--port PORT]
"""

import argparse
import grpc
import numpy as np
import time
import json
from typing import Dict, Any

# Import EPII protocol definitions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from leeq.epii.proto import epii_pb2, epii_pb2_grpc


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_result(label, value, indent=2):
    """Print a formatted result line."""
    print(" " * indent + f"• {label}: {value}")


def test_connection(stub):
    """Test basic connectivity to EPII service."""
    print_header("Testing Connection")
    try:
        response = stub.Ping(epii_pb2.Empty())
        print_result("Status", "Connected")
        print_result("Service", response.message)
        print_result("Timestamp", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response.timestamp/1000)))
        return True
    except grpc.RpcError as e:
        print_result("Status", f"Failed - {e.code()}: {e.details()}")
        return False


def get_capabilities(stub):
    """Query and display service capabilities."""
    print_header("Service Capabilities")
    try:
        response = stub.GetCapabilities(epii_pb2.Empty())
        print_result("Framework", f"{response.framework_name} v{response.framework_version}")
        print_result("EPII Version", response.epii_version)
        print_result("Backends", ", ".join(response.supported_backends))
        print_result("Experiments Available", len(response.experiment_types))
        
        print("\n  Available Experiments:")
        for exp in response.experiment_types[:5]:  # Show first 5
            print(f"    - {exp.name}: {exp.description}")
        
        return True
    except grpc.RpcError as e:
        print_result("Error", f"{e.code()}: {e.details()}")
        return False


def run_resonator_spectroscopy(stub, qubit_id="q0"):
    """Run a basic Rabi as resonator spectroscopy substitute (not available in EPII)."""
    print_header(f"Quick Rabi Check - {qubit_id}")
    
    try:
        # Use a quick Rabi scan as substitute since resonator spectroscopy isn't available
        request = epii_pb2.ExperimentRequest(
            experiment_type="rabi",
            parameters={
                "qubit": qubit_id,
                "start": "0",
                "stop": "0.2",
                "step": "0.02"
            },
            return_raw_data=True
        )
        
        print("  Running experiment...")
        start_time = time.time()
        response = stub.RunExperiment(request)
        duration = time.time() - start_time
        
        if response.success:
            print_result("Status", "Success", 4)
            print_result("Duration", f"{duration:.2f}s", 4)
            
            # Display calibration results
            if response.calibration_results:
                print_result("Quick Check", 
                           f"Amplitude at {response.calibration_results.get('pi_amp', 0):.4f}", 4)
            
            # Display data statistics
            if response.measurement_data:
                data = response.measurement_data[0]
                print_result("Data Points", len(data.shape), 4)
                
            return True
        else:
            print_result("Status", "Failed", 4)
            print_result("Error", response.error_message, 4)
            return False
            
    except grpc.RpcError as e:
        print_result("Error", f"{e.code()}: {e.details()}", 4)
        return False


def run_rabi_experiment(stub, qubit_id="q0"):
    """Run Rabi oscillation experiment (from TuneUpDemo)."""
    print_header(f"Rabi Oscillation - {qubit_id}")
    
    try:
        request = epii_pb2.ExperimentRequest(
            experiment_type="rabi",
            parameters={
                "qubit": qubit_id,
                "start": "0",
                "stop": "0.5",
                "step": "0.01"
            },
            return_raw_data=True
        )
        
        print("  Running experiment...")
        start_time = time.time()
        response = stub.RunExperiment(request)
        duration = time.time() - start_time
        
        if response.success:
            print_result("Status", "Success", 4)
            print_result("Duration", f"{duration:.2f}s", 4)
            
            # Display calibration results
            if response.calibration_results:
                print_result("Pi Amplitude", 
                           f"{response.calibration_results.get('pi_amp', 0):.4f}", 4)
                print_result("Rabi Frequency", 
                           f"{response.calibration_results.get('rabi_freq', 0):.2f} MHz", 4)
            
            return True
        else:
            print_result("Status", "Failed", 4)
            print_result("Error", response.error_message, 4)
            return False
            
    except grpc.RpcError as e:
        print_result("Error", f"{e.code()}: {e.details()}", 4)
        return False


def run_ramsey_experiment(stub, qubit_id="q0"):
    """Run Ramsey experiment for frequency calibration (from TuneUpDemo)."""
    print_header(f"Ramsey Experiment - {qubit_id}")
    
    try:
        # Fine frequency calibration
        request = epii_pb2.ExperimentRequest(
            experiment_type="ramsey",
            parameters={
                "qubit": qubit_id,
                "stop": "3",  # microseconds
                "step": "0.05",
                "set_offset": "1"  # MHz offset
            },
            return_raw_data=True
        )
        
        print("  Running experiment...")
        start_time = time.time()
        response = stub.RunExperiment(request)
        duration = time.time() - start_time
        
        if response.success:
            print_result("Status", "Success", 4)
            print_result("Duration", f"{duration:.2f}s", 4)
            
            # Display calibration results
            if response.calibration_results:
                print_result("Frequency Offset", 
                           f"{response.calibration_results.get('frequency_offset', 0):.3f} MHz", 4)
                print_result("T2*", 
                           f"{response.calibration_results.get('t2_star', 0):.2f} μs", 4)
            
            return True
        else:
            print_result("Status", "Failed", 4)
            print_result("Error", response.error_message, 4)
            return False
            
    except grpc.RpcError as e:
        print_result("Error", f"{e.code()}: {e.details()}", 4)
        return False


def run_t1_experiment(stub, qubit_id="q0"):
    """Run T1 relaxation time measurement (from TuneUpDemo)."""
    print_header(f"T1 Measurement - {qubit_id}")
    
    try:
        request = epii_pb2.ExperimentRequest(
            experiment_type="t1",
            parameters={
                "qubit": qubit_id,
                "time_length": "100",  # microseconds
                "time_resolution": "2"
            },
            return_raw_data=True
        )
        
        print("  Running experiment...")
        start_time = time.time()
        response = stub.RunExperiment(request)
        duration = time.time() - start_time
        
        if response.success:
            print_result("Status", "Success", 4)
            print_result("Duration", f"{duration:.2f}s", 4)
            
            # Display calibration results
            if response.calibration_results:
                print_result("T1", 
                           f"{response.calibration_results.get('t1', 0):.2f} μs", 4)
            
            return True
        else:
            print_result("Status", "Failed", 4)
            print_result("Error", response.error_message, 4)
            return False
            
    except grpc.RpcError as e:
        print_result("Error", f"{e.code()}: {e.details()}", 4)
        return False


def run_echo_experiment(stub, qubit_id="q0"):
    """Run spin echo experiment for T2 measurement (from TuneUpDemo)."""
    print_header(f"Spin Echo (T2) - {qubit_id}")
    
    try:
        request = epii_pb2.ExperimentRequest(
            experiment_type="echo",
            parameters={
                "qubit": qubit_id,
                "free_evolution_time": "50",  # microseconds
                "time_resolution": "1"
            },
            return_raw_data=True
        )
        
        print("  Running experiment...")
        start_time = time.time()
        response = stub.RunExperiment(request)
        duration = time.time() - start_time
        
        if response.success:
            print_result("Status", "Success", 4)
            print_result("Duration", f"{duration:.2f}s", 4)
            
            # Display calibration results
            if response.calibration_results:
                print_result("T2 Echo", 
                           f"{response.calibration_results.get('t2_echo', 0):.2f} μs", 4)
            
            return True
        else:
            print_result("Status", "Failed", 4)
            print_result("Error", response.error_message, 4)
            return False
            
    except grpc.RpcError as e:
        print_result("Error", f"{e.code()}: {e.details()}", 4)
        return False


def main():
    """Main client execution."""
    parser = argparse.ArgumentParser(
        description="EPII test client with Chronicle integration"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="EPII daemon host (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="EPII daemon port (default: 50051)"
    )
    
    parser.add_argument(
        "--qubit",
        type=str,
        default="q0",
        help="Qubit to test (default: q0)"
    )
    
    args = parser.parse_args()
    
    # Create gRPC channel and stub
    channel_address = f"{args.host}:{args.port}"
    print(f"\nConnecting to EPII daemon at {channel_address}...")
    
    channel = grpc.insecure_channel(channel_address)
    stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
    
    try:
        # Test connection
        if not test_connection(stub):
            print("\nFailed to connect to EPII daemon.")
            print("Please ensure the daemon is running:")
            print("  python scripts/epii_chronicle_daemon.py --launch-viewer")
            return 1
        
        # Get capabilities
        get_capabilities(stub)
        
        # Run calibration sequence similar to TuneUpDemo
        print("\n" + "="*60)
        print(" RUNNING CALIBRATION SEQUENCE")
        print("="*60)
        print("\nThis sequence mimics the TuneUpDemo notebook experiments.")
        print("Watch the Chronicle dashboard at http://localhost:8051")
        
        experiments = [
            ("Quick Rabi Check", run_resonator_spectroscopy),
            ("Rabi Oscillation", run_rabi_experiment),
            ("Ramsey (Frequency)", run_ramsey_experiment),
            ("T1 Relaxation", run_t1_experiment),
            ("Spin Echo (T2)", run_echo_experiment),
        ]
        
        results = []
        for name, func in experiments:
            success = func(stub, args.qubit)
            results.append((name, success))
            
            if success:
                print(f"\n  ✓ {name} completed - check Chronicle dashboard!")
            else:
                print(f"\n  ✗ {name} failed")
            
            # Small delay between experiments
            time.sleep(1)
        
        # Summary
        print_header("Calibration Summary")
        successful = sum(1 for _, success in results if success)
        total = len(results)
        print_result("Experiments Run", total)
        print_result("Successful", successful)
        print_result("Failed", total - successful)
        
        if successful == total:
            print("\n✓ All calibration experiments completed successfully!")
            print("  Check the Chronicle dashboard for detailed results and plots.")
        else:
            print("\n⚠ Some experiments failed. Check logs for details.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nClient stopped by user.")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        channel.close()


if __name__ == "__main__":
    sys.exit(main())