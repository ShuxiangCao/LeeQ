#!/usr/bin/env python3
"""
Advanced EPII client example showing a complete calibration workflow.
"""

import grpc
import numpy as np
import time
from leeq.epii.proto import epii_pb2, epii_pb2_grpc

class EPIIClient:
    """EPII client with convenience methods."""
    
    def __init__(self, address='localhost:50051'):
        self.channel = grpc.insecure_channel(address)
        self.stub = epii_pb2_grpc.ExperimentPlatformServiceStub(self.channel)
    
    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
    
    def serialize_array(self, array):
        """Convert NumPy array to protobuf bytes."""
        return array.astype(np.float64).tobytes()
    
    def deserialize_array(self, data):
        """Convert protobuf bytes back to NumPy array."""
        return np.frombuffer(data, dtype=np.float64)
    
    def ping(self):
        """Test service connectivity."""
        response = self.stub.Ping(epii_pb2.PingRequest())
        return response.message
    
    def get_parameter(self, name):
        """Get a parameter value."""
        request = epii_pb2.ParameterRequest(name=name)
        response = self.stub.GetParameter(request)
        return response.value
    
    def set_parameter(self, name, value):
        """Set a parameter value."""
        request = epii_pb2.SetParameterRequest(name=name, value=str(value))
        response = self.stub.SetParameter(request)
        return response.success
    
    def run_experiment(self, experiment_name, parameters):
        """Run an experiment with given parameters."""
        # Convert numpy arrays in parameters
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                parameters[key] = self.serialize_array(value)
            else:
                parameters[key] = str(value)
        
        request = epii_pb2.ExperimentRequest(
            experiment_name=experiment_name,
            parameters=parameters
        )
        
        response = self.stub.RunExperiment(request)
        
        return {
            'data': self.deserialize_array(response.data),
            'fit_params': dict(response.fit_params),
            'metadata': dict(response.metadata)
        }

def run_rabi_calibration(client, qubit):
    """Run Rabi experiment to calibrate pi pulse."""
    print(f"Running Rabi calibration for {qubit}...")
    
    amplitudes = np.linspace(0, 1, 51)
    result = client.run_experiment("rabi", {
        "qubit": qubit,
        "amplitudes": amplitudes,
        "num_shots": 2000
    })
    
    pi_amplitude = result['fit_params'].get('pi_amplitude')
    if pi_amplitude:
        print(f"  ✓ Pi amplitude: {float(pi_amplitude):.4f}")
        # Update the parameter
        client.set_parameter(f"{qubit}.pi_amplitude", pi_amplitude)
        return float(pi_amplitude)
    else:
        print("  ✗ Failed to fit pi amplitude")
        return None

def measure_t1(client, qubit):
    """Measure T1 relaxation time."""
    print(f"Measuring T1 for {qubit}...")
    
    delays = np.logspace(-6, -3, 30)  # 1μs to 1ms
    result = client.run_experiment("t1", {
        "qubit": qubit,
        "delays": delays,
        "num_shots": 1000
    })
    
    t1 = result['fit_params'].get('t1')
    if t1:
        t1_us = float(t1) * 1e6
        print(f"  ✓ T1: {t1_us:.1f} μs")
        return float(t1)
    else:
        print("  ✗ Failed to fit T1")
        return None

def measure_ramsey(client, qubit):
    """Measure T2* with Ramsey experiment."""
    print(f"Measuring T2* for {qubit}...")
    
    delays = np.linspace(0, 50e-6, 51)  # 0 to 50μs
    result = client.run_experiment("ramsey", {
        "qubit": qubit,
        "delays": delays,
        "detuning": 0.5e6,  # 500 kHz detuning
        "num_shots": 1000
    })
    
    t2_star = result['fit_params'].get('t2_star')
    frequency = result['fit_params'].get('frequency')
    
    if t2_star:
        t2_star_us = float(t2_star) * 1e6
        print(f"  ✓ T2*: {t2_star_us:.1f} μs")
        if frequency:
            freq_khz = float(frequency) * 1e-3
            print(f"  ✓ Detuning: {freq_khz:.1f} kHz")
        return float(t2_star)
    else:
        print("  ✗ Failed to fit T2*")
        return None

def measure_echo(client, qubit):
    """Measure T2 with echo experiment."""
    print(f"Measuring T2 echo for {qubit}...")
    
    delays = np.linspace(0, 100e-6, 51)  # 0 to 100μs
    result = client.run_experiment("echo", {
        "qubit": qubit,
        "delays": delays,
        "num_shots": 1000
    })
    
    t2 = result['fit_params'].get('t2')
    if t2:
        t2_us = float(t2) * 1e6
        print(f"  ✓ T2 echo: {t2_us:.1f} μs")
        return float(t2)
    else:
        print("  ✗ Failed to fit T2")
        return None

def run_randomized_benchmarking(client, qubit, max_length=100):
    """Run randomized benchmarking to measure gate fidelity."""
    print(f"Running randomized benchmarking for {qubit}...")
    
    lengths = np.logspace(0, np.log10(max_length), 10).astype(int)
    result = client.run_experiment("randomized_benchmarking", {
        "qubit": qubit,
        "lengths": lengths,
        "num_sequences": 30,
        "num_shots": 1000
    })
    
    fidelity = result['fit_params'].get('fidelity')
    if fidelity:
        fidelity_percent = float(fidelity) * 100
        print(f"  ✓ Gate fidelity: {fidelity_percent:.2f}%")
        return float(fidelity)
    else:
        print("  ✗ Failed to fit fidelity")
        return None

def full_qubit_calibration(client, qubit):
    """Run complete calibration sequence for a qubit."""
    print(f"\n{'='*50}")
    print(f"Full calibration for {qubit}")
    print(f"{'='*50}")
    
    results = {}
    
    # Step 1: Rabi calibration
    results['pi_amplitude'] = run_rabi_calibration(client, qubit)
    time.sleep(1)  # Brief pause between experiments
    
    # Step 2: T1 measurement
    results['t1'] = measure_t1(client, qubit)
    time.sleep(1)
    
    # Step 3: Ramsey (T2*)
    results['t2_star'] = measure_ramsey(client, qubit)
    time.sleep(1)
    
    # Step 4: Echo (T2)
    results['t2'] = measure_echo(client, qubit)
    time.sleep(1)
    
    # Step 5: Randomized benchmarking
    results['fidelity'] = run_randomized_benchmarking(client, qubit)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Calibration Summary for {qubit}")
    print(f"{'='*50}")
    
    if results['pi_amplitude']:
        print(f"Pi amplitude: {results['pi_amplitude']:.4f}")
    
    if results['t1']:
        print(f"T1: {results['t1']*1e6:.1f} μs")
    
    if results['t2_star']:
        print(f"T2*: {results['t2_star']*1e6:.1f} μs")
    
    if results['t2']:
        print(f"T2: {results['t2']*1e6:.1f} μs")
        if results['t1']:
            ratio = results['t2'] / results['t1']
            print(f"T2/T1 ratio: {ratio:.2f}")
    
    if results['fidelity']:
        print(f"Gate fidelity: {results['fidelity']*100:.2f}%")
    
    return results

def main():
    """Main calibration workflow."""
    client = EPIIClient()
    
    try:
        # Test connection
        print("Connecting to EPII service...")
        message = client.ping()
        print(f"✓ Connected: {message}")
        
        # Get available qubits (assuming we know q0 exists)
        qubits = ["q0"]  # Could be extended to discover qubits dynamically
        
        # Run calibration for each qubit
        all_results = {}
        for qubit in qubits:
            all_results[qubit] = full_qubit_calibration(client, qubit)
        
        # Overall summary
        print(f"\n{'='*60}")
        print("OVERALL CALIBRATION SUMMARY")
        print(f"{'='*60}")
        
        for qubit, results in all_results.items():
            print(f"\n{qubit}:")
            for param, value in results.items():
                if value is not None:
                    if param in ['t1', 't2', 't2_star']:
                        print(f"  {param}: {value*1e6:.1f} μs")
                    elif param == 'fidelity':
                        print(f"  {param}: {value*100:.2f}%")
                    else:
                        print(f"  {param}: {value:.4f}")
        
        print("\n✓ Calibration complete!")
        
    except grpc.RpcError as e:
        print(f"✗ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()