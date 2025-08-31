#!/usr/bin/env python
"""
Chronicle Session Viewer Example

This script demonstrates how to use the Chronicle.launch_viewer() method
to monitor experiments in real-time during a calibration session.

Usage:
    python chronicle_session_example.py
    
The viewer will open at http://localhost:8051 and update every 5 seconds
as new experiments complete.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from leeq.chronicle import Chronicle
from leeq.setups.built_in.setup_simulation_from_yaml import (
    setup_simulation_from_yaml
)
from leeq.experiments.builtin import (
    QubitSpectroscopyFrequency,
    ResonatorSweepTransmissionWithExtraInitialLPB
)


def setup_simulation():
    """Setup a simulated quantum system for demonstration."""
    print("Setting up simulated quantum system...")
    
    # Create a simple yaml configuration
    yaml_config = """
simulation:
  hamiltonian:
    lab_frame: True
    
virtual_qubits:
  - name: "Q01"
    parameters:
      frequency: 5000.0
      anharmonicity: -200.0
      resonator_frequency: 6500.0
      coupling_strength: 100.0
"""
    
    # Save temporary config
    config_path = Path("temp_simulation_config.yaml")
    config_path.write_text(yaml_config)
    
    try:
        # Setup simulation
        setup = setup_simulation_from_yaml(str(config_path))
        return setup
    finally:
        # Clean up temp file
        if config_path.exists():
            config_path.unlink()


def run_demonstration():
    """Run a demonstration of the Chronicle session viewer."""
    
    print("=" * 60)
    print("Chronicle Session Viewer Demonstration")
    print("=" * 60)
    
    # Initialize Chronicle singleton
    chronicle = Chronicle()
    print("\n1. Chronicle singleton initialized")
    
    # Launch the session viewer
    print("\n2. Launching Chronicle session viewer...")
    try:
        # Launch viewer in background thread if possible
        import threading
        viewer_thread = threading.Thread(
            target=chronicle.launch_viewer,
            kwargs={'port': 8051, 'debug': False},
            daemon=True
        )
        viewer_thread.start()
        time.sleep(3)  # Give viewer time to start
        
        print("   Session viewer launched successfully!")
        print("   Open http://localhost:8051 in your browser")
        print("   The viewer will auto-refresh every 5 seconds")
        
    except Exception as e:
        print(f"   Note: Could not launch viewer in background: {e}")
        print("   You can launch it manually with: chronicle.launch_viewer()")
    
    # Setup simulation
    print("\n3. Setting up simulated quantum system...")
    setup = setup_simulation()
    qubit = setup.get_virtual_qubit("Q01")
    print(f"   Created virtual qubit: {qubit.name}")
    
    # Run some example experiments
    print("\n4. Running example experiments...")
    print("   Watch them appear in the viewer as they complete!")
    
    experiments_to_run = [
        ("Resonator Spectroscopy", {
            'dut_qubit': qubit,
            'start': 6400.0,
            'stop': 6600.0,
            'step': 5.0,
            'num_avs': 100
        }),
        ("Qubit Spectroscopy", {
            'dut_qubit': qubit,
            'start': 4900.0,
            'stop': 5100.0,
            'step': 2.0,
            'num_avs': 500
        }),
        ("Fine Qubit Spectroscopy", {
            'dut_qubit': qubit,
            'start': 4990.0,
            'stop': 5010.0,
            'step': 0.5,
            'num_avs': 1000
        })
    ]
    
    for i, (exp_name, params) in enumerate(experiments_to_run, 1):
        print(f"\n   Running experiment {i}/3: {exp_name}")
        
        try:
            if "Resonator" in exp_name:
                exp = ResonatorSweepTransmissionWithExtraInitialLPB(**params)
            else:
                exp = QubitSpectroscopyFrequency(**params)
            
            print(f"   ✓ {exp_name} completed!")
            print("   Check the viewer - it should appear within 5 seconds")
            
            # Pause between experiments
            if i < len(experiments_to_run):
                time.sleep(3)
                
        except Exception as e:
            print(f"   ✗ {exp_name} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nKey Points Demonstrated:")
    print("✓ Chronicle singleton initialization")
    print("✓ Launching session viewer with chronicle.launch_viewer()")
    print("✓ Running experiments that appear in viewer")
    print("✓ Automatic 5-second refresh updates")
    
    print("\nNext Steps:")
    print("1. Open http://localhost:8051 to see your experiments")
    print("2. Click on experiments in the tree to see details")
    print("3. Use the manual refresh button for immediate updates")
    print("4. Try generating plots for visualization")
    
    print("\nPress Ctrl+C to stop the viewer and exit")
    
    # Keep script running so viewer stays active
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")


def main():
    """Main entry point."""
    try:
        run_demonstration()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()