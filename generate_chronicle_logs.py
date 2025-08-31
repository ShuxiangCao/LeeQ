#!/usr/bin/env python3
"""
Generate Chronicle Logs Script

This script replicates the TuneUpDemo.ipynb notebook functionality to generate
chronicle log files with various experiment types for testing the dashboard.

Based on: notebooks/SimulatedSystem/TuneUpDemo.ipynb
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the notebooks directory to path to import simulated_setup
notebook_path = Path(__file__).parent / "notebooks" / "SimulatedSystem"
sys.path.insert(0, str(notebook_path))

try:
    import leeq
    from simulated_setup import *
    from scipy import optimize as so
    from leeq.experiments.builtin import *
    import plotly.graph_objects as go
    from leeq.chronicle import log_and_record, register_browser_function
    from leeq.utils.compatibility import *
    from leeq.core.elements.built_in.qudit_transmon import TransmonElement
except ImportError as e:
    print(f"Error importing LeeQ modules: {e}")
    print("Make sure you're in the leeq-nvidia environment")
    sys.exit(1)

def setup_qubits():
    """Initialize qubits configuration"""
    print("Setting up simulation...")
    simulation_setup()
    
    # Configure setup parameters
    setup().status().set_param("Shot_Number", 500)
    setup().status().set_param("Shot_Period", 500)
    
    dut_dict = {
        'Q1': {'Active': True, 'Tuneup': False, 'FromLog': False, 'Params': configuration_a},
        'Q2': {'Active': True, 'Tuneup': False, 'FromLog': False, 'Params': configuration_b}
    }
    
    duts_dict = {}
    for hrid, dd in dut_dict.items():
        if dd['Active']:
            if dd['FromLog']:
                dut = TransmonElement.load_from_calibration_log(dd['Params']['hrid'])
            else:
                dut = TransmonElement(name=dd['Params']['hrid'], parameters=dd['Params'])
            
            if dd['Tuneup']:
                dut.save_calibration_log()
            else:
                lpb_scan = (dut.get_c1('f01')['I'], dut.get_c1('f01')['X'])
                calib = MeasurementCalibrationMultilevelGMM(dut, mprim_index=0, sweep_lpb_list=lpb_scan)
            
            dut.print_config_info()
            duts_dict[hrid] = dut
    
    return duts_dict

def run_resonator_spectroscopy(duts_dict):
    """Run resonator spectroscopy experiments"""
    print("\n=== Running Resonator Spectroscopy ===")
    
    dut = duts_dict['Q1']
    mprim = dut.get_measurement_prim_intlist(0)
    c1 = dut.get_c1('f01')
    
    print("Running ResonatorSweepTransmissionWithExtraInitialLPB...")
    resonator_exp = ResonatorSweepTransmissionWithExtraInitialLPB(
        dut,
        start=9645-10,
        stop=9645+10,
        step=0.5,
        num_avs=10000,
        rep_rate=0.0,
        mp_width=8,
        amp=0.03
    )
    print(f"Resonator spectroscopy completed: {type(resonator_exp).__name__}")
    return resonator_exp

def run_qubit_tuneup(duts_dict):
    """Run qubit tuneup experiments"""
    print("\n=== Running Qubit Tuneup ===")
    
    dut = duts_dict['Q1']
    
    # Calibrate single qubit pulse amplitude
    print("Running Rabi experiment...")
    rabi = NormalisedRabi(
        dut_qubit=dut,
        step=0.01,
        stop=0.5,
        amp=0.19905818643939352,
        update=True
    )
    print(f"Rabi experiment completed: {type(rabi).__name__}")
    
    # Ramsey experiments with different parameters
    print("Running Ramsey experiments...")
    ramsey1 = SimpleRamseyMultilevel(dut=dut, set_offset=10, stop=0.3, step=0.005)
    print(f"Ramsey 1 completed: {type(ramsey1).__name__}")
    
    ramsey2 = SimpleRamseyMultilevel(dut=dut, set_offset=1, stop=3, step=0.05)
    print(f"Ramsey 2 completed: {type(ramsey2).__name__}")
    
    ramsey3 = SimpleRamseyMultilevel(dut=dut, set_offset=0.1, stop=30, step=0.5)
    print(f"Ramsey 3 completed: {type(ramsey3).__name__}")
    
    return rabi, ramsey1, ramsey2, ramsey3

def run_advanced_calibration(duts_dict):
    """Run advanced calibration experiments"""
    print("\n=== Running Advanced Calibration ===")
    
    dut = duts_dict['Q1']
    
    # Pingpong calibration
    print("Running Pingpong calibration...")
    pingpong = AmpPingpongCalibrationSingleQubitMultilevel(dut=dut)
    print(f"Pingpong calibration completed: {type(pingpong).__name__}")
    
    # DRAG calibration
    print("Running DRAG calibration...")
    drag = CrossAllXYDragMultiRunSingleQubitMultilevel(dut=dut)
    print(f"DRAG calibration completed: {type(drag).__name__}")
    
    return pingpong, drag

def run_coherence_experiments(duts_dict):
    """Run coherence experiments (T1, T2, echo)"""
    print("\n=== Running Coherence Experiments ===")
    
    dut = duts_dict['Q1']
    
    # T1 measurement
    print("Running T1 measurement...")
    t1_exp = SimpleT1(qubit=dut, time_length=300, time_resolution=5)
    print(f"T1 experiment completed: {type(t1_exp).__name__}")
    
    # Spin echo
    print("Running Spin Echo...")
    echo = SpinEchoMultiLevel(dut=dut, free_evolution_time=200, time_resolution=5)
    print(f"Spin Echo completed: {type(echo).__name__}")
    
    # Final Ramsey
    print("Running final Ramsey...")
    final_ramsey = SimpleRamseyMultilevel(
        dut=dut,
        stop=50,
        step=0.25,
        set_offset=0.2
    )
    print(f"Final Ramsey completed: {type(final_ramsey).__name__}")
    
    return t1_exp, echo, final_ramsey

def main():
    """Main function to run all experiments and generate chronicle logs"""
    print("Starting Chronicle Log Generation Script")
    print("=====================================")
    
    try:
        # Setup qubits
        duts_dict = setup_qubits()
        print(f"Successfully initialized {len(duts_dict)} qubits")
        
        # Launch Chronicle viewer for monitoring
        from leeq.chronicle import Chronicle
        chronicle = Chronicle()
        chronicle.launch_viewer(port=8051)  # Launches in background
        print("\n" + "="*50)
        print("Chronicle viewer launched at http://localhost:8051")
        print("You can monitor experiments as they complete in real-time")
        print("="*50 + "\n")
        
        experiments = []
        
        # Run resonator spectroscopy
        resonator_exp = run_resonator_spectroscopy(duts_dict)
        experiments.append(resonator_exp)
        input("\n📊 Check viewer for completed resonator spectroscopy. Press Enter to continue...")
        
        # Run qubit tuneup
        rabi, ramsey1, ramsey2, ramsey3 = run_qubit_tuneup(duts_dict)
        experiments.extend([rabi, ramsey1, ramsey2, ramsey3])
        input("\n📊 Check viewer for Rabi and Ramsey experiments. Press Enter to continue...")
        
        # Run advanced calibration
        pingpong, drag = run_advanced_calibration(duts_dict)
        experiments.extend([pingpong, drag])
        input("\n📊 Check viewer for advanced calibration experiments. Press Enter to continue...")
        
        # Run coherence experiments
        t1_exp, echo, final_ramsey = run_coherence_experiments(duts_dict)
        experiments.extend([t1_exp, echo, final_ramsey])
        input("\n📊 Check viewer for coherence measurements. Press Enter to continue...")
        
        print(f"\n=== Experiment Generation Complete ===")
        print(f"Total experiments created: {len(experiments)}")
        
        # Print experiment types for verification
        print("\nExperiment types generated:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i:2d}. {type(exp).__name__}")
        
        # Get chronicle log location
        log_path = chronicle._config["log_path"]
        print(f"\nChronicle logs saved to: {log_path}")
        
        print("\n✅ Chronicle log generation completed successfully!")
        print("You can now use these logs with the chronicle viewer dashboard.")
        
    except Exception as e:
        print(f"\n❌ Error during experiment generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)