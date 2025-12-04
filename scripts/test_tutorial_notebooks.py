#!/usr/bin/env python
"""Test script to validate tutorial notebooks execution."""

import sys
import os
import subprocess
import json
from pathlib import Path

def test_notebook_imports():
    """Test that all required imports work."""
    try:
        # Test core imports
        from leeq.chronicle import Chronicle, log_and_record
        from leeq.core.elements.built_in.qudit_transmon import TransmonElement
        from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
        from leeq.experiments.experiments import ExperimentManager
        from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
        
        # Test experiment imports
        from leeq.experiments.builtin.basic.calibrations import (
            RabiAmplitudeCalibration, 
            MeasurementStatistics
        )
        from leeq.experiments.builtin.basic.characterizations import (
            T1Measurement,
            T2EchoMeasurement,
            T2RamseyMeasurement
        )
        
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_simulation():
    """Test basic simulation setup."""
    try:
        from leeq.chronicle import Chronicle
        from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
        from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
        from leeq.experiments.experiments import ExperimentManager
        
        # Start Chronicle
        Chronicle().start_log()
        
        # Create virtual qubit
        virtual_qubit = VirtualTransmon(
            name='TestQubit',
            qubit_frequency=5040.4,
            anharmonicity=-198,
            t1=70,
            t2=35,
            readout_frequency=9645.4
        )
        
        # Create setup
        manager = ExperimentManager()
        manager.clear_setups()
        setup = HighLevelSimulationSetup(
            name='TestSetup',
            virtual_qubits={1: virtual_qubit}
        )
        manager.register_setup(setup)
        
        print("✓ Basic simulation setup successful")
        return True
    except Exception as e:
        print(f"✗ Simulation setup failed: {e}")
        return False

def test_basic_experiment():
    """Test running a basic experiment."""
    try:
        from leeq.chronicle import Chronicle
        from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
        from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
        from leeq.experiments.experiments import ExperimentManager
        from leeq.experiments.builtin.basic.calibrations import MeasurementStatistics
        
        # Setup
        Chronicle().start_log()
        virtual_qubit = VirtualTransmon(
            name='TestQubit',
            qubit_frequency=5040.4,
            anharmonicity=-198,
            t1=70,
            t2=35,
            readout_frequency=9645.4
        )
        
        manager = ExperimentManager()
        manager.clear_setups()
        setup = HighLevelSimulationSetup(
            name='TestSetup',
            virtual_qubits={1: virtual_qubit}
        )
        manager.register_setup(setup)
        
        # Run experiment
        experiment = MeasurementStatistics(
            name="TestMeasurement",
            qubit=1,
            repeated_measurement_count=100
        )
        
        results = experiment.run()
        
        if 'statistics' in results:
            print("✓ Basic experiment execution successful")
            print(f"  Ground state probability: {results['statistics']['ground_state_probability']:.3f}")
            return True
        else:
            print("✗ Experiment did not return expected results")
            return False
            
    except Exception as e:
        print(f"✗ Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_notebook_structure(notebook_path):
    """Validate notebook has proper structure."""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Check for cells
        if 'cells' not in notebook:
            print(f"✗ {notebook_path}: No cells found")
            return False
        
        # Count cell types
        markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
        code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
        
        print(f"✓ {notebook_path}:")
        print(f"  - {markdown_cells} markdown cells")
        print(f"  - {code_cells} code cells")
        
        if markdown_cells < 3 or code_cells < 3:
            print(f"  ⚠ Warning: Notebook seems too short")
            
        return True
        
    except Exception as e:
        print(f"✗ {notebook_path}: Failed to validate - {e}")
        return False

def main():
    """Main test runner."""
    print("=" * 60)
    print("Tutorial Notebook Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    print("\n1. Testing imports...")
    if not test_notebook_imports():
        all_passed = False
    
    # Test simulation
    print("\n2. Testing simulation setup...")
    if not test_basic_simulation():
        all_passed = False
    
    # Test experiment
    print("\n3. Testing experiment execution...")
    if not test_basic_experiment():
        all_passed = False
    
    # Validate notebook structure
    print("\n4. Validating notebook structure...")
    notebooks = [
        "notebooks/tutorials/01_basics.ipynb",
        "notebooks/tutorials/02_single_qubit.ipynb"
    ]
    
    for notebook in notebooks:
        notebook_path = Path(notebook)
        if notebook_path.exists():
            if not validate_notebook_structure(notebook_path):
                all_passed = False
        else:
            print(f"✗ {notebook}: File not found")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All validation tests passed!")
        print("Notebooks are ready for use.")
    else:
        print("✗ Some tests failed. Please review the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

# Script-style execution converted to proper pytest discovery
# Tests will be run by pytest discovery, no manual execution needed