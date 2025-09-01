#!/usr/bin/env python3
"""
Validation script for LeeQ experiment documentation.
Checks that all experiments listed in leeq_builtin_experiments.md have:
1. EPII_INFO static variable
2. Proper docstrings for run() method
3. Proper docstrings for run_simulated() method (if it exists)
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import inspect
import importlib.util

# Base path for experiments
BASE_PATH = Path("/home/coxious/Projects/VILA_training/epii-documentation/leeq/experiments/builtin")

# All experiments from leeq_builtin_experiments.md
EXPERIMENT_CLASSES = {
    "Basic Calibrations": {
        "qubit_spectroscopy.py": [
            "QubitSpectroscopyFrequency",
            "QubitSpectroscopyAmplitudeFrequency"
        ],
        "two_tone_spectroscopy.py": [
            "TwoToneQubitSpectroscopy"
        ],
        "resonator_spectroscopy.py": [
            "ResonatorSweepTransmissionWithExtraInitialLPB",
            "ResonatorSweepAmpFreqWithExtraInitialLPB",
            "ResonatorSweepTransmissionXiComparison",
            "ResonatorPowerSweepSpectroscopy",
            "ResonatorBistabilityCharacterization",
            "ResonatorThreeRegimeCharacterization",
            "MeasurementScanParams"
        ],
        "rabi.py": [
            "NormalisedRabi",
            "PowerRabi",
            "MultiQubitRabi"
        ],
        "ramsey.py": [
            "SimpleRamseyMultilevel",
            "MultiQubitRamseyMultilevel"
        ],
        "drag.py": [
            "DragCalibrationSingleQubitMultilevel",
            "CrossAllXYDragMultiRunSingleQubitMultilevel",
            "DragPhaseCalibrationMultiQubitsMultilevel"
        ],
        "pingpong.py": [
            "PingPongSingleQubitMultilevel",
            "AmpPingpongCalibrationSingleQubitMultilevel",
            "PingPongMultiQubitMultilevel",
            "AmpTuneUpMultiQubitMultilevel"
        ],
        "residual_zz.py": [
            "CalibrateOptimizedFrequencyWith2QZZShift",
            "ZZShiftTwoQubitMultilevel"
        ],
        "state_discrimination/assignment.py": [
            "CalibrateFullAssignmentMatrices",
            "CalibrateSingleDutAssignmentMatrices"
        ],
        "state_discrimination/gaussian_mixture.py": [
            "MeasurementCalibrationMultilevelGMM"
        ],
        "state_discrimination/windowing_functions.py": [
            "MeasurementCollectTraces"
        ],
        "transmon_tuneup.py": [
            "MultilevelTransmonTuneup"
        ]
    },
    "Basic Characterizations": {
        "t1.py": [
            "SimpleT1",
            "MultiQubitT1",
            "MultiQuditT1Decay"
        ],
        "t2.py": [
            "SpinEchoMultiLevel"
        ],
        "randomized_benchmarking.py": [
            "RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem",
            "SingleQubitRandomizedBenchmarking"
        ]
    },
    "Multi-Qubit Gates": {
        "randomized_benchmarking.py": [
            "RandomizedBenchmarking2Qubits",
            "RandomizedBenchmarking2QubitsInterleavedComparison"
        ],
        "ac_stark/ac_stark_shift.py": [
            "StarkSingleQubitT1",
            "StarkTwoQubitsSWAP",
            "StarkTwoQubitsSWAPTwoDrives",
            "StarkRamseyMultilevel",
            "StarkDriveRamseyTwoQubits",
            "StarkDriveRamseyTwoQubitsTwoStarkDrives",
            "StarkDriveRamseyMultiQubits",
            "StarkZZShiftTwoQubitMultilevel",
            "StarkRepeatedGateRabi",
            "StarkContinuesRabi",
            "StarkRepeatedGateDRAGLeakageCalibration"
        ],
        "sizzel/calibration.py": [
            "ConditionalStarkTuneUpRabiXY",
            "ConditionalStarkTuneUpRepeatedGateXY",
            "ConditionalStarkEchoTuneUp"
        ],
        "sizzel/hamiltonian_tomography.py": [
            "ConditionalStarkFineFrequencyTuneUp",
            "ConditionalStarkFineAmpTuneUp",
            "ConditionalStarkFinePhaseTuneUp",
            "ConditionalStarkFineRiseTuneUp",
            "ConditionalStarkFineTruncTuneUp"
        ],
        "sizzel/expectation_value_difference.py": [
            # Base class excluded: ConsidtionalStarkSpectroscopyDifferenceBase
            "ConditionalStarkSpectroscopyDiffAmpFreq",
            "ConditionalStarkSpectroscopyDiffAmpTargetFreq",
            "ConditionalStarkSpectroscopyDiffPhaseFreq",
            "ConditionalStarkSpectroscopyDiffAmpPhase"
        ]
    },
    "Tomography": {
        "base.py": [
            "GeneralisedSingleDutStateTomography",
            "GeneralisedSingleDutProcessTomography",
            "GeneralisedStateTomography",
            "GeneralisedProcessTomography"
        ],
        "qubits.py": [
            "SingleQubitStateTomography",
            "MultiQubitsStateTomography",
            "MultiQubitsProcessTomography"
        ],
        "qutrits.py": [
            "MultiQutritsStateTomography",
            "MultiQutritsProcessTomography"
        ],
        "qudits.py": [
            "MultiQuditsStateTomography",
            "MultiQuditsProcessTomography"
        ]
    },
    "Hamiltonian Tomography": {
        # base.py contains only base classes - excluded
        "single_qubit.py": [
            # Base classes excluded: HamiltonianTomographySingleQubitBase, HamiltonianTomographySingleQubitXYBase
            "HamiltonianTomographySingleQubitStarkShift",
            "HamiltonianTomographySingleQubitOffresonanceDrive"
        ]
    },
    "Optimal Control": {
        "single_qubit_gates.py": [
            "GRAPESingleQubitGate"
        ]
    }
}

# Map category to base directory
CATEGORY_PATHS = {
    "Basic Calibrations": "basic/calibrations",
    "Basic Characterizations": "basic/characterizations",
    "Multi-Qubit Gates": "multi_qubit_gates",
    "Tomography": "tomography",
    "Hamiltonian Tomography": "hamiltonian_tomography",
    "Optimal Control": "optimal_control"
}


def check_class_has_epii_info(source_code: str, class_name: str) -> bool:
    """Check if a class has EPII_INFO static variable."""
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == "EPII_INFO":
                                return True
    except:
        pass
    return False


def check_method_has_docstring(source_code: str, class_name: str, method_name: str) -> Tuple[bool, bool]:
    """
    Check if a method exists and has a docstring.
    Returns (method_exists, has_docstring)
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        has_docstring = (
                            isinstance(item.body[0], ast.Expr) and
                            isinstance(item.body[0].value, ast.Constant) and
                            isinstance(item.body[0].value.value, str) and
                            len(item.body[0].value.value.strip()) > 10
                        )
                        return True, has_docstring
        return False, False
    except:
        return False, False


def validate_experiment_class(file_path: Path, class_name: str) -> Dict[str, bool]:
    """Validate a single experiment class."""
    results = {
        "exists": False,
        "has_epii_info": False,
        "run_exists": False,
        "run_has_docstring": False,
        "run_simulated_exists": False,
        "run_simulated_has_docstring": False
    }
    
    if not file_path.exists():
        return results
    
    results["exists"] = True
    
    try:
        source_code = file_path.read_text()
        
        # Check for EPII_INFO
        results["has_epii_info"] = check_class_has_epii_info(source_code, class_name)
        
        # Check run() method
        results["run_exists"], results["run_has_docstring"] = check_method_has_docstring(
            source_code, class_name, "run"
        )
        
        # Check run_simulated() method
        results["run_simulated_exists"], results["run_simulated_has_docstring"] = check_method_has_docstring(
            source_code, class_name, "run_simulated"
        )
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return results


def main():
    """Main validation function."""
    print("=" * 80)
    print("LeeQ Experiment Documentation Validation")
    print("=" * 80)
    print()
    
    total_experiments = 0
    passed_experiments = 0
    failed_experiments = []
    
    for category, files in EXPERIMENT_CLASSES.items():
        print(f"\n{category}")
        print("-" * len(category))
        
        category_path = BASE_PATH / CATEGORY_PATHS[category]
        
        for file_name, classes in files.items():
            file_path = category_path / file_name
            
            for class_name in classes:
                total_experiments += 1
                results = validate_experiment_class(file_path, class_name)
                
                # Determine pass/fail
                passed = (
                    results["exists"] and
                    results["has_epii_info"] and
                    results["run_exists"] and
                    results["run_has_docstring"]
                    # Note: run_simulated is optional
                )
                
                if passed:
                    passed_experiments += 1
                    status = "✅ PASS"
                else:
                    failed_experiments.append((category, file_name, class_name, results))
                    status = "❌ FAIL"
                
                print(f"  {class_name}: {status}")
                
                if not passed:
                    if not results["exists"]:
                        print(f"    - File not found: {file_path}")
                    else:
                        if not results["has_epii_info"]:
                            print(f"    - Missing EPII_INFO")
                        if not results["run_exists"]:
                            print(f"    - Missing run() method")
                        elif not results["run_has_docstring"]:
                            print(f"    - run() missing/incomplete docstring")
                        
                        # Optional: report on run_simulated
                        if results["run_simulated_exists"] and not results["run_simulated_has_docstring"]:
                            print(f"    - run_simulated() missing/incomplete docstring")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Passed: {passed_experiments}")
    print(f"Failed: {total_experiments - passed_experiments}")
    print(f"Success rate: {passed_experiments/total_experiments*100:.1f}%")
    
    if failed_experiments:
        print("\n" + "=" * 80)
        print("FAILED EXPERIMENTS DETAILS")
        print("=" * 80)
        for category, file_name, class_name, results in failed_experiments:
            print(f"\n{category} / {file_name} / {class_name}:")
            print(f"  File exists: {results['exists']}")
            if results['exists']:
                print(f"  Has EPII_INFO: {results['has_epii_info']}")
                print(f"  Has run(): {results['run_exists']}")
                print(f"  run() has docstring: {results['run_has_docstring']}")
                print(f"  Has run_simulated(): {results['run_simulated_exists']}")
                if results['run_simulated_exists']:
                    print(f"  run_simulated() has docstring: {results['run_simulated_has_docstring']}")
    
    return passed_experiments == total_experiments


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)