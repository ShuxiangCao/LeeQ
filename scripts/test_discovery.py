#!/usr/bin/env python3
"""Test script for dynamic experiment discovery."""

import sys
import os

# Add current directory to path BEFORE any imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leeq.epii.experiments import ExperimentRouter

# Create router with dynamic discovery
router = ExperimentRouter()

print(f"Discovered: {len(router.experiment_map)} experiments")

# List first 10 experiments discovered
print("\nFirst 10 experiments discovered:")
for i, name in enumerate(sorted(router.experiment_map.keys())[:10]):
    print(f"  {i+1}. {name}")

# Verify known experiments exist (using module prefix naming)
expected_experiments = [
    'calibrations.NormalisedRabi',
    'characterizations.SimpleT1',
    'calibrations.DragCalibrationSingleQubitMultilevel',
    'characterizations.SpinEchoMultiLevel',
    'calibrations.SimpleRamseyMultilevel'
]

print("\nVerifying expected experiments:")
all_found = True
for exp_name in expected_experiments:
    if exp_name in router.experiment_map:
        print(f"  ✓ {exp_name} found")
    else:
        print(f"  ✗ {exp_name} NOT found")
        all_found = False

# Check if we found at least 74 experiments as expected
if len(router.experiment_map) >= 74:
    print(f"\n✅ Core discovery working - found {len(router.experiment_map)} experiments (expected 74+)")
else:
    print(f"\n⚠️ Found only {len(router.experiment_map)} experiments (expected 74+)")

# Test get_experiment_info for one experiment
if 'calibrations.NormalisedRabi' in router.experiment_map:
    print("\nTesting get_experiment_info for calibrations.NormalisedRabi:")
    info = router.get_experiment_info('calibrations.NormalisedRabi')
    has_epii = 'epii_info' in info and info['epii_info'] is not None
    has_docstring = 'run_docstring' in info and info['run_docstring'] is not None
    print(f"  Has EPII_INFO: {has_epii}")
    print(f"  Has run docstring: {has_docstring}")
    if has_epii:
        epii_info = info['epii_info']
        print(f"  EPII description: {epii_info.get('description', 'N/A')}")

print("\n" + "="*50)
if all_found and len(router.experiment_map) >= 74:
    print("✅ All validation checks passed!")
else:
    print("⚠️ Some validation checks failed")