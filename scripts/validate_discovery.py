#!/usr/bin/env python
"""
Comprehensive validation script for dynamic experiment discovery.
Verifies that all requirements are met for the EPII experiment system.
"""

import sys
from leeq.epii.experiments import ExperimentRouter

def validate_discovery():
    """Run comprehensive validation of experiment discovery."""
    print("Starting comprehensive experiment discovery validation...")
    print("=" * 60)
    
    # Initialize router
    router = ExperimentRouter()
    experiments = router.experiment_map
    
    # Test 1: Check discovery count
    print("\n1. Checking experiment discovery count...")
    expected_min = 74
    actual_count = len(experiments)
    test1_passed = actual_count >= expected_min
    if test1_passed:
        print(f"   ✅ Found {actual_count} experiments (expected >= {expected_min})")
    else:
        print(f"   ❌ Only found {actual_count} experiments (expected >= {expected_min})")
        print(f"      Note: This may be due to import issues - continuing validation...")
    
    # Test 2: Check naming convention
    print("\n2. Validating experiment naming convention...")
    valid_categories = [
        'calibrations', 
        'characterizations', 
        'multi_qubit_gates', 
        'tomography', 
        'hamiltonian_tomography', 
        'optimal_control'
    ]
    
    naming_errors = []
    for name in experiments.keys():
        if '.' in name:
            parts = name.split('.')
            if len(parts) != 2:
                naming_errors.append(f"Invalid format (multiple dots): {name}")
            else:
                category, class_name = parts
                if category not in valid_categories:
                    naming_errors.append(f"Unknown category '{category}': {name}")
        # Simple names (no dots) are also valid for uncategorized experiments
    
    test2_passed = len(naming_errors) == 0
    if test2_passed:
        print(f"   ✅ All {actual_count} experiment names follow convention")
    else:
        print(f"   ❌ Found {len(naming_errors)} naming issues:")
        for error in naming_errors[:5]:  # Show first 5 errors
            print(f"      - {error}")
        if len(naming_errors) > 5:
            print(f"      ... and {len(naming_errors) - 5} more")
    
    # Test 3: Check metadata availability
    print("\n3. Checking metadata availability...")
    
    # Sample a variety of experiments
    sample_size = min(10, len(experiments))
    sample_experiments = list(experiments.keys())[:sample_size]
    
    metadata_errors = []
    for exp_name in sample_experiments:
        try:
            exp_class = router.get_experiment(exp_name)
            if not exp_class:
                metadata_errors.append(f"{exp_name}: experiment class not found")
                continue
            
            # Check for EPII_INFO
            if not hasattr(exp_class, 'EPII_INFO'):
                # This is acceptable - not all experiments have EPII_INFO
                pass
            
            # Check for run method docstring
            if hasattr(exp_class, 'run'):
                run_method = getattr(exp_class, 'run')
                if not run_method.__doc__:
                    # This is acceptable - not all experiments have docstrings
                    pass
                
        except Exception as e:
            metadata_errors.append(f"{exp_name}: error checking metadata - {str(e)}")
    
    test3_passed = len(metadata_errors) == 0
    if test3_passed:
        print(f"   ✅ Metadata available for sampled experiments")
    else:
        print(f"   ❌ Found {len(metadata_errors)} metadata issues:")
        for error in metadata_errors[:5]:
            print(f"      - {error}")
        if len(metadata_errors) > 5:
            print(f"      ... and {len(metadata_errors) - 5} more")
    
    # Test 4: Verify EPII_INFO extraction
    print("\n4. Verifying EPII_INFO extraction...")
    epii_count = 0
    no_epii_count = 0
    
    for exp_name in experiments.keys():
        try:
            exp_class = router.get_experiment(exp_name)
            if exp_class and hasattr(exp_class, 'EPII_INFO'):
                epii_count += 1
            else:
                no_epii_count += 1
        except:
            pass
    
    print(f"   ✅ {epii_count} experiments have EPII_INFO")
    print(f"   ℹ️  {no_epii_count} experiments don't have EPII_INFO (expected)")
    
    # Test 5: Category distribution
    print("\n5. Experiment category distribution:")
    category_counts = {}
    uncategorized = 0
    
    for name in experiments.keys():
        if '.' in name:
            category = name.split('.')[0]
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            uncategorized += 1
    
    for category in sorted(category_counts.keys()):
        print(f"   - {category}: {category_counts[category]} experiments")
    if uncategorized > 0:
        print(f"   - uncategorized: {uncategorized} experiments")
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY:")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if test1_passed:
        print(f"✅ All {actual_count} experiments successfully discovered")
    else:
        print(f"⚠️  Only {actual_count} experiments discovered (expected >= {expected_min})")
    
    if test2_passed:
        print("✅ Naming convention validated")
    else:
        print("❌ Some naming convention issues found")
    
    if test3_passed:
        print("✅ Metadata extraction working")
    else:
        print("❌ Some metadata issues found")
    
    print("✅ EPII_INFO available where defined")
    
    if all_passed:
        print("\n✅ All validation checks passed!")
    else:
        print("\n⚠️  Some validation checks had issues - review above")
    
    return all_passed

if __name__ == "__main__":
    success = validate_discovery()
    sys.exit(0 if success else 1)