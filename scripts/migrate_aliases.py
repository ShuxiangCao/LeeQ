#!/usr/bin/env python3
"""
Migration tool to replace all experiment aliases with canonical names.

This script will update all Python files to use canonical experiment names
like 'calibrations.NormalisedRabi' instead of 'rabi'.

Usage:
    # Dry run (default) - shows what would change
    python scripts/migrate_aliases.py
    
    # Apply changes
    python scripts/migrate_aliases.py --apply
    
    # Specific file
    python scripts/migrate_aliases.py --file tests/epii/test_experiments.py
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Comprehensive alias to canonical mapping
ALIAS_TO_CANONICAL = {
    'rabi': 'calibrations.NormalisedRabi',
    't1': 'characterizations.SimpleT1',
    'ramsey': 'calibrations.SimpleRamseyMultilevel',
    'echo': 'characterizations.SpinEchoMultiLevel',
    'spin_echo': 'characterizations.SpinEchoMultiLevel',
    'drag': 'calibrations.DragCalibrationSingleQubitMultilevel',
    'randomized_benchmarking': 'characterizations.RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem',
    'multi_qubit_rabi': 'calibrations.MultiQubitRabi',
    'multi_qubit_t1': 'characterizations.MultiQubitT1',
    'multi_qubit_ramsey': 'calibrations.MultiQubitRamseyMultilevel',
    'qubit_spectroscopy_frequency': 'calibrations.QubitSpectroscopyFrequency',
    
    # Additional mappings for special cases
    'drag_phase': 'calibrations.DragPhaseCalibrationMultiQubitsMultilevel',
    'single_qubit_rb': 'characterizations.SingleQubitRandomizedBenchmarking',
}

# Files to migrate
FILES_TO_MIGRATE = [
    # Core implementation
    'leeq/epii/experiments.py',
    'leeq/epii/daemon.py',
    'leeq/epii/utils.py',
    'leeq/epii/config.py',
    
    # Tests
    'tests/epii/test_experiments.py',
    'tests/epii/test_leeq_backend_integration.py',
    'tests/integration_tests/test_epii_daemon.py',
    'tests/epii/test_service.py',
    'tests/epii/test_logging_debugging.py',
    'tests/epii/test_protobuf_messages.py',
    'tests/epii/test_parameters.py',
    'tests/epii/test_structure.py',
    'tests/epii/test_epii_config.py',
    'tests/epii/test_fixtures_new.py',
    'tests/epii/conftest.py',
    
    # Examples
    'examples/epii/calibration_client.py',
    'examples/epii/simple_client.py',
    'examples/epii/chronicle_test_client.py',
    
    # Documentation helpers
    'docs/notebooks/shared/experiment_helpers.py',
    'docs/notebooks/shared/validation_utils.py',
    
    # Additional files with aliases
    'test_epii_qubit_mock.py',
    'test_extended_data.py',
    'scripts/notebook_test_runner.py',
    'scripts/check_outputs.py',
]


def create_replacement_patterns() -> List[Tuple[str, str]]:
    """Create all replacement patterns for migration."""
    patterns = []
    
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        # String literals with quotes
        patterns.extend([
            (f'"{alias}"', f'"{canonical}"'),
            (f"'{alias}'", f"'{canonical}'"),
            
            # In parameter contexts
            (f'experiment_type="{alias}"', f'experiment_type="{canonical}"'),
            (f"experiment_type='{alias}'", f"experiment_type='{canonical}'"),
            (f'experiment_name="{alias}"', f'experiment_name="{canonical}"'),
            (f"experiment_name='{alias}'", f"experiment_name='{canonical}'"),
            
            # In test assertions
            (f'== "{alias}"', f'== "{canonical}"'),
            (f"== '{alias}'", f"== '{canonical}'"),
            (f'in ["{alias}"', f'in ["{canonical}"'),
            (f"in ['{alias}'", f"in ['{canonical}'"),
        ])
    
    return patterns


def migrate_file(file_path: Path, dry_run: bool = True) -> Tuple[List[Tuple[str, str]], str]:
    """
    Migrate a single file from aliases to canonical names.
    
    Returns:
        Tuple of (replacements made, modified content)
    """
    if not file_path.exists():
        return [], ""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    replacements = []
    
    # Apply individual replacements
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        # Count occurrences before replacement
        count_before = content.count(f'"{alias}"') + content.count(f"'{alias}'")
        
        if count_before > 0:
            # Replace all variations
            for old, new in create_replacement_patterns():
                if old.startswith(f'"{alias}"') or old.startswith(f"'{alias}'"):
                    if old in content:
                        content = content.replace(old, new)
                        replacements.append((old, new))
    
    # Special case: lists of experiments (common in tests)
    list_patterns = [
        (
            r'\["rabi", "t1", "ramsey", "echo", "drag", "randomized_benchmarking"\]',
            '["calibrations.NormalisedRabi", "characterizations.SimpleT1", '
            '"calibrations.SimpleRamseyMultilevel", "characterizations.SpinEchoMultiLevel", '
            '"calibrations.DragCalibrationSingleQubitMultilevel", '
            '"characterizations.RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem"]'
        ),
        (
            r"'rabi', 't1', 'ramsey'",
            "'calibrations.NormalisedRabi', 'characterizations.SimpleT1', 'calibrations.SimpleRamseyMultilevel'"
        ),
    ]
    
    for pattern, replacement in list_patterns:
        if pattern in content or re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            replacements.append((pattern[:30] + "...", replacement[:30] + "..."))
    
    # Handle parameter map definitions (in experiments.py)
    if 'parameter_map = {' in content:
        # This section needs to be removed entirely in experiments.py
        if 'leeq/epii/experiments.py' in str(file_path):
            # Mark for manual removal
            replacements.append(("parameter_map definitions", "REMOVE MANUALLY"))
    
    # Save changes if not dry run
    if not dry_run and content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return replacements, content


def analyze_file(file_path: Path) -> Dict[str, int]:
    """Analyze a file for alias usage."""
    if not file_path.exists():
        return {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    usage = defaultdict(int)
    for alias in ALIAS_TO_CANONICAL.keys():
        # Count various forms
        usage[alias] += content.count(f'"{alias}"')
        usage[alias] += content.count(f"'{alias}'")
        usage[alias] += content.count(f'experiment_type="{alias}"')
        usage[alias] += content.count(f"experiment_name='{alias}'")
    
    return dict(usage)


def migrate_codebase(dry_run: bool = True, specific_file: Optional[str] = None):
    """Migrate entire codebase or specific file."""
    
    if specific_file:
        files = [specific_file]
    else:
        files = FILES_TO_MIGRATE
    
    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN MODE' if dry_run else 'APPLYING CHANGES'}")
    print(f"{'=' * 60}\n")
    
    total_replacements = 0
    files_changed = 0
    
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  Skipping (not found): {file_path}")
            continue
        
        # Analyze before migration
        usage = analyze_file(path)
        if not any(usage.values()):
            continue
        
        # Perform migration
        replacements, new_content = migrate_file(path, dry_run)
        
        if replacements:
            files_changed += 1
            total_replacements += len(replacements)
            
            print(f"📝 {file_path}")
            print(f"   Aliases found: {', '.join(k for k, v in usage.items() if v > 0)}")
            print(f"   Replacements: {len(replacements)}")
            
            # Show first few replacements
            for old, new in replacements[:3]:
                if len(old) > 50:
                    old = old[:47] + "..."
                if len(new) > 50:
                    new = new[:47] + "..."
                print(f"     {old} → {new}")
            
            if len(replacements) > 3:
                print(f"     ... and {len(replacements) - 3} more")
            print()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("MIGRATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files analyzed: {len(files)}")
    print(f"Files with changes: {files_changed}")
    print(f"Total replacements: {total_replacements}")
    
    if dry_run:
        print("\n✅ Dry run complete. No files were modified.")
        print("   Use --apply to make changes.")
    else:
        print("\n✅ Migration complete! Files have been updated.")
        print("\n⚠️  Next steps:")
        print("   1. Update leeq/epii/experiments.py manually:")
        print("      - Remove _add_backward_compatibility_aliases()")
        print("      - Remove parameter_map initialization")
        print("   2. Run tests: pytest tests/epii/ -v")
        print("   3. Commit changes: git add -u && git commit -m 'Remove experiment aliases'")


def validate_migration():
    """Validate that migration was successful."""
    print("\n" + "=" * 60)
    print("POST-MIGRATION VALIDATION")
    print("=" * 60)
    
    # Check for remaining aliases
    remaining = defaultdict(list)
    
    for file_path in FILES_TO_MIGRATE:
        path = Path(file_path)
        if not path.exists():
            continue
        
        usage = analyze_file(path)
        for alias, count in usage.items():
            if count > 0:
                remaining[file_path].append(alias)
    
    if remaining:
        print("\n❌ Aliases still found in:")
        for file, aliases in remaining.items():
            print(f"   {file}: {', '.join(aliases)}")
    else:
        print("\n✅ No aliases found - migration successful!")
    
    # Test import
    try:
        from leeq.epii.experiments import ExperimentRouter
        router = ExperimentRouter()
        
        # Check that aliases don't exist
        bad_aliases = []
        for alias in ALIAS_TO_CANONICAL.keys():
            if alias in router.experiment_map:
                bad_aliases.append(alias)
        
        if bad_aliases:
            print(f"\n❌ Aliases still in ExperimentRouter: {', '.join(bad_aliases)}")
            print("   Manual update needed in leeq/epii/experiments.py")
        else:
            print("\n✅ ExperimentRouter has no aliases")
        
        # Check canonical names exist
        missing = []
        for canonical in ALIAS_TO_CANONICAL.values():
            if canonical not in router.experiment_map:
                missing.append(canonical)
        
        if missing:
            print(f"\n⚠️  Missing canonical experiments: {', '.join(missing[:3])}")
        else:
            print("✅ All canonical experiments found")
            
    except ImportError as e:
        print(f"\n⚠️  Cannot validate ExperimentRouter: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate experiment aliases to canonical names")
    parser.add_argument('--apply', action='store_true', help='Apply changes (default is dry run)')
    parser.add_argument('--file', type=str, help='Migrate specific file only')
    parser.add_argument('--validate', action='store_true', help='Validate migration')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_migration()
    else:
        migrate_codebase(dry_run=not args.apply, specific_file=args.file)


if __name__ == "__main__":
    main()