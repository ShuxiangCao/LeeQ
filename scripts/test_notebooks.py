#!/usr/bin/env python3
"""
Notebook testing infrastructure for LeeQ documentation notebooks.

This script provides basic testing capabilities for Jupyter notebooks
in the tutorials, examples, and workflows directories.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def check_notebook_syntax(notebook_path):
    """Check if a notebook has valid JSON syntax."""
    try:
        import json
        with open(notebook_path, 'r') as f:
            json.load(f)
        return True, "Valid JSON syntax"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_notebook_structure(notebook_path):
    """Check if a notebook has required structural elements."""
    try:
        import json
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        # Check for required fields
        required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        missing_fields = [field for field in required_fields if field not in nb]
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Check for at least one cell
        if not nb.get('cells', []):
            return False, "Notebook has no cells"
        
        # Check if first cell is markdown with title
        first_cell = nb['cells'][0]
        if first_cell.get('cell_type') != 'markdown':
            return False, "First cell should be markdown with title"
        
        return True, "Valid notebook structure"
    
    except Exception as e:
        return False, f"Error validating structure: {e}"


def run_notebook_tests(notebooks_dir):
    """Run tests on all notebooks in the specified directory."""
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    notebook_paths = list(Path(notebooks_dir).rglob('*.ipynb'))
    
    if not notebook_paths:
        print(f"No notebooks found in {notebooks_dir}")
        return results
    
    print(f"Testing {len(notebook_paths)} notebooks in {notebooks_dir}...")
    
    for notebook_path in notebook_paths:
        print(f"\nTesting {notebook_path.relative_to(notebooks_dir)}...")
        
        # Test syntax
        syntax_ok, syntax_msg = check_notebook_syntax(notebook_path)
        if not syntax_ok:
            results['failed'] += 1
            results['errors'].append(f"{notebook_path}: {syntax_msg}")
            print(f"  ❌ Syntax: {syntax_msg}")
            continue
        else:
            print(f"  ✅ Syntax: {syntax_msg}")
        
        # Test structure
        structure_ok, structure_msg = check_notebook_structure(notebook_path)
        if not structure_ok:
            results['failed'] += 1
            results['errors'].append(f"{notebook_path}: {structure_msg}")
            print(f"  ❌ Structure: {structure_msg}")
            continue
        else:
            print(f"  ✅ Structure: {structure_msg}")
        
        results['passed'] += 1
        print(f"  ✅ All tests passed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test LeeQ documentation notebooks')
    parser.add_argument('--notebooks-dir', '-d', 
                       default='notebooks',
                       help='Directory containing notebooks to test')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    notebooks_dir = project_root / args.notebooks_dir
    
    if not notebooks_dir.exists():
        print(f"Error: Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)
    
    print(f"LeeQ Notebook Testing")
    print(f"====================")
    print(f"Notebooks directory: {notebooks_dir}")
    
    # Run tests
    results = run_notebook_tests(notebooks_dir)
    
    # Summary
    print(f"\n\nTest Summary")
    print(f"============")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Total:  {results['passed'] + results['failed']}")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Exit with appropriate code
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == '__main__':
    main()