#!/usr/bin/env python3
"""
Notebook output validation script for LeeQ quantum experiments.

This script validates that notebooks produce expected quantum experimental results,
including oscillation patterns, decay measurements, and parameter fits.

Usage:
    python scripts/check_outputs.py [notebook_files...]
    
    # Check specific quantum experiment patterns
    python scripts/check_outputs.py --check-oscillations notebooks/examples/rabi_experiments.ipynb
    python scripts/check_outputs.py --check-decay notebooks/examples/t1_t2_measurements.ipynb
    
    # Check all outputs comprehensively
    python scripts/check_outputs.py --comprehensive notebooks/tutorials/
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import re
import math


class OutputValidator:
    """Validates quantum experiment outputs in notebook results."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.validation_results = {}
        
    def check_rabi_oscillations(self, notebook_path: Path) -> Dict[str, Any]:
        """Check for Rabi oscillation patterns in notebook outputs."""
        results = {
            'found_oscillations': False,
            'oscillation_count': 0,
            'amplitude_variation': 0.0,
            'frequency_detected': False,
            'fit_quality': 'unknown'
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    outputs = cell.get('outputs', [])
                    
                    for output in outputs:
                        # Check for numerical data that might be oscillations
                        if output.get('output_type') == 'execute_result':
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                text = data['text/plain']
                                if isinstance(text, list):
                                    text = ''.join(text)
                                
                                # Look for array-like data that could be oscillations
                                if self._contains_oscillation_data(text):
                                    results['found_oscillations'] = True
                        
                        # Check for plot data
                        elif output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            if 'application/vnd.plotly.v1+json' in data:
                                plot_data = data['application/vnd.plotly.v1+json']
                                if self._analyze_plotly_oscillations(plot_data):
                                    results['found_oscillations'] = True
                                    results['oscillation_count'] = self._count_oscillations_in_plot(plot_data)
                        
                        # Check for text output mentioning fits or parameters
                        elif output.get('output_type') == 'stream':
                            text = output.get('text', '')
                            if isinstance(text, list):
                                text = ''.join(text)
                            
                            if any(term in text.lower() for term in ['rabi frequency', 'fit', 'amplitude', 'period']):
                                results['frequency_detected'] = True
                                
                                # Extract fit quality if mentioned
                                if 'r-squared' in text.lower() or 'r²' in text.lower():
                                    r_squared = self._extract_r_squared(text)
                                    if r_squared:
                                        results['fit_quality'] = 'good' if r_squared > 0.8 else 'poor'
        
        except Exception as e:
            if self.verbose:
                print(f"Error checking Rabi oscillations in {notebook_path}: {e}")
        
        return results
    
    def check_decay_measurements(self, notebook_path: Path) -> Dict[str, Any]:
        """Check for T1/T2 decay measurements in notebook outputs."""
        results = {
            'found_t1_decay': False,
            'found_t2_decay': False,
            'decay_fits': [],
            'time_constants': [],
            'fit_quality': 'unknown'
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    outputs = cell.get('outputs', [])
                    
                    # Check if this is T1 or T2 experiment
                    is_t1 = 'T1' in source or 't1' in source.lower()
                    is_t2 = 'T2' in source or 't2' in source.lower() or 'ramsey' in source.lower()
                    
                    for output in outputs:
                        # Check for exponential decay patterns
                        if output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            if 'application/vnd.plotly.v1+json' in data:
                                plot_data = data['application/vnd.plotly.v1+json']
                                if self._analyze_decay_pattern(plot_data):
                                    if is_t1:
                                        results['found_t1_decay'] = True
                                    if is_t2:
                                        results['found_t2_decay'] = True
                        
                        # Check for time constant outputs
                        elif output.get('output_type') == 'stream':
                            text = output.get('text', '')
                            if isinstance(text, list):
                                text = ''.join(text)
                            
                            # Look for time constants (T1, T2)
                            time_constants = self._extract_time_constants(text)
                            if time_constants:
                                results['time_constants'].extend(time_constants)
                                
                                # Check if values are reasonable (typically microseconds)
                                if any(10 < tc < 1000 for tc in time_constants):  # 10-1000 μs range
                                    if is_t1:
                                        results['found_t1_decay'] = True
                                    if is_t2:
                                        results['found_t2_decay'] = True
        
        except Exception as e:
            if self.verbose:
                print(f"Error checking decay measurements in {notebook_path}: {e}")
        
        return results
    
    def check_entanglement_states(self, notebook_path: Path) -> Dict[str, Any]:
        """Check for entanglement or multi-qubit state indicators."""
        results = {
            'found_bell_states': False,
            'found_two_qubit_gates': False,
            'found_entanglement_measures': False,
            'crosstalk_analyzed': False
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    outputs = cell.get('outputs', [])
                    
                    # Check source for entanglement indicators
                    if any(term in source.lower() for term in ['bell', 'entangl', 'cnot', 'cx']):
                        results['found_bell_states'] = True
                    
                    if any(term in source.lower() for term in ['two_qubit', 'two qubit', 'crosstalk']):
                        results['found_two_qubit_gates'] = True
                        if 'crosstalk' in source.lower():
                            results['crosstalk_analyzed'] = True
                    
                    # Check outputs for entanglement measures
                    for output in outputs:
                        if output.get('output_type') == 'stream':
                            text = output.get('text', '')
                            if isinstance(text, list):
                                text = ''.join(text)
                            
                            if any(term in text.lower() for term in ['fidelity', 'concurrence', 'entanglement']):
                                results['found_entanglement_measures'] = True
        
        except Exception as e:
            if self.verbose:
                print(f"Error checking entanglement states in {notebook_path}: {e}")
        
        return results
    
    def check_parameter_persistence(self, notebook_path: Path) -> Dict[str, Any]:
        """Check for parameter persistence and Chronicle logging."""
        results = {
            'chronicle_logging': False,
            'parameter_saving': False,
            'parameter_loading': False,
            'log_entries_found': False
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    outputs = cell.get('outputs', [])
                    
                    # Check for Chronicle usage
                    if 'Chronicle' in source or 'log_and_record' in source:
                        results['chronicle_logging'] = True
                    
                    # Check for parameter persistence patterns
                    if any(term in source.lower() for term in ['save', 'store', 'persist']):
                        results['parameter_saving'] = True
                    
                    if any(term in source.lower() for term in ['load', 'restore', 'retrieve']):
                        results['parameter_loading'] = True
                    
                    # Check outputs for log entries
                    for output in outputs:
                        if output.get('output_type') == 'stream':
                            text = output.get('text', '')
                            if isinstance(text, list):
                                text = ''.join(text)
                            
                            if any(term in text.lower() for term in ['logged', 'recorded', 'saved to chronicle']):
                                results['log_entries_found'] = True
        
        except Exception as e:
            if self.verbose:
                print(f"Error checking parameter persistence in {notebook_path}: {e}")
        
        return results
    
    def _contains_oscillation_data(self, text: str) -> bool:
        """Check if text contains array data that looks like oscillations."""
        # Look for array patterns
        array_patterns = [
            r'\[.*\]',  # Python list
            r'array\([^\)]*\)',  # numpy array
        ]
        
        for pattern in array_patterns:
            if re.search(pattern, text):
                # Extract numerical values
                numbers = re.findall(r'-?\d+\.?\d*', text)
                if len(numbers) > 10:  # Need enough points
                    values = [float(n) for n in numbers[:50]]  # Check first 50 values
                    return self._has_oscillation_pattern(values)
        
        return False
    
    def _has_oscillation_pattern(self, values: List[float]) -> bool:
        """Check if numerical values show oscillation pattern."""
        if len(values) < 10:
            return False
        
        # Simple oscillation detection: count sign changes
        sign_changes = 0
        for i in range(1, len(values)):
            if (values[i] > 0) != (values[i-1] > 0):
                sign_changes += 1
        
        # Should have multiple sign changes for oscillations
        return sign_changes > 4
    
    def _analyze_plotly_oscillations(self, plot_data: Any) -> bool:
        """Analyze Plotly data for oscillation patterns."""
        try:
            if isinstance(plot_data, str):
                plot_data = json.loads(plot_data)
            
            # Look for trace data
            if 'data' in plot_data:
                for trace in plot_data['data']:
                    if 'y' in trace:
                        y_values = trace['y']
                        if isinstance(y_values, list) and len(y_values) > 10:
                            return self._has_oscillation_pattern([float(y) for y in y_values])
        except:
            pass
        
        return False
    
    def _count_oscillations_in_plot(self, plot_data: Any) -> int:
        """Count the number of oscillation cycles in plot data."""
        try:
            if isinstance(plot_data, str):
                plot_data = json.loads(plot_data)
            
            if 'data' in plot_data:
                for trace in plot_data['data']:
                    if 'y' in trace:
                        y_values = [float(y) for y in trace['y']]
                        # Count zero crossings / 2 for full cycles
                        crossings = 0
                        for i in range(1, len(y_values)):
                            if (y_values[i] > 0) != (y_values[i-1] > 0):
                                crossings += 1
                        return crossings // 2
        except:
            pass
        
        return 0
    
    def _analyze_decay_pattern(self, plot_data: Any) -> bool:
        """Analyze plot data for exponential decay patterns."""
        try:
            if isinstance(plot_data, str):
                plot_data = json.loads(plot_data)
            
            if 'data' in plot_data:
                for trace in plot_data['data']:
                    if 'y' in trace and 'x' in trace:
                        y_values = [float(y) for y in trace['y']]
                        if len(y_values) > 5:
                            # Check if values generally decrease
                            first_half = np.mean(y_values[:len(y_values)//2])
                            second_half = np.mean(y_values[len(y_values)//2:])
                            return first_half > second_half * 1.2  # At least 20% decrease
        except:
            pass
        
        return False
    
    def _extract_time_constants(self, text: str) -> List[float]:
        """Extract time constant values from text output."""
        time_constants = []
        
        # Look for patterns like "T1 = 45.2 μs" or "T2 = 23.1 us"
        patterns = [
            r'T1\s*[=:]\s*(\d+\.?\d*)\s*[μu]?s',
            r'T2\s*[=:]\s*(\d+\.?\d*)\s*[μu]?s',
            r'time constant\s*[=:]\s*(\d+\.?\d*)\s*[μu]?s',
            r'decay time\s*[=:]\s*(\d+\.?\d*)\s*[μu]?s'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    time_constants.append(float(match))
                except:
                    continue
        
        return time_constants
    
    def _extract_r_squared(self, text: str) -> Optional[float]:
        """Extract R-squared value from text."""
        patterns = [
            r'r-squared\s*[=:]\s*(\d+\.?\d*)',
            r'r²\s*[=:]\s*(\d+\.?\d*)',
            r'R\^2\s*[=:]\s*(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        return None
    
    def validate_notebook_outputs(self, notebook_path: Path, checks: List[str]) -> Dict[str, Any]:
        """Run specified validation checks on a notebook."""
        results = {
            'notebook': str(notebook_path),
            'checks_requested': checks,
            'results': {}
        }
        
        if 'oscillations' in checks:
            results['results']['oscillations'] = self.check_rabi_oscillations(notebook_path)
        
        if 'decay' in checks:
            results['results']['decay'] = self.check_decay_measurements(notebook_path)
        
        if 'entanglement' in checks:
            results['results']['entanglement'] = self.check_entanglement_states(notebook_path)
        
        if 'persistence' in checks:
            results['results']['persistence'] = self.check_parameter_persistence(notebook_path)
        
        # Overall assessment
        passed_checks = 0
        total_checks = 0
        
        for check_type, check_results in results['results'].items():
            for key, value in check_results.items():
                if isinstance(value, bool):
                    total_checks += 1
                    if value:
                        passed_checks += 1
        
        results['pass_rate'] = passed_checks / max(total_checks, 1)
        results['passed'] = results['pass_rate'] > 0.5  # At least 50% of checks pass
        
        self.validation_results[str(notebook_path)] = results
        return results


def main():
    """Main output validation function."""
    parser = argparse.ArgumentParser(description='Validate notebook experimental outputs')
    parser.add_argument('notebooks', nargs='*',
                       help='Notebook files to validate')
    parser.add_argument('--check-oscillations', action='store_true',
                       help='Check for Rabi oscillation patterns')
    parser.add_argument('--check-decay', action='store_true',
                       help='Check for T1/T2 decay measurements')
    parser.add_argument('--check-entanglement', action='store_true',
                       help='Check for entanglement/multi-qubit indicators')
    parser.add_argument('--check-persistence', action='store_true',
                       help='Check for parameter persistence')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run all checks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.notebooks:
        print("No notebooks specified. Provide notebook paths.")
        sys.exit(1)
    
    # Determine which checks to run
    checks = []
    if args.check_oscillations or args.comprehensive:
        checks.append('oscillations')
    if args.check_decay or args.comprehensive:
        checks.append('decay')
    if args.check_entanglement or args.comprehensive:
        checks.append('entanglement')
    if args.check_persistence or args.comprehensive:
        checks.append('persistence')
    
    if not checks:
        checks = ['oscillations', 'decay', 'persistence']  # Default checks
    
    print(f"Notebook Output Validation")
    print(f"=========================")
    print(f"Checks: {', '.join(checks)}")
    print(f"Notebooks: {len(args.notebooks)}")
    print()
    
    validator = OutputValidator(verbose=args.verbose)
    all_passed = True
    
    for notebook_file in args.notebooks:
        notebook_path = Path(notebook_file)
        if not notebook_path.exists():
            print(f"❌ File not found: {notebook_path}")
            all_passed = False
            continue
        
        if not notebook_path.suffix == '.ipynb':
            print(f"⚠️  Skipping non-notebook file: {notebook_path}")
            continue
        
        print(f"Validating {notebook_path.name}...")
        results = validator.validate_notebook_outputs(notebook_path, checks)
        
        if results['passed']:
            print(f"  ✅ Passed ({results['pass_rate']:.1%} success rate)")
        else:
            print(f"  ❌ Failed ({results['pass_rate']:.1%} success rate)")
            all_passed = False
        
        if args.verbose:
            for check_type, check_results in results['results'].items():
                print(f"    {check_type}:")
                for key, value in check_results.items():
                    if isinstance(value, bool):
                        status = "✅" if value else "❌"
                        print(f"      {status} {key}")
                    elif isinstance(value, (int, float)) and value > 0:
                        print(f"      📊 {key}: {value}")
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All notebooks passed output validation!")
    else:
        print("❌ Some notebooks failed output validation")
    
    # Summary stats
    if validator.validation_results:
        avg_pass_rate = np.mean([r['pass_rate'] for r in validator.validation_results.values()])
        print(f"Average pass rate: {avg_pass_rate:.1%}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()