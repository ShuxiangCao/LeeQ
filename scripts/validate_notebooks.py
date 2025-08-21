#!/usr/bin/env python3
"""
Comprehensive notebook validation script for LeeQ notebooks.

This script performs both static validation and execution testing of notebooks.
It supports validation criteria for quantum experiment outputs and provides
detailed pass/fail reporting.

Usage:
    python scripts/validate_notebooks.py [notebook_files...]
    
    # Validate all notebooks with execution
    python scripts/validate_notebooks.py --execute docs/notebooks/**/*.ipynb
    
    # Static validation only
    python scripts/validate_notebooks.py docs/notebooks/tutorials/01_basics.ipynb
    
    # Comprehensive validation with output checking
    python scripts/validate_notebooks.py --comprehensive --check-outputs docs/notebooks/
"""

import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import time


class NotebookValidator:
    """Comprehensive notebook validation with optional execution testing."""
    
    def __init__(self, verbose: bool = False, execute: bool = False, timeout: int = 300):
        self.verbose = verbose
        self.execute = execute
        self.timeout = timeout
        self.errors = []
        self.warnings = []
        self.execution_stats = {
            'total_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'execution_time': {}
        }
    
    def validate_json_syntax(self, notebook_path: Path) -> bool:
        """Check if notebook has valid JSON syntax."""
        try:
            with open(notebook_path, 'r') as f:
                json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {notebook_path}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {notebook_path}: {e}")
            return False
    
    def validate_structure(self, notebook_path: Path) -> bool:
        """Validate notebook structure."""
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            # Check required fields
            required = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
            for field in required:
                if field not in nb:
                    self.errors.append(f"Missing required field '{field}' in {notebook_path}")
                    return False
            
            # Check cells
            if not nb['cells']:
                self.warnings.append(f"Empty notebook: {notebook_path}")
                return True
            
            # First cell should be markdown with title
            first_cell = nb['cells'][0]
            if first_cell.get('cell_type') != 'markdown':
                self.warnings.append(f"First cell should be markdown with title: {notebook_path}")
            else:
                source = ''.join(first_cell.get('source', []))
                if not source.startswith('#'):
                    self.warnings.append(f"First cell should start with # title: {notebook_path}")
            
            return True
        
        except Exception as e:
            self.errors.append(f"Error validating structure of {notebook_path}: {e}")
            return False
    
    def check_imports(self, notebook_path: Path) -> Dict[str, bool]:
        """Check for required imports."""
        imports_found = {
            'leeq': False,
            'chronicle': False,
            'numpy': False,
            'matplotlib': False
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    if re.search(r'from\s+leeq|import\s+leeq', source):
                        imports_found['leeq'] = True
                    if re.search(r'chronicle|Chronicle', source):
                        imports_found['chronicle'] = True
                    if re.search(r'import\s+numpy|from\s+numpy', source):
                        imports_found['numpy'] = True
                    if re.search(r'import\s+matplotlib|from\s+matplotlib|import\s+pyplot', source):
                        imports_found['matplotlib'] = True
            
            # Check if it's a LeeQ notebook (should have LeeQ imports)
            if not imports_found['leeq']:
                self.warnings.append(f"No LeeQ imports found in {notebook_path}")
            
            return imports_found
        
        except Exception as e:
            self.errors.append(f"Error checking imports in {notebook_path}: {e}")
            return imports_found
    
    def check_outputs(self, notebook_path: Path) -> Dict[str, int]:
        """Check notebook outputs."""
        stats = {
            'total_cells': 0,
            'code_cells': 0,
            'cells_with_output': 0,
            'cells_with_errors': 0,
            'plot_cells': 0
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            stats['total_cells'] = len(nb['cells'])
            
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    stats['code_cells'] += 1
                    
                    outputs = cell.get('outputs', [])
                    if outputs:
                        stats['cells_with_output'] += 1
                        
                        # Check for errors
                        for output in outputs:
                            if output.get('output_type') == 'error':
                                stats['cells_with_errors'] += 1
                                break
                        
                        # Check for plots
                        for output in outputs:
                            if output.get('output_type') == 'display_data':
                                data = output.get('data', {})
                                if any(mime in data for mime in ['image/png', 'image/svg+xml', 
                                                                  'application/vnd.plotly.v1+json']):
                                    stats['plot_cells'] += 1
                                    break
            
            # Warnings based on stats
            if stats['cells_with_errors'] > 0:
                self.warnings.append(f"{notebook_path} has {stats['cells_with_errors']} cells with errors")
            
            if stats['code_cells'] > 0 and stats['cells_with_output'] == 0:
                self.warnings.append(f"{notebook_path} has no cell outputs (not executed?)")
            
            return stats
        
        except Exception as e:
            self.errors.append(f"Error checking outputs in {notebook_path}: {e}")
            return stats
    
    def check_best_practices(self, notebook_path: Path) -> List[str]:
        """Check for best practices."""
        issues = []
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            # Check for print statements (should use logging)
            print_count = 0
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    print_count += len(re.findall(r'\bprint\s*\(', source))
            
            if print_count > 5:
                issues.append(f"Too many print statements ({print_count}), consider using logging")
            
            # Check for hardcoded paths
            hardcoded_paths = []
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    # Look for absolute paths
                    paths = re.findall(r'["\']/(home|Users|var|tmp|opt)/[^"\']+["\']', source)
                    hardcoded_paths.extend(paths)
            
            if hardcoded_paths:
                issues.append(f"Hardcoded absolute paths found: {hardcoded_paths[:3]}")
            
            # Check for large cells
            for i, cell in enumerate(nb['cells']):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    lines = source.count('\n')
                    if lines > 50:
                        issues.append(f"Cell {i} is too long ({lines} lines), consider splitting")
            
            # Check for proper markdown structure
            has_sections = False
            for cell in nb['cells']:
                if cell.get('cell_type') == 'markdown':
                    source = ''.join(cell.get('source', []))
                    if re.search(r'^##\s+', source, re.MULTILINE):
                        has_sections = True
                        break
            
            if not has_sections and len(nb['cells']) > 10:
                issues.append("No section headers (##) found, consider adding structure")
            
            for issue in issues:
                self.warnings.append(f"{notebook_path}: {issue}")
            
            return issues
        
        except Exception as e:
            self.errors.append(f"Error checking best practices in {notebook_path}: {e}")
            return issues
    
    def execute_notebook(self, notebook_path: Path) -> bool:
        """Execute notebook and check for errors."""
        if not self.execute:
            return True
        
        start_time = time.time()
        
        try:
            # Create temporary copy for execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as tmp_file:
                with open(notebook_path, 'r') as original:
                    tmp_file.write(original.read())
                tmp_notebook = Path(tmp_file.name)
            
            if self.verbose:
                print(f"    Executing notebook (timeout: {self.timeout}s)...")
            
            # Execute notebook using nbconvert
            cmd = [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout={}'.format(self.timeout),
                '--ExecutePreprocessor.kernel_name=python3',
                str(tmp_notebook)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 30  # Extra timeout buffer
            )
            
            execution_time = time.time() - start_time
            self.execution_stats['execution_time'][str(notebook_path)] = execution_time
            self.execution_stats['total_executed'] += 1
            
            if result.returncode == 0:
                self.execution_stats['successful_executions'] += 1
                
                # Check executed notebook for cell errors
                execution_errors = self._check_execution_errors(tmp_notebook)
                
                if execution_errors:
                    self.errors.extend([f"{notebook_path}: {error}" for error in execution_errors])
                    return False
                
                if self.verbose:
                    print(f"    ✅ Executed successfully in {execution_time:.1f}s")
                return True
            else:
                self.execution_stats['failed_executions'] += 1
                error_msg = result.stderr.strip() or result.stdout.strip()
                self.errors.append(f"{notebook_path}: Execution failed - {error_msg}")
                if self.verbose:
                    print(f"    ❌ Execution failed in {execution_time:.1f}s")
                return False
        
        except subprocess.TimeoutExpired:
            self.execution_stats['failed_executions'] += 1
            self.errors.append(f"{notebook_path}: Execution timeout after {self.timeout}s")
            if self.verbose:
                print(f"    ❌ Execution timeout after {self.timeout}s")
            return False
        except Exception as e:
            self.execution_stats['failed_executions'] += 1
            self.errors.append(f"{notebook_path}: Execution error - {str(e)}")
            if self.verbose:
                print(f"    ❌ Execution error: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            try:
                tmp_notebook.unlink()
            except:
                pass
    
    def _check_execution_errors(self, notebook_path: Path) -> List[str]:
        """Check executed notebook for cell execution errors."""
        errors = []
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for i, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') == 'code':
                    outputs = cell.get('outputs', [])
                    for output in outputs:
                        if output.get('output_type') == 'error':
                            error_name = output.get('ename', 'Unknown')
                            error_value = output.get('evalue', 'Unknown error')
                            errors.append(f"Cell {i}: {error_name} - {error_value}")
        except Exception as e:
            errors.append(f"Failed to check execution errors: {str(e)}")
        
        return errors
    
    def check_quantum_experiment_outputs(self, notebook_path: Path) -> Dict[str, bool]:
        """Check for quantum experiment specific outputs."""
        checks = {
            'has_rabi_oscillations': False,
            'has_decay_measurements': False,
            'has_chronicle_logs': False,
            'has_visualizations': False,
            'has_parameter_fits': False
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    outputs = cell.get('outputs', [])
                    
                    # Check source code patterns
                    if 'Rabi' in source and ('amplitude' in source.lower() or 'frequency' in source.lower()):
                        checks['has_rabi_oscillations'] = True
                    
                    if any(pattern in source for pattern in ['T1', 'T2', 'decay', 'coherence']):
                        checks['has_decay_measurements'] = True
                    
                    if 'Chronicle' in source or 'log_and_record' in source:
                        checks['has_chronicle_logs'] = True
                    
                    # Check outputs
                    for output in outputs:
                        if output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            # Check for plots
                            if any(mime in data for mime in ['image/png', 'image/svg+xml', 
                                                           'application/vnd.plotly.v1+json']):
                                checks['has_visualizations'] = True
                        
                        elif output.get('output_type') == 'stream':
                            text = output.get('text', '')
                            if isinstance(text, list):
                                text = ''.join(text)
                            # Check for fitting results
                            if any(term in text.lower() for term in ['fit', 'optimize', 'parameter', 'coefficient']):
                                checks['has_parameter_fits'] = True
        
        except Exception as e:
            self.warnings.append(f"Failed to check quantum outputs in {notebook_path}: {str(e)}")
        
        return checks
    
    def validate_notebook(self, notebook_path: Path) -> bool:
        """Validate a single notebook."""
        if self.verbose:
            print(f"Validating {notebook_path}...")
        
        initial_error_count = len(self.errors)
        
        # Run all validation checks
        json_valid = self.validate_json_syntax(notebook_path)
        if not json_valid:
            return False
        
        structure_valid = self.validate_structure(notebook_path)
        if not structure_valid:
            return False
        
        imports = self.check_imports(notebook_path)
        outputs = self.check_outputs(notebook_path)
        issues = self.check_best_practices(notebook_path)
        
        # Execute notebook if requested
        execution_valid = self.execute_notebook(notebook_path)
        
        # Check quantum experiment outputs if execution was successful
        quantum_checks = {}
        if execution_valid and self.execute:
            quantum_checks = self.check_quantum_experiment_outputs(notebook_path)
        
        if self.verbose:
            print(f"  Imports: LeeQ={imports['leeq']}, Chronicle={imports['chronicle']}")
            print(f"  Outputs: {outputs['cells_with_output']}/{outputs['code_cells']} cells")
            if outputs['cells_with_errors'] > 0:
                print(f"  ⚠️  {outputs['cells_with_errors']} cells with errors")
            if issues:
                print(f"  ⚠️  {len(issues)} best practice issues")
            if quantum_checks:
                print(f"  Quantum experiments: {sum(quantum_checks.values())}/{len(quantum_checks)} checks passed")
                if not all(quantum_checks.values()):
                    failed_checks = [k for k, v in quantum_checks.items() if not v]
                    print(f"    Missing: {', '.join(failed_checks)}")
        
        # Return True if no new errors were added during this validation
        return len(self.errors) == initial_error_count
    
    def validate_notebooks(self, notebook_paths: List[Path]) -> bool:
        """Validate multiple notebooks."""
        all_valid = True
        
        for notebook_path in notebook_paths:
            if not notebook_path.exists():
                self.errors.append(f"File not found: {notebook_path}")
                all_valid = False
                continue
            
            if not notebook_path.suffix == '.ipynb':
                self.warnings.append(f"Not a notebook file: {notebook_path}")
                continue
            
            if not self.validate_notebook(notebook_path):
                all_valid = False
        
        return all_valid


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Comprehensive notebook validation')
    parser.add_argument('notebooks', nargs='*', 
                       help='Notebook files to validate')
    parser.add_argument('--all', action='store_true',
                       help='Validate all notebooks in docs/notebooks/')
    parser.add_argument('--execute', action='store_true',
                       help='Execute notebooks for validation')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive validation (includes execution)')
    parser.add_argument('--check-outputs', action='store_true',
                       help='Check quantum experiment outputs')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Execution timeout in seconds (default: 300)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed validation report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set execution mode
    execute = args.execute or args.comprehensive
    
    # Determine which notebooks to validate
    if args.all:
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / 'docs' / 'notebooks'
        if not notebooks_dir.exists():
            notebooks_dir = project_root / 'notebooks'  # Fallback
        notebook_paths = list(notebooks_dir.rglob('*.ipynb'))
    elif args.notebooks:
        notebook_paths = [Path(nb) for nb in args.notebooks]
    else:
        print("No notebooks specified. Use --all or provide notebook paths.")
        sys.exit(1)
    
    # Filter out checkpoint notebooks
    notebook_paths = [nb for nb in notebook_paths 
                     if '.ipynb_checkpoints' not in str(nb)]
    
    print(f"LeeQ Notebook Validation")
    print(f"=======================")
    if execute:
        print(f"Mode: Execution validation (timeout: {args.timeout}s)")
    else:
        print(f"Mode: Static validation only")
    print(f"Notebooks: {len(notebook_paths)} found")
    print()
    
    # Run validation
    validator = NotebookValidator(
        verbose=args.verbose, 
        execute=execute, 
        timeout=args.timeout
    )
    valid = validator.validate_notebooks(notebook_paths)
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    # Execution stats
    if execute and validator.execution_stats['total_executed'] > 0:
        stats = validator.execution_stats
        print(f"\n📊 EXECUTION STATISTICS:")
        print(f"  Total executed: {stats['total_executed']}")
        print(f"  Successful: {stats['successful_executions']}")
        print(f"  Failed: {stats['failed_executions']}")
        if stats['execution_time']:
            avg_time = sum(stats['execution_time'].values()) / len(stats['execution_time'])
            print(f"  Average time: {avg_time:.1f}s")
            max_time = max(stats['execution_time'].values())
            print(f"  Max time: {max_time:.1f}s")
    
    if validator.errors:
        print(f"\n❌ ERRORS ({len(validator.errors)}):")
        for error in validator.errors:
            print(f"  - {error}")
    
    if validator.warnings:
        print(f"\n⚠️  WARNINGS ({len(validator.warnings)}):")
        for warning in validator.warnings:
            print(f"  - {warning}")
    
    # Generate report if requested
    if args.generate_report:
        report_path = Path('notebook_validation_report.json')
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'execution' if execute else 'static',
            'total_notebooks': len(notebook_paths),
            'validation_passed': valid,
            'errors': validator.errors,
            'warnings': validator.warnings,
            'execution_stats': validator.execution_stats
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📋 Detailed report saved to: {report_path}")
    
    if valid:
        print("\n✅ All notebooks passed validation!")
        if execute:
            success_rate = (validator.execution_stats['successful_executions'] / 
                          max(validator.execution_stats['total_executed'], 1) * 100)
            print(f"   Execution success rate: {success_rate:.1f}%")
        sys.exit(0)
    else:
        print(f"\n❌ Validation failed with {len(validator.errors)} errors")
        sys.exit(1)


if __name__ == '__main__':
    main()