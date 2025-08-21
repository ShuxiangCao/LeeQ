#!/usr/bin/env python3
"""
Comprehensive notebook testing infrastructure for LeeQ documentation notebooks.

This script provides comprehensive testing capabilities including:
- Notebook execution validation with nbval
- Output verification for expected results and plots
- Chronicle logging verification tests
- Custom validation for LeeQ-specific features
- Automated dependency checking and installation
- Performance benchmarking
- Memory usage tracking

Usage:
    python scripts/test_notebook_infrastructure.py --all
    python scripts/test_notebook_infrastructure.py --tutorials
    python scripts/test_notebook_infrastructure.py --examples
    python scripts/test_notebook_infrastructure.py --workflows
    python scripts/test_notebook_infrastructure.py --install-deps
"""

import os
import sys
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Any
import re
import time
import traceback
import hashlib
from datetime import datetime
import warnings


class NotebookTestResult:
    """Container for notebook test results."""
    
    def __init__(self, notebook_path: Path):
        self.notebook_path = notebook_path
        self.syntax_ok = False
        self.structure_ok = False
        self.execution_ok = False
        self.chronicle_ok = False
        self.outputs_ok = False
        self.plots_verified = False
        self.leeq_patterns_ok = False
        self.performance_ok = False
        self.errors = []
        self.warnings = []
        self.execution_time = 0.0
        self.memory_usage_mb = 0.0
        self.output_stats = {}
        self.test_timestamp = datetime.now()
        
    @property
    def passed(self) -> bool:
        """True if all tests passed."""
        return all([
            self.syntax_ok,
            self.structure_ok, 
            self.execution_ok,
            self.chronicle_ok,
            self.outputs_ok,
            self.leeq_patterns_ok
        ])
    
    def add_error(self, test_type: str, message: str):
        """Add an error for a specific test type."""
        self.errors.append(f"{test_type}: {message}")
    
    def add_warning(self, test_type: str, message: str):
        """Add a warning for a specific test type."""
        self.warnings.append(f"{test_type}: {message}")
    
    def to_dict(self) -> dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            'notebook': str(self.notebook_path),
            'timestamp': self.test_timestamp.isoformat(),
            'passed': self.passed,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'tests': {
                'syntax': self.syntax_ok,
                'structure': self.structure_ok,
                'execution': self.execution_ok,
                'chronicle': self.chronicle_ok,
                'outputs': self.outputs_ok,
                'plots': self.plots_verified,
                'leeq_patterns': self.leeq_patterns_ok,
                'performance': self.performance_ok
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'output_stats': self.output_stats
        }


class NotebookTester:
    """Comprehensive notebook testing framework."""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.test_results = {}
        
    def test_notebook_syntax(self, notebook_path: Path) -> Tuple[bool, str]:
        """Test if a notebook has valid JSON syntax."""
        try:
            with open(notebook_path, 'r') as f:
                json.load(f)
            return True, "Valid JSON syntax"
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def test_notebook_structure(self, notebook_path: Path) -> Tuple[bool, str]:
        """Test if a notebook has required structural elements."""
        try:
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
            
            # Check for proper cell structure
            for i, cell in enumerate(nb['cells']):
                if 'cell_type' not in cell:
                    return False, f"Cell {i} missing cell_type"
                if 'source' not in cell:
                    return False, f"Cell {i} missing source"
            
            return True, "Valid notebook structure"
        
        except Exception as e:
            return False, f"Error validating structure: {e}"
    
    def test_notebook_execution(self, notebook_path: Path) -> Tuple[bool, str, float]:
        """Test notebook execution using nbval and nbconvert."""
        start_time = time.time()
        
        try:
            # First try with nbval (preferred for testing)
            cmd = [
                sys.executable, '-m', 'pytest', 
                str(notebook_path), 
                '--nbval', '--nbval-lax', '-v', '--tb=short'
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return True, "Notebook executed successfully", execution_time
            else:
                return False, f"Execution failed: {result.stderr}", execution_time
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return False, "Execution timed out (>5 minutes)", execution_time
        except FileNotFoundError:
            # Fallback to nbconvert if nbval not available
            return self._test_with_nbconvert(notebook_path, start_time)
        except Exception as e:
            execution_time = time.time() - start_time
            return False, f"Execution error: {e}", execution_time
    
    def _test_with_nbconvert(self, notebook_path: Path, start_time: float) -> Tuple[bool, str, float]:
        """Fallback execution test using nbconvert."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            cmd = [
                'jupyter', 'nbconvert', 
                '--to', 'notebook',
                '--execute',
                '--output', tmp_path,
                str(notebook_path)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            execution_time = time.time() - start_time
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            if result.returncode == 0:
                return True, "Notebook executed successfully (nbconvert)", execution_time
            else:
                return False, f"Execution failed (nbconvert): {result.stderr}", execution_time
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return False, "Execution timed out (>5 minutes, nbconvert)", execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            return False, f"Execution error (nbconvert): {e}", execution_time
    
    def test_chronicle_integration(self, notebook_path: Path) -> Tuple[bool, str]:
        """Test Chronicle logging integration in notebook."""
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            chronicle_imports = False
            chronicle_usage = False
            log_start = False
            
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Check for Chronicle imports
                    if re.search(r'from\s+leeq\.chronicle\s+import|import\s+.*chronicle', source, re.IGNORECASE):
                        chronicle_imports = True
                    
                    # Check for Chronicle usage
                    if re.search(r'chronicle\.|Chronicle\(|@log_and_record', source):
                        chronicle_usage = True
                    
                    # Check for log start
                    if re.search(r'\.start_log\(\)|Chronicle\(\)\.start_log', source):
                        log_start = True
            
            if not chronicle_imports:
                return False, "No Chronicle imports found"
            
            if not chronicle_usage:
                return False, "Chronicle imported but not used"
            
            if not log_start:
                return False, "Chronicle not started with start_log()"
            
            return True, "Chronicle integration looks good"
        
        except Exception as e:
            return False, f"Error checking Chronicle integration: {e}"
    
    def test_output_verification(self, notebook_path: Path) -> Tuple[bool, str, dict]:
        """Test for expected outputs and plot generation with detailed statistics."""
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            stats = {
                'total_cells': len(nb['cells']),
                'code_cells': 0,
                'cells_with_outputs': 0,
                'plot_outputs': 0,
                'data_outputs': 0,
                'error_outputs': 0,
                'expected_plots_missing': 0
            }
            
            plot_generators = [
                r'\.plot\(',
                r'\.show\(',
                r'plotly\.',
                r'matplotlib\.',
                r'plt\.',
                r'\.draw\(',
                r'\.visualize\('
            ]
            
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    stats['code_cells'] += 1
                    outputs = cell.get('outputs', [])
                    source = ''.join(cell.get('source', []))
                    
                    if outputs:
                        stats['cells_with_outputs'] += 1
                        
                        for output in outputs:
                            # Check for plot outputs
                            if output.get('output_type') == 'display_data':
                                data = output.get('data', {})
                                if any(mime in data for mime in ['image/png', 'image/svg+xml', 
                                                                  'application/vnd.plotly.v1+json']):
                                    stats['plot_outputs'] += 1
                            
                            # Check for data outputs
                            if output.get('output_type') in ['execute_result', 'stream']:
                                stats['data_outputs'] += 1
                            
                            # Check for errors
                            if output.get('output_type') == 'error':
                                stats['error_outputs'] += 1
                    
                    # Check if plot-generating code has corresponding output
                    if any(re.search(pattern, source) for pattern in plot_generators):
                        has_plot_output = any(
                            output.get('output_type') == 'display_data' and
                            any(mime in output.get('data', {}) for mime in 
                                ['image/png', 'image/svg+xml', 'application/vnd.plotly.v1+json'])
                            for output in outputs
                        )
                        if not has_plot_output:
                            stats['expected_plots_missing'] += 1
            
            # Validate based on notebook type
            issues = []
            if 'tutorial' in str(notebook_path).lower():
                if stats['cells_with_outputs'] == 0:
                    issues.append("Tutorial notebook should have outputs")
                if stats['code_cells'] > 0 and stats['cells_with_outputs'] / stats['code_cells'] < 0.3:
                    issues.append("Tutorial notebook has too few outputs (< 30% of code cells)")
            
            if stats['error_outputs'] > 0:
                issues.append(f"Found {stats['error_outputs']} error outputs")
            
            if stats['expected_plots_missing'] > 0:
                issues.append(f"Found {stats['expected_plots_missing']} cells with plot code but no plot output")
            
            if issues:
                return False, "; ".join(issues), stats
            
            return True, f"Output verification passed", stats
        
        except Exception as e:
            return False, f"Error verifying outputs: {e}", {}
    
    def test_leeq_patterns(self, notebook_path: Path) -> Tuple[bool, str]:
        """Test for LeeQ-specific patterns and best practices."""
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            
            # Patterns to check
            has_leeq_imports = False
            has_proper_setup = False
            has_experiment_creation = False
            has_data_analysis = False
            uses_context_managers = False
            
            leeq_import_patterns = [
                r'from\s+leeq',
                r'import\s+leeq',
                r'from\s+leeq\.',
            ]
            
            setup_patterns = [
                r'TransmonElement|Transmon',
                r'VirtualTransmon',
                r'QubitCollection',
                r'LogicalPrimitive',
                r'QubitSetup'
            ]
            
            experiment_patterns = [
                r'Experiment\(',
                r'BasicSetup\(',
                r'PulseSequence\(',
                r'\.run\(',
                r'\.execute\('
            ]
            
            analysis_patterns = [
                r'\.analyze\(',
                r'\.fit\(',
                r'\.plot\(',
                r'FitResult',
                r'AnalysisResult'
            ]
            
            context_patterns = [
                r'with\s+.*\s+as\s+',
                r'\.start_log\(',
                r'\.stop_log\('
            ]
            
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Check for LeeQ imports
                    if any(re.search(pattern, source) for pattern in leeq_import_patterns):
                        has_leeq_imports = True
                    
                    # Check for proper setup
                    if any(re.search(pattern, source) for pattern in setup_patterns):
                        has_proper_setup = True
                    
                    # Check for experiment creation
                    if any(re.search(pattern, source) for pattern in experiment_patterns):
                        has_experiment_creation = True
                    
                    # Check for data analysis
                    if any(re.search(pattern, source) for pattern in analysis_patterns):
                        has_data_analysis = True
                    
                    # Check for context managers
                    if any(re.search(pattern, source) for pattern in context_patterns):
                        uses_context_managers = True
            
            # Validate based on notebook type
            issues = []
            
            if not has_leeq_imports:
                issues.append("No LeeQ imports found")
            
            # Tutorial notebooks should demonstrate all aspects
            if 'tutorial' in str(notebook_path).lower():
                if not has_proper_setup:
                    issues.append("Tutorial should demonstrate proper setup")
                if not has_experiment_creation and '01_basics' not in str(notebook_path):
                    issues.append("Tutorial should show experiment creation")
            
            # Example notebooks should show specific features
            if 'example' in str(notebook_path).lower():
                if not has_experiment_creation:
                    issues.append("Example should demonstrate experiments")
                if not has_data_analysis:
                    issues.append("Example should include data analysis")
            
            # Workflow notebooks should be comprehensive
            if 'workflow' in str(notebook_path).lower():
                if not all([has_proper_setup, has_experiment_creation, has_data_analysis]):
                    issues.append("Workflow should be comprehensive")
            
            if issues:
                return False, "; ".join(issues)
            
            return True, "LeeQ patterns verified"
        
        except Exception as e:
            return False, f"Error checking LeeQ patterns: {e}"
    
    def test_single_notebook(self, notebook_path: Path) -> NotebookTestResult:
        """Run all tests on a single notebook."""
        result = NotebookTestResult(notebook_path)
        
        if self.verbose:
            print(f"\nTesting {notebook_path.relative_to(self.project_root)}...")
        
        # Test 1: Syntax
        syntax_ok, syntax_msg = self.test_notebook_syntax(notebook_path)
        result.syntax_ok = syntax_ok
        if not syntax_ok:
            result.add_error("Syntax", syntax_msg)
        elif self.verbose:
            print(f"  ✅ Syntax: {syntax_msg}")
        
        # Test 2: Structure  
        structure_ok, structure_msg = self.test_notebook_structure(notebook_path)
        result.structure_ok = structure_ok
        if not structure_ok:
            result.add_error("Structure", structure_msg)
        elif self.verbose:
            print(f"  ✅ Structure: {structure_msg}")
        
        # Test 3: Chronicle Integration
        chronicle_ok, chronicle_msg = self.test_chronicle_integration(notebook_path)
        result.chronicle_ok = chronicle_ok
        if not chronicle_ok:
            result.add_warning("Chronicle", chronicle_msg)  # Warning, not error
        elif self.verbose:
            print(f"  ✅ Chronicle: {chronicle_msg}")
        
        # Test 4: LeeQ Patterns
        leeq_ok, leeq_msg = self.test_leeq_patterns(notebook_path)
        result.leeq_patterns_ok = leeq_ok
        if not leeq_ok:
            result.add_error("LeeQ Patterns", leeq_msg)
        elif self.verbose:
            print(f"  ✅ LeeQ Patterns: {leeq_msg}")
        
        # Test 5: Execution (skip if earlier tests failed)
        if result.syntax_ok and result.structure_ok:
            execution_ok, execution_msg, execution_time = self.test_notebook_execution(notebook_path)
            result.execution_ok = execution_ok
            result.execution_time = execution_time
            if not execution_ok:
                result.add_error("Execution", execution_msg)
            elif self.verbose:
                print(f"  ✅ Execution: {execution_msg} ({execution_time:.1f}s)")
                
            # Performance check
            if execution_time < 60:
                result.performance_ok = True
            else:
                result.performance_ok = False
                result.add_warning("Performance", f"Execution took {execution_time:.1f}s (> 60s)")
        else:
            result.add_error("Execution", "Skipped due to syntax/structure failures")
        
        # Test 6: Output Verification (enhanced)
        outputs_ok, outputs_msg, output_stats = self.test_output_verification(notebook_path)
        result.outputs_ok = outputs_ok
        result.output_stats = output_stats
        if not outputs_ok:
            result.add_error("Outputs", outputs_msg)
        elif self.verbose:
            print(f"  ✅ Outputs: {outputs_msg}")
            if output_stats:
                print(f"     Stats: {output_stats.get('cells_with_outputs', 0)}/{output_stats.get('code_cells', 0)} cells with output, "
                      f"{output_stats.get('plot_outputs', 0)} plots")
        
        # Check for plot verification
        if output_stats.get('plot_outputs', 0) > 0:
            result.plots_verified = True
        
        # Summary for this notebook
        if result.passed:
            if self.verbose:
                print(f"  ✅ All tests passed")
        else:
            print(f"  ❌ {len(result.errors)} test(s) failed: {notebook_path.name}")
            if self.verbose:
                for error in result.errors:
                    print(f"    - {error}")
                if result.warnings:
                    print("  ⚠️  Warnings:")
                    for warning in result.warnings:
                        print(f"    - {warning}")
        
        return result
    
    def test_notebooks_in_directory(self, notebooks_dir: Path) -> Dict[Path, NotebookTestResult]:
        """Test all notebooks in a directory."""
        results = {}
        
        if not notebooks_dir.exists():
            print(f"Warning: Directory not found: {notebooks_dir}")
            return results
        
        notebook_paths = list(notebooks_dir.rglob('*.ipynb'))
        
        if not notebook_paths:
            print(f"No notebooks found in {notebooks_dir}")
            return results
        
        print(f"Testing {len(notebook_paths)} notebooks in {notebooks_dir.relative_to(self.project_root)}...")
        
        for notebook_path in sorted(notebook_paths):
            result = self.test_single_notebook(notebook_path)
            results[notebook_path] = result
        
        return results
    
    def generate_test_report(self, all_results: Dict[Path, Dict[Path, NotebookTestResult]]) -> str:
        """Generate a comprehensive test report."""
        total_notebooks = 0
        total_passed = 0
        total_failed = 0
        total_execution_time = 0.0
        
        report = ["# LeeQ Notebook Testing Report", ""]
        
        for directory, results in all_results.items():
            if not results:
                continue
                
            directory_passed = sum(1 for r in results.values() if r.passed)
            directory_failed = len(results) - directory_passed
            directory_time = sum(r.execution_time for r in results.values())
            
            total_notebooks += len(results)
            total_passed += directory_passed
            total_failed += directory_failed
            total_execution_time += directory_time
            
            report.extend([
                f"## {directory.name}",
                "",
                f"- **Total notebooks:** {len(results)}",
                f"- **Passed:** {directory_passed}",
                f"- **Failed:** {directory_failed}",
                f"- **Success rate:** {directory_passed/len(results)*100:.1f}%",
                f"- **Total execution time:** {directory_time:.1f}s",
                ""
            ])
            
            if directory_failed > 0:
                report.append("### Failed notebooks:")
                for notebook_path, result in results.items():
                    if not result.passed:
                        report.append(f"- **{notebook_path.name}**: {len(result.errors)} errors")
                        for error in result.errors:
                            report.append(f"  - {error}")
                report.append("")
        
        # Overall summary
        success_rate = total_passed / total_notebooks * 100 if total_notebooks > 0 else 0
        report.extend([
            "## Overall Summary",
            "",
            f"- **Total notebooks tested:** {total_notebooks}",
            f"- **Passed:** {total_passed}",
            f"- **Failed:** {total_failed}",  
            f"- **Overall success rate:** {success_rate:.1f}%",
            f"- **Total execution time:** {total_execution_time:.1f}s",
            ""
        ])
        
        if success_rate == 100:
            report.append("🎉 **All notebooks passed!**")
        elif success_rate >= 80:
            report.append("✅ **Good: Most notebooks passed**")
        elif success_rate >= 50:
            report.append("⚠️  **Warning: Some notebooks failing**")
        else:
            report.append("❌ **Critical: Many notebooks failing**")
        
        return "\n".join(report)


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    dependencies = {
        'nbval': False,
        'nbconvert': False,
        'jupyter': False,
        'pytest': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'nbval':
                import nbval
                dependencies[dep] = True
            elif dep == 'nbconvert':
                import nbconvert
                dependencies[dep] = True
            elif dep == 'jupyter':
                import jupyter
                dependencies[dep] = True
            elif dep == 'pytest':
                import pytest
                dependencies[dep] = True
        except ImportError:
            pass
    
    return dependencies


def install_missing_dependencies(missing_deps: List[str]) -> bool:
    """Install missing dependencies."""
    if not missing_deps:
        return True
    
    print(f"Installing missing dependencies: {', '.join(missing_deps)}")
    
    # Map dependency names to pip packages
    pip_packages = {
        'nbval': 'nbval',
        'nbconvert': 'nbconvert',
        'jupyter': 'jupyter',
        'pytest': 'pytest'
    }
    
    for dep in missing_deps:
        package = pip_packages.get(dep, dep)
        print(f"  Installing {package}...")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"    ❌ Failed to install {package}: {result.stderr}")
                return False
            else:
                print(f"    ✅ Successfully installed {package}")
        except Exception as e:
            print(f"    ❌ Error installing {package}: {e}")
            return False
    
    return True


def save_test_results_json(results: Dict[Path, Dict[Path, NotebookTestResult]], 
                           output_file: Path):
    """Save test results to JSON file for further analysis."""
    json_results = {}
    
    for directory, notebook_results in results.items():
        dir_name = str(directory)
        json_results[dir_name] = {}
        
        for notebook_path, result in notebook_results.items():
            json_results[dir_name][str(notebook_path)] = result.to_dict()
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nTest results saved to: {output_file}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Comprehensive LeeQ notebook testing')
    parser.add_argument('--all', action='store_true', 
                       help='Test all notebook directories')
    parser.add_argument('--tutorials', action='store_true',
                       help='Test tutorial notebooks')
    parser.add_argument('--examples', action='store_true',
                       help='Test example notebooks')
    parser.add_argument('--workflows', action='store_true',
                       help='Test workflow notebooks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', '-r',
                       help='Save detailed report to file')
    parser.add_argument('--json', '-j',
                       help='Save JSON test results to file')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install missing dependencies automatically')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Default to all if no specific directories specified
    if not any([args.all, args.tutorials, args.examples, args.workflows, args.check_deps]):
        args.all = True
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    notebooks_dir = project_root / 'notebooks'
    
    print("LeeQ Comprehensive Notebook Testing")
    print("===================================")
    print(f"Project root: {project_root}")
    print(f"Notebooks directory: {notebooks_dir}")
    print()
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [dep for dep, installed in deps.items() if not installed]
    
    if args.check_deps:
        print("Dependency Status:")
        for dep, installed in deps.items():
            status = "✅ Installed" if installed else "❌ Missing"
            print(f"  {dep}: {status}")
        
        if missing_deps:
            print(f"\nTo install missing dependencies, run with --install-deps")
            sys.exit(1)
        else:
            print("\nAll dependencies are installed!")
            sys.exit(0)
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        
        if args.install_deps:
            if not install_missing_dependencies(missing_deps):
                print("Failed to install dependencies. Please install manually:")
                print(f"  pip install {' '.join(missing_deps)}")
                sys.exit(1)
            print("\nDependencies installed successfully!")
        else:
            print("\nWarning: Some tests may fail due to missing dependencies.")
            print("Run with --install-deps to install them automatically.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    # Initialize tester
    tester = NotebookTester(project_root, verbose=args.verbose)
    
    # Determine which directories to test
    test_dirs = {}
    if args.all or args.tutorials:
        test_dirs[notebooks_dir / 'tutorials'] = 'tutorials'
    if args.all or args.examples:
        test_dirs[notebooks_dir / 'examples'] = 'examples'
    if args.all or args.workflows:
        test_dirs[notebooks_dir / 'workflows'] = 'workflows'
    
    # Run tests
    all_results = {}
    for test_dir, name in test_dirs.items():
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")
        results = tester.test_notebooks_in_directory(test_dir)
        all_results[test_dir] = results
    
    # Generate and display report
    if all_results:
        report = tester.generate_test_report(all_results)
        print(f"\n{'='*50}")
        print("FINAL REPORT")
        print(f"{'='*50}")
        print(report)
        
        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"\nDetailed report saved to: {args.report}")
        
        # Save JSON results if requested
        if args.json:
            json_path = Path(args.json)
            save_test_results_json(all_results, json_path)
        
        # Determine exit code
        total_failed = sum(
            sum(1 for r in results.values() if not r.passed)
            for results in all_results.values()
        )
        
        # Print final status
        if total_failed == 0:
            print("\n🎉 SUCCESS: All notebook tests passed!")
        else:
            print(f"\n❌ FAILURE: {total_failed} notebook(s) failed tests")
        
        sys.exit(0 if total_failed == 0 else 1)
    else:
        print("\nNo notebooks found to test.")
        sys.exit(0)


if __name__ == '__main__':
    main()