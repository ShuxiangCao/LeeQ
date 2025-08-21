#!/usr/bin/env python3
"""
Automated notebook test runner for LeeQ documentation notebooks.

This script orchestrates comprehensive testing of notebook suites, including
execution validation, output checking, and performance benchmarking.

Usage:
    python scripts/notebook_test_runner.py --all
    
    # Test specific categories
    python scripts/notebook_test_runner.py --tutorials --examples
    
    # Performance testing
    python scripts/notebook_test_runner.py --all --performance-check
    
    # Generate comprehensive report
    python scripts/notebook_test_runner.py --all --generate-report
"""

import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class NotebookTestRunner:
    """Orchestrates comprehensive testing of notebook suites."""
    
    def __init__(self, verbose: bool = False, parallel: bool = True, max_workers: int = 4):
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers
        self.test_results = {}
        self.performance_data = {}
        self.lock = threading.Lock()
        
        # Validation criteria thresholds
        self.thresholds = {
            'execution_success_rate': 0.90,  # 90% of notebooks must execute successfully
            'average_execution_time': 300,   # Average execution time under 5 minutes
            'max_execution_time': 600,       # No notebook should take longer than 10 minutes
            'output_validation_rate': 0.80,  # 80% of output checks should pass
            'static_validation_rate': 0.95   # 95% of static checks should pass
        }
    
    def discover_notebooks(self, project_root: Path, categories: List[str]) -> Dict[str, List[Path]]:
        """Discover notebooks by category."""
        notebooks = {}
        
        # Standard directory structure
        base_dirs = {
            'tutorials': project_root / 'docs' / 'notebooks' / 'tutorials',
            'examples': project_root / 'docs' / 'notebooks' / 'examples', 
            'workflows': project_root / 'docs' / 'notebooks' / 'workflows'
        }
        
        # Fallback to old structure if needed
        for category in base_dirs:
            if not base_dirs[category].exists():
                base_dirs[category] = project_root / 'notebooks' / category
        
        for category in categories:
            if category == 'all':
                # Include all categories
                for cat_name, cat_dir in base_dirs.items():
                    if cat_dir.exists():
                        notebooks[cat_name] = list(cat_dir.glob('*.ipynb'))
            elif category in base_dirs:
                cat_dir = base_dirs[category]
                if cat_dir.exists():
                    notebooks[category] = list(cat_dir.glob('*.ipynb'))
                else:
                    if self.verbose:
                        print(f"Warning: Directory not found: {cat_dir}")
                    notebooks[category] = []
        
        # Filter out checkpoint files
        for category in notebooks:
            notebooks[category] = [
                nb for nb in notebooks[category] 
                if '.ipynb_checkpoints' not in str(nb)
            ]
        
        return notebooks
    
    def run_static_validation(self, notebook_path: Path) -> Dict[str, any]:
        """Run static validation on a notebook."""
        try:
            cmd = [
                'python', 'scripts/validate_notebooks.py',
                '--verbose' if self.verbose else '',
                str(notebook_path)
            ]
            cmd = [c for c in cmd if c]  # Remove empty strings
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'validation_type': 'static'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'validation_type': 'static'
            }
    
    def run_execution_validation(self, notebook_path: Path, timeout: int = 300) -> Dict[str, any]:
        """Run execution validation on a notebook."""
        start_time = time.time()
        
        try:
            cmd = [
                'python', 'scripts/validate_notebooks.py',
                '--execute',
                '--timeout', str(timeout),
                '--verbose' if self.verbose else '',
                str(notebook_path)
            ]
            cmd = [c for c in cmd if c]  # Remove empty strings
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 30
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'validation_type': 'execution'
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Execution timeout after {timeout}s',
                'execution_time': timeout,
                'validation_type': 'execution'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'validation_type': 'execution'
            }
    
    def run_output_validation(self, notebook_path: Path) -> Dict[str, any]:
        """Run output validation on a notebook."""
        try:
            # Determine which checks to run based on notebook category/name
            checks = self._determine_output_checks(notebook_path)
            
            cmd = ['python', 'scripts/check_outputs.py'] + checks + [str(notebook_path)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                'success': result.returncode == 0,
                'checks_run': checks,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'validation_type': 'output'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'validation_type': 'output'
            }
    
    def _determine_output_checks(self, notebook_path: Path) -> List[str]:
        """Determine which output checks to run for a notebook."""
        notebook_name = notebook_path.name.lower()
        
        checks = []
        
        if any(term in notebook_name for term in ['rabi', 'amplitude', 'frequency']):
            checks.append('--check-oscillations')
        
        if any(term in notebook_name for term in ['t1', 't2', 'decay', 'coherence']):
            checks.append('--check-decay')
        
        if any(term in notebook_name for term in ['multi_qubit', 'entangle', 'bell', 'two_qubit']):
            checks.append('--check-entanglement')
        
        if any(term in notebook_name for term in ['calibration', 'workflow', 'daily']):
            checks.append('--check-persistence')
        
        # Default to comprehensive if no specific checks determined
        if not checks:
            checks = ['--comprehensive']
        
        return checks
    
    def test_notebook(self, notebook_path: Path, run_execution: bool = True, 
                     run_output_validation: bool = True, timeout: int = 300) -> Dict[str, any]:
        """Test a single notebook comprehensively."""
        test_start = time.time()
        
        results = {
            'notebook': str(notebook_path),
            'category': self._get_notebook_category(notebook_path),
            'tests_run': [],
            'overall_success': True,
            'test_time': 0
        }
        
        if self.verbose:
            print(f"  Testing {notebook_path.name}...")
        
        # 1. Static validation (always run)
        static_result = self.run_static_validation(notebook_path)
        results['static_validation'] = static_result
        results['tests_run'].append('static')
        
        if not static_result['success']:
            results['overall_success'] = False
            if self.verbose:
                print(f"    ❌ Static validation failed")
        elif self.verbose:
            print(f"    ✅ Static validation passed")
        
        # 2. Execution validation
        if run_execution:
            execution_result = self.run_execution_validation(notebook_path, timeout)
            results['execution_validation'] = execution_result
            results['tests_run'].append('execution')
            
            if not execution_result['success']:
                results['overall_success'] = False
                if self.verbose:
                    print(f"    ❌ Execution failed ({execution_result.get('execution_time', 0):.1f}s)")
            else:
                exec_time = execution_result.get('execution_time', 0)
                if self.verbose:
                    print(f"    ✅ Execution passed ({exec_time:.1f}s)")
                
                # Store performance data
                with self.lock:
                    if notebook_path not in self.performance_data:
                        self.performance_data[str(notebook_path)] = {}
                    self.performance_data[str(notebook_path)]['execution_time'] = exec_time
        
        # 3. Output validation (only if execution succeeded)
        if (run_output_validation and run_execution and 
            results.get('execution_validation', {}).get('success', False)):
            
            output_result = self.run_output_validation(notebook_path)
            results['output_validation'] = output_result
            results['tests_run'].append('output')
            
            if not output_result['success']:
                results['overall_success'] = False
                if self.verbose:
                    print(f"    ❌ Output validation failed")
            elif self.verbose:
                print(f"    ✅ Output validation passed")
        
        results['test_time'] = time.time() - test_start
        
        with self.lock:
            self.test_results[str(notebook_path)] = results
        
        return results
    
    def _get_notebook_category(self, notebook_path: Path) -> str:
        """Determine the category of a notebook from its path."""
        path_parts = notebook_path.parts
        
        for part in path_parts:
            if part in ['tutorials', 'examples', 'workflows']:
                return part
        
        return 'unknown'
    
    def run_test_suite(self, notebooks: Dict[str, List[Path]], 
                      run_execution: bool = True, run_output_validation: bool = True,
                      timeout: int = 300) -> Dict[str, any]:
        """Run comprehensive test suite on notebook collections."""
        
        suite_start = time.time()
        
        print(f"Notebook Test Suite")
        print(f"==================")
        print(f"Execution: {'Yes' if run_execution else 'No'}")
        print(f"Output validation: {'Yes' if run_output_validation else 'No'}")
        print(f"Timeout: {timeout}s")
        print(f"Parallel: {'Yes' if self.parallel else 'No'} (max workers: {self.max_workers})")
        print()
        
        # Count total notebooks
        total_notebooks = sum(len(nbs) for nbs in notebooks.values())
        print(f"Found {total_notebooks} notebooks across {len(notebooks)} categories")
        
        for category, notebook_list in notebooks.items():
            print(f"  {category}: {len(notebook_list)} notebooks")
        print()
        
        # Run tests
        if self.parallel and total_notebooks > 1:
            self._run_parallel_tests(notebooks, run_execution, run_output_validation, timeout)
        else:
            self._run_sequential_tests(notebooks, run_execution, run_output_validation, timeout)
        
        suite_time = time.time() - suite_start
        
        # Compile results
        suite_results = self._compile_suite_results(suite_time)
        
        return suite_results
    
    def _run_parallel_tests(self, notebooks: Dict[str, List[Path]], 
                          run_execution: bool, run_output_validation: bool, timeout: int):
        """Run tests in parallel."""
        print("Running tests in parallel...")
        
        all_notebooks = []
        for category, notebook_list in notebooks.items():
            all_notebooks.extend(notebook_list)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test jobs
            future_to_notebook = {
                executor.submit(
                    self.test_notebook, nb, run_execution, run_output_validation, timeout
                ): nb for nb in all_notebooks
            }
            
            # Process completed tests
            completed = 0
            for future in as_completed(future_to_notebook):
                notebook = future_to_notebook[future]
                try:
                    result = future.result()
                    completed += 1
                    if not self.verbose:
                        status = "✅" if result['overall_success'] else "❌"
                        print(f"  {status} {notebook.name} ({completed}/{len(all_notebooks)})")
                except Exception as e:
                    print(f"  ❌ {notebook.name} - Exception: {e}")
    
    def _run_sequential_tests(self, notebooks: Dict[str, List[Path]], 
                            run_execution: bool, run_output_validation: bool, timeout: int):
        """Run tests sequentially."""
        print("Running tests sequentially...")
        
        for category, notebook_list in notebooks.items():
            if notebook_list:
                print(f"\nTesting {category} notebooks:")
                for notebook in notebook_list:
                    result = self.test_notebook(notebook, run_execution, run_output_validation, timeout)
                    if not self.verbose:
                        status = "✅" if result['overall_success'] else "❌"
                        print(f"  {status} {notebook.name}")
    
    def _compile_suite_results(self, suite_time: float) -> Dict[str, any]:
        """Compile comprehensive suite results."""
        
        # Count successes and failures by test type
        static_passed = sum(1 for r in self.test_results.values() 
                          if r.get('static_validation', {}).get('success', False))
        static_total = sum(1 for r in self.test_results.values() 
                         if 'static_validation' in r)
        
        execution_passed = sum(1 for r in self.test_results.values() 
                             if r.get('execution_validation', {}).get('success', False))
        execution_total = sum(1 for r in self.test_results.values() 
                            if 'execution_validation' in r)
        
        output_passed = sum(1 for r in self.test_results.values() 
                          if r.get('output_validation', {}).get('success', False))
        output_total = sum(1 for r in self.test_results.values() 
                         if 'output_validation' in r)
        
        overall_passed = sum(1 for r in self.test_results.values() 
                           if r['overall_success'])
        overall_total = len(self.test_results)
        
        # Performance statistics
        execution_times = [
            data.get('execution_time', 0) 
            for data in self.performance_data.values()
        ]
        
        suite_results = {
            'suite_execution_time': suite_time,
            'total_notebooks': overall_total,
            'overall_success_rate': overall_passed / max(overall_total, 1),
            'static_validation': {
                'passed': static_passed,
                'total': static_total,
                'success_rate': static_passed / max(static_total, 1)
            },
            'execution_validation': {
                'passed': execution_passed,
                'total': execution_total,
                'success_rate': execution_passed / max(execution_total, 1)
            },
            'output_validation': {
                'passed': output_passed,
                'total': output_total,
                'success_rate': output_passed / max(output_total, 1)
            },
            'performance': {
                'average_execution_time': sum(execution_times) / max(len(execution_times), 1),
                'max_execution_time': max(execution_times) if execution_times else 0,
                'min_execution_time': min(execution_times) if execution_times else 0
            },
            'threshold_compliance': {},
            'failed_notebooks': [
                path for path, result in self.test_results.items()
                if not result['overall_success']
            ]
        }
        
        # Check threshold compliance
        thresholds = suite_results['threshold_compliance']
        
        thresholds['execution_success_rate'] = {
            'value': suite_results['execution_validation']['success_rate'],
            'threshold': self.thresholds['execution_success_rate'],
            'passed': suite_results['execution_validation']['success_rate'] >= self.thresholds['execution_success_rate']
        }
        
        thresholds['average_execution_time'] = {
            'value': suite_results['performance']['average_execution_time'],
            'threshold': self.thresholds['average_execution_time'],
            'passed': suite_results['performance']['average_execution_time'] <= self.thresholds['average_execution_time']
        }
        
        thresholds['max_execution_time'] = {
            'value': suite_results['performance']['max_execution_time'],
            'threshold': self.thresholds['max_execution_time'],
            'passed': suite_results['performance']['max_execution_time'] <= self.thresholds['max_execution_time']
        }
        
        thresholds['output_validation_rate'] = {
            'value': suite_results['output_validation']['success_rate'],
            'threshold': self.thresholds['output_validation_rate'],
            'passed': suite_results['output_validation']['success_rate'] >= self.thresholds['output_validation_rate']
        }
        
        thresholds['static_validation_rate'] = {
            'value': suite_results['static_validation']['success_rate'],
            'threshold': self.thresholds['static_validation_rate'],
            'passed': suite_results['static_validation']['success_rate'] >= self.thresholds['static_validation_rate']
        }
        
        return suite_results
    
    def print_summary(self, suite_results: Dict[str, any]):
        """Print comprehensive test summary."""
        print("\n" + "="*70)
        print("TEST SUITE RESULTS")
        print("="*70)
        
        # Overall results
        overall_rate = suite_results['overall_success_rate']
        total_notebooks = suite_results['total_notebooks']
        print(f"\n📊 OVERALL RESULTS:")
        print(f"  Total notebooks: {total_notebooks}")
        print(f"  Success rate: {overall_rate:.1%} ({int(overall_rate * total_notebooks)}/{total_notebooks})")
        print(f"  Suite execution time: {suite_results['suite_execution_time']:.1f}s")
        
        # Test type results
        print(f"\n📋 TEST TYPE RESULTS:")
        for test_type in ['static_validation', 'execution_validation', 'output_validation']:
            if test_type in suite_results and suite_results[test_type]['total'] > 0:
                data = suite_results[test_type]
                print(f"  {test_type.replace('_', ' ').title()}: {data['success_rate']:.1%} ({data['passed']}/{data['total']})")
        
        # Performance results
        if suite_results['performance']['average_execution_time'] > 0:
            print(f"\n⏱️  PERFORMANCE RESULTS:")
            perf = suite_results['performance']
            print(f"  Average execution time: {perf['average_execution_time']:.1f}s")
            print(f"  Max execution time: {perf['max_execution_time']:.1f}s")
            print(f"  Min execution time: {perf['min_execution_time']:.1f}s")
        
        # Threshold compliance
        print(f"\n✅ THRESHOLD COMPLIANCE:")
        all_thresholds_passed = True
        for threshold_name, threshold_data in suite_results['threshold_compliance'].items():
            status = "✅" if threshold_data['passed'] else "❌"
            print(f"  {status} {threshold_name.replace('_', ' ').title()}: "
                  f"{threshold_data['value']:.3g} (threshold: {threshold_data['threshold']:.3g})")
            if not threshold_data['passed']:
                all_thresholds_passed = False
        
        # Failed notebooks
        if suite_results['failed_notebooks']:
            print(f"\n❌ FAILED NOTEBOOKS ({len(suite_results['failed_notebooks'])}):")
            for notebook_path in suite_results['failed_notebooks'][:10]:  # Show up to 10
                notebook_name = Path(notebook_path).name
                result = self.test_results[notebook_path]
                failed_tests = []
                if not result.get('static_validation', {}).get('success', True):
                    failed_tests.append('static')
                if not result.get('execution_validation', {}).get('success', True):
                    failed_tests.append('execution')
                if not result.get('output_validation', {}).get('success', True):
                    failed_tests.append('output')
                print(f"  - {notebook_name}: {', '.join(failed_tests)} failed")
            
            if len(suite_results['failed_notebooks']) > 10:
                print(f"  ... and {len(suite_results['failed_notebooks']) - 10} more")
        
        # Final verdict
        print(f"\n" + "="*70)
        if overall_rate >= 0.9 and all_thresholds_passed:
            print("🎉 SUITE PASSED - All criteria met!")
        elif overall_rate >= 0.8:
            print("⚠️  SUITE PARTIALLY PASSED - Some issues detected")
        else:
            print("❌ SUITE FAILED - Significant issues found")
        print("="*70)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Automated notebook test runner')
    
    # Notebook selection
    parser.add_argument('--all', action='store_true',
                       help='Test all notebooks')
    parser.add_argument('--tutorials', action='store_true',
                       help='Test tutorial notebooks')
    parser.add_argument('--examples', action='store_true',
                       help='Test example notebooks')
    parser.add_argument('--workflows', action='store_true',
                       help='Test workflow notebooks')
    
    # Test options
    parser.add_argument('--static-only', action='store_true',
                       help='Run only static validation (no execution)')
    parser.add_argument('--no-output-validation', action='store_true',
                       help='Skip output validation checks')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Execution timeout in seconds (default: 300)')
    
    # Performance and reporting
    parser.add_argument('--performance-check', action='store_true',
                       help='Include performance benchmarking')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed JSON report')
    parser.add_argument('--sequential', action='store_true',
                       help='Run tests sequentially instead of parallel')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers (default: 4)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine categories to test
    categories = []
    if args.all:
        categories = ['all']
    else:
        if args.tutorials:
            categories.append('tutorials')
        if args.examples:
            categories.append('examples')
        if args.workflows:
            categories.append('workflows')
    
    if not categories:
        print("No test categories specified. Use --all, --tutorials, --examples, or --workflows")
        sys.exit(1)
    
    # Setup test runner
    runner = NotebookTestRunner(
        verbose=args.verbose,
        parallel=not args.sequential,
        max_workers=args.max_workers
    )
    
    # Discover notebooks
    project_root = Path(__file__).parent.parent
    notebooks = runner.discover_notebooks(project_root, categories)
    
    if not any(notebooks.values()):
        print("No notebooks found to test!")
        sys.exit(1)
    
    # Run test suite
    suite_results = runner.run_test_suite(
        notebooks,
        run_execution=not args.static_only,
        run_output_validation=not args.no_output_validation,
        timeout=args.timeout
    )
    
    # Print summary
    runner.print_summary(suite_results)
    
    # Generate report if requested
    if args.generate_report:
        report_path = Path('notebook_test_report.json')
        
        # Compile full report
        full_report = {
            'test_configuration': {
                'categories': categories,
                'static_only': args.static_only,
                'output_validation': not args.no_output_validation,
                'timeout': args.timeout,
                'parallel': not args.sequential,
                'max_workers': args.max_workers
            },
            'suite_results': suite_results,
            'individual_results': runner.test_results,
            'performance_data': runner.performance_data
        }
        
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n📄 Comprehensive report saved to: {report_path}")
    
    # Exit with appropriate code
    success_rate = suite_results['overall_success_rate']
    if success_rate >= 0.9:
        sys.exit(0)  # Success
    elif success_rate >= 0.8:
        sys.exit(2)  # Partial success
    else:
        sys.exit(1)  # Failure


if __name__ == '__main__':
    main()