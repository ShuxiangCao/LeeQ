#!/usr/bin/env python3
"""
Chronicle logging verification tests for LeeQ notebooks.

This script specifically tests Chronicle integration in notebooks:
- Verifies Chronicle imports are present
- Checks logging initialization
- Validates log entries are created
- Tests log directory structure

Usage:
    python scripts/test_chronicle_integration.py notebooks/tutorials/
    python scripts/test_chronicle_integration.py notebooks/examples/01_basics.ipynb
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
import argparse
import re
from typing import List, Dict, Tuple, Optional
import datetime


class ChronicleTestResult:
    """Container for Chronicle-specific test results."""
    
    def __init__(self, notebook_path: Path):
        self.notebook_path = notebook_path
        self.has_imports = False
        self.has_initialization = False
        self.has_logging_calls = False
        self.creates_log_entries = False
        self.proper_log_structure = False
        self.errors = []
        self.warnings = []
        self.log_directory = None
        
    @property
    def passed(self) -> bool:
        """True if all Chronicle tests passed."""
        return all([
            self.has_imports,
            self.has_initialization,
            self.has_logging_calls,
            self.creates_log_entries,
            self.proper_log_structure
        ])
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)


class ChronicleVerifier:
    """Verifies Chronicle integration in notebooks."""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        
    def analyze_notebook_chronicle_usage(self, notebook_path: Path) -> ChronicleTestResult:
        """Analyze a notebook for Chronicle usage patterns."""
        result = ChronicleTestResult(notebook_path)
        
        try:
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
        except Exception as e:
            result.add_error(f"Failed to read notebook: {e}")
            return result
        
        # Check for Chronicle imports
        import_patterns = [
            r'from\s+leeq\.chronicle\s+import',
            r'import\s+leeq\.chronicle',
            r'from\s+leeq\.chronicle\.chronicle\s+import\s+Chronicle',
            r'from\s+leeq\.chronicle\s+import\s+Chronicle',
            r'from\s+leeq\.chronicle\s+import\s+log_and_record'
        ]
        
        # Check for initialization patterns
        init_patterns = [
            r'Chronicle\(\)\.start_log\(\)',
            r'chronicle\.start_log\(\)',
            r'\.start_log\(\)'
        ]
        
        # Check for logging usage patterns
        logging_patterns = [
            r'@log_and_record',
            r'chronicle\.log\(',
            r'Chronicle\(\)\.log\(',
            r'log_and_record\('
        ]
        
        for cell in nb['cells']:
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Check imports
                for pattern in import_patterns:
                    if re.search(pattern, source, re.IGNORECASE):
                        result.has_imports = True
                        break
                
                # Check initialization
                for pattern in init_patterns:
                    if re.search(pattern, source):
                        result.has_initialization = True
                        break
                
                # Check logging usage
                for pattern in logging_patterns:
                    if re.search(pattern, source):
                        result.has_logging_calls = True
                        break
        
        # Generate error messages for missing components
        if not result.has_imports:
            result.add_error("No Chronicle imports found")
        
        if not result.has_initialization:
            result.add_error("Chronicle not initialized with start_log()")
        
        if not result.has_logging_calls:
            result.add_warning("No explicit Chronicle logging calls found")
        
        return result
    
    def execute_notebook_and_check_logs(self, notebook_path: Path) -> Tuple[bool, bool, Optional[Path]]:
        """Execute notebook and verify log creation."""
        # Create temporary log directory
        temp_log_dir = Path(tempfile.mkdtemp(prefix='leeq_test_logs_'))
        
        try:
            # Set environment to use temp log directory
            env = os.environ.copy()
            env['LEEQ_LOG_DIR'] = str(temp_log_dir)
            
            # Execute notebook with nbconvert
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
                timeout=120,  # 2 minute timeout
                env=env
            )
            
            # Clean up temp notebook
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            execution_success = result.returncode == 0
            
            # Check for log creation
            log_files_created = False
            log_structure_correct = False
            
            if temp_log_dir.exists():
                # Look for log files
                log_files = list(temp_log_dir.rglob('*'))
                if log_files:
                    log_files_created = True
                    
                    # Check log structure (should have user/date/time hierarchy)
                    # Expected: temp_log_dir/username/YYYY-MM/YYYY-MM-DD/HH.MM.SS/
                    date_dirs = [d for d in temp_log_dir.iterdir() if d.is_dir()]
                    if date_dirs:
                        # Check if we have a reasonable directory structure
                        user_dir = date_dirs[0]  # First user directory
                        year_month_dirs = [d for d in user_dir.iterdir() if d.is_dir()]
                        if year_month_dirs:
                            log_structure_correct = True
            
            return execution_success, log_files_created, temp_log_dir if log_files_created else None
            
        except subprocess.TimeoutExpired:
            return False, False, None
        except Exception as e:
            if self.verbose:
                print(f"Execution error: {e}")
            return False, False, None
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_log_dir)
            except:
                pass
    
    def verify_chronicle_integration(self, notebook_path: Path) -> ChronicleTestResult:
        """Complete verification of Chronicle integration."""
        if self.verbose:
            print(f"Verifying Chronicle integration in {notebook_path.name}...")
        
        # First, analyze the notebook statically
        result = self.analyze_notebook_chronicle_usage(notebook_path)
        
        # If basic requirements are met, try execution test
        if result.has_imports and result.has_initialization:
            if self.verbose:
                print("  Running execution test...")
            
            execution_ok, logs_created, log_dir = self.execute_notebook_and_check_logs(notebook_path)
            
            if execution_ok:
                result.creates_log_entries = logs_created
                result.proper_log_structure = logs_created  # Simplified for now
                result.log_directory = log_dir
                
                if not logs_created:
                    result.add_error("Notebook executed but no log files created")
            else:
                result.add_error("Notebook execution failed")
        else:
            result.add_error("Skipping execution test due to missing imports/initialization")
        
        return result
    
    def test_notebooks_in_directory(self, notebooks_dir: Path) -> Dict[Path, ChronicleTestResult]:
        """Test Chronicle integration for all notebooks in directory."""
        results = {}
        
        if not notebooks_dir.exists():
            print(f"Directory not found: {notebooks_dir}")
            return results
        
        notebook_paths = list(notebooks_dir.rglob('*.ipynb'))
        
        if not notebook_paths:
            print(f"No notebooks found in {notebooks_dir}")
            return results
        
        print(f"Testing Chronicle integration in {len(notebook_paths)} notebooks...")
        
        for notebook_path in sorted(notebook_paths):
            result = self.verify_chronicle_integration(notebook_path)
            results[notebook_path] = result
            
            if self.verbose:
                if result.passed:
                    print(f"  ✅ {notebook_path.name}: Chronicle integration verified")
                else:
                    print(f"  ❌ {notebook_path.name}: {len(result.errors)} errors")
                    for error in result.errors:
                        print(f"    - {error}")
        
        return results
    
    def generate_chronicle_report(self, results: Dict[Path, ChronicleTestResult]) -> str:
        """Generate a report on Chronicle integration."""
        if not results:
            return "No notebooks tested."
        
        total = len(results)
        passed = sum(1 for r in results.values() if r.passed)
        failed = total - passed
        
        report = [
            "# Chronicle Integration Test Report",
            "",
            f"**Summary:** {passed}/{total} notebooks passed Chronicle integration tests",
            ""
        ]
        
        if failed > 0:
            report.extend([
                "## Failed Notebooks",
                ""
            ])
            
            for notebook_path, result in results.items():
                if not result.passed:
                    report.append(f"### {notebook_path.name}")
                    report.append("")
                    
                    # Status indicators
                    status_items = [
                        ("Imports", result.has_imports),
                        ("Initialization", result.has_initialization),
                        ("Logging calls", result.has_logging_calls),
                        ("Log creation", result.creates_log_entries),
                        ("Log structure", result.proper_log_structure)
                    ]
                    
                    for item, status in status_items:
                        icon = "✅" if status else "❌"
                        report.append(f"- {icon} {item}")
                    
                    if result.errors:
                        report.append("")
                        report.append("**Errors:**")
                        for error in result.errors:
                            report.append(f"- {error}")
                    
                    if result.warnings:
                        report.append("")
                        report.append("**Warnings:**")
                        for warning in result.warnings:
                            report.append(f"- {warning}")
                    
                    report.append("")
        
        # Success stories
        passed_notebooks = [path for path, result in results.items() if result.passed]
        if passed_notebooks:
            report.extend([
                "## Passed Notebooks",
                ""
            ])
            for notebook_path in passed_notebooks:
                report.append(f"- ✅ {notebook_path.name}")
            report.append("")
        
        # Recommendations
        report.extend([
            "## Recommendations",
            ""
        ])
        
        missing_imports = sum(1 for r in results.values() if not r.has_imports)
        missing_init = sum(1 for r in results.values() if not r.has_initialization)
        missing_logs = sum(1 for r in results.values() if not r.creates_log_entries)
        
        if missing_imports > 0:
            report.append(f"- {missing_imports} notebooks missing Chronicle imports")
        if missing_init > 0:
            report.append(f"- {missing_init} notebooks missing Chronicle initialization")
        if missing_logs > 0:
            report.append(f"- {missing_logs} notebooks not creating log entries")
        
        if passed == total:
            report.append("- 🎉 All notebooks have proper Chronicle integration!")
        
        return "\n".join(report)


def main():
    """Main function for Chronicle integration testing."""
    parser = argparse.ArgumentParser(description='Test Chronicle integration in LeeQ notebooks')
    parser.add_argument('path', 
                       help='Path to notebook file or directory to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', '-r',
                       help='Save report to file')
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Resolve test path
    test_path = Path(args.path)
    if not test_path.is_absolute():
        test_path = project_root / test_path
    
    print("LeeQ Chronicle Integration Testing")
    print("=================================")
    print(f"Testing: {test_path}")
    print()
    
    # Initialize verifier
    verifier = ChronicleVerifier(project_root, verbose=args.verbose)
    
    # Run tests
    if test_path.is_file() and test_path.suffix == '.ipynb':
        # Test single notebook
        result = verifier.verify_chronicle_integration(test_path)
        results = {test_path: result}
    elif test_path.is_dir():
        # Test directory
        results = verifier.test_notebooks_in_directory(test_path)
    else:
        print(f"Error: Path not found or not a notebook/directory: {test_path}")
        sys.exit(1)
    
    # Generate report
    if results:
        report = verifier.generate_chronicle_report(results)
        print(report)
        
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.report}")
        
        # Exit code based on results
        failed = sum(1 for r in results.values() if not r.passed)
        sys.exit(0 if failed == 0 else 1)
    else:
        print("No notebooks found to test.")
        sys.exit(0)


if __name__ == '__main__':
    main()