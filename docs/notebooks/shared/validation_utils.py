"""
Validation utilities for LeeQ notebook testing and quality assurance.

This module provides functions for validating notebook execution, checking
experiment outputs, and ensuring content quality across all notebooks.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def validate_notebook_execution(notebook_path, timeout=300):
    """
    Validate that a notebook executes without errors.
    
    Args:
        notebook_path (str): Path to the notebook file
        timeout (int): Timeout in seconds for execution
        
    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        # Read the notebook
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create executor
        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
        
        # Execute the notebook
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            return True, "Notebook executed successfully"
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
            
    except Exception as e:
        return False, f"Failed to read notebook: {str(e)}"


def check_notebook_outputs(notebook_path):
    """
    Check that notebook has meaningful outputs in cells.
    
    Args:
        notebook_path (str): Path to the notebook file
        
    Returns:
        dict: Summary of output validation
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        output_stats = {
            'total_cells': len(nb.cells),
            'code_cells': 0,
            'cells_with_outputs': 0,
            'cells_with_plots': 0,
            'cells_with_errors': 0,
            'empty_output_cells': 0
        }
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                output_stats['code_cells'] += 1
                
                if hasattr(cell, 'outputs') and cell.outputs:
                    output_stats['cells_with_outputs'] += 1
                    
                    # Check for plots/figures
                    for output in cell.outputs:
                        if output.get('output_type') == 'display_data':
                            if 'image/png' in output.get('data', {}):
                                output_stats['cells_with_plots'] += 1
                                break
                        
                        # Check for errors
                        if output.get('output_type') == 'error':
                            output_stats['cells_with_errors'] += 1
                            break
                else:
                    output_stats['empty_output_cells'] += 1
        
        # Calculate percentages
        if output_stats['code_cells'] > 0:
            output_stats['output_percentage'] = (
                output_stats['cells_with_outputs'] / output_stats['code_cells'] * 100
            )
        else:
            output_stats['output_percentage'] = 0
            
        return output_stats
        
    except Exception as e:
        return {'error': f"Failed to analyze outputs: {str(e)}"}


def validate_experiment_results(notebook_path, expected_experiments=None):
    """
    Validate that specific experiments produce expected results.
    
    Args:
        notebook_path (str): Path to the notebook file
        expected_experiments (list): List of expected experiment types
        
    Returns:
        dict: Validation results for experiments
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        results = {
            'found_experiments': [],
            'missing_experiments': [],
            'validation_passed': True
        }
        
        # Look for experiment patterns in code cells
        experiment_patterns = {
            'NormalisedRabi': 'calibrations.NormalisedRabi',
            'SimpleRamseyMultilevel': 'calibrations.SimpleRamseyMultilevel', 
            'SimpleT1': 'characterizations.SimpleT1',
            'SpinEchoMultiLevel': 't2_echo',
            'AmpPingpongCalibrationSingleQubitMultilevel': 'pingpong',
            'CrossAllXYDragMultiRunSingleQubitMultilevel': 'calibrations.DragCalibrationSingleQubitMultilevel'
        }
        
        found_patterns = set()
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                for pattern, name in experiment_patterns.items():
                    if pattern in source:
                        found_patterns.add(name)
                        results['found_experiments'].append(name)
        
        # Check against expected experiments
        if expected_experiments:
            expected_set = set(expected_experiments)
            missing = expected_set - found_patterns
            results['missing_experiments'] = list(missing)
            results['validation_passed'] = len(missing) == 0
        
        return results
        
    except Exception as e:
        return {'error': f"Failed to validate experiments: {str(e)}"}


def check_plot_generation(notebook_path):
    """
    Check if notebook generates plots and visualizations.
    
    Args:
        notebook_path (str): Path to the notebook file
        
    Returns:
        dict: Plot validation results
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        plot_stats = {
            'has_plotly_imports': False,
            'has_matplotlib_imports': False,
            'plot_cells': 0,
            'cells_with_plot_outputs': 0,
            'total_plots': 0
        }
        
        # Check imports
        import_patterns = ['plotly', 'matplotlib', 'plt.', 'go.']
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source.lower()
                
                # Check for plotting imports
                if 'plotly' in source or 'go.' in source:
                    plot_stats['has_plotly_imports'] = True
                if 'matplotlib' in source or 'plt.' in source:
                    plot_stats['has_matplotlib_imports'] = True
                
                # Check for plotting commands
                plot_commands = ['plot(', 'figure(', 'show()', '.plot_', 'add_trace']
                if any(cmd in source for cmd in plot_commands):
                    plot_stats['plot_cells'] += 1
                
                # Check outputs for plots
                if hasattr(cell, 'outputs'):
                    for output in cell.outputs:
                        if output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            if any(mime in data for mime in ['image/png', 'image/jpeg', 'text/html']):
                                plot_stats['cells_with_plot_outputs'] += 1
                                plot_stats['total_plots'] += 1
                                break
        
        return plot_stats
        
    except Exception as e:
        return {'error': f"Failed to check plots: {str(e)}"}


def validate_notebook_content_quality(notebook_path):
    """
    Comprehensive content quality validation.
    
    Args:
        notebook_path (str): Path to the notebook file
        
    Returns:
        dict: Comprehensive validation report
    """
    notebook_name = os.path.basename(notebook_path)
    
    report = {
        'notebook': notebook_name,
        'execution': {},
        'outputs': {},
        'experiments': {},
        'plots': {},
        'overall_score': 0,
        'recommendations': []
    }
    
    # Test execution
    success, error_msg = validate_notebook_execution(notebook_path)
    report['execution'] = {
        'passed': success,
        'error': error_msg if not success else None
    }
    
    if success:
        # Check outputs
        report['outputs'] = check_notebook_outputs(notebook_path)
        
        # Check plots
        report['plots'] = check_plot_generation(notebook_path)
        
        # Check experiments based on notebook type
        if 'tutorial' in notebook_path.lower():
            if '01_basics' in notebook_path:
                expected = ['calibrations.NormalisedRabi']
            elif '02_single_qubit' in notebook_path:
                expected = ['calibrations.NormalisedRabi', 'calibrations.SimpleRamseyMultilevel', 'characterizations.SimpleT1', 't2_echo']
            else:
                expected = None
            
            if expected:
                report['experiments'] = validate_experiment_results(notebook_path, expected)
    
    # Calculate overall score
    score = 0
    if report['execution']['passed']:
        score += 40
    
    if report['outputs'].get('output_percentage', 0) > 50:
        score += 30
    
    if report['plots'].get('total_plots', 0) > 0:
        score += 20
    
    if report['experiments'].get('validation_passed', False):
        score += 10
    
    report['overall_score'] = score
    
    # Generate recommendations
    if not report['execution']['passed']:
        report['recommendations'].append("Fix execution errors before proceeding")
    
    if report['outputs'].get('output_percentage', 0) < 30:
        report['recommendations'].append("Add more cells with meaningful outputs")
    
    if report['plots'].get('total_plots', 0) == 0:
        report['recommendations'].append("Add data visualizations and plots")
    
    if report['experiments'].get('missing_experiments'):
        missing = report['experiments']['missing_experiments']
        report['recommendations'].append(f"Add missing experiments: {', '.join(missing)}")
    
    return report


def run_notebook_test_suite(notebook_directory, pattern="*.ipynb"):
    """
    Run validation tests on all notebooks in a directory.
    
    Args:
        notebook_directory (str): Directory containing notebooks
        pattern (str): File pattern to match
        
    Returns:
        dict: Test suite results
    """
    from glob import glob
    
    notebook_dir = Path(notebook_directory)
    notebook_files = list(notebook_dir.glob(pattern))
    
    suite_results = {
        'total_notebooks': len(notebook_files),
        'passed': 0,
        'failed': 0,
        'results': {},
        'summary': {}
    }
    
    for notebook_file in notebook_files:
        print(f"Testing {notebook_file.name}...")
        
        report = validate_notebook_content_quality(str(notebook_file))
        suite_results['results'][notebook_file.name] = report
        
        if report['overall_score'] >= 70:
            suite_results['passed'] += 1
            status = "PASSED"
        else:
            suite_results['failed'] += 1
            status = "FAILED"
        
        print(f"  {status} (Score: {report['overall_score']}/100)")
    
    # Generate summary
    suite_results['summary'] = {
        'pass_rate': suite_results['passed'] / suite_results['total_notebooks'] * 100,
        'average_score': sum(r['overall_score'] for r in suite_results['results'].values()) / len(suite_results['results']),
        'common_issues': []
    }
    
    # Find common issues
    all_recommendations = []
    for result in suite_results['results'].values():
        all_recommendations.extend(result.get('recommendations', []))
    
    from collections import Counter
    common_issues = Counter(all_recommendations).most_common(5)
    suite_results['summary']['common_issues'] = [issue for issue, count in common_issues]
    
    return suite_results


def generate_validation_report(test_results, output_file=None):
    """
    Generate a formatted validation report.
    
    Args:
        test_results (dict): Results from run_notebook_test_suite
        output_file (str): Optional file to save report
        
    Returns:
        str: Formatted report
    """
    report = f"""
# LeeQ Notebook Validation Report

## Summary
- Total Notebooks: {test_results['total_notebooks']}
- Passed: {test_results['passed']}
- Failed: {test_results['failed']}
- Pass Rate: {test_results['summary']['pass_rate']:.1f}%
- Average Score: {test_results['summary']['average_score']:.1f}/100

## Individual Results
"""
    
    for notebook, result in test_results['results'].items():
        status = "✓ PASSED" if result['overall_score'] >= 70 else "✗ FAILED"
        report += f"""
### {notebook} - {status} ({result['overall_score']}/100)

- Execution: {'✓' if result['execution']['passed'] else '✗'}
- Output Coverage: {result['outputs'].get('output_percentage', 0):.1f}%
- Plot Generation: {result['plots'].get('total_plots', 0)} plots
- Experiments: {'✓' if result['experiments'].get('validation_passed', False) else '✗'}

"""
        if result.get('recommendations'):
            report += "Recommendations:\n"
            for rec in result['recommendations']:
                report += f"- {rec}\n"
    
    if test_results['summary']['common_issues']:
        report += "\n## Common Issues\n"
        for issue in test_results['summary']['common_issues']:
            report += f"- {issue}\n"
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report
