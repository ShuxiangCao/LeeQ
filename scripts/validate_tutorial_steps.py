#!/usr/bin/env python3
"""
Tutorial Validation Framework

This script provides comprehensive validation for LeeQ tutorial content by:
1. Step-by-step verification for tutorial accuracy
2. Concept explanation validation
3. Learning progression checkpoints  
4. Code example validation

Usage:
    python scripts/validate_tutorial_steps.py [tutorial_file] [--verbose] [--section SECTION]
    
Exit codes:
    0: All validations passed
    1: Tutorial structure issues
    2: Code example failures
    3: Concept validation failures
    4: Learning progression issues
    5: Unexpected error occurred
"""

import argparse
import ast
import importlib.util
import logging
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import textwrap


class TutorialValidator:
    """Comprehensive tutorial validation framework."""
    
    def __init__(self, tutorial_file: Path, verbose: bool = False):
        self.tutorial_file = tutorial_file
        self.verbose = verbose
        self.setup_logging()
        self.validation_results = {
            'structure': [],
            'code_examples': [],
            'concepts': [],
            'progression': []
        }
        self.tutorial_content = ""
        self.sections = {}
        
    def setup_logging(self):
        """Configure logging for validation output."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=level
        )
        self.logger = logging.getLogger(__name__)
        
    def load_tutorial_content(self) -> Tuple[bool, str]:
        """Load and parse tutorial content."""
        try:
            with open(self.tutorial_file, 'r', encoding='utf-8') as f:
                self.tutorial_content = f.read()
            
            # Parse sections
            self._parse_sections()
            
            return True, f"Tutorial loaded: {len(self.sections)} sections"
            
        except Exception as e:
            return False, f"Failed to load tutorial: {e}"
            
    def _parse_sections(self):
        """Parse tutorial into sections based on headers."""
        lines = self.tutorial_content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('# '):
                # Main title
                if current_section:
                    self.sections[current_section] = '\n'.join(current_content)
                current_section = line[2:].strip()
                current_content = [line]
            elif line.startswith('## '):
                # Section header
                if current_section:
                    self.sections[current_section] = '\n'.join(current_content)
                current_section = line[3:].strip()
                current_content = [line]
            else:
                current_content.append(line)
                
        # Add final section
        if current_section:
            self.sections[current_section] = '\n'.join(current_content)
            
    def validate_tutorial_structure(self) -> Tuple[bool, str]:
        """Validate overall tutorial structure and organization."""
        self.logger.info("Validating tutorial structure...")
        
        issues = []
        
        # Check for required sections
        required_sections = [
            'Parameter Storage and Update',
            'Orchestrating Pulses', 
            'Adjusting Runtime Parameters',
            'Customizing Your Setup',
            'Creating a Customized Experiment',
            'Data Persistence'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in self.sections:
                missing_sections.append(section)
                
        if missing_sections:
            issues.append(f"Missing required sections: {missing_sections}")
            
        # Check section ordering and progression
        section_order = list(self.sections.keys())
        
        # Validate logical progression
        progression_issues = self._validate_section_progression(section_order)
        if progression_issues:
            issues.extend(progression_issues)
            
        # Check for proper introduction
        if 'Tutorial' not in section_order[0] if section_order else True:
            issues.append("Tutorial should start with proper introduction")
            
        success = len(issues) == 0
        message = "Tutorial structure valid" if success else f"Structure issues: {'; '.join(issues)}"
        
        self.validation_results['structure'].append({
            'test': 'Tutorial Structure',
            'success': success,
            'message': message,
            'issues': issues
        })
        
        return success, message
        
    def _validate_section_progression(self, section_order: List[str]) -> List[str]:
        """Validate logical progression of tutorial sections."""
        issues = []
        
        # Define expected progression patterns
        basic_concepts = ['Parameter Storage', 'DUT Object', 'Collection', 'Measurement Primitives']
        intermediate_concepts = ['Orchestrating Pulses', 'Single Qubit Operations']
        advanced_concepts = ['Customizing', 'Creating']
        
        # Check if basic concepts come before advanced ones
        basic_found = False
        advanced_found = False
        
        for section in section_order:
            if any(concept in section for concept in basic_concepts):
                basic_found = True
            elif any(concept in section for concept in advanced_concepts):
                if not basic_found:
                    issues.append(f"Advanced concept '{section}' appears before basic concepts")
                advanced_found = True
                
        return issues
        
    def extract_code_examples(self) -> List[Dict[str, Any]]:
        """Extract all code examples from tutorial content."""
        code_blocks = []
        
        # Pattern to match code blocks
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, self.tutorial_content, re.DOTALL)
        
        for i, code in enumerate(matches):
            # Determine which section this code belongs to
            section = self._find_section_for_code(code, i)
            
            code_blocks.append({
                'index': i,
                'code': code.strip(),
                'section': section,
                'type': self._classify_code_example(code)
            })
            
        return code_blocks
        
    def _find_section_for_code(self, code: str, index: int) -> str:
        """Find which section a code example belongs to."""
        # Simple approach: find the closest preceding section header
        lines_before_code = []
        current_pos = 0
        
        for match in re.finditer(r'```python\n.*?\n```', self.tutorial_content, re.DOTALL):
            if match.group(0).replace('```python\n', '').replace('\n```', '').strip() == code.strip():
                lines_before_code = self.tutorial_content[:match.start()].split('\n')
                break
                
        # Find most recent section header
        for line in reversed(lines_before_code):
            if line.startswith('## '):
                return line[3:].strip()
        return "Unknown"
        
    def _classify_code_example(self, code: str) -> str:
        """Classify the type of code example."""
        if 'TransmonElement' in code:
            return 'dut_creation'
        elif 'lpb_collections' in code or 'measurement_primitives' in code:
            return 'configuration'
        elif 'class' in code and 'Experiment' in code:
            return 'experiment_definition'
        elif 'def' in code:
            return 'function_definition'
        elif any(method in code for method in ['get_c1', 'get_measurement_prim', 'basic']):
            return 'experiment_execution'
        elif 'Chronicle' in code:
            return 'data_persistence'
        else:
            return 'general_example'
            
    def validate_code_examples(self) -> Tuple[bool, str]:
        """Validate all code examples in the tutorial."""
        self.logger.info("Validating code examples...")
        
        code_examples = self.extract_code_examples()
        total_examples = len(code_examples)
        
        if total_examples == 0:
            return False, "No code examples found in tutorial"
            
        validation_results = []
        
        for example in code_examples:
            result = self._validate_single_code_example(example)
            validation_results.append(result)
            
        passed = sum(1 for r in validation_results if r['success'])
        
        success = passed == total_examples
        message = f"Code examples: {passed}/{total_examples} passed"
        
        self.validation_results['code_examples'] = validation_results
        
        return success, message
        
    def _validate_single_code_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single code example."""
        code = example['code']
        
        try:
            # First, check for syntax errors
            ast.parse(code)
            syntax_valid = True
            syntax_error = None
        except SyntaxError as e:
            syntax_valid = False
            syntax_error = str(e)
            
        # Check for common imports and patterns
        import_issues = self._check_imports(code)
        pattern_issues = self._check_code_patterns(code, example['type'])
        
        # Check for completeness based on type
        completeness_issues = self._check_code_completeness(code, example['type'])
        
        all_issues = import_issues + pattern_issues + completeness_issues
        
        success = syntax_valid and len(all_issues) == 0
        
        return {
            'example': example,
            'success': success,
            'syntax_valid': syntax_valid,
            'syntax_error': syntax_error,
            'issues': all_issues
        }
        
    def _check_imports(self, code: str) -> List[str]:
        """Check for necessary imports in code examples."""
        issues = []
        
        # Common patterns that need imports
        patterns_requiring_imports = {
            'TransmonElement': 'from leeq.core.elements.built_in.qudit_transmon import TransmonElement',
            'Chronicle': 'from leeq.chronicle import Chronicle', 
            'Experiment': 'from leeq.experiments.experiments import Experiment',
            'go.Figure': 'import plotly.graph_objects as go',
            'np.': 'import numpy as np',
            'PulseShapeFactory': 'from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory',
            'get_t_list': 'from leeq.compiler.utils.time_base import get_t_list',
            'Sweeper': 'from leeq.experiments.sweeper import Sweeper',
            'prims.': 'from leeq.core import primitives as prims',
            'sparam.': 'from leeq.experiments import sweeper as sparam',
            'basic': 'from leeq.experiments.experiments import basic',
            'log_and_record': 'from leeq.chronicle.decorators import log_and_record',
            'register_browser_function': 'from leeq.experiments.plots.live_dash_app import register_browser_function'
        }
        
        for pattern, expected_import in patterns_requiring_imports.items():
            if pattern in code and expected_import not in code:
                # Check if a similar import exists (flexible matching)
                import_base = expected_import.split('import')[-1].strip().split('.')[0]
                if f'import {import_base}' not in code and f'from {import_base}' not in code:
                    issues.append(f"Missing import for {pattern}: consider {expected_import}")
                    
        return issues
        
    def _check_code_patterns(self, code: str, example_type: str) -> List[str]:
        """Check for proper coding patterns based on example type."""
        issues = []
        
        if example_type == 'dut_creation':
            if 'TransmonElement' in code:
                if 'name=' not in code:
                    issues.append("TransmonElement should include name parameter")
                if 'parameters=' not in code:
                    issues.append("TransmonElement should include parameters dictionary")
                    
        elif example_type == 'configuration':
            if 'lpb_collections' in code:
                required_fields = ['type', 'freq', 'channel']
                for field in required_fields:
                    if f"'{field}'" not in code and f'"{field}"' not in code:
                        issues.append(f"LPB collection should include {field} field")
                        
        elif example_type == 'experiment_definition':
            if 'class' in code:
                if 'Experiment' not in code:
                    issues.append("Experiment class should inherit from Experiment")
                if 'def run(' not in code:
                    issues.append("Experiment class should have run method")
                    
        elif example_type == 'data_persistence':
            if 'Chronicle' in code:
                if 'start_log()' not in code:
                    issues.append("Chronicle should call start_log() method")
                    
        return issues
        
    def _check_code_completeness(self, code: str, example_type: str) -> List[str]:
        """Check if code examples are complete and functional."""
        issues = []
        
        # Check for incomplete examples (common indicators)
        if '...' in code:
            issues.append("Code example contains '...' indicating incompleteness")
            
        if '# TODO' in code or '# FIXME' in code:
            issues.append("Code example contains TODO/FIXME comments")
            
        # Check for proper error handling in experiment examples
        if example_type == 'experiment_definition':
            if 'try:' not in code and 'except' not in code:
                # This is not always required, so just a warning
                pass  # Could add warning about error handling
                
        # Check for proper docstrings in function definitions
        if 'def ' in code and '"""' not in code and "'''" not in code:
            issues.append("Function definition should include docstring")
            
        return issues
        
    def validate_concept_explanations(self) -> Tuple[bool, str]:
        """Validate concept explanations for clarity and completeness."""
        self.logger.info("Validating concept explanations...")
        
        concepts = {
            'DUT Object': {
                'section': 'DUT Object',
                'required_explanations': [
                    'Device Under Test',
                    'TransmonElement',
                    'parameter storage',
                    'configuration'
                ]
            },
            'Collection': {
                'section': 'Collection', 
                'required_explanations': [
                    'group of pulses',
                    'common parameters',
                    'SimpleDriveCollection'
                ]
            },
            'Measurement Primitives': {
                'section': 'Measurement Primitives',
                'required_explanations': [
                    'measurement pulses',
                    'data acquisition',
                    'SimpleDispersiveMeasurement'
                ]
            },
            'Logical Primitive': {
                'section': 'Orchestrating Pulses',
                'required_explanations': [
                    'basic operation',
                    'single pulse',
                    'tree leaf'
                ]
            },
            'LPB': {
                'section': 'Orchestrating Pulses', 
                'required_explanations': [
                    'Logical Primitive Block',
                    'composite element',
                    'tree structure'
                ]
            }
        }
        
        validation_results = []
        
        for concept_name, concept_info in concepts.items():
            result = self._validate_single_concept(concept_name, concept_info)
            validation_results.append(result)
            
        passed = sum(1 for r in validation_results if r['success'])
        total = len(validation_results)
        
        success = passed == total
        message = f"Concept explanations: {passed}/{total} validated"
        
        self.validation_results['concepts'] = validation_results
        
        return success, message
        
    def _validate_single_concept(self, concept_name: str, concept_info: Dict) -> Dict[str, Any]:
        """Validate explanation of a single concept."""
        section_name = concept_info['section']
        required_explanations = concept_info['required_explanations']
        
        # Get section content
        section_content = self.sections.get(section_name, '')
        
        missing_explanations = []
        for explanation in required_explanations:
            if explanation.lower() not in section_content.lower():
                missing_explanations.append(explanation)
                
        # Check if concept is properly introduced
        concept_introduced = concept_name.lower() in section_content.lower()
        
        # Check for adequate detail (at least 2 sentences)
        sentences = len([s for s in section_content.split('.') if s.strip()])
        adequate_detail = sentences >= 2
        
        issues = []
        if missing_explanations:
            issues.append(f"Missing explanations: {missing_explanations}")
        if not concept_introduced:
            issues.append(f"Concept '{concept_name}' not clearly introduced")
        if not adequate_detail:
            issues.append("Insufficient detail in explanation")
            
        success = len(issues) == 0
        
        return {
            'concept': concept_name,
            'section': section_name,
            'success': success,
            'issues': issues,
            'missing_explanations': missing_explanations
        }
        
    def validate_learning_progression(self) -> Tuple[bool, str]:
        """Validate learning progression and checkpoints."""
        self.logger.info("Validating learning progression...")
        
        # Define learning progression checkpoints
        checkpoints = [
            {
                'name': 'Basic Concepts',
                'prerequisite': None,
                'concepts': ['DUT Object', 'Collection', 'Measurement Primitives'],
                'skills': ['Create TransmonElement', 'Define collections']
            },
            {
                'name': 'Pulse Orchestration',
                'prerequisite': 'Basic Concepts',
                'concepts': ['Logical Primitive', 'LPB', 'Pulse scheduling'],
                'skills': ['Build pulse sequences', 'Use operators']
            },
            {
                'name': 'Single Qubit Operations', 
                'prerequisite': 'Pulse Orchestration',
                'concepts': ['Single qubit gates', 'Measurement primitives'],
                'skills': ['Access gates', 'Run measurements']
            },
            {
                'name': 'Runtime Control',
                'prerequisite': 'Single Qubit Operations',
                'concepts': ['Parameter updates', 'Calibration persistence'],
                'skills': ['Update parameters', 'Save/load configurations']
            },
            {
                'name': 'Custom Experiments',
                'prerequisite': 'Runtime Control',
                'concepts': ['Experiment class', 'Data visualization'],
                'skills': ['Create experiments', 'Plot results']
            }
        ]
        
        validation_results = []
        
        for checkpoint in checkpoints:
            result = self._validate_progression_checkpoint(checkpoint)
            validation_results.append(result)
            
        passed = sum(1 for r in validation_results if r['success'])
        total = len(validation_results)
        
        success = passed == total
        message = f"Learning progression: {passed}/{total} checkpoints validated"
        
        self.validation_results['progression'] = validation_results
        
        return success, message
        
    def _validate_progression_checkpoint(self, checkpoint: Dict) -> Dict[str, Any]:
        """Validate a single learning progression checkpoint."""
        name = checkpoint['name']
        concepts = checkpoint['concepts']
        skills = checkpoint['skills']
        
        issues = []
        
        # Check if concepts are covered
        missing_concepts = []
        for concept in concepts:
            if not self._concept_covered_before_checkpoint(concept, name):
                missing_concepts.append(concept)
                
        if missing_concepts:
            issues.append(f"Missing concepts: {missing_concepts}")
            
        # Check if skills can be demonstrated 
        missing_skills = []
        for skill in skills:
            if not self._skill_demonstrable(skill):
                missing_skills.append(skill)
                
        if missing_skills:
            issues.append(f"Skills not demonstrable: {missing_skills}")
            
        # Check prerequisite ordering
        if checkpoint['prerequisite']:
            prereq_position = self._get_checkpoint_position(checkpoint['prerequisite'])
            current_position = self._get_checkpoint_position(name)
            
            if prereq_position >= current_position:
                issues.append(f"Prerequisite '{checkpoint['prerequisite']}' comes after current checkpoint")
                
        success = len(issues) == 0
        
        return {
            'checkpoint': name,
            'success': success,
            'issues': issues,
            'concepts_covered': len(concepts) - len(missing_concepts),
            'total_concepts': len(concepts),
            'skills_demonstrable': len(skills) - len(missing_skills),
            'total_skills': len(skills)
        }
        
    def _concept_covered_before_checkpoint(self, concept: str, checkpoint_name: str) -> bool:
        """Check if concept is covered before the given checkpoint."""
        # Simple check: see if concept appears in tutorial content
        return concept.lower() in self.tutorial_content.lower()
        
    def _skill_demonstrable(self, skill: str) -> bool:
        """Check if skill is demonstrable through code examples."""
        skill_keywords = {
            'Create TransmonElement': ['TransmonElement('],
            'Define collections': ['lpb_collections'],
            'Build pulse sequences': ['+', '*', 'LogicalPrimitiveBlock'],
            'Use operators': ['+', '*'],
            'Access gates': ['get_c1', "['X']", "['Y']"],
            'Run measurements': ['get_measurement_prim', 'basic('],
            'Update parameters': ['set_', 'save_calibration_log'],
            'Save/load configurations': ['save_calibration_log', 'load_from_calibration_log'],
            'Create experiments': ['class', 'Experiment', 'def run('],
            'Plot results': ['plot_', 'go.Figure', '@register_browser_function']
        }
        
        keywords = skill_keywords.get(skill, [])
        return any(keyword in self.tutorial_content for keyword in keywords)
        
    def _get_checkpoint_position(self, checkpoint_name: str) -> int:
        """Get position of checkpoint in tutorial (rough approximation)."""
        # Map checkpoint names to section positions
        section_positions = {name: i for i, name in enumerate(self.sections.keys())}
        
        checkpoint_sections = {
            'Basic Concepts': 0,
            'Pulse Orchestration': 1, 
            'Single Qubit Operations': 2,
            'Runtime Control': 3,
            'Custom Experiments': 4
        }
        
        return checkpoint_sections.get(checkpoint_name, 999)
        
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 80)
        report.append("LEEQ TUTORIAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.validation_results.items():
            if isinstance(results, list):
                total_tests += len(results)
                passed_tests += sum(1 for r in results if r.get('success', False))
                
        report.append(f"Overall Results: {passed_tests}/{total_tests} tests passed")
        report.append("")
        
        # Detailed results by category
        for category, results in self.validation_results.items():
            report.append(f"{category.upper().replace('_', ' ')} VALIDATION")
            report.append("-" * 50)
            
            if isinstance(results, list):
                for result in results:
                    status = "PASS" if result.get('success', False) else "FAIL"
                    name = result.get('test', result.get('concept', result.get('checkpoint', 'Unknown')))
                    report.append(f"{name:.<40} {status}")
                    
                    if not result.get('success', False) and result.get('issues'):
                        for issue in result['issues']:
                            report.append(f"  └─ {issue}")
            report.append("")
            
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"• {rec}")
            
        return "\n".join(report)
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for common issues
        failed_code_examples = [r for r in self.validation_results.get('code_examples', []) 
                               if not r.get('success', False)]
        if failed_code_examples:
            recommendations.append("Review and fix code examples for syntax and completeness")
            
        failed_concepts = [r for r in self.validation_results.get('concepts', [])
                          if not r.get('success', False)]
        if failed_concepts:
            recommendations.append("Enhance concept explanations with more detail and examples")
            
        failed_progression = [r for r in self.validation_results.get('progression', [])
                             if not r.get('success', False)]
        if failed_progression:
            recommendations.append("Improve learning progression by adding prerequisite concepts")
            
        # General recommendations
        if not recommendations:
            recommendations.append("Tutorial validation passed - consider adding more examples")
            
        return recommendations
        
    def run_validation(self, target_section: Optional[str] = None) -> int:
        """Run complete validation workflow."""
        self.logger.info("Starting tutorial validation...")
        
        try:
            # Load tutorial content
            success, msg = self.load_tutorial_content()
            if not success:
                self.logger.error(msg)
                return 5
                
            self.logger.info(msg)
            
            # Run validations
            validations = []
            
            if not target_section or target_section == 'structure':
                success, msg = self.validate_tutorial_structure()
                validations.append(('Structure', success, msg))
                
            if not target_section or target_section == 'code':
                success, msg = self.validate_code_examples() 
                validations.append(('Code Examples', success, msg))
                
            if not target_section or target_section == 'concepts':
                success, msg = self.validate_concept_explanations()
                validations.append(('Concepts', success, msg))
                
            if not target_section or target_section == 'progression':
                success, msg = self.validate_learning_progression()
                validations.append(('Learning Progression', success, msg))
                
            # Generate and display report
            report = self.generate_validation_report()
            print("\n" + report)
            
            # Determine exit code
            all_passed = all(success for _, success, _ in validations)
            
            if all_passed:
                self.logger.info("🎉 Tutorial validation SUCCESSFUL!")
                return 0
            else:
                failed_categories = [name for name, success, _ in validations if not success]
                self.logger.error(f"❌ Tutorial validation FAILED in: {', '.join(failed_categories)}")
                
                # Return specific error codes
                if any('Structure' in cat for cat in failed_categories):
                    return 1
                elif any('Code' in cat for cat in failed_categories):
                    return 2
                elif any('Concepts' in cat for cat in failed_categories):
                    return 3
                elif any('Progression' in cat for cat in failed_categories):
                    return 4
                else:
                    return 5
                    
        except Exception as e:
            self.logger.error(f"Unexpected error during validation: {e}")
            if self.verbose:
                self.logger.debug(traceback.format_exc())
            return 5


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate LeeQ tutorial content comprehensively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'tutorial_file',
        nargs='?',
        default='docs/tutorial.md',
        help='Path to tutorial file (default: docs/tutorial.md)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true', 
        help='Enable verbose output with detailed logging'
    )
    
    parser.add_argument(
        '--section', '-s',
        choices=['structure', 'code', 'concepts', 'progression'],
        help='Validate specific section only'
    )
    
    args = parser.parse_args()
    
    # Resolve tutorial file path
    tutorial_path = Path(args.tutorial_file)
    if not tutorial_path.is_absolute():
        # Assume relative to script directory's parent (project root)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        tutorial_path = project_root / args.tutorial_file
        
    if not tutorial_path.exists():
        print(f"Error: Tutorial file not found: {tutorial_path}")
        sys.exit(5)
        
    validator = TutorialValidator(tutorial_path, verbose=args.verbose)
    exit_code = validator.run_validation(target_section=args.section)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()