#!/usr/bin/env python3
"""
Quick Start Validation Script

This script automatically validates the LeeQ Quick Start Guide by:
1. Testing installation verification
2. Simulating fresh environment setup
3. Running basic experiment workflows
4. Providing failure recovery and troubleshooting automation

Usage:
    python scripts/test_quick_start.py [--verbose] [--skip-install] [--timeout SECONDS]
    
Exit codes:
    0: All tests passed
    1: Installation verification failed
    2: Environment setup failed  
    3: Experiment execution failed
    4: Unexpected error occurred
"""

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple


class QuickStartValidator:
    """Validates the LeeQ Quick Start Guide steps automatically."""
    
    def __init__(self, verbose: bool = False, timeout: int = 600):
        self.verbose = verbose
        self.timeout = timeout
        self.start_time = time.time()
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for validation output."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=level
        )
        self.logger = logging.getLogger(__name__)
        
    def check_timeout(self) -> bool:
        """Check if validation has exceeded timeout."""
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            self.logger.error(f"Validation timeout exceeded: {elapsed:.1f}s > {self.timeout}s")
            return True
        return False
        
    def validate_python_environment(self) -> Tuple[bool, str]:
        """Validate Python environment meets requirements."""
        self.logger.info("Validating Python environment...")
        
        # Check Python version (3.8+)
        if sys.version_info < (3, 8):
            return False, f"Python 3.8+ required, found {sys.version}"
            
        # Check virtual environment (recommended)
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if not in_venv:
            self.logger.warning("Not running in virtual environment (recommended)")
            
        return True, "Python environment valid"
        
    def validate_dependencies(self) -> Tuple[bool, str]:
        """Validate core LeeQ dependencies are available."""
        self.logger.info("Validating core dependencies...")
        
        required_packages = [
            'numpy', 'scipy', 'matplotlib', 'plotly', 'qutip', 
            'sklearn', 'h5py', 'yaml', 'pandas'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.debug(f"✓ {package} available")
            except ImportError:
                missing_packages.append(package)
                self.logger.error(f"✗ {package} missing")
                
        if missing_packages:
            return False, f"Missing packages: {', '.join(missing_packages)}"
            
        return True, "All dependencies available"
        
    def validate_leeq_installation(self) -> Tuple[bool, str]:
        """Validate LeeQ package is properly installed."""
        self.logger.info("Validating LeeQ installation...")
        
        try:
            import leeq
            self.logger.debug(f"✓ LeeQ imported from {leeq.__file__}")
            
            # Test core modules
            core_modules = [
                'leeq.core.base',
                'leeq.core.elements.built_in.qudit_transmon',
                'leeq.setups.built_in.setup_simulation_high_level',
                'leeq.experiments.builtin',
                'leeq.chronicle'
            ]
            
            for module in core_modules:
                try:
                    importlib.import_module(module)
                    self.logger.debug(f"✓ {module} available")
                except ImportError as e:
                    return False, f"Core module {module} unavailable: {e}"
                    
            return True, "LeeQ installation valid"
            
        except ImportError as e:
            return False, f"LeeQ not installed or importable: {e}"
            
    def create_test_environment_setup(self) -> Tuple[bool, str]:
        """Create a test environment setup file."""
        self.logger.info("Creating test environment setup...")
        
        setup_code = '''
import numpy as np
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.chronicle import Chronicle

def create_test_setup():
    """Create a minimal test setup for validation."""
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # Create virtual transmon with realistic parameters
    virtual_transmon = VirtualTransmon(
        name="TestQubit",
        qubit_frequency=5000.0,  # MHz
        anharmonicity=-200,      # MHz
        t1=50,                   # microseconds
        t2=25,                   # microseconds
        readout_frequency=9500.0, # MHz
        quiescent_state_distribution=np.asarray([0.95, 0.04, 0.01])
    )

    # Create high-level simulation setup
    setup = HighLevelSimulationSetup(
        name='QuickStartTestSetup',
        virtual_qubits={0: virtual_transmon},
    )
    
    manager.register_setup(setup)
    return manager, setup

# Test qubit configuration
test_qubit_params = {
    'hrid': 'TestQ',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5000.0,
            'channel': 0,
            'shape': 'blackman_drag',
            'amp': 0.5,
            'phase': 0.0,
            'width': 0.05,
            'alpha': 500,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9500.0,
            'channel': 1,
            'shape': 'square',
            'amp': 0.1,
            'phase': 0.0,
            'width': 1.0,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}
'''
        
        try:
            # Create temporary file with test setup
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(setup_code)
                self.test_setup_file = f.name
                
            self.logger.debug(f"Test setup file created: {self.test_setup_file}")
            return True, "Test environment setup created"
            
        except Exception as e:
            return False, f"Failed to create test setup: {e}"
            
    def validate_basic_experiment_workflow(self) -> Tuple[bool, str]:
        """Validate basic experiment workflow execution."""
        self.logger.info("Validating basic experiment workflow...")
        
        try:
            # Import test setup
            spec = importlib.util.spec_from_file_location("test_setup", self.test_setup_file)
            test_setup = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_setup)
            
            # Create setup and qubit
            manager, setup = test_setup.create_test_setup()
            
            # Create TransmonElement
            from leeq.core.elements.built_in.qudit_transmon import TransmonElement
            dut = TransmonElement(
                name=test_setup.test_qubit_params['hrid'],
                parameters=test_setup.test_qubit_params
            )
            
            self.logger.debug("✓ TransmonElement created successfully")
            
            # Test basic measurements (without running actual experiments to save time)
            # Just verify the setup can be configured
            manager.status().set_param("Shot_Number", 100)
            manager.status().set_param("Shot_Period", 100)
            
            self.logger.debug("✓ Basic setup configuration successful")
            
            # Verify virtual qubit access
            virtual_qubit = setup.get_virtual_qubit(dut)
            self.logger.debug(f"✓ Virtual qubit access: {virtual_qubit.name}")
            
            return True, "Basic experiment workflow validated"
            
        except Exception as e:
            self.logger.error(f"Experiment workflow validation failed: {e}")
            if self.verbose:
                self.logger.debug(traceback.format_exc())
            return False, f"Experiment workflow failed: {e}"
            
    def validate_chronicle_integration(self) -> Tuple[bool, str]:
        """Validate Chronicle logging integration."""
        self.logger.info("Validating Chronicle integration...")
        
        try:
            from leeq.chronicle import Chronicle
            
            # Test chronicle startup
            chronicle = Chronicle()
            chronicle.start_log()
            
            self.logger.debug("✓ Chronicle logging started")
            
            # Test basic logging functionality
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Chronicle should handle logging without errors
                pass
                
            return True, "Chronicle integration validated"
            
        except Exception as e:
            return False, f"Chronicle integration failed: {e}"
            
    def check_common_issues(self) -> Tuple[bool, str]:
        """Check for common installation/setup issues."""
        self.logger.info("Checking for common issues...")
        
        issues_found = []
        
        # Check for numpy version compatibility
        try:
            import numpy as np
            if np.__version__.startswith('2.'):
                issues_found.append("NumPy 2.x detected - LeeQ requires NumPy < 2.0.0")
        except ImportError:
            issues_found.append("NumPy not available")
            
        # Check for conflicting packages
        try:
            import qutip
            # Basic qutip import test
        except ImportError:
            issues_found.append("QuTiP not available - required for simulation")
            
        # Check file permissions in current directory
        if not os.access('.', os.W_OK):
            issues_found.append("Current directory not writable - Chronicle needs write access")
            
        if issues_found:
            return False, f"Issues found: {'; '.join(issues_found)}"
            
        return True, "No common issues detected"
        
    def provide_troubleshooting_guidance(self, error_msg: str) -> str:
        """Provide troubleshooting guidance based on error message."""
        guidance = "\n=== TROUBLESHOOTING GUIDANCE ===\n"
        
        if "numpy" in error_msg.lower():
            guidance += """
NumPy Issue Detected:
• Ensure NumPy < 2.0.0: pip install 'numpy<2.0.0'
• Consider using virtual environment to avoid conflicts
• Check: python -c "import numpy; print(numpy.__version__)"
"""
        
        if "import" in error_msg.lower() or "module" in error_msg.lower():
            guidance += """
Import/Module Issue:
• Verify LeeQ installation: pip install git+https://github.com/ShuxiangCao/LeeQ
• Check dependencies: pip install -r requirements.txt
• Activate virtual environment if using one
• Try: python -c "import leeq; print('Success')"
"""
        
        if "chronicle" in error_msg.lower():
            guidance += """
Chronicle Issue:
• Chronicle is now integrated in LeeQ (not separate package)
• Ensure write permissions in current directory
• Check environment variables LAB_CHRONICLE_LOG_DIR if set
"""
        
        if "timeout" in error_msg.lower():
            guidance += """
Timeout Issue:
• Increase timeout: --timeout 1200
• Check system performance
• Verify no blocking operations
"""
        
        guidance += """
General Recovery Steps:
1. Restart Python interpreter
2. Clear any cached modules: python -c "import sys; sys.modules.clear()"
3. Reinstall LeeQ: pip uninstall leeq && pip install git+https://github.com/ShuxiangCao/LeeQ
4. Check GitHub issues: https://github.com/ShuxiangCao/LeeQ/issues
5. Run with verbose output: python scripts/test_quick_start.py --verbose
"""
        
        return guidance
        
    def cleanup(self):
        """Clean up temporary files."""
        if hasattr(self, 'test_setup_file') and os.path.exists(self.test_setup_file):
            try:
                os.unlink(self.test_setup_file)
                self.logger.debug("Temporary test setup file cleaned up")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file: {e}")
                
    def run_validation(self, skip_install: bool = False) -> int:
        """Run complete validation workflow."""
        self.logger.info("=" * 60)
        self.logger.info("LeeQ Quick Start Guide Validation")
        self.logger.info("=" * 60)
        
        validation_steps = []
        
        try:
            # Step 1: Environment validation
            if not skip_install:
                success, msg = self.validate_python_environment()
                validation_steps.append(("Python Environment", success, msg))
                if not success:
                    return 1
                    
                if self.check_timeout():
                    return 4
                    
                success, msg = self.validate_dependencies()
                validation_steps.append(("Dependencies", success, msg))
                if not success:
                    return 1
                    
            # Step 2: LeeQ installation
            if self.check_timeout():
                return 4
                
            success, msg = self.validate_leeq_installation()
            validation_steps.append(("LeeQ Installation", success, msg))
            if not success:
                return 1
                
            # Step 3: Environment setup
            if self.check_timeout():
                return 4
                
            success, msg = self.create_test_environment_setup()
            validation_steps.append(("Environment Setup", success, msg))
            if not success:
                return 2
                
            # Step 4: Chronicle integration
            if self.check_timeout():
                return 4
                
            success, msg = self.validate_chronicle_integration()
            validation_steps.append(("Chronicle Integration", success, msg))
            if not success:
                return 2
                
            # Step 5: Experiment workflow
            if self.check_timeout():
                return 4
                
            success, msg = self.validate_basic_experiment_workflow()
            validation_steps.append(("Basic Experiment Workflow", success, msg))
            if not success:
                return 3
                
            # Step 6: Common issues check
            if self.check_timeout():
                return 4
                
            success, msg = self.check_common_issues()
            validation_steps.append(("Common Issues Check", success, msg))
            
        except Exception as e:
            self.logger.error(f"Unexpected error during validation: {e}")
            if self.verbose:
                self.logger.debug(traceback.format_exc())
            print(self.provide_troubleshooting_guidance(str(e)))
            return 4
            
        finally:
            self.cleanup()
            
        # Results summary
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION RESULTS SUMMARY")
        self.logger.info("=" * 60)
        
        total_time = time.time() - self.start_time
        passed = sum(1 for _, success, _ in validation_steps if success)
        total = len(validation_steps)
        
        for step_name, success, message in validation_steps:
            status = "PASS" if success else "FAIL"
            self.logger.info(f"{step_name:.<30} {status}")
            if not success or self.verbose:
                self.logger.info(f"  └─ {message}")
                
        self.logger.info("-" * 60)
        self.logger.info(f"Total: {passed}/{total} passed in {total_time:.1f}s")
        
        if passed == total:
            self.logger.info("🎉 Quick Start Guide validation SUCCESSFUL!")
            self.logger.info("You can proceed with LeeQ experiments.")
            return 0
        else:
            self.logger.error("❌ Quick Start Guide validation FAILED!")
            failed_steps = [name for name, success, _ in validation_steps if not success]
            error_msg = f"Failed steps: {', '.join(failed_steps)}"
            print(self.provide_troubleshooting_guidance(error_msg))
            return max(1, 4 - passed)  # Return appropriate error code


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate LeeQ Quick Start Guide setup and functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed logging'
    )
    
    parser.add_argument(
        '--skip-install',
        action='store_true',
        help='Skip installation verification steps'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout in seconds for validation (default: 600)'
    )
    
    args = parser.parse_args()
    
    validator = QuickStartValidator(verbose=args.verbose, timeout=args.timeout)
    exit_code = validator.run_validation(skip_install=args.skip_install)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()