# Quick Start Validation Script

The `test_quick_start.py` script provides automated testing and validation of the LeeQ Quick Start Guide to ensure users can successfully set up and run their first quantum experiment.

## Features

### 1. Fresh Environment Simulation
- Tests Python environment compatibility (3.8+)
- Validates core dependencies availability
- Checks for common compatibility issues

### 2. Installation Verification
- Verifies LeeQ package installation and import
- Tests core module availability
- Validates integrated Chronicle functionality

### 3. Automated Testing
- Creates temporary test environment setup
- Simulates basic experiment workflow
- Tests TransmonElement creation and configuration
- Validates virtual qubit simulation

### 4. Failure Recovery & Troubleshooting
- Comprehensive error detection and classification
- Context-aware troubleshooting guidance
- Common issue detection and solutions
- Recovery recommendations based on failure type

## Usage

### Basic Usage
```bash
# Run full validation (recommended for fresh installs)
python scripts/test_quick_start.py

# Run with verbose output for debugging
python scripts/test_quick_start.py --verbose

# Skip installation checks (if already validated)
python scripts/test_quick_start.py --skip-install

# Set custom timeout (default: 600 seconds)
python scripts/test_quick_start.py --timeout 300
```

### Exit Codes
- `0`: All tests passed successfully
- `1`: Installation verification failed
- `2`: Environment setup failed
- `3`: Experiment execution failed
- `4`: Unexpected error occurred

## Validation Steps

The script performs the following validation steps:

1. **Python Environment Validation**
   - Python version check (≥3.8)
   - Virtual environment detection
   - Basic environment sanity checks

2. **Dependencies Validation**
   - Core package availability (numpy, scipy, matplotlib, etc.)
   - Version compatibility checks
   - Import validation

3. **LeeQ Installation Validation**
   - LeeQ package import
   - Core module availability
   - Component integration testing

4. **Environment Setup Validation**
   - Test setup file creation
   - Simulation environment configuration
   - Virtual qubit setup

5. **Chronicle Integration Validation**
   - Chronicle logging initialization
   - Log directory creation
   - Basic logging functionality

6. **Basic Experiment Workflow Validation**
   - TransmonElement creation
   - Setup configuration
   - Virtual qubit access
   - Parameter management

7. **Common Issues Check**
   - NumPy version compatibility
   - File permissions
   - Package conflicts

## Troubleshooting Integration

When validation fails, the script provides:

### Targeted Guidance
- **NumPy Issues**: Version downgrade instructions
- **Import Errors**: Installation and dependency guidance
- **Chronicle Issues**: File permission and integration help
- **Timeout Issues**: Performance optimization suggestions

### General Recovery Steps
1. Python interpreter restart
2. Module cache clearing
3. LeeQ reinstallation
4. GitHub issue reporting guidance
5. Verbose output recommendations

## Integration with Quick Start Guide

This script validates the exact steps documented in the Quick Start Guide:

1. **Installation Process**: Verifies `pip install git+https://github.com/ShuxiangCao/LeeQ`
2. **Environment Setup**: Tests creation of experiment setup files
3. **Simulation Backend**: Validates high-level simulation functionality
4. **Basic Experiments**: Confirms experiment execution capability

## Development Notes

### Extending Validation
To add new validation steps:

1. Create a new validation method following the pattern:
   ```python
   def validate_new_feature(self) -> Tuple[bool, str]:
       """Validate new feature functionality."""
       try:
           # Validation logic here
           return True, "Success message"
       except Exception as e:
           return False, f"Error message: {e}"
   ```

2. Add the step to `run_validation()` method
3. Update troubleshooting guidance if needed

### Testing the Validator
```bash
# Test script syntax
python -m py_compile scripts/test_quick_start.py

# Test help functionality
python scripts/test_quick_start.py --help

# Quick validation test
python scripts/test_quick_start.py --skip-install --timeout 60
```

## Example Output

### Successful Validation
```
============================================================
LeeQ Quick Start Guide Validation
============================================================
LeeQ Installation............. PASS
Environment Setup............. PASS
Chronicle Integration......... PASS
Basic Experiment Workflow..... PASS
Common Issues Check........... PASS
------------------------------------------------------------
Total: 5/5 passed in 6.4s
🎉 Quick Start Guide validation SUCCESSFUL!
You can proceed with LeeQ experiments.
```

### Failed Validation with Guidance
```
❌ Quick Start Guide validation FAILED!

=== TROUBLESHOOTING GUIDANCE ===

Import/Module Issue:
• Verify LeeQ installation: pip install git+https://github.com/ShuxiangCao/LeeQ
• Check dependencies: pip install -r requirements.txt
• Activate virtual environment if using one
• Try: python -c "import leeq; print('Success')"

General Recovery Steps:
1. Restart Python interpreter
2. Clear any cached modules
3. Reinstall LeeQ
4. Check GitHub issues
5. Run with verbose output
```

This validation script ensures users can reliably follow the Quick Start Guide and quickly identify and resolve any setup issues.