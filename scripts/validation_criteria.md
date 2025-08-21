# LeeQ Notebook Validation Criteria

## Overview
This document defines the validation criteria and pass/fail thresholds for LeeQ documentation notebooks.

## Validation Types

### 1. Static Validation
**Purpose**: Validates notebook structure and syntax without execution
**Tools**: `validate_notebooks.py`

**Checks**:
- JSON syntax validity
- Required notebook structure (cells, metadata, etc.)
- Import patterns (LeeQ, Chronicle, numpy, etc.)
- Best practices compliance
- Code structure and formatting

**Pass Criteria**: 
- All JSON syntax checks pass
- Required structural elements present
- At least one LeeQ import detected
- No critical structural issues

### 2. Execution Validation
**Purpose**: Validates that notebooks execute without errors
**Tools**: `validate_notebooks.py --execute`

**Checks**:
- Notebook executes completely within timeout
- No cell execution errors
- All imports resolve successfully
- Simulation setup completes

**Pass Criteria**:
- Execution completes within timeout (default: 300s)
- No Python exceptions or errors in any cell
- All required imports available

### 3. Output Validation
**Purpose**: Validates quantum experiment results and patterns
**Tools**: `check_outputs.py`

**Quantum Experiment Checks**:
- **Oscillations**: Rabi frequency oscillations with >3 cycles
- **Decay**: T1/T2 exponential decay patterns
- **Entanglement**: Bell states, two-qubit gates, entanglement measures
- **Persistence**: Chronicle logging, parameter saving/loading

**Pass Criteria**:
- At least 50% of relevant checks pass per notebook
- Time constants in reasonable range (10-1000 μs)
- Visualization outputs present
- Fit quality indicators when available

## Suite-Level Thresholds

### Critical Thresholds (Must Pass)
```python
'static_validation_rate': 0.95,      # 95% of notebooks pass static validation
'execution_success_rate': 0.90,      # 90% of notebooks execute successfully
'max_execution_time': 600,           # No notebook >10 minutes execution
```

### Quality Thresholds (Should Pass)
```python
'average_execution_time': 300,       # Average execution <5 minutes
'output_validation_rate': 0.80,      # 80% of output checks pass
```

## Category-Specific Criteria

### Tutorial Notebooks (01-05)
**Requirements**:
- All must execute successfully (100% execution rate)
- Progressive complexity validation
- Chronicle logging demonstration
- Clear visualizations in each notebook
- Execution time <5 minutes per notebook

**Specific Validations**:
- `01_basics.ipynb`: Chronicle setup, basic measurement
- `02_single_qubit.ipynb`: Rabi oscillations, T1/T2 decay
- `03_multi_qubit.ipynb`: Entanglement, crosstalk analysis
- `04_calibration.ipynb`: Parameter persistence
- `05_ai_integration.ipynb`: AI experiment suggestions

### Example Notebooks
**Requirements**:
- 90% execution success rate acceptable
- Specialized experiment patterns validated
- Performance optimization for <5 minute execution

**Specific Validations**:
- `rabi_experiments.ipynb`: Multiple oscillation patterns
- `t1_t2_measurements.ipynb`: Decay fitting validation
- `tomography.ipynb`: State fidelity analysis
- `randomized_benchmarking.ipynb`: Benchmarking metrics
- `custom_experiments.ipynb`: Framework validation

### Workflow Notebooks
**Requirements**:
- 95% execution success rate (critical for automation)
- Parameter persistence validation
- Error handling and recovery
- Performance <10 minutes per workflow

**Specific Validations**:
- `daily_calibration.ipynb`: Automated routine validation
- `qubit_characterization.ipynb`: Complete characterization
- `two_qubit_calibration.ipynb`: Multi-qubit procedures
- `experiment_analysis.ipynb`: Data analysis validation

## Pass/Fail Classification

### PASS - Suite Level
- Overall success rate ≥90%
- All critical thresholds met
- No execution timeouts >10 minutes
- Static validation rate ≥95%

### PARTIAL PASS - Suite Level
- Overall success rate ≥80% but <90%
- Some quality thresholds missed
- Minor execution issues
- Recoverable validation failures

### FAIL - Suite Level
- Overall success rate <80%
- Critical thresholds not met
- Multiple execution failures
- Structural or import problems

### Individual Notebook Classification
**PASS**: All validation types pass
**RECOVERABLE FAIL**: Execution or output validation fails, but static passes
**NON-RECOVERABLE FAIL**: Static validation fails (syntax, structure, imports)

## Usage Examples

```bash
# Quick static validation
python scripts/validate_notebooks.py docs/notebooks/tutorials/01_basics.ipynb

# Comprehensive validation with execution
python scripts/validate_notebooks.py --comprehensive docs/notebooks/tutorials/

# Output validation only
python scripts/check_outputs.py --comprehensive docs/notebooks/examples/rabi_experiments.ipynb

# Full test suite
python scripts/notebook_test_runner.py --all --generate-report

# Performance benchmarking
python scripts/notebook_test_runner.py --all --performance-check --timeout 600
```

## Validation Commands for Phase 1

As specified in the implementation plan, Phase 1 validation commands are:

```bash
# Check shared utilities work (when created)
python -c "from docs.notebooks.shared.setup_templates import get_standard_setup; print('✓')"
python -c "from docs.notebooks.shared.experiment_helpers import *; print('✓')"

# Execute core tutorials
jupyter nbconvert --execute --to notebook docs/notebooks/tutorials/01_basics.ipynb
jupyter nbconvert --execute --to notebook docs/notebooks/tutorials/02_single_qubit.ipynb

# Validate content quality
python scripts/validate_notebooks.py docs/notebooks/tutorials/01_basics.ipynb
python scripts/validate_notebooks.py docs/notebooks/tutorials/02_single_qubit.ipynb
```

## Error Recovery Guidelines

### Execution Timeouts
1. Check for infinite loops or blocking operations
2. Reduce simulation complexity or data points
3. Optimize expensive operations
4. Consider increasing timeout for complex workflows

### Import Errors
1. Verify LeeQ installation and version
2. Check for missing dependencies
3. Add graceful fallbacks for optional imports
4. Document required packages

### Output Validation Failures
1. Check if experiments produce expected patterns
2. Verify parameter ranges are realistic
3. Ensure visualizations are generated
4. Validate fitting procedures converge

### Static Validation Issues
1. Fix JSON syntax errors
2. Add required notebook metadata
3. Ensure proper markdown structure
4. Address best practice violations

## Continuous Integration Integration

The validation infrastructure supports CI/CD integration:

```yaml
# Example GitHub Actions integration
- name: Validate Notebooks
  run: |
    python scripts/notebook_test_runner.py --all --static-only --generate-report
    python scripts/validate_notebooks.py --execute --timeout 600 docs/notebooks/tutorials/
```

Exit codes:
- `0`: Success (≥90% pass rate)
- `1`: Failure (<80% pass rate) 
- `2`: Partial success (80-90% pass rate)

## Reporting

All validation tools support detailed JSON reporting for analysis:
- Individual test results
- Performance metrics
- Threshold compliance
- Failed notebook details
- Execution statistics

Reports are saved as `notebook_validation_report.json` or `notebook_test_report.json`.