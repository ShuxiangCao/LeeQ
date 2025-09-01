# LeeQ Experiment Documentation Plan

## Objective

Update documentation for all 85 LeeQ experiment classes to ensure consistency and completeness:
1. Align `run()` and `run_simulated()` method documentation
2. Add `EPII_INFO` static variable to each experiment class with structured information

## Task 1: Synchronize run/run_simulated Documentation

### Goal
Ensure `run()` and `run_simulated()` methods have matching, comprehensive docstrings.

### Documentation Standard
```python
def run(self, dut_qubit, start=4900.0, stop=5100.0, step=2.0, num_avs=1000, ...):
    """
    Execute the experiment on hardware.
    
    Parameters
    ----------
    dut_qubit : Any
        The device under test (qubit object).
    start : float, optional
        Start frequency for the sweep (MHz). Default: 4900.0
    stop : float, optional
        Stop frequency for the sweep (MHz). Default: 5100.0
    step : float, optional
        Frequency increment (MHz). Default: 2.0
    num_avs : int, optional
        Number of averages. Default: 1000
        
    Returns
    -------
    dict or None
        Experiment results if available.
    """

def run_simulated(self, dut_qubit, start=4900.0, stop=5100.0, step=2.0, num_avs=1000, ...):
    """
    Execute the experiment in simulation mode.
    
    Parameters
    ----------
    [MUST MATCH run() parameters exactly]
    
    Returns
    -------
    dict or None
        Simulated experiment results if available.
    """
```

### Implementation Steps
1. Parse each experiment class to extract parameter signatures
2. Compare run() vs run_simulated() parameters
3. Generate consistent docstrings for both methods
4. Apply updates to source files

## Task 2: Add EPII_INFO Static Variable

### Goal
Add structured metadata to each experiment class as a static variable.

### EPII_INFO Structure
```python
class ExperimentName(Experiment):
    EPII_INFO = {
        "name": "ExperimentName",
        "description": "Brief one-line description of the experiment",
        "purpose": "Detailed explanation of what the experiment does and when to use it",
        "attributes": {
            "mp": {
                "type": "MeasurementPrimitive",
                "description": "The measurement primitive used"
            },
            "trace": {
                "type": "np.ndarray[complex]",
                "description": "Raw IQ trace data",
                "shape": "(n_frequency_points,)"
            },
            "result": {
                "type": "dict",
                "description": "Processed results",
                "keys": {
                    "Magnitude": "np.ndarray[float] - Magnitude of IQ response",
                    "Phase": "np.ndarray[float] - Unwrapped phase"
                }
            },
            "freq_arr": {
                "type": "np.ndarray[float]",
                "description": "Frequency array (MHz)",
                "shape": "(n_frequency_points,)"
            }
        },
        "notes": [
            "The frequency_guess uses first 10 points as baseline",
            "Phase is automatically unwrapped",
            "In simulation, disable_noise=True provides deterministic results"
        ]
    }
```

### Implementation Steps
1. Analyze each experiment class to identify all attributes
2. Determine attribute types
3. Generate EPII_INFO dictionary
4. Add as first class attribute after class declaration

## Experiment Priority List

### Tier 1: Core Calibrations (25 experiments)
**Immediate focus - Most frequently used**

1. **Qubit Spectroscopy** (5 experiments)
   - QubitSpectroscopyFrequency
   - QubitSpectroscopyAmplitudeFrequency
   - TwoToneQubitSpectroscopy
   - QubitSpectroscopyPower
   - QubitSpectroscopyMultiFrequency

2. **Rabi Experiments** (4 experiments)
   - NormalisedRabi
   - PowerRabi
   - MultiQubitRabi
   - DrivingRabi

3. **Ramsey Experiments** (3 experiments)
   - RamseyFrequency
   - RamseyAmplitude
   - RamseyDetuning

4. **T1/T2 Experiments** (4 experiments)
   - T1Experiment
   - T2EchoExperiment
   - T2RamseyExperiment
   - MultiQubitT1

5. **Resonator Spectroscopy** (3 experiments)
   - ResonatorSweepTransmission
   - ResonatorSweepTransmissionWithExtraInitialLPB
   - ResonatorPower

6. **Basic Calibrations** (6 experiments)
   - PiCalibration
   - AmplitudeCalibration
   - FrequencyCalibration
   - PhaseCalibration
   - ReadoutCalibration
   - StateTomography

### Tier 2: Advanced Calibrations (30 experiments)
**Secondary priority - Specialized but common**

- DRAG calibrations
- PingPong experiments
- State discrimination
- Randomized benchmarking
- Cross-resonance experiments
- Stark shift measurements

### Tier 3: Specialized Features (30 experiments)
**Lower priority - Research-specific**

- Hamiltonian tomography
- Optimal control experiments
- Advanced multi-qubit gates
- Process tomography
- Quantum volume

## Automation Script Structure

```python
# epii_documentation_generator.py

class ExperimentDocumentationGenerator:
    def __init__(self):
        self.experiments = self.discover_all_experiments()
    
    def process_experiment(self, exp_class):
        """Process single experiment class."""
        # 1. Extract and align run/run_simulated docs
        self.synchronize_method_docs(exp_class)
        
        # 2. Generate EPII_INFO
        epii_info = self.generate_epii_info(exp_class)
        
        # 3. Update source file
        self.update_source_file(exp_class, epii_info)
    
    def synchronize_method_docs(self, exp_class):
        """Ensure run and run_simulated have matching docs."""
        run_params = self.extract_parameters(exp_class, 'run')
        run_sim_params = self.extract_parameters(exp_class, 'run_simulated')
        
        # Generate consistent docstrings
        run_doc = self.generate_docstring(run_params, "Execute on hardware")
        run_sim_doc = self.generate_docstring(run_sim_params, "Execute in simulation")
        
        return run_doc, run_sim_doc
    
    def generate_epii_info(self, exp_class):
        """Create EPII_INFO dictionary."""
        return {
            "name": exp_class.__name__,
            "description": self.extract_brief_description(exp_class),
            "purpose": self.extract_detailed_purpose(exp_class),
            "attributes": self.extract_attributes(exp_class),
            "notes": self.extract_notes(exp_class)
        }
    
    def extract_attributes(self, exp_class):
        """Analyze class to find all attributes."""
        attributes = {}
        source = inspect.getsource(exp_class)
        
        # Parse for self.* assignments
        # Determine types from context
        
        return attributes
```

## Validation Checklist

For each experiment class, verify:

- [ ] run() method has complete parameter documentation
- [ ] run_simulated() documentation matches run()
- [ ] All parameters have type hints and descriptions
- [ ] EPII_INFO contains all required fields
- [ ] All instance attributes are documented in EPII_INFO
- [ ] Attribute types and shapes are specified
- [ ] Notes section includes important usage information

## Timeline

### Week 1: Foundation & Tier 1
- **Day 1-2**: Build automation tools and process first 5 experiments manually as templates
- **Day 3-4**: Process remaining Tier 1 experiments (20 experiments)
- **Day 5**: Review and validate Tier 1 documentation

### Week 2: Tier 2 & Automation
- **Day 6-8**: Process Tier 2 experiments with automation (30 experiments)
- **Day 9-10**: Quality review and fixes

### Week 3: Completion
- **Day 11-13**: Process Tier 3 experiments (30 experiments)
- **Day 14**: Final validation and testing
- **Day 15**: Delivery and integration

## Success Metrics

1. **Documentation Consistency**: 100% of experiments have matching run/run_simulated docs
2. **EPII_INFO Coverage**: All 85 experiments have complete EPII_INFO
3. **Attribute Documentation**: Every instance attribute is documented with type
4. **Validation Pass Rate**: All experiments pass automated validation checks

## Deliverable

Modified source files for all 85 LeeQ experiment classes with:
1. Synchronized run/run_simulated documentation
2. EPII_INFO static variable with complete metadata (name, description, purpose, attributes with types, notes)
3. Validation report confirming documentation completeness