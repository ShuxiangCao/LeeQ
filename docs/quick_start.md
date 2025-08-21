# Quick Start Guide: Your First Quantum Experiment in 10 Minutes

Get LeeQ running and execute your first quantum experiment using simulation - no hardware required! This guide will have you running a Rabi oscillation experiment in under 10 minutes.

## Step 1: Installation (2 minutes)

### Install LeeQ
```bash
pip install git+https://github.com/ShuxiangCao/LeeQ
```

### Verify Installation
```bash
python -c "import leeq; print('LeeQ installed successfully!')"
```

If this command runs without errors, you're ready to proceed!

## Step 2: Set Up Simulation Environment (3 minutes)

Create a new Python file called `quick_start_experiment.py` and add this simulation setup:

```python
# quick_start_experiment.py - Your first LeeQ experiment
import numpy as np
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.experiments.builtin.basic.calibrations.rabi import NormalisedRabi
from leeq.chronicle import Chronicle

def setup_simulation():
    """Set up a simulated quantum system with one qubit."""
    
    # Start experiment logging
    Chronicle().start_log()
    
    # Create experiment manager and clear any existing setups
    manager = ExperimentManager()
    manager.clear_setups()
    
    # Create a virtual qubit with realistic parameters
    virtual_qubit = VirtualTransmon(
        name="QuickStartQubit",
        qubit_frequency=5040.4,  # MHz
        anharmonicity=-198,      # MHz
        t1=70,                   # microseconds
        t2=35,                   # microseconds
        readout_frequency=9645.4, # MHz
        quiescent_state_distribution=np.array([0.8, 0.15, 0.04, 0.01])
    )
    
    # Create simulation setup
    setup = HighLevelSimulationSetup(
        name='QuickStartSimulation',
        virtual_qubits={0: virtual_qubit}  # Channel 0
    )
    
    # Register the setup
    manager.register_setup(setup)
    return manager

# Qubit configuration for our simulated system
qubit_config = {
    'hrid': 'Q0',
    'lpb_collections': {
        'f01': {  # 0->1 transition
            'type': 'SimpleDriveCollection',
            'freq': 5040.4,      # Match qubit frequency
            'channel': 0,
            'shape': 'blackman_drag',
            'amp': 0.1,          # Start with small amplitude
            'phase': 0.0,
            'width': 0.05,       # 50 ns pulse
            'alpha': 500,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9645.4,     # Match readout frequency
            'channel': 1,
            'shape': 'square',
            'amp': 0.15,
            'phase': 0.0,
            'width': 1.0,       # 1 us readout
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

if __name__ == "__main__":
    # Initialize the simulation
    manager = setup_simulation()
    print("Simulation environment ready!")
```

## Step 3: Run Your First Quantum Experiment (3 minutes)

Now add the experiment code to your `quick_start_experiment.py` file:

```python
# Add this to the end of quick_start_experiment.py

def run_rabi_experiment():
    """Run a Rabi oscillation experiment to calibrate qubit drive amplitude."""
    
    # Initialize simulation
    manager = setup_simulation()
    
    # Create the qubit element
    qubit = TransmonElement(name=qubit_config['hrid'], parameters=qubit_config)
    
    # Configure experiment parameters
    from leeq.experiments import setup
    setup().status().set_param("Shot_Number", 1000)
    setup().status().set_param("Sampling_Noise", True)  # Add realistic noise
    
    print("Running Rabi oscillation experiment...")
    print("This will sweep pulse duration to find optimal drive parameters.\n")
    
    # Create and run the Rabi experiment (experiment runs automatically)
    rabi_exp = NormalisedRabi(
        dut_qubit=qubit,
        amp=0.05,          # Drive amplitude
        start=0.01,        # Start time (µs)
        stop=0.3,          # Stop time (µs) 
        step=0.005,        # Time step (µs)
        fit=True,          # Fit oscillations
        update=True        # Update qubit parameters
    )
    
    # Show results
    print("Experiment completed!")
    print(f"Fitted frequency: {rabi_exp.fit_params['Frequency']:.3f} MHz")
    print(f"Oscillation amplitude: {rabi_exp.fit_params['Amplitude']:.3f}")
    print(f"Suggested drive amplitude: {rabi_exp.guess_amp:.3f}")
    
    # Plot the results
    try:
        fig = rabi_exp.plot()
        fig.show()
        print("\nPlot displayed! You should see Rabi oscillations.")
    except Exception as e:
        print(f"Plotting requires a display. Results saved to data.")
    
    return rabi_exp

if __name__ == "__main__":
    # Run the complete experiment
    experiment_result = run_rabi_experiment()
```
## Step 4: Execute Your Experiment (2 minutes)

Run your first quantum experiment:

```bash
python quick_start_experiment.py
```

**Expected Output:**
```
Simulation environment ready!
Running Rabi oscillation experiment...
This will sweep pulse duration to find optimal drive parameters.

Experiment completed!
Fitted frequency: 3.125 MHz
Oscillation amplitude: 0.856
Suggested drive amplitude: 0.160
Amplitude updated: 0.160

Plot displayed! You should see Rabi oscillations.
```

## What You Just Accomplished

Congratulations! You just:

1. **Set up a quantum simulation environment** - Created a virtual transmon qubit with realistic parameters
2. **Ran a Rabi oscillation experiment** - Swept pulse duration to observe quantum oscillations
3. **Automatically calibrated qubit parameters** - Found optimal drive amplitude for π pulses
4. **Analyzed quantum data** - Fitted oscillations and extracted meaningful parameters

The Rabi experiment you ran is fundamental to quantum computing - it demonstrates coherent control of a qubit state and is used to calibrate the strength of quantum gates.

## Troubleshooting

### Installation Issues

**Problem**: `ImportError: No module named 'leeq'`
```bash
# Solution: Install in development mode
pip install -e git+https://github.com/ShuxiangCao/LeeQ#egg=leeq
```

**Problem**: `ModuleNotFoundError: No module named 'plotly'`
```bash
# Solution: Install plotting dependencies
pip install plotly kaleido
```

### Runtime Issues

**Problem**: `KeyError: 'Frequency'` in fit results
- **Cause**: Insufficient oscillations in the data
- **Solution**: Increase the stop time or decrease step size in the Rabi experiment

**Problem**: No plot displayed
- **Cause**: Running in environment without display
- **Solution**: Save plot to file instead:
```python
fig = rabi_exp.plot()
fig.write_html("rabi_results.html")
print("Plot saved to rabi_results.html")
```

**Problem**: "No virtual qubit found" error
- **Cause**: Setup not properly registered
- **Solution**: Ensure `manager.register_setup(setup)` is called

### Getting Help

If you encounter issues:
1. Check the [Issues page](https://github.com/ShuxiangCao/LeeQ/issues) for similar problems
2. Verify all dependencies are installed: `pip list | grep -E "(numpy|scipy|plotly)"`
3. Follow the [comprehensive tutorial](tutorial.md) for more detailed explanations

## Next Steps

Ready to explore more? Here's your learning path:

### Immediate Next Steps (15-30 minutes)
- **Tutorial**: Read the [complete tutorial](tutorial.md) for deeper understanding
- **Interactive Learning**: Work through the [tutorial](tutorial.md) for hands-on practice
- **More Experiments**: Explore T1, T2, and spectroscopy experiments

### Intermediate Learning (1-2 hours)
- **Multi-qubit Systems**: Learn about two-qubit gates and entanglement
- **Real Hardware**: Connect to actual quantum hardware when available
- **Custom Experiments**: Build your own experiment sequences

### Advanced Features (2+ hours)
- **AI Integration**: Use LeeQ's AI agents for automated calibration
- **Data Analysis**: Master the Chronicle logging and analysis tools
- **Hardware Integration**: Connect to QubiC or other quantum control systems

### Recommended Learning Order
1. **Start Here**: [Complete Tutorial](tutorial.md) - Build on what you learned
2. **Core Concepts**: [User Guide](guide/concepts.md) - Understand LeeQ architecture
3. **Experiments Guide**: [Experiments](guide/experiments.md) - Learn about available experiments
4. **Documentation**: [API Reference](api/core/base.md) - Detailed technical documentation

## Environment Configuration (Optional)

For persistent settings, you can set these environment variables:

```bash
# Optional: Set custom data directories
export LAB_CHRONICLE_LOG_DIR="/path/to/experiment/logs"
export LEEQ_CALIBRATION_LOG_PATH="/path/to/calibration/logs"
```

If not set, LeeQ creates these directories in your working folder automatically.

---

**Congratulations on running your first quantum experiment with LeeQ!** 

You're now ready to dive deeper into quantum computing with a powerful, flexible framework at your fingertips.
