## Orchestrating Pulses

LeeQ employs a tree structure for scheduling rather than a predefined schedule, introducing two key concepts:

- **Logical Primitive (LP):** The basic operation, typically a single pulse or delay, serving as a tree's leaf.

- **Logical Primitive Block (LPB):** A composite element within the tree,
  including `LogicalPrimitiveBlockSeries`, `LogicalPrimitiveBlockParallel`, and `LogicalPrimitiveBlockSweep`.

`LogicalPrimitiveBlockSeries` signifies sequential execution of its children. It can be constructed using the `+`
operator to combine LPs or LPBs.

`LogicalPrimitiveBlockParallel` indicates simultaneous start times for its children, created using the `*` operator.

`LogicalPrimitiveBlockSweep` pairs with a `Sweeper` for dynamic pulse adjustments during a sequence sweep.

Example:

```python
lpb_1 = LogicalPrimitiveBlockSeries([lp_1, lp_2, lp_3])

lpb_2 = LogicalPrimitiveParallel([lpb_1, lp_4])  # Mixing LPBs and LPs

lpb_3 = lpb_1 + lpb_2  # Series combination

lpb_4 = lpb_1 * lpb_2  # Parallel combination
```

### Single Qubit Operations

Single qubit gates are accessible through the DUT object's collection, which organizes the operations by subspace. For
instance:

```python
dut = duts_dict['Q1']
c1 = dut.get_c1('f01')  # Access the single qubit drive collection for subspace 0,1
lp = c1['X']  # X gate
lp = c1['Y']  # Y gate
lp = c1['Yp']  # +pi/2 Y gate
lp = c1['Ym']  # -pi/2 Y gate
```

Shortcut methods are available for composite gates, like:

```python
gate = dut.get_gate('qutrit_hadamard')
```

The returned object, typically an LPB, consists of a sequence of gates. Detailed documentation is available
for `get_gate`.

To access measurement primitives:

```python
mprim = dut.get_measurement_prim_intlist(name='0')
```

`get_measurement_prim_intlist` offers single-shot, demodulated, and aggregated readouts, among other options detailed in
the documentation.

Delay operations are available through the `DelayPrimitive` class to create a lpb.

```
delay = DelayPrimitive({'delay_time':1})  # 1us delay
```

## How to Use the Sweeper Classes to Sweep Parameters

The code provided defines a framework for performing parameter sweeps with side effects, which can be highly useful in
simulations, hardware testing, or any scenario where changing a parameter needs to invoke a specific action. Below is a
tutorial on how to utilize these classes effectively.

#### Overview of Classes

- **SweepParametersSideEffect**: Base class for creating side effects.
- **SweepParametersSideEffectFunction**: Derived class for function call-based side effects.
- **SweepParametersSideEffectAttribute**: Derived class for attribute setting-based side effects.
- **SweepParametersSideEffectFactory**: Factory class for creating side effect objects.
- **Sweeper**: Class that handles the logic of sweeping through parameters and applying side effects.

#### Step 1: Define Your Side Effect Functions or Attributes

First, you need to define the functions or object attributes that will be altered during the sweep.

**Example Function:**

```python
def update_frequency(frequency):
    print(f"Frequency set to {frequency} Hz")
```

**Example Object and Attribute:**

```python
class Oscillator:
    def __init__(self):
        self.frequency = 0


oscillator = Oscillator()
```

#### Step 2: Create Side Effect Objects

Using the `SweepParametersSideEffectFactory`, create side effect objects that wrap your function or attribute.

**For Function:**

```python
function_side_effect = SweepParametersSideEffectFactory.from_function(
    function=update_frequency,
    argument_name='frequency'
)
```

**For Attribute:**

```python
attribute_side_effect = SweepParametersSideEffectFactory.from_attribute(
    object_instance=oscillator,
    attribute_name='frequency'
)
```

#### Step 3: Define Sweep Parameters

Determine the parameters over which you want to perform the sweep. This can be any iterable or a callable that generates
parameters.

**Example:**

```python
frequencies = [1000, 2000, 3000]  # Hz
```

#### Step 4: Create the Sweeper

Instantiate a `Sweeper` object with the sweep parameters and the side effect objects created in step 2.

```python
sweeper = Sweeper(
    sweep_parameters=frequencies,
    params=[function_side_effect, attribute_side_effect]
)
```

This will print out the frequency set by both the function call and the attribute setting, demonstrating how
the `Sweeper` applies both types of side effects.

#### Advanced Usage: Chaining Sweepers

You can chain multiple sweepers to perform complex nested sweeps.

**Example:**

```python
# Define another sweeper for a different parameter
amplitudes = [0.1, 0.5, 1.0]  # Example amplitude values

# Assume update_amplitude is a defined function
amplitude_side_effect = SweepParametersSideEffectFactory.from_function(
    function=update_amplitude,
    argument_name='amplitude'
)

amplitude_sweeper = Sweeper(
    sweep_parameters=amplitudes,
    params=[amplitude_side_effect]
)

# Chain sweepers
combined_sweeper = sweeper + amplitude_sweeper
```

Full Examples:

```python
    def run(self,
            dut_qubit: Any,
            amp: float = 0.05,
            start: float = 0.01,
            stop: float = 0.3,
            step: float = 0.002,
            fit: bool = True,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            pulse_discretization: bool = True,
            update=True,
            initial_lpb: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """
    Run a Rabi experiment on a given qubit and analyze the results.

    Parameters:
    dut_qubit (Any): Device under test (DUT) qubit object.
    amp (float): Amplitude of the Rabi pulse. Default is 0.05.
    start (float): Start width for the pulse width sweep. Default is 0.01.
    stop (float): Stop width for the pulse width sweep. Default is 0.15.
    step (float): Step width for the pulse width sweep. Default is 0.001.
    fit (bool): Whether to fit the resulting data to a sinusoidal function. Default is True.
    collection_name (str): Collection name for retrieving c1. Default is 'f01'.
    mprim_index (int): Index for retrieving measurement primitive. Default is 0.
    pulse_discretization (bool): Whether to discretize the pulse. Default is False.
    update (bool): Whether to update the qubit parameters If you are tuning up the qubit set it to True. Default is False.
    initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.

    Returns:
    Dict[str, Any]: Fitted parameters if fit is True, None otherwise.

    Example:
        >>> # Run an experiment to calibrate the driving amplitude of a single qubit gate
        >>> rabi_experiment = NormalisedRabi(
        >>> dut_qubit=dut, amp=0.05, start=0.01, stop=0.3, step=0.002, fit=True,
        >>> collection_name='f01', mprim_index=0, pulse_discretization=True, update=True)
    """
    # Get c1 from the DUT qubit
    c1 = dut_qubit.get_c1(collection_name)
    rabi_pulse = c1['X'].clone()

    if amp is not None:
        rabi_pulse.update_pulse_args(
            amp=amp, phase=0., shape='square', width=step)
    else:
        amp = rabi_pulse.amp

    if not pulse_discretization:
        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            rabi_pulse.update_pulse_args, {}, 'width'
        )]
        swp = Sweeper(
            np.arange,
            n_kwargs={'start': start, 'stop': stop, 'step': step},
            params=swpparams
        )
        pulse = rabi_pulse
    else:
        # Sometimes it is expensive to update the pulse envelope everytime, so we can keep the envelope the same
        # and just change the number of pulses
        pulse = LogicalPrimitiveBlockSweep([prims.SerialLPB(
            [rabi_pulse] * k, name='rabi_pulse') for k in range(int((stop - start) / step + 0.5))])
        swp = Sweeper.from_sweep_lpb(pulse)

    # Get the measurement primitive
    mprim = dut_qubit.get_measurement_prim_intlist(mprim_index)
    self.mp = mprim

    # Create the loopback pulse (lpb)
    lpb = pulse + mprim

    if initial_lpb is not None:
        lpb = initial_lpb + lpb

    # Run the basic experiment
    basic(lpb, swp, '<z>')

    # Extract the data
    self.data = np.squeeze(mprim.result())

    if not fit:
        return None

    # Fit data to a sinusoidal function and return the fit parameters
    self.fit_params = self.fit_sinusoidal(self.data, time_step=step)

    # Update the qubit parameters, to make one pulse width correspond to a pi pulse
    # Here we suppose all pulse envelopes give unit area when width=1,
    # amp=1
    normalised_pulse_area = c1['X'].calculate_envelope_area() / c1['X'].amp
    two_pi_area = amp * (1 / self.fit_params['Frequency'])
    new_amp = two_pi_area / 2 / normalised_pulse_area
    self.guess_amp = new_amp

    if update:
        c1.update_parameters(amp=new_amp)
        print(f"Amplitude updated: {new_amp}")

```