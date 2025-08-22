# Two-Tone Qubit Spectroscopy

## Overview

Two-tone spectroscopy is an advanced characterization technique that simultaneously applies two frequency-swept tones to a quantum system. This technique is essential for probing:

- **Multi-photon transitions**: Identifying higher-order excitation pathways
- **Sideband effects**: Characterizing modulation-induced sidebands
- **Qubit-resonator coupling**: Measuring dispersive shifts and coupling strengths
- **AC Stark shifts**: Quantifying power-dependent frequency shifts
- **Cross-Kerr interactions**: Studying inter-qubit coupling effects

## Physics Background

In two-tone spectroscopy, the system Hamiltonian includes two driving terms:

```
H = H_0 + Ω₁(t)cos(ω₁t)σ_x^(1) + Ω₂(t)cos(ω₂t)σ_x^(2)
```

Where:
- `H_0` is the unperturbed system Hamiltonian
- `Ω₁,₂` are the drive amplitudes
- `ω₁,₂` are the drive frequencies
- `σ_x^(1,2)` are the Pauli operators (may act on same or different transitions)

The two tones can interact through various mechanisms:
1. **Direct multi-photon processes**: ω₁ + ω₂ = ω_transition
2. **Parametric processes**: ω₁ - ω₂ = ω_modulation
3. **Cross-Kerr effects**: Simultaneous excitation modifies transition frequencies

## Usage

### Basic Two-Tone Spectroscopy

```python
from leeq.experiments.builtin.basic.calibrations.two_tone_spectroscopy import TwoToneQubitSpectroscopy

# Run two-tone spectroscopy with different channels
exp = TwoToneQubitSpectroscopy(
    dut_qubit=qubit,
    tone1_start=4950.0,  # MHz
    tone1_stop=5050.0,   # MHz
    tone1_step=2.0,      # MHz
    tone1_amp=0.1,       # Drive amplitude
    tone2_start=4750.0,  # MHz
    tone2_stop=4850.0,   # MHz
    tone2_step=2.0,      # MHz
    tone2_amp=0.1,       # Drive amplitude
    same_channel=False,  # Use different channels
    num_avs=1000,        # Number of averages
    mp_width=1.0         # Measurement pulse width (μs)
)

# Results are automatically plotted
# Access data programmatically
magnitude = exp.result['Magnitude']
phase = exp.result['Phase']
```

### Same Channel Mode

For superimposed tones on the same drive channel:

```python
exp = TwoToneQubitSpectroscopy(
    dut_qubit=qubit,
    tone1_start=4980.0,
    tone1_stop=5020.0,
    tone1_step=1.0,
    tone2_start=5000.0,
    tone2_stop=5040.0,
    tone2_step=1.0,
    same_channel=True,  # Superimpose on same channel
    tone1_amp=0.05,
    tone2_amp=0.05
)
```

### Analysis Methods

#### Peak Detection
```python
peaks = exp.find_peaks()
print(f"Peak at Tone1: {peaks['peak_freq1']} MHz")
print(f"Peak at Tone2: {peaks['peak_freq2']} MHz")
print(f"Peak magnitude: {peaks['peak_magnitude']}")
```

#### Cross-Sections
```python
# Extract 1D slice at peak frequency
cross_section = exp.get_cross_section(axis='freq1')
frequencies = cross_section['frequencies']
magnitude = cross_section['magnitude']

# Or at specific frequency
cross_section = exp.get_cross_section(axis='freq2', value=5000.0)
```

## Parameters

### Required Parameters
- `dut_qubit`: TransmonElement to perform spectroscopy on

### Tone 1 Parameters
- `tone1_start`: Start frequency in MHz (default: 4950.0)
- `tone1_stop`: Stop frequency in MHz (default: 5050.0)
- `tone1_step`: Frequency step size in MHz (default: 10.0)
- `tone1_amp`: Drive amplitude (default: 0.1)

### Tone 2 Parameters
- `tone2_start`: Start frequency in MHz (default: 4750.0)
- `tone2_stop`: Stop frequency in MHz (default: 4850.0)
- `tone2_step`: Frequency step size in MHz (default: 10.0)
- `tone2_amp`: Drive amplitude (default: 0.1)

### Configuration Parameters
- `same_channel`: If True, both tones on same channel (default: False)
- `num_avs`: Number of averages (default: 1000)
- `rep_rate`: Repetition rate in Hz (default: 0.0)
- `mp_width`: Measurement pulse width in μs (default: 1.0)
- `set_qubit`: Transition for tone 1 ('f01' or 'f12', default: 'f01')

## Implementation Details

### Pulse Sequence

The experiment constructs a pulse sequence with two drive pulses applied in parallel:

```python
lpb = (drive1 * drive2) + measurement
```

The `*` operator creates a `LogicalPrimitiveBlockParallel` ensuring simultaneous application.

### Sweep Engine

The 2D frequency sweep uses chained sweepers:

```python
swp = swp_freq1 + swp_freq2  # freq2 is inner loop
```

This creates an efficient nested sweep where:
- Outer loop: tone1 frequency
- Inner loop: tone2 frequency

### Simulation Mode

In simulation mode, the experiment uses `CWSpectroscopySimulator` which:
1. Calculates dressed states for each drive configuration
2. Includes crosstalk effects between channels
3. Simulates realistic IQ responses
4. Adds shot noise scaled by `num_avs`

## Visualization

The experiment automatically generates:

1. **Magnitude Heatmap**: 2D plot of response magnitude vs both frequencies
2. **Phase Heatmap**: Unwrapped phase response revealing dispersive effects

Both plots are automatically displayed when the experiment completes.

## Advanced Applications

### Identifying Multi-Photon Transitions
```python
# Look for sum/difference frequency transitions
exp = TwoToneQubitSpectroscopy(
    dut_qubit=qubit,
    tone1_start=2400.0,  # ~f01/2
    tone1_stop=2600.0,
    tone2_start=2400.0,  # ~f01/2
    tone2_stop=2600.0,
    same_channel=True
)
```

### AC Stark Shift Measurement
```python
# Fix one tone, sweep the other with varying power
stark_shifts = []
for amp in [0.01, 0.05, 0.1, 0.2]:
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=qubit,
        tone1_start=5000.0,
        tone1_stop=5000.0,  # Fixed frequency
        tone1_step=10.0,
        tone1_amp=amp,
        tone2_start=4900.0,
        tone2_stop=5100.0,
        tone2_step=2.0,
        tone2_amp=0.01  # Weak probe
    )
    peaks = exp.find_peaks()
    stark_shifts.append(peaks['peak_freq2'])
```

### Sideband Spectroscopy
```python
# Look for modulation sidebands
modulation_freq = 50.0  # MHz
exp = TwoToneQubitSpectroscopy(
    dut_qubit=qubit,
    tone1_start=5000.0 - modulation_freq - 10,
    tone1_stop=5000.0 - modulation_freq + 10,
    tone2_start=5000.0 + modulation_freq - 10,
    tone2_stop=5000.0 + modulation_freq + 10,
    same_channel=False
)
```

## Troubleshooting

### No Clear Resonances
- Increase `num_avs` for better SNR
- Reduce frequency step size for finer resolution
- Check amplitude levels - too low won't excite, too high causes power broadening

### Unexpected Peak Locations
- Verify qubit frequency calibration
- Check for AC Stark shifts at high power
- Consider crosstalk if using same channel

### Asymmetric Response
- May indicate nonlinear effects
- Check for heating (reduce rep_rate)
- Verify measurement settings

## Related Experiments
- `QubitSpectroscopyFrequency`: Single-tone frequency sweep
- `QubitSpectroscopyAmplitudeFrequency`: 2D amplitude-frequency sweep
- `ResonatorSpectroscopy`: Cavity mode characterization

## References
1. Blais et al., "Cavity quantum electrodynamics for superconducting electrical circuits" (2004)
2. Schuster et al., "AC Stark shift and dephasing of a superconducting qubit" (2005)
3. Wallraff et al., "Approaching unit visibility for control of a superconducting qubit" (2005)