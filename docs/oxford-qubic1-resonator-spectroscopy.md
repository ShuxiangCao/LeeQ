# Oxford QubiC1 Resonator Spectroscopy Plan

Status: executed 2026-09-09. A three-point probe, a 501-point forward sweep, a
501-point reverse sweep, and a 501-point serial sweep completed on QubiC1.

## Candidate-window follow-up (2026-09-10)

Twenty-two requested candidate centers were each scanned over a 40 MHz window
using the same initialized Oxford QubiC1 setup, existing Q2/channel-3 DUT,
`ResonatorSweepTransmissionWithExtraInitialLPB`, amplitude 0.02, width 8 us,
800 points (0.05 MHz spacing), and 500 averages. No convincing resonator was
resolved in any requested window.

The fine grid reveals that the sharp features form a 6.25 MHz instrumental
comb. The strongest sustained anomaly in every window falls within 0.1 MHz of
that comb. Several off-grid requested centers snap to it—for example, the 9374,
9149, 9722, and 8937 MHz windows peak near 9375, 9150, 9725, and 8950 MHz—not at
their requested centers. The 9000 and 9500 MHz responses show symmetric
ringing around exact comb points rather than a localized resonator line shape.
After removing a smooth 5 MHz complex-I/Q background, no repeatable off-comb
notch with a corresponding phase rotation is visible.

Review classification for all requested centers is therefore **no convincing
resonator detected**: 9937.5, 9900, 9875, 9850, 9812.5, 9800, 9760, 9722,
9625, 9600, 9562, 9550, 9500, 9374, 9149, 9124, 9100, 9000, 8937, 8900,
8875, and 8625 MHz.

Evidence:

- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260910T010619Z-candidate-window-scans/`
- `all-candidate-windows-normalized.png` is the 22-window comparison image.
- Each `center-*-MHz.png` is an individual raw magnitude/phase/gradient image.
- Each matching `.npz` retains the physical/programmed grids and complex I/Q.
- `analysis-metrics.json` records the strongest sustained feature and distance
  to the 6.25 MHz comb for each window.

The same 22-window acquisition was repeated at normalized amplitude 0.2 after
a separate 9500 MHz saturation guard. The guard and full run contained 800
finite, distinct complex-I/Q values per window with no flat-topping. Relative
to amplitude 0.02, the median `|IQ|` response scaled by 9.965 and the median
normalized-magnitude correlation was 0.987. All 22 strongest features still
coincide with the 6.25 MHz comb, and no new convincing resonator appeared.

Amplitude-0.2 evidence:

- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260910T013933Z-amp-0p2-candidate-window-scans/` (9500 MHz guard)
- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260910T014027Z-amp-0p2-candidate-window-scans/` (complete 22-window run)
- `amplitude-comparison-normalized.png` overlays the amplitude-0.02 and
  amplitude-0.2 normalized magnitude responses.

## Result

No convincing resonator was resolved between 9.000 and 9.500 GHz at 1 MHz
spacing with the existing Oxford Q2/channel-3 measurement primitive, amplitude
0.02, width 8 us, and 100 averages.

The response is dominated by a smooth approximately 136 ns electrical-delay
rotation and an approximately 50 MHz standing-wave ripple. Sharp single-point
outliers concentrate on a 20/25/50 MHz comb and vary substantially in height,
so they are acquisition/DDS artifacts rather than credible resonator calls.
The historical 9386.8 MHz location is smooth and repeatable across forward,
reverse, and serial acquisition; no localized dip or phase feature is visible
there. Forward/reverse magnitude correlation is 0.933 and their median relative
difference is 2.4% away from the endpoints.

Evidence:

- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260909T191503Z-coarse-resonator-spectroscopy/`
- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260909T191638Z-coarse-reverse-resonator-spectroscopy/`
- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260909T191803Z-coarse-serial-resonator-spectroscopy/`
- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260909-repeat-comparison.png`
- `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260909-repeat-analysis.json`

The sweep did not load an overlay, restart the RPC server, change clocks or
Nyquist zones, invoke JTAG, or write detected frequencies into calibration.

## Target and transport

| Item | Value |
|---|---|
| Logical board name | `QubiC1` |
| Board alias | `qubit80_1` |
| Role | Oxford board connected to the real fridge |
| SSH endpoint | `127.0.0.1:2222` |
| Existing RPC forward | `http://127.0.0.1:19095` |
| Board RPC destination | `192.168.137.80:9095` |
| Firmware named by PYNQ state | `024756ff` |

The `19095` mapping was correlated with the board's live TCP connection table.
Local port `29095` reaches `QubiC2` / `qubic81` instead and must not be used for
the fridge scan.

The live RPC notebook created its server with firmware `024756ff` and listens on
port `9095`. Do not restart that notebook: constructing its runner reloads the
overlay. Reuse the existing service without changing LMK, RFDC Nyquist zones,
firmware, or PYNQ state.

## Frequency convention

The checked-in LeeQ QubiC setup applies the readout conversion

```text
QubiC programmed frequency [MHz] = 15000 - physical frequency [MHz]
```

Subject to confirmation against the configuration used for the day's qubit
measurement, the requested physical scan maps as follows:

| Physical frequency | QubiC programmed frequency |
|---|---|
| 9.000 GHz | 6.000 GHz |
| 9.500 GHz | 5.500 GHz |

Do not send 9--9.5 GHz directly to the QubiC compiler and do not change the live
server's `dac_nyquist_zone: 2`, `adc_nyquist_zone: 1`, or `lmk_freq: 500.18`
settings to make the range fit. First confirm that the 15 GHz conversion is the
same one used in the successful fridge measurement.

## Execution parameters

- Reused `q2_params` directly from `notebooks/RealSystem/example_setup.py`.
- Readout channel: 3; qubit drive was not included in the experiment LPB.
- Physical frequency: 9000--9500 MHz inclusive; 15 GHz mixer conversion.
- Readout amplitude: 0.02; measurement width: 8 us.
- Coarse passes: 100 averages per frequency.
- The RPC endpoint had no established client before the initial probe.

## Client compatibility

Latest LeeQ main intentionally uses QubiC's pre-Executable load/run protocol,
whereas the signature-development software checkout uses the newer Executable
API. The isolated `.venv` therefore uses source snapshots compatible with the
deployed Oxford RPC server:

- QubiC software `2e26182f6d064d34ef552877359bcf3383bd2b69`
- distproc `82a54c60814131da757f45b5e499fae42576339b`
- qubitconfig `c3a12fce91d615dca69569438b7afa38782bf89f`
- MinimalLLM main `107daa1` (exports `p_map` without forcing NumPy 1.26)

LeeQ now preserves the wrapper's real `dt` signature for QubiC validation and
maps the old assembler's per-core list back to the core-keyed raw-ASM mapping
expected by the live server. These are interface fixes; pulse shapes still come
from LeeQ and the DUT definition is the existing example Q2 definition.

## LeeQ initialization and execution model

LeeQ requires a registered experiment setup before a device or experiment is
used. The QubiC1-specific setup is implemented in
`notebooks/RealSystem/oxford_qubic1_setup.py`. Its lifecycle is:

```python
from oxford_qubic1_setup import initialize_oxford_qubic1_setup
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.builtin.basic.calibrations.resonator_spectroscopy import (
    ResonatorSweepTransmissionWithExtraInitialLPB,
)

# Creates the client-side RPC proxy and registers it as LeeQ's default setup.
# This does not submit an experiment.
qubic1_setup = initialize_oxford_qubic1_setup()

# Build this from the confirmed channel and today's approved measurement
# parameters; do not substitute LeeQ's generic example configuration.
dut = TransmonElement(name="<confirmed DUT>", parameters=confirmed_dut_params)

# EXECUTION BOUNDARY: constructing a LeeQ Experiment calls its run() method.
experiment = ResonatorSweepTransmissionWithExtraInitialLPB(
    dut_qubit=dut,
    start=9000.0,
    stop=9501.0,  # LeeQ uses numpy.arange; stop is exclusive.
    step=1.0,
    num_avs=100,
    rep_rate=0.0,
    mp_width=confirmed_pulse_width_us,
    amp=confirmed_readout_amplitude,
)
```

Do not instantiate the experiment merely to inspect it. LeeQ's `Experiment`
constructor immediately enters its `run()` method and reaches the registered
setup. Perform offline review of the DUT parameters and sweep grid before that
last statement is evaluated.

## Proposed acquisition

1. Compile locally and inspect the full physical/programmed frequency table.
   Confirm that the circuit contains only the intended readout-drive and
   readout-LO pulses on the selected channel.
2. Capture an amplitude-zero baseline using otherwise identical timing.
3. Run a coarse 9.000--9.500 GHz physical sweep at 1 MHz spacing (501 points;
   pass `stop=9501.0` because LeeQ's stop is exclusive),
   initially with 100 averages per point and the approved readout settings.
   Submit small batches and checkpoint raw complex I/Q after every batch.
4. Detect candidates using both magnitude and unwrapped-phase gradient. Repeat
   the coarse sweep in reverse order to reject drift and one-off artifacts.
5. For each repeatable candidate, scan approximately +/-5 MHz using 50--100 kHz
   spacing and 500--1000 averages, then fit center frequency, linewidth/Q,
   complex background, and uncertainty.
6. Retain the complete physical and programmed frequency arrays, raw complex
   I/Q, channel mapping, amplitude, attenuation, pulse width, averaging,
   timestamps, client commit, and firmware metadata.

The repository's example configuration mentions a historical candidate near
9.3868 GHz. Treat it as a search hint only, not as evidence of a currently
connected resonator.

## Stop conditions

Stop without retrying or changing the board configuration if the endpoint or
channel mapping is ambiguous, another RPC client is active, approved RF limits
are unavailable, I/Q clips or saturates, the RPC/SSH transport becomes
unhealthy, or the observed response changes between forward and reverse scans
without an understood cause.

This plan does not authorize an overlay load, RPC-server restart, JTAG action,
clock change, Nyquist-zone change, calibration-file update, or automatic write
back of a detected resonance.

## 2026-09-10 broad two-board sweep

The existing LeeQ
`ResonatorSweepTransmissionWithExtraInitialLPB` experiment was run concurrently
through the already-running RPC services for both Oxford boards. The DUT was
the existing `example_setup.q2_params` definition; no pulse was authored for
this scan and neither RPC service nor overlay was restarted.

| Parameter | Value |
|---|---|
| Physical range | 8.000--10.9998 GHz |
| Step | 0.2 MHz |
| Points | 15,000 per board |
| Readout amplitude | 1.0 |
| Averages | 500 |
| Measurement width | 8 us |
| QubiC1 endpoint | `http://127.0.0.1:19095` |
| QubiC2 endpoint | `http://127.0.0.1:29095` |

Both result arrays passed the exact-grid, finite-IQ, and 15,000-distinct-point
checks. QubiC1 is dominated by a large, approximately periodic standing-wave
ripple and sharp grid-aligned dropouts. QubiC2 has a much smaller received
level, a broad transfer-function envelope, and many isolated single-bin/grid-
periodic excursions. No unambiguous sustained resonator line is visible in
this high-power broad scan. In particular, single-bin excursions and the
previously observed 6.25 MHz-related comb must not be promoted to resonator
candidates.

This run is qualitative: the earlier amplitude guard showed QubiC1 scaling
sublinearly between amplitudes 0.2 and 1.0, consistent with compression at the
requested full-scale setting. A lower-power follow-up is needed before
concluding that the fridge has no resonators in this band.

Phase-gradient follow-up on the saved QubiC1 IQ found 138 one-bin magnitude
outliers. The largest raw phase-gradient excursions occur at exact grid-like
frequencies, led by 10.100 and 10.900 GHz; inspection of adjacent samples shows
that these are discontinuous single-bin/phase-unwrap events rather than finite-
width line shapes. After median-of-three rejection of those samples, 61 broad
minima remain from 8.0572 through 10.7752 GHz. Their median spacing is 45.2 MHz
(standard deviation 1.4 MHz) and median FWHM is 16.6 MHz. That band-wide
periodicity corresponds to an approximately 22.1 ns delay and is consistent
with a standing-wave/reflection background.

Within 9.0--9.5 GHz the repeating minima are 9.0118, 9.0542, 9.1006, 9.1480,
9.1926, 9.2380, 9.2838, 9.3290, 9.3754, 9.4212, and 9.4666 GHz. They should be
treated as ripple minima, not independent resonator candidates. After removing
the one-bin events and local phase-gradient background, no isolated phase
feature establishes a resonator in this scan.

Artifacts:

- QubiC1 raw result and plot:
  `/local/data/projects/qubic_validation/artifacts/oxford-qubic1/20260910T070316Z-amp-1-broad-resonator-spectroscopy/`
- QubiC2 raw result and plot:
  `/local/data/projects/qubic_validation/artifacts/oxford-qubic2/20260910T070929Z-amp-1-broad-resonator-spectroscopy/`
- Side-by-side raw and detrended comparison:
  `/local/data/projects/qubic_validation/artifacts/oxford-broad-resonator-spectroscopy/20260910-amp-1-500avgs-qubic1-qubic2-comparison.png`
- QubiC1 phase-gradient/dip classification:
  `/local/data/projects/qubic_validation/artifacts/oxford-broad-resonator-spectroscopy/20260910-qubic1-phase-gradient-dip-analysis.png`

## Sub-5-MHz two-board candidate follow-up

The broad QubiC1 and QubiC2 traces were screened again for features with fitted
widths from 0.4 through 5 MHz. A provisional candidate required a multi-bin
magnitude depression, a coincident phase-gradient excursion, robust magnitude
and phase scores of at least five, and separation from the obvious grid bins.
This admitted four locations per board:

- QubiC1: 8.4210, 9.2822, 10.5520, and 10.7780 GHz.
- QubiC2: 8.5020, 8.9992, 10.5020, and 10.6830 GHz.

Each was then acquired with the same existing LeeQ resonator experiment and DUT
definition using a 20 MHz total span, 0.02 MHz step, 1,000 points, amplitude 1,
and 500 averages. The two boards ran concurrently; windows on an individual
board ran serially.

None of the eight candidates was confirmed as an isolated resonator. The four
QubiC1 candidate frequencies did not reproduce; the dominant fine-scan events
were instead one-bin/grid events at such frequencies as 8.425, 9.275, 10.550,
and 10.780 GHz. Three QubiC2 windows resolved into ringing centered exactly at
8.500, 9.000, and 10.500 GHz, showing that the broad candidates were ringing
skirts. The QubiC2 10.683 GHz window was noise-dominated and did not form a
coherent finite-width magnitude/phase line.

The comparison plot and machine-readable adjudication are:

- `/local/data/projects/qubic_validation/artifacts/oxford-broad-resonator-spectroscopy/20260910-two-board-fine-candidate-adjudication.png`
- `/local/data/projects/qubic_validation/artifacts/oxford-broad-resonator-spectroscopy/20260910-two-board-fine-candidate-adjudication.json`
