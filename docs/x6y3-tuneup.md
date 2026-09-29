# X6Y3 Ramsey and ping-pong

Use `notebooks/RealSystem/TuneUp_X6Y3_LeeQ.ipynb` for a staged workflow on the existing Huracan executable-RPC service. Hardware execution is disabled by default. The original X6Y3 YAML/ZIP remains the baseline; candidate settings are written to a separate host-side YAML only after validation.

## Ramsey

The three scans follow `TuneUpExample.ipynb` cells 14/20:

| Stage | Drive offset (MHz) | Delay stop (µs, exclusive) | Step (µs) |
| --- | ---: | ---: | ---: |
| 0 | 10 | 0.3 | 0.005 |
| 1 | 1 | 3 | 0.05 |
| 2 | 0.1 | 30 | 0.5 |

Each sequence is X90–delay–negative X90 at `baseline frequency + center correction + stage offset`. This changes physical drive frequency, unlike the archived qcal virtual-Z Ramsey diagnostic. The LeeQ damped-sine fit gives the update `new center = old center + stage offset - fitted positive frequency`. Check residuals and fit uncertainty before feeding an accepted center to the next stage. Repeat the fine scan after amplitude tuning.

## Amplitude

The X6Y3 notebook defines independent 35 ns X90 and 70 ns X FAST_DRAG envelopes, with gate-specific virtual-Z corrections. The generic LeeQ collection derives X90 from X/2; applying that assumption here would replace the supplied pulse definitions. The native implementation instead uses the same repetition protocol as LeeQ ping-pong while preserving each composite gate.

Repeat X an even number of times or X90 a multiple of four, followed by a fixed X90. Interleave a negative-X90 final-phase control, and randomize conditions across repeated blocks. Subtract paired IQ projections and fit their slope versus repeated gate count. Interpolate the amplitude at zero slope using a local bracket. Report uncertainty from the slope fits and inflate it when the amplitude-linear fit has excess residuals.

Start with scales 0.98, 1, 1.02 and counts 0/4/8/12 for X90 or 0/2/4/6 for X. Narrow the bracket and increase repetitions only while the response is locally linear. Validate the candidate against the original amplitude in fresh randomized blocks. Tune X90 first, then keep its validated value fixed while tuning X. A zero crossing is an amplitude-error estimate, not a gate-fidelity measurement; detuning, phase errors, drift and imperfect preparation can limit it.

## Command line

`scripts/run_x6y3_tuneup.py` takes `--calibration`, matching current two-core `--channel-config`, optional modern QubiC `--client-path`, and a complete baseline `--readout-reference` directory. Select `--qubit 0|1 --mode ramsey --stage 0|1|2` or `--mode pingpong --gate X90|X --scales 0.98,1,1.02 --counts 0,4,8,12`. `--center-offset` is relative to the immutable baseline in MHz; `--x90-scale` is the fixed half pulse's amplitude multiplier. Choose a new `--output` directory for every scan.

Without `--execute`, the command blocks networking and verifies the complete plan against independent qcal schedules and waveform/frequency memories. With `--execute`, it precompiles all points, submits native LeeQ LPBs, saves raw per-point IQ and provenance, and fits the result. Failed acquisitions are not retried automatically. Inspect saved data if fitting fails; do not repeat hardware shots merely to rerun analysis.

No board-side install, firmware download, service restart, reboot or persistent RF reconfiguration is part of this workflow. The hardware path uses ordinary IQ acquisition, a 500 µs passive wait from the supplied configuration, and no heralding or active reset.
