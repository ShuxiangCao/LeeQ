# X6Y3 notebook experiments in LeeQ

The supplied qcal notebook parameter set is the already largely calibrated
baseline. This integration preserves it and uses native LeeQ gates, delays,
measurement primitives and the sweep engine. It never promotes fit results or
writes the input configuration. Start with
[`Calibration_X6Y3_LeeQ.ipynb`](../notebooks/RealSystem/Calibration_X6Y3_LeeQ.ipynb).

## Inputs and environment

The development host has the original notebook, archive and extracted YAML at
`/local/data/projects/qubic_validation/X6Y3/`. The importer reads `config.yaml`;
it does not load either supplied pickle. Use the matching two-core aa01c78f
`channel_config.json`, not the archive's eight-core metadata. Construction checks
channel mappings, direct-RF assumptions and waveform sample rates.

LeeQ source is on `feature/lbnl-qubic-preparation`. Validation uses Python 3.11,
LeeQ's existing environment and qcal 4.0.1 / lbl-qubic 25.8.0 / distproc 25.8.0
from the existing X6Y3 environment. No board package installation is needed.
The optional qcal dependency supplies its original FAST_DRAG/cosine_square
envelope functions; it is not the experiment execution engine. The integration
does not use qcal's saved gate cache or initialize a qcal QPU.

```bash
/local/data/projects/LeeQ/.venv/bin/python scripts/check_x6y3_reproduction.py \
  --calibration /local/data/projects/qubic_validation/X6Y3/configs_new/config.yaml \
  --channel-config /local/data/artifacts/zcu216/20260903T_aa01c78f_rdrv230_d4_candidate/bits_aa01c78f/channel_config.json \
  --client-path /local/data/projects/qubic_validation/artifacts/x6y3/venv/lib/python3.11/site-packages \
  --full --output /tmp/x6y3-leeq-parity.json
```

The validator disables sockets and XML-RPC before imports and disables the
optional LLM dependency's import-time price-map download. It compares each
protocol with independent qcal `cycle_pulse` translation: scheduled start times,
frequencies, amplitudes, phases, complex samples and byte-identical assembled
waveform/frequency memories. This is offline evidence, not physical fidelity.

## Preserved calibration and protocol details

- Frequencies convert Hz to MHz; seconds convert to microseconds. Phase values
  retain qcal's numerical/radian convention, including large demodulation phases.
- X and X90 remain independent calibrated FAST_DRAG sequences. X90 is not made
  by halving X. Each original virtual-Z correction remains in the sequence.
- Readout drive and demodulation have independent duration, phase and envelope.
  Demodulation delay is relative to readout onset on rdlo only. Legacy LeeQ
  measurement primitives without `demodulation` retain their prior behavior.
- Readout and T1 nominal excited preparation uses X90 + X90, as in the source
  notebook's default protocol. The independent native X remains available.
- Amplitude refinement uses four fixed-duration X90 pulses and 31 relative
  amplitude points from 0.7 to 1.3. One-pulse diagnostic mode is also available.
- GE Ramsey uses X90–idle–Rz–X90, with virtual-Z detunings ±2.5/±5 MHz,
  30 delays from 0 to 1 us (qcal Frequency's default grid). The drive frequency
  stays calibrated. GE T1 uses 50 delays from 0 to 350 us.
- The 500 us passive preparation wait occurs once per shot. The initial subset
  is unheralded; original notebook selections outside Q0/Q1, heralding, EF, CZ,
  signature processing and active reset are not reproduced here.

## Physical acquisition

The notebook defaults to `RUN_HARDWARE = False`. With explicit execution selected,
reuse Huracan's existing `http://127.0.0.1:9095` RPC service. Confirm the current
image, matching metadata and instrument availability first. No board reboot,
bitstream reload, software deployment, service restart or RF setting change is
part of this workflow. Normal RPC acquisition necessarily loads transient
program/waveform buffers and advances shot/readout registers.

The CLI `scripts/run_x6y3_experiment.py` also defaults to offline compilation.
It accepts the same three input-path options above, plus `--experiment readout`,
`--qubit 0`, `--shots 512`, `--blocks 4`, and `--output <new-directory>`.
Only `--execute` submits shots. Other experiment names are `amplitude`, `ramsey`
and `t1`; pass `--readout-reference <completed-readout-directory>` for fitting.
Use `--shots 2000` to match the notebook T1 shot count.

Each point runs through LeeQ's native sweep engine in a separate submission and
is saved before continuing. Failures are not retried. Partial result directories
contain completed point files and an error manifest. Reusing an existing output
directory is rejected. Local LeeQ status parameters are restored on completion
or failure; input calibration is never modified.

Artifacts include per-shot IQ, point definitions, calibration snapshot/hash,
channel metadata/hash, package versions, LeeQ revision/source hashes and UTC
timestamps. Readout analysis trains LDA on earlier preparation blocks and tests
on held-out later blocks. Its score measures assignment to nominal preparation
labels, not independent state fidelity. Sweep fits use a baseline-referenced IQ
projection, not corrected populations. Fits and candidate corrections are local
reports only; readout/preparation drift can invalidate their interpretation.

Focused regressions are in `tests/experiments/test_x6y3.py`; run with the modern
QubiC/qcal packages importable and `LITELLM_LOCAL_MODEL_COST_MAP=True`.
