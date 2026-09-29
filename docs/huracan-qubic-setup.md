# Huracan / X6Y3 LeeQ preparation

This branch adds an integrated-I/Q LeeQ setup for Huracan's existing QubiC
RPC server at `http://127.0.0.1:9095`. The local tunnel forwards to
`192.168.1.236:9095` through `LBNL-simulator-auto` and the Berkeley gateway.
The backend uses QubiC's `Executable` / `run_circuit_batch` API. Existing
legacy QubiC setups keep their load/run interface.

## Firmware and configuration

Saved deployment evidence on 2026-09-29 supersedes the original eight-core
`99eb0a53` configuration: Huracan was updated to **aa01c78f, Q0/Q1 only**.
The deployment record identifies software source `dcceeeda`; the saved 03:59 UTC
Q0 acquisition contains paired I/Q and D4 signature packets. These are records
from the existing QubiC workflow, not LeeQ hardware validation.

Use `channel_config.json` from the complete aa01c78f package. On this host:

```text
/local/data/artifacts/zcu216/20260903T_aa01c78f_rdrv230_d4_candidate/bits_aa01c78f/channel_config.json
```

`create_huracan_setup` rejects the older eight-core channel configuration.
LeeQ channels 0/1 map to Q0 drive/readout, and 2/3 map to Q1 drive/readout.
Compilation uses the clock frequency and memory names in the supplied metadata.
Metadata validation does not establish which image is currently running.

## Local setup

Use a Python environment containing LeeQ and the modern QubiC toolchain.
The compatibility baseline is `lbl-qubic==25.8.0`, `distproc==25.8.0`, and
`qubitconfig==25.5.1`. The separately maintained signature client extends the
same batch API. The original Oxford environment's pinned legacy QubiC source
paths must not take precedence for this setup.

```python
from leeq.setups.huracan import create_huracan_setup

huracan = create_huracan_setup("/path/to/bits_aa01c78f/channel_config.json")
huracan.status.set_parameters(Acquisition_Type="IQ", Shot_Number=10)
```

Construction reads local metadata and creates a client object without sending
RPC requests. It does not register a default LeeQ setup or start an experiment.
`huracan.compile_circuits(qubic_circuit_list, batch_size=1)` returns an executable
locally. A later experiment uses the usual LeeQ setup registration and device
primitives with reviewed Huracan calibration. This change does not translate
the qcal calibration YAML or supply drive/readout pulse calibrations.

## Offline verification

The check below compiles synthetic LeeQ measurement primitives for Q0 and Q1
through the real QubiC compiler and assembler. It blocks socket connections
and XML-RPC requests before importing LeeQ. There is no acquisition mode.

```sh
python scripts/check_huracan_setup.py \
  --channel-config /path/to/bits_aa01c78f/channel_config.json
```

On this development host, existing environments can be reused without changing
their installed packages:

```sh
/local/data/projects/LeeQ/.venv/bin/python scripts/check_huracan_setup.py \
  --client-path /local/data/projects/qubic_validation/artifacts/x6y3/venv/lib/python3.11/site-packages \
  --channel-config /local/data/artifacts/zcu216/20260903T_aa01c78f_rdrv230_d4_candidate/bits_aa01c78f/channel_config.json
```

Focused tests cover real executable compilation, serialized RPC payloads with
a fake remote proxy, signed I/Q decoding, sweep ordering, averaging, malformed
results, incompatible metadata/client rejection, and failure without retries.
All network connections are blocked for these tests.

Validation on 2026-09-29: **29 tests passed** with the released 25.8.0 client,
and the same 29 passed with the signature software snapshot at
`dcceeedae1f0857768b72743a535726212e6bec4`. The real LeeQ-to-executable offline
check also passed with the package above and reported zero RPC attempts.
The supplied channel metadata SHA-256 is
`b62ded6a386e6b631b9aa2f6163aad5d3040ea7f6e4310c1e0f90eefd4287778`.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/setups/test_qubic_executable_setups.py \
  tests/setups/test_qubic_setups.py \
  tests/compiler/lbnl_qubic/test_qubic_envelope_adapter.py
```

## Scope and current status

The adapter supports `IQ` and `IQ_average`, with one readout per measured
channel per sweep point. It maps named `Qn.rdlo` results to the core-indexed
arrays expected by LeeQ. Raw ADC traces, signature-feature exposure, and
mid-circuit feedback need separate work.

Huracan is occupied. Implementation and validation are local only. A requested
status check on 2026-09-29 confirmed the existing tunnel master and local 9095
listener; the board rejected SSH key authentication, so no board-side status
command ran. No RPC request, deployment, server restart, FPGA access, RF setting
change, or measurement was made by this preparation.

Before a future authorized measurement, verify the current server/image and
calibrated primitives for this two-core mapping. Hardware validation remains
pending; a successful offline compilation is not a hardware result.
