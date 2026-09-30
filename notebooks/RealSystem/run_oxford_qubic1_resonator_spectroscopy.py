#!/usr/bin/env python3
"""Run a guarded resonator sweep on Oxford QubiC1.

The default is a three-point transport/protocol probe.  Passing ``--full``
selects the 501-point 9.0--9.5 GHz coarse sweep.  In both cases ``--execute``
is required before LeeQ is allowed to contact the board.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Latest LeeQ main intentionally uses QubiC's pre-Executable RPC interface.
# Put the matching, pinned source snapshots ahead of globally/editably installed
# QubiC packages before importing LeeQ.
LEEQ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEEQ_ROOT / ".venv/src/oxford-distproc/python"))
sys.path.insert(0, str(LEEQ_ROOT / ".venv/src/oxford-qubic"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments import setup
from leeq.experiments.builtin.basic.calibrations.resonator_spectroscopy import (
    ResonatorSweepTransmissionWithExtraInitialLPB,
)

from oxford_qubic1_setup import initialize_oxford_qubic1_setup
from example_setup import q2_params


ARTIFACT_ROOT = Path(
    "/local/data/projects/qubic_validation/artifacts/oxford-qubic1"
)


def build_dut() -> TransmonElement:
    """Reuse the repository's established Oxford Q2/channel-3 definition."""
    parameters = copy.deepcopy(q2_params)
    return TransmonElement(name=parameters["hrid"], parameters=parameters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="run 9000--9500 MHz at 1 MHz spacing and 100 averages",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement that this will transmit RF",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="run the full coarse grid from 9500 down to 9000 MHz",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="use one independently compiled/acquired point per batch",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reverse:
        start_mhz, stop_mhz, step_mhz = 9500.0, 8999.0, -1.0
        num_averages, batch_size, run_kind = 100, 20, "coarse-reverse"
    elif args.full:
        start_mhz, stop_mhz, step_mhz = 9000.0, 9501.0, 1.0
        num_averages, batch_size, run_kind = 100, 20, "coarse"
    else:
        start_mhz, stop_mhz, step_mhz = 9000.0, 9501.0, 250.0
        num_averages, batch_size, run_kind = 10, 1, "probe"

    if args.serial:
        batch_size = 1
        run_kind += "-serial"

    frequencies_mhz = np.arange(start_mhz, stop_mhz, step_mhz)
    planned = {
        "board": "QubiC1",
        "alias": "qubit80_1",
        "rpc_uri": "http://127.0.0.1:19095",
        "run_kind": run_kind,
        "physical_frequencies_mhz": frequencies_mhz.tolist(),
        "programmed_frequencies_mhz": (15000.0 - frequencies_mhz).tolist(),
        "measurement_channel": 3,
        "amplitude": 0.02,
        "pulse_width_us": 8.0,
        "num_averages": num_averages,
        "batch_size": batch_size,
    }
    print(json.dumps(planned, indent=2))
    if not args.execute:
        print("Dry run only; pass --execute to transmit RF.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ARTIFACT_ROOT / f"{timestamp}-{run_kind}-resonator-spectroscopy"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(planned | {"started_utc": timestamp}, indent=2) + "\n"
    )

    initialize_oxford_qubic1_setup()
    setup().status().set_param("Engine_Batch_Size", batch_size)
    dut = build_dut()
    experiment = ResonatorSweepTransmissionWithExtraInitialLPB(
        dut_qubit=dut,
        start=start_mhz,
        stop=stop_mhz,
        step=step_mhz,
        num_avs=num_averages,
        rep_rate=0.0,
        mp_width=8.0,
        initial_lpb=None,
        amp=0.02,
    )

    iq = np.asarray(experiment.data).reshape(-1)
    if iq.size != frequencies_mhz.size:
        raise RuntimeError(
            f"expected {frequencies_mhz.size} IQ values, received {iq.size}"
        )
    magnitude = np.abs(iq)
    phase = np.unwrap(np.angle(iq))
    phase_gradient = np.gradient(phase, frequencies_mhz)
    np.savez(
        output_dir / "results.npz",
        physical_frequencies_mhz=frequencies_mhz,
        programmed_frequencies_mhz=15000.0 - frequencies_mhz,
        iq=iq,
        magnitude=magnitude,
        unwrapped_phase_rad=phase,
        phase_gradient_rad_per_mhz=phase_gradient,
    )

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    axes[0].plot(frequencies_mhz, magnitude, marker=".")
    axes[0].set_ylabel("|IQ|")
    axes[1].plot(frequencies_mhz, phase, marker=".")
    axes[1].set_ylabel("phase (rad)")
    axes[2].plot(frequencies_mhz, phase_gradient, marker=".")
    axes[2].set_ylabel("d phase / df")
    axes[2].set_xlabel("physical frequency (MHz)")
    fig.suptitle(f"Oxford QubiC1 resonator spectroscopy ({run_kind})")
    fig.tight_layout()
    fig.savefig(output_dir / "response.png", dpi=160)
    plt.close(fig)

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "points": int(iq.size),
        "iq_real": iq.real.tolist(),
        "iq_imag": iq.imag.tolist(),
        "largest_phase_gradient_frequency_mhz": float(
            frequencies_mhz[np.argmax(np.abs(phase_gradient))]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved results to {output_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
