#!/usr/bin/env python3
"""Run the same broad LeeQ resonator sweep on either Oxford QubiC board."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LEEQ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEEQ_ROOT / ".venv/src/oxford-distproc/python"))
sys.path.insert(0, str(LEEQ_ROOT / ".venv/src/oxford-qubic"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leeq.experiments import setup
from leeq.experiments.builtin.basic.calibrations.resonator_spectroscopy import (
    ResonatorSweepTransmissionWithExtraInitialLPB,
)

from oxford_qubic1_setup import (
    initialize_oxford_qubic1_setup,
    initialize_oxford_qubic2_setup,
)
from run_oxford_qubic1_resonator_spectroscopy import build_dut


BOARD_CONFIG = {
    "qubic1": {
        "logical_name": "QubiC1",
        "alias": "qubit80_1",
        "role": "Oxford fridge",
        "rpc_uri": "http://127.0.0.1:19095",
        "artifact_root": Path(
            "/local/data/projects/qubic_validation/artifacts/oxford-qubic1"
        ),
        "initializer": initialize_oxford_qubic1_setup,
    },
    "qubic2": {
        "logical_name": "QubiC2",
        "alias": "qubic81",
        "role": "Oxford bench",
        "rpc_uri": "http://127.0.0.1:29095",
        "artifact_root": Path(
            "/local/data/projects/qubic_validation/artifacts/oxford-qubic2"
        ),
        "initializer": initialize_oxford_qubic2_setup,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=BOARD_CONFIG, required=True)
    parser.add_argument("--start-mhz", type=float, default=8000.0)
    parser.add_argument("--stop-mhz", type=float, default=11000.0)
    parser.add_argument("--step-mhz", type=float, default=0.2)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--averages", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.amplitude <= 1.0:
        raise ValueError("amplitude must be in the normalized interval (0, 1]")
    if args.step_mhz <= 0 or args.stop_mhz <= args.start_mhz:
        raise ValueError("require stop > start and step > 0")
    board = BOARD_CONFIG[args.board]
    frequencies_mhz = np.arange(args.start_mhz, args.stop_mhz, args.step_mhz)
    plan = {
        "board": board["logical_name"],
        "alias": board["alias"],
        "role": board["role"],
        "rpc_uri": board["rpc_uri"],
        "physical_start_mhz": args.start_mhz,
        "physical_stop_exclusive_mhz": args.stop_mhz,
        "step_mhz": args.step_mhz,
        "points": int(frequencies_mhz.size),
        "programmed_start_mhz": 15000.0 - args.start_mhz,
        "programmed_last_mhz": float(15000.0 - frequencies_mhz[-1]),
        "measurement_channel": 3,
        "amplitude": args.amplitude,
        "pulse_width_us": 8.0,
        "num_averages": args.averages,
        "batch_size": args.batch_size,
        "experiment": "ResonatorSweepTransmissionWithExtraInitialLPB",
        "dut_definition": "existing example_setup.q2_params via build_dut",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to transmit RF.", flush=True)
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    amplitude_label = f"{args.amplitude:g}".replace(".", "p")
    output_dir = board["artifact_root"] / (
        f"{timestamp}-amp-{amplitude_label}-broad-resonator-spectroscopy"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(plan | {"started_utc": timestamp}, indent=2) + "\n"
    )

    board["initializer"](chronicle_name=f"-{args.board}")
    setup().status().set_param("Engine_Batch_Size", args.batch_size)
    setup().status().set_param("Plot_Result_In_Jupyter", False)
    dut = build_dut()
    experiment = ResonatorSweepTransmissionWithExtraInitialLPB(
        dut_qubit=dut,
        start=args.start_mhz,
        stop=args.stop_mhz,
        step=args.step_mhz,
        num_avs=args.averages,
        rep_rate=0.0,
        mp_width=8.0,
        initial_lpb=None,
        amp=args.amplitude,
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

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
    axes[0].plot(frequencies_mhz, magnitude, linewidth=0.65)
    axes[0].set_ylabel("|IQ|")
    axes[1].plot(frequencies_mhz, phase, linewidth=0.65)
    axes[1].set_ylabel("unwrapped phase (rad)")
    axes[2].plot(frequencies_mhz, phase_gradient, linewidth=0.65)
    axes[2].set_ylabel("d phase / df")
    axes[2].set_xlabel("physical frequency (MHz)")
    fig.suptitle(
        f"{board['logical_name']} broad resonator spectroscopy: "
        f"{args.start_mhz:g}–{args.stop_mhz:g} MHz, amp {args.amplitude:g}, "
        f"{args.averages} averages"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "response.png", dpi=180)
    plt.close(fig)

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "points": int(iq.size),
        "finite_iq": bool(np.all(np.isfinite(iq))),
        "distinct_iq": int(np.unique(iq).size),
        "magnitude_min": float(magnitude.min()),
        "magnitude_median": float(np.median(magnitude)),
        "magnitude_max": float(magnitude.max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SWEEP_DONE output_dir={output_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
