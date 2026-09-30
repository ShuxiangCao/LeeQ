#!/usr/bin/env python3
"""Scan requested resonator-candidate windows on Oxford QubiC1.

This deliberately reuses the established Q2/channel-3 device definition and
the already-running RPC service.  It does not load firmware, change clocks, or
write fitted frequencies back to calibration.
"""

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

from oxford_qubic1_setup import initialize_oxford_qubic1_setup
from run_oxford_qubic1_resonator_spectroscopy import build_dut


ARTIFACT_ROOT = Path(
    "/local/data/projects/qubic_validation/artifacts/oxford-qubic1"
)
CENTERS_MHZ = [
    9937.5,
    9900.0,
    9875.0,
    9850.0,
    9812.5,
    9800.0,
    9760.0,
    9722.0,
    9625.0,
    9600.0,
    9562.0,
    9550.0,
    9500.0,
    9374.0,
    9149.0,
    9124.0,
    9100.0,
    9000.0,
    8937.0,
    8900.0,
    8875.0,
    8625.0,
]
SPAN_MHZ = 40.0
POINTS = 800
STEP_MHZ = SPAN_MHZ / POINTS
NUM_AVERAGES = 500
BATCH_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement that the scans will transmit RF",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.02,
        help="normalized readout pulse amplitude (default: 0.02)",
    )
    parser.add_argument(
        "--centers-mhz",
        type=float,
        nargs="+",
        default=None,
        help="optional subset of physical-frequency centers",
    )
    return parser.parse_args()


def center_label(center_mhz: float) -> str:
    return f"{center_mhz:09.3f}".replace(".", "p")


def save_window(
    output_dir: Path,
    center_mhz: float,
    frequencies_mhz: np.ndarray,
    iq: np.ndarray,
    amplitude: float,
) -> dict:
    magnitude = np.abs(iq)
    phase = np.unwrap(np.angle(iq))
    phase_gradient = np.gradient(phase, frequencies_mhz)
    label = center_label(center_mhz)

    np.savez(
        output_dir / f"center-{label}-MHz.npz",
        center_mhz=center_mhz,
        physical_frequencies_mhz=frequencies_mhz,
        programmed_frequencies_mhz=15000.0 - frequencies_mhz,
        iq=iq,
        magnitude=magnitude,
        unwrapped_phase_rad=phase,
        phase_gradient_rad_per_mhz=phase_gradient,
    )

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(frequencies_mhz, magnitude, linewidth=1)
    axes[0].set_ylabel("|IQ|")
    axes[1].plot(frequencies_mhz, phase, linewidth=1)
    axes[1].set_ylabel("unwrapped phase (rad)")
    axes[2].plot(frequencies_mhz, phase_gradient, linewidth=1)
    axes[2].set_ylabel("d phase / df")
    axes[2].set_xlabel("physical frequency (MHz)")
    fig.suptitle(
        f"Oxford QubiC1: {center_mhz:.3f} MHz ±20 MHz "
        f"({POINTS} points, {NUM_AVERAGES} averages, amp {amplitude:g})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"center-{label}-MHz.png", dpi=180)
    plt.close(fig)

    return {
        "center_mhz": center_mhz,
        "start_mhz": float(frequencies_mhz[0]),
        "stop_mhz": float(frequencies_mhz[-1]),
        "points": int(iq.size),
        "largest_phase_gradient_frequency_mhz": float(
            frequencies_mhz[np.argmax(np.abs(phase_gradient))]
        ),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.amplitude <= 1.0:
        raise ValueError("amplitude must be in the normalized interval (0, 1]")
    centers_mhz = CENTERS_MHZ if args.centers_mhz is None else args.centers_mhz
    plan = {
        "board": "QubiC1",
        "alias": "qubit80_1",
        "rpc_uri": "http://127.0.0.1:19095",
        "centers_mhz": centers_mhz,
        "span_mhz": SPAN_MHZ,
        "points_per_window": POINTS,
        "step_mhz": STEP_MHZ,
        "num_averages": NUM_AVERAGES,
        "batch_size": BATCH_SIZE,
        "measurement_channel": 3,
        "amplitude": args.amplitude,
        "pulse_width_us": 8.0,
        "frequency_conversion": "programmed_mhz = 15000 - physical_mhz",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to transmit RF.", flush=True)
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    amplitude_label = f"{args.amplitude:g}".replace(".", "p")
    output_dir = ARTIFACT_ROOT / (
        f"{timestamp}-amp-{amplitude_label}-candidate-window-scans"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(plan | {"started_utc": timestamp}, indent=2) + "\n"
    )

    initialize_oxford_qubic1_setup()
    setup().status().set_param("Engine_Batch_Size", BATCH_SIZE)
    # Experiments normally open their registered Plotly browser views after
    # acquisition.  This unattended driver writes deterministic PNG/NPZ
    # artifacts itself, so suppress only that interactive post-run display.
    setup().status().set_param("Plot_Result_In_Jupyter", False)
    dut = build_dut()
    completed: list[dict] = []

    for index, center_mhz in enumerate(centers_mhz, start=1):
        start_mhz = center_mhz - SPAN_MHZ / 2
        stop_mhz = center_mhz + SPAN_MHZ / 2
        frequencies_mhz = np.arange(start_mhz, stop_mhz, STEP_MHZ)
        if frequencies_mhz.size != POINTS:
            raise RuntimeError(
                f"grid for {center_mhz} MHz has {frequencies_mhz.size} points, "
                f"expected {POINTS}"
            )

        print(
            f"WINDOW_START {index}/{len(centers_mhz)} center={center_mhz:.3f}MHz",
            flush=True,
        )
        experiment = ResonatorSweepTransmissionWithExtraInitialLPB(
            dut_qubit=dut,
            start=start_mhz,
            stop=stop_mhz,
            step=STEP_MHZ,
            num_avs=NUM_AVERAGES,
            rep_rate=0.0,
            mp_width=8.0,
            initial_lpb=None,
            amp=args.amplitude,
        )
        iq = np.asarray(experiment.data).reshape(-1)
        if iq.size != POINTS:
            raise RuntimeError(
                f"center {center_mhz} MHz returned {iq.size} IQ values, "
                f"expected {POINTS}"
            )
        completed.append(
            save_window(
                output_dir, center_mhz, frequencies_mhz, iq, args.amplitude
            )
        )
        (output_dir / "progress.json").write_text(
            json.dumps(completed, indent=2) + "\n"
        )
        print(
            f"WINDOW_DONE {index}/{len(centers_mhz)} center={center_mhz:.3f}MHz",
            flush=True,
        )

    completed_utc = datetime.now(timezone.utc).isoformat()
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"completed_utc": completed_utc, "windows": completed}, indent=2
        )
        + "\n"
    )
    print(f"ALL_WINDOWS_DONE output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
