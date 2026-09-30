#!/usr/bin/env python3
"""Compare completed broad LeeQ resonator sweeps from the two Oxford boards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubic1", type=Path, required=True)
    parser.add_argument("--qubic2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as result:
        frequency = np.asarray(result["physical_frequencies_mhz"], dtype=float)
        iq = np.asarray(result["iq"], dtype=complex).reshape(-1)
    if frequency.shape != iq.shape:
        raise ValueError(f"frequency/IQ shape mismatch in {path}")
    if not np.all(np.isfinite(iq)):
        raise ValueError(f"non-finite IQ values in {path}")
    return frequency, iq


def derived_traces(
    frequency: np.ndarray, iq: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(iq), 1.0))
    magnitude_db -= np.median(magnitude_db)

    # A 501-point window is 100.2 MHz on the requested 0.2 MHz grid. It removes
    # only the broad transfer-function envelope; narrow and ripple-like features
    # remain visible in the residual.
    magnitude_residual_db = magnitude_db - savgol_filter(magnitude_db, 501, 3)
    phase = np.unwrap(np.angle(iq))
    phase_residual = phase - savgol_filter(phase, 501, 2)
    return magnitude_db, magnitude_residual_db, phase_residual


def main() -> int:
    args = parse_args()
    f1, iq1 = load_result(args.qubic1)
    f2, iq2 = load_result(args.qubic2)
    if not np.array_equal(f1, f2):
        raise ValueError("QubiC1 and QubiC2 frequency grids differ")
    expected = 8000.0 + 0.2 * np.arange(15000)
    if not np.allclose(f1, expected, rtol=0.0, atol=1e-8):
        raise ValueError("results do not use the requested 8–11 GHz, 0.2 MHz grid")

    traces = [derived_traces(f1, iq1), derived_traces(f2, iq2)]
    labels = ["QubiC1 / qubit80_1 (fridge)", "QubiC2 / qubic81 (bench)"]
    colors = ["#1764ab", "#c24d2c"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), sharex=True)
    for row, (label, color, trace) in enumerate(zip(labels, colors, traces)):
        magnitude_db, magnitude_residual_db, phase_residual = trace
        axes[row, 0].plot(f1 / 1000.0, magnitude_db, lw=0.55, color=color)
        axes[row, 1].plot(f1 / 1000.0, magnitude_residual_db, lw=0.5, color=color)
        axes[row, 2].plot(f1 / 1000.0, phase_residual, lw=0.45, color=color)
        axes[row, 0].set_ylabel(f"{label}\nrelative magnitude (dB)")
        axes[row, 1].set_ylabel("magnitude residual (dB)")
        axes[row, 2].set_ylabel("phase residual (rad)")
        axes[row, 1].set_ylim(-15, 15)
        axes[row, 2].set_ylim(-3.5, 3.5)
        for axis in axes[row]:
            axis.grid(alpha=0.18)
    axes[0, 0].set_title("Raw response (median-normalized)")
    axes[0, 1].set_title("100 MHz baseline removed")
    axes[0, 2].set_title("100 MHz phase trend removed")
    for axis in axes[-1]:
        axis.set_xlabel("physical frequency (GHz)")
    fig.suptitle(
        "Oxford broad resonator spectroscopy — amp 1, 500 averages, "
        "0.2 MHz steps (15,000 points/board)\n"
        "Single-bin/grid-periodic excursions are shown but should not be "
        "interpreted as resonators",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary = {
        "output_png": str(args.output.resolve()),
        "qubic1_results": str(args.qubic1.resolve()),
        "qubic2_results": str(args.qubic2.resolve()),
        "points_per_board": int(f1.size),
        "physical_start_mhz": float(f1[0]),
        "physical_stop_mhz": float(f1[-1]),
        "step_mhz": float(f1[1] - f1[0]),
        "amplitude": 1.0,
        "averages": 500,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
