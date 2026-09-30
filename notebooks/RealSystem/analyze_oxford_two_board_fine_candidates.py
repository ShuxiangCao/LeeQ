#!/usr/bin/env python3
"""Render the 2026-09-10 Oxford two-board fine-candidate adjudication."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HARNESS = Path("/local/data/projects/qubic_validation")
OUTPUT = HARNESS / "artifacts/oxford-broad-resonator-spectroscopy"
SCANS = [
    ("QubiC1", 8421.0, "oxford-qubic1/20260910T073904Z-amp-1-broad-resonator-spectroscopy", 8425.0,
     "rejected: candidate absent; one-bin event at 8425 MHz"),
    ("QubiC1", 9282.2, "oxford-qubic1/20260910T073929Z-amp-1-broad-resonator-spectroscopy", 9275.0,
     "rejected: candidate absent; grid events dominate"),
    ("QubiC1", 10552.0, "oxford-qubic1/20260910T073954Z-amp-1-broad-resonator-spectroscopy", 10550.0,
     "rejected: candidate absent; one-bin event at 10550 MHz"),
    ("QubiC1", 10778.0, "oxford-qubic1/20260910T074019Z-amp-1-broad-resonator-spectroscopy", 10780.0,
     "rejected: candidate absent; one-bin event at 10780 MHz"),
    ("QubiC2", 8502.0, "oxford-qubic2/20260910T073905Z-amp-1-broad-resonator-spectroscopy", 8500.0,
     "rejected: ringing centered at exact 8500 MHz grid"),
    ("QubiC2", 8999.2, "oxford-qubic2/20260910T073929Z-amp-1-broad-resonator-spectroscopy", 9000.0,
     "rejected: ringing centered at exact 9000 MHz grid"),
    ("QubiC2", 10502.0, "oxford-qubic2/20260910T073954Z-amp-1-broad-resonator-spectroscopy", 10500.0,
     "rejected: ringing centered at exact 10500 MHz grid"),
    ("QubiC2", 10683.0, "oxford-qubic2/20260910T074019Z-amp-1-broad-resonator-spectroscopy", None,
     "rejected: noise-dominated; no coherent finite-width line"),
]


def main() -> int:
    fig, axes = plt.subplots(4, 2, figsize=(18, 15), sharex=False)
    metrics = []
    for axis, (board, candidate, relative_dir, artifact_frequency, decision) in zip(
        axes.T.flat, SCANS
    ):
        result_path = HARNESS / "artifacts" / relative_dir / "results.npz"
        with np.load(result_path) as result:
            frequency = np.asarray(result["physical_frequencies_mhz"], dtype=float)
            iq = np.asarray(result["iq"], dtype=complex).reshape(-1)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(iq), 1.0))
        phase = np.unwrap(np.angle(iq))
        # Ten-MHz local trends remove cable delay and the standing-wave baseline
        # while retaining the requested <=5 MHz structures.
        magnitude_residual = magnitude_db - savgol_filter(magnitude_db, 501, 3)
        phase_residual = phase - savgol_filter(phase, 501, 3)

        phase_axis = axis.twinx()
        axis.plot(frequency, magnitude_residual, color="#1764ab", lw=0.7,
                  label="magnitude residual")
        phase_axis.plot(frequency, phase_residual, color="#e07a1f", lw=0.55,
                        alpha=0.72, label="phase residual")
        axis.axvline(candidate, color="black", ls="--", lw=1.0,
                     label="broad-sweep candidate")
        if artifact_frequency is not None:
            axis.axvline(artifact_frequency, color="#d62728", ls=":", lw=1.1,
                         label="resolved grid artifact")
        axis.set_title(f"{board}: {candidate:.1f} MHz — {decision}", fontsize=10)
        axis.set_ylabel("magnitude residual (dB)", color="#1764ab")
        phase_axis.set_ylabel("phase residual (rad)", color="#e07a1f")
        axis.set_xlabel("physical frequency (MHz)")
        axis.grid(alpha=0.18)
        metrics.append(
            {
                "board": board,
                "candidate_mhz": candidate,
                "scan_start_mhz": float(frequency[0]),
                "scan_stop_mhz": float(frequency[-1]),
                "step_mhz": float(np.median(np.diff(frequency))),
                "points": int(frequency.size),
                "artifact_frequency_mhz": artifact_frequency,
                "decision": decision,
                "result_path": str(result_path),
            }
        )

    # Build an explicit legend because each panel uses twinned y axes.
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#1764ab", label="magnitude residual"),
        Line2D([0], [0], color="#e07a1f", label="phase residual"),
        Line2D([0], [0], color="black", ls="--", label="broad candidate"),
        Line2D([0], [0], color="#d62728", ls=":", label="grid artifact center"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4)
    fig.suptitle(
        "Oxford QubiC1/QubiC2 fine candidate scans — 20 MHz span, "
        "0.02 MHz step, amp 1, 500 averages\n"
        "No candidate reproduces as an isolated resonator with κ ≤ 5 MHz",
        fontsize=15,
        y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    output_png = OUTPUT / "20260910-two-board-fine-candidate-adjudication.png"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    report = {
        "output_png": str(output_png),
        "acquisition": {
            "span_mhz": 20.0,
            "step_mhz": 0.02,
            "points_per_candidate": 1000,
            "averages": 500,
            "amplitude": 1.0,
        },
        "confirmed_resonators": [],
        "candidates": metrics,
        "conclusion": (
            "All eight broad-sweep candidates are rejected. QubiC1 candidates "
            "do not reproduce at their proposed frequencies. Three QubiC2 "
            "windows resolve into ringing at exact round-number grid frequencies; "
            "the 10683 MHz window is noise-dominated without a coherent line."
        ),
    }
    output_png.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
