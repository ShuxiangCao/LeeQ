#!/usr/bin/env python3
"""Create comparison views for completed Oxford QubiC1 candidate scans."""

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
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="optional matching scan directory for normalized overlay plots",
    )
    return parser.parse_args()


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    return median, max(mad, np.finfo(float).eps)


def main() -> int:
    args = parse_args()
    paths = sorted(args.scan_dir.glob("center-*-MHz.npz"), reverse=True)
    if len(paths) != 22:
        raise RuntimeError(f"expected 22 scan files, found {len(paths)}")

    analyses = []
    traces = []
    for path in paths:
        data = np.load(path)
        center = float(data["center_mhz"])
        frequency = data["physical_frequencies_mhz"]
        iq = data["iq"]
        magnitude = np.abs(iq)
        phase = np.unwrap(np.angle(iq))

        # Five-MHz Savitzky-Golay baselines remove the smooth standing-wave
        # envelope and electrical delay while retaining resonator-scale detail.
        smooth_magnitude = savgol_filter(magnitude, 101, 3, mode="interp")
        smooth_phase = savgol_filter(phase, 101, 3, mode="interp")
        relative_magnitude = magnitude / np.maximum(
            smooth_magnitude, np.median(smooth_magnitude) * 1e-9
        ) - 1.0
        phase_residual = phase - smooth_phase
        complex_score = np.hypot(relative_magnitude, phase_residual)
        sustained_score = np.convolve(complex_score, np.ones(5) / 5, mode="same")

        core = slice(60, -60)
        score_median, score_scale = robust_scale(sustained_score[core])
        core_index = int(np.argmax(sustained_score[core])) + 60
        peak_snr = float(
            (sustained_score[core_index] - score_median) / score_scale
        )
        peak_frequency = float(frequency[core_index])
        distance_to_6p25_comb = float(
            abs(peak_frequency / 6.25 - round(peak_frequency / 6.25)) * 6.25
        )
        analyses.append(
            {
                "center_mhz": center,
                "strongest_sustained_feature_mhz": peak_frequency,
                "sustained_feature_robust_snr": peak_snr,
                "distance_to_6p25_mhz_comb_mhz": distance_to_6p25_comb,
                "note": (
                    "comb-coincident"
                    if distance_to_6p25_comb <= 0.15
                    else "off-comb"
                ),
            }
        )
        traces.append(
            (center, frequency, relative_magnitude, phase_residual, peak_frequency)
        )

    fig, axes = plt.subplots(6, 4, figsize=(20, 22), sharey=False)
    for axis, trace in zip(axes.flat, traces):
        center, frequency, relative_magnitude, phase_residual, peak_frequency = trace
        axis.plot(frequency, relative_magnitude, linewidth=0.8, label="Δ|IQ| / baseline")
        axis.plot(frequency, phase_residual, linewidth=0.8, alpha=0.75, label="phase residual")
        first_comb = np.ceil(frequency[0] / 6.25) * 6.25
        for comb_frequency in np.arange(first_comb, frequency[-1] + 0.01, 6.25):
            axis.axvline(comb_frequency, color="tab:red", alpha=0.16, linewidth=0.7)
        axis.axvline(center, color="black", linestyle="--", alpha=0.35, linewidth=0.8)
        axis.axvline(peak_frequency, color="tab:purple", alpha=0.35, linewidth=0.8)
        axis.set_title(f"center {center:g} MHz")
        axis.set_xlim(frequency[0], frequency[-1])
        axis.set_ylim(-1.5, 1.5)
        axis.grid(alpha=0.15)
    for axis in axes.flat[len(traces):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Oxford QubiC1 candidate windows — 5 MHz baseline removed\n"
        "red: 6.25 MHz instrumental comb; dashed black: requested center; purple: strongest sustained anomaly",
        y=0.995,
    )
    fig.supxlabel("physical frequency (MHz)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(args.scan_dir / "all-candidate-windows-normalized.png", dpi=170)
    plt.close(fig)

    (args.scan_dir / "analysis-metrics.json").write_text(
        json.dumps({"windows": analyses}, indent=2) + "\n"
    )

    if args.reference_dir is not None:
        fig, axes = plt.subplots(6, 4, figsize=(20, 22), sharey=False)
        correlations = []
        for axis, path in zip(axes.flat, paths):
            reference_path = args.reference_dir / path.name
            if not reference_path.exists():
                raise RuntimeError(f"missing reference scan {reference_path}")
            current = np.load(path)
            reference = np.load(reference_path)
            frequency = current["physical_frequencies_mhz"]
            if not np.array_equal(frequency, reference["physical_frequencies_mhz"]):
                raise RuntimeError(f"frequency-grid mismatch for {path.name}")
            current_magnitude = np.abs(current["iq"])
            reference_magnitude = np.abs(reference["iq"])
            current_normalized = current_magnitude / np.median(current_magnitude)
            reference_normalized = reference_magnitude / np.median(reference_magnitude)
            correlation = float(
                np.corrcoef(current_normalized, reference_normalized)[0, 1]
            )
            correlations.append(correlation)
            center = float(current["center_mhz"])
            axis.plot(frequency, reference_normalized, linewidth=0.8, label="amp 0.02")
            axis.plot(frequency, current_normalized, linewidth=0.8, label="amp 0.2")
            axis.set_title(f"{center:g} MHz; r={correlation:.3f}")
            axis.set_xlim(frequency[0], frequency[-1])
            axis.set_ylim(0, 2)
            axis.grid(alpha=0.15)
        for axis in axes.flat[len(paths):]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(
            "Oxford QubiC1 normalized magnitude: amplitude 0.02 vs 0.2",
            y=0.995,
        )
        fig.supxlabel("physical frequency (MHz)")
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))
        fig.savefig(args.scan_dir / "amplitude-comparison-normalized.png", dpi=170)
        plt.close(fig)
        (args.scan_dir / "amplitude-comparison.json").write_text(
            json.dumps(
                {
                    "reference_dir": str(args.reference_dir),
                    "normalized_magnitude_correlations": correlations,
                    "median_correlation": float(np.median(correlations)),
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps({"windows": analyses}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
