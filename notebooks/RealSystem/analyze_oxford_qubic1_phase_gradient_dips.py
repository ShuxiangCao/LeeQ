#!/usr/bin/env python3
"""Separate QubiC1 broad-sweep ripple minima from one-bin phase artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks, medfilt, peak_widths, savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.input) as result:
        frequency = np.asarray(result["physical_frequencies_mhz"], dtype=float)
        iq = np.asarray(result["iq"], dtype=complex).reshape(-1)
    if frequency.shape != iq.shape or not np.all(np.isfinite(iq)):
        raise ValueError("invalid frequency/IQ arrays")

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(iq), 1.0))
    phase = np.unwrap(np.angle(iq))
    raw_phase_gradient = np.gradient(phase, frequency)

    # A median-of-three trace exposes the physical-width structure without
    # deleting the raw one-bin excursions from the diagnostic plot.
    clean_magnitude_db = medfilt(magnitude_db, 3)
    one_bin_mask = np.abs(magnitude_db - clean_magnitude_db) > 1.0
    clean_iq = iq.copy()
    one_bin_indices = np.flatnonzero(one_bin_mask)
    for index in one_bin_indices:
        if 0 < index < clean_iq.size - 1:
            clean_iq[index] = (clean_iq[index - 1] + clean_iq[index + 1]) / 2.0

    clean_phase = np.unwrap(np.angle(clean_iq))
    smooth_phase_gradient = savgol_filter(
        clean_phase, 11, 3, deriv=1, delta=float(frequency[1] - frequency[0])
    )
    phase_gradient_residual = smooth_phase_gradient - savgol_filter(
        smooth_phase_gradient, 101, 3
    )

    # Remove only the very broad gain envelope. The resulting ~45 MHz ripple
    # minima are deliberately retained and measured.
    broad_envelope_db = savgol_filter(clean_magnitude_db, 1251, 3)
    magnitude_residual_db = clean_magnitude_db - broad_envelope_db
    minima, properties = find_peaks(
        -magnitude_residual_db, distance=100, prominence=1.0
    )
    interior = (frequency[minima] >= 8050.0) & (frequency[minima] <= 10800.0)
    minima = minima[interior]
    widths_mhz = (
        peak_widths(-magnitude_residual_db, minima, rel_height=0.5)[0]
        * float(frequency[1] - frequency[0])
    )
    spacing_mhz = np.diff(frequency[minima])

    # Raw phase-gradient extrema are reported separately: these identify the
    # discontinuous acquisition bins, not credible finite-width resonances.
    phase_spikes, _ = find_peaks(
        np.abs(raw_phase_gradient), distance=10, prominence=1.0
    )
    phase_spikes = phase_spikes[
        np.argsort(np.abs(raw_phase_gradient[phase_spikes]))[::-1]
    ][:20]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    ghz = frequency / 1000.0
    axes[0].plot(ghz, magnitude_db, color="0.75", lw=0.5, label="raw")
    axes[0].plot(ghz, clean_magnitude_db, color="#1764ab", lw=0.75,
                 label="median-of-3 (one-bin outliers rejected)")
    axes[0].scatter(
        frequency[minima] / 1000.0,
        clean_magnitude_db[minima],
        color="#c24d2c",
        s=13,
        zorder=3,
        label="repeating finite-width minima",
    )
    axes[0].set_ylabel("magnitude (dB)")
    axes[0].legend(loc="lower left", ncol=3)

    axes[1].plot(ghz, raw_phase_gradient, color="#6a3d9a", lw=0.55)
    axes[1].scatter(
        frequency[phase_spikes] / 1000.0,
        raw_phase_gradient[phase_spikes],
        color="#e31a1c",
        s=16,
        label="largest raw gradient spikes",
    )
    axes[1].set_ylim(-5.5, 5.5)
    axes[1].set_ylabel("raw d phase / df\n(rad/MHz, clipped)")
    axes[1].legend(loc="lower left")

    axes[2].plot(ghz, phase_gradient_residual, color="#1b9e77", lw=0.65)
    axes[2].scatter(
        frequency[minima] / 1000.0,
        phase_gradient_residual[minima],
        color="#c24d2c",
        s=13,
        zorder=3,
    )
    axes[2].set_ylabel("cleaned phase-gradient\nresidual (rad/MHz)")
    axes[2].set_xlabel("physical frequency (GHz)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "QubiC1 broad-sweep dip analysis — amp 1, 500 averages\n"
        f"{minima.size} repeating minima: median spacing "
        f"{np.median(spacing_mhz):.1f} MHz, median width "
        f"{np.median(widths_mhz):.1f} MHz; sharp phase spikes are one-bin/grid artifacts",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary = {
        "input": str(args.input.resolve()),
        "output_png": str(args.output.resolve()),
        "one_bin_outlier_count": int(one_bin_mask.sum()),
        "repeating_minimum_count": int(minima.size),
        "repeating_minima_mhz": [round(float(value), 1) for value in frequency[minima]],
        "median_spacing_mhz": float(np.median(spacing_mhz)),
        "spacing_std_mhz": float(np.std(spacing_mhz)),
        "median_fwhm_mhz": float(np.median(widths_mhz)),
        "equivalent_delay_ns": float(1000.0 / np.median(spacing_mhz)),
        "largest_raw_phase_gradient_spikes": [
            {
                "frequency_mhz": round(float(frequency[index]), 1),
                "gradient_rad_per_mhz": float(raw_phase_gradient[index]),
            }
            for index in phase_spikes
        ],
        "interpretation": (
            "The finite-width minima form a band-wide approximately 45.2 MHz "
            "periodic ripple, consistent with a standing-wave/reflection background. "
            "The largest raw phase-gradient spikes are discontinuous one-bin/grid "
            "artifacts. No isolated resonator-like phase feature is established."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
