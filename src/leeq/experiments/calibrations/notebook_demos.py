"""Lightweight demo experiments used by tutorial notebooks.

These classes provide deterministic, fast-running experiment-shaped objects for
documentation notebooks. They intentionally avoid hardware or pulse-stack
requirements while preserving the small constructor/run API used in tutorials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MeasurementStatistics",
    "RabiAmplitudeCalibration",
    "RabiFrequencyCalibration",
]


@dataclass
class _DemoExperiment:
    name: str | None = None
    qubit: Any = None

    def __init__(self, name: str | None = None, qubit: Any = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.qubit = qubit
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.results = None

    def _rng(self) -> np.random.Generator:
        seed = sum(ord(ch) for ch in f"{self.__class__.__name__}:{self.name}") % (2**32)
        return np.random.default_rng(seed)


class MeasurementStatistics(_DemoExperiment):
    """Fast deterministic measurement-statistics demo."""

    def __init__(self, repeated_measurement_count: int = 1000, **kwargs):
        super().__init__(repeated_measurement_count=repeated_measurement_count, **kwargs)

    def run(self):
        ground = 0.90 + 0.03 * self._rng().random()
        excited = 1.0 - ground
        fidelity = 0.95 + 0.03 * self._rng().random()
        shots = int(self.repeated_measurement_count)

        self.results = {
            "statistics": {
                "ground_state_probability": ground,
                "excited_state_probability": excited,
                "measurement_fidelity": fidelity,
                "ground_state_counts": int(round(ground * shots)),
                "excited_state_counts": int(round(excited * shots)),
                "shots": shots,
            }
        }
        return self.results


class RabiAmplitudeCalibration(_DemoExperiment):
    """Fast deterministic Rabi-amplitude calibration demo."""

    def __init__(
        self,
        drive_frequency: float = 5000.0,
        amplitude_start: float = 0.0,
        amplitude_stop: float = 1.0,
        amplitude_points: int = 51,
        pulse_width: float = 0.05,
        repeated_measurement_count: int = 1000,
        **kwargs,
    ):
        super().__init__(
            drive_frequency=drive_frequency,
            amplitude_start=amplitude_start,
            amplitude_stop=amplitude_stop,
            amplitude_points=amplitude_points,
            pulse_width=pulse_width,
            repeated_measurement_count=repeated_measurement_count,
            **kwargs,
        )

    def run(self):
        amplitudes = np.linspace(self.amplitude_start, self.amplitude_stop, int(self.amplitude_points))
        pi_amplitude = max((self.amplitude_stop - self.amplitude_start) * 0.52, 1e-9)
        probabilities = 0.05 + 0.90 * np.sin(0.5 * np.pi * amplitudes / pi_amplitude) ** 2
        probabilities += self._rng().normal(0.0, 0.01, size=probabilities.shape)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        peak_index = int(np.argmax(probabilities))
        self.pi_amplitude = float(amplitudes[peak_index])
        self.fit_params = {
            "pi_amplitude": self.pi_amplitude,
            "pi_half_amplitude": self.pi_amplitude / 2.0,
            "contrast": float(np.max(probabilities) - np.min(probabilities)),
        }
        self.results = {
            "sweep_values": amplitudes,
            "measurement_probabilities": probabilities,
            "fit_parameters": self.fit_params,
        }
        return self.results


class RabiFrequencyCalibration(_DemoExperiment):
    """Fast deterministic Rabi-frequency calibration demo."""

    def __init__(
        self,
        frequency_start: float = 4995.0,
        frequency_stop: float = 5005.0,
        frequency_points: int = 41,
        repeated_measurement_count: int = 1000,
        **kwargs,
    ):
        super().__init__(
            frequency_start=frequency_start,
            frequency_stop=frequency_stop,
            frequency_points=frequency_points,
            repeated_measurement_count=repeated_measurement_count,
            **kwargs,
        )

    def run(self):
        frequencies = np.linspace(self.frequency_start, self.frequency_stop, int(self.frequency_points))
        center = 0.5 * (self.frequency_start + self.frequency_stop)
        width = max((self.frequency_stop - self.frequency_start) / 8.0, 1e-9)
        probabilities = 0.10 + 0.80 / (1.0 + ((frequencies - center) / width) ** 2)
        probabilities += self._rng().normal(0.0, 0.01, size=probabilities.shape)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        peak_index = int(np.argmax(probabilities))
        self.calibrated_frequency = float(frequencies[peak_index])
        self.fit_params = {
            "center_frequency": self.calibrated_frequency,
            "linewidth": width,
        }
        self.results = {
            "sweep_values": frequencies,
            "measurement_probabilities": probabilities,
            "fit_parameters": self.fit_params,
        }
        return self.results
