"""Lightweight characterization demos used by tutorial notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["T1Measurement", "T2RamseyMeasurement", "T2EchoMeasurement"]


class _DemoCharacterization:
    def __init__(self, name: str | None = None, qubit: Any = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.qubit = qubit
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.results = None

    def _rng(self) -> np.random.Generator:
        seed = sum(ord(ch) for ch in f"{self.__class__.__name__}:{self.name}") % (2**32)
        return np.random.default_rng(seed)

    def _delay_axis(self):
        return np.linspace(self.delay_start, self.delay_stop, int(self.delay_points))


class T1Measurement(_DemoCharacterization):
    """Fast deterministic T1 measurement demo."""

    def __init__(
        self,
        delay_start: float = 0.0,
        delay_stop: float = 200.0,
        delay_points: int = 41,
        pi_pulse_amplitude: float = 0.5,
        repeated_measurement_count: int = 1000,
        **kwargs,
    ):
        super().__init__(
            delay_start=delay_start,
            delay_stop=delay_stop,
            delay_points=delay_points,
            pi_pulse_amplitude=pi_pulse_amplitude,
            repeated_measurement_count=repeated_measurement_count,
            **kwargs,
        )

    def run(self):
        delays = self._delay_axis()
        t1 = max(self.delay_stop / 3.0, 1.0)
        probabilities = 0.08 + 0.86 * np.exp(-delays / t1)
        probabilities += self._rng().normal(0.0, 0.01, size=probabilities.shape)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        self.fit_params = {"t1": t1, "amplitude": 0.86, "offset": 0.08}
        self.results = {
            "sweep_values": delays,
            "measurement_probabilities": probabilities,
            "fit_parameters": self.fit_params,
        }
        return self.results


class T2RamseyMeasurement(_DemoCharacterization):
    """Fast deterministic T2 Ramsey measurement demo."""

    def __init__(
        self,
        delay_start: float = 0.0,
        delay_stop: float = 100.0,
        delay_points: int = 51,
        pi_half_pulse_amplitude: float = 0.25,
        detuning: float = 0.1,
        repeated_measurement_count: int = 1000,
        **kwargs,
    ):
        super().__init__(
            delay_start=delay_start,
            delay_stop=delay_stop,
            delay_points=delay_points,
            pi_half_pulse_amplitude=pi_half_pulse_amplitude,
            detuning=detuning,
            repeated_measurement_count=repeated_measurement_count,
            **kwargs,
        )

    def run(self):
        delays = self._delay_axis()
        t2 = max(self.delay_stop / 2.8, 1.0)
        envelope = np.exp(-delays / t2)
        probabilities = 0.50 + 0.42 * envelope * np.cos(2.0 * np.pi * self.detuning * delays)
        probabilities += self._rng().normal(0.0, 0.01, size=probabilities.shape)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        self.fit_params = {"t2_star": t2, "detuning": self.detuning, "offset": 0.50}
        self.results = {
            "sweep_values": delays,
            "measurement_probabilities": probabilities,
            "fit_parameters": self.fit_params,
        }
        return self.results


class T2EchoMeasurement(_DemoCharacterization):
    """Fast deterministic T2 echo measurement demo."""

    def __init__(
        self,
        delay_start: float = 0.0,
        delay_stop: float = 150.0,
        delay_points: int = 31,
        pi_pulse_amplitude: float = 0.5,
        pi_half_pulse_amplitude: float = 0.25,
        repeated_measurement_count: int = 1000,
        **kwargs,
    ):
        super().__init__(
            delay_start=delay_start,
            delay_stop=delay_stop,
            delay_points=delay_points,
            pi_pulse_amplitude=pi_pulse_amplitude,
            pi_half_pulse_amplitude=pi_half_pulse_amplitude,
            repeated_measurement_count=repeated_measurement_count,
            **kwargs,
        )

    def run(self):
        delays = self._delay_axis()
        t2_echo = max(self.delay_stop / 2.2, 1.0)
        probabilities = 0.10 + 0.82 * np.exp(-(delays / t2_echo) ** 1.3)
        probabilities += self._rng().normal(0.0, 0.01, size=probabilities.shape)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        self.fit_params = {"t2_echo": t2_echo, "amplitude": 0.82, "offset": 0.10}
        self.results = {
            "sweep_values": delays,
            "measurement_probabilities": probabilities,
            "fit_parameters": self.fit_params,
        }
        return self.results
