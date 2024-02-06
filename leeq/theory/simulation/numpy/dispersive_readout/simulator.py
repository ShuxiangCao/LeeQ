from leeq.theory.simulation.numpy.dispersive_readout.utils import *
import numpy as np
from typing import List, Tuple, Union


class DispersiveReadoutSimulator:
    """
    A simulator for dispersive readout in quantum computing.

    Methods:
        _simulate_trace(state: int, f_prob: float, noise_std: float) -> np.ndarray:
            Simulates the signal trace for a given quantum state.
        plot_iq(states: List[int], f_prob: float, noise_std: float) -> None:
            Plots the I-Q trajectory for different quantum states.
    """

    def simulate_trace_with_decay(
            self,
            state: int,
            f_prob: float,
            noise_std: float = None,
            return_states_over_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulates a signal trace for a given quantum state with decay.
        Args:
            state: The initial state of the qubit.
            f_prob: The readout frequency used for probing the resonator.
            noise_std: the standard deviation for noise.
            return_states_over_time: whether to return the states over time.

        Returns:
            np.ndarray: Simulated signal trace with noise.
        """

        # Get the time list based on the sampling rate and the envelope width
        t_list = get_t_list(self.sampling_rate, width=self.width)

        t = 0
        current_state = state

        states_over_time = []

        while t < t_list[-1]:
            states_over_time.append((t, current_state))
            if current_state == 0:
                break
            expected_duration = self.t1s[current_state - 1]
            t += np.random.exponential(scale=expected_duration)
            current_state -= 1

        data, states_at_t = self._simulate_trace(
            state=states_over_time, f_prob=f_prob, noise_std=noise_std,
            return_states_over_time=return_states_over_time
        )

        if return_states_over_time:
            return data, np.asarray(states_at_t)
        else:
            return data

    def plot_iq(self, states: List[int], f_prob: float, noise_std: float = 0) -> None:
        """
        Plots the I-Q trajectory for different quantum states.

        Args:
            states (List[int]): Quantum states to plot.
            f_prob (float): The readout frequency used for probing the resonator.
            noise_std (float): Standard deviation for noise.
        """
        plt.figure()
        for state in states:
            trace = self._simulate_trace(state, f_prob, noise_std)
            trajectory = np.cumsum(trace)
            Is = np.real(trajectory)
            Qs = np.imag(trajectory)
            plt.plot(Is, Qs, label=str(state))
        plt.legend()
        plt.show()


class DispersiveReadoutSimulatorSyntheticData(DispersiveReadoutSimulator):
    """
    A simulator for dispersive readout in quantum computing.

    Attributes:
        f_r (float): Resonance frequency of the readout resonator.
        kappa (float): Bandwidth of the readout resonator.
        chis (List[float]): Chi shift for different quantum states.
        amp (float): Amplitude of the readout signal.
        baseline (float): Baseline level of the signal.
        width (float): Width of the signal pulse.
        rise (float): Rise time of the signal pulse.
        trunc (float): Truncation factor for the signal pulse.
        sampling_rate (float): Sampling rate for generating the signal.
    """

    def __init__(
            self,
            f_r: float,
            kappa: float,
            chis: List[float],
            amp: float = 1,
            baseline: float = 0.1,
            width: float = 10,
            rise: float = 0.0001,
            trunc: float = 1.2,
            sampling_rate: float = 1e3,
            t1s: List[float] = ([100.0, 50.0, 100 / 3],),
    ):
        self.f_r = f_r
        self.kappa = kappa
        self.chis = chis
        self.amp = amp
        self.baseline = baseline
        self.sampling_rate = sampling_rate
        self.rise = rise
        self.trunc = trunc
        self.width = width
        self.t1s = t1s

    def _simulate_trace(
            self,
            state: Union[int, List[Tuple[float, int]]],
            f_prob: float,
            noise_std: float = 0,
    ) -> np.ndarray:
        """
        Simulates a signal trace for a given quantum state.

        Args:
            state (int): Quantum state to simulate.
            f_prob (float): The readout frequency used for probing the resonator.
            noise_std (float): Standard deviation for noise.

        Returns:
            np.ndarray: Simulated signal trace with noise.
        """
        envelope = soft_square(
            sampling_rate=self.sampling_rate,
            amp=self.amp,
            phase=0,
            width=self.width,
            rise=self.rise,
            trunc=self.trunc,
        )

        # Get the time list based on the sampling rate and the envelope width
        t_list = get_t_list(self.sampling_rate, len(envelope) / self.sampling_rate)

        assert len(t_list) == len(envelope)

        if isinstance(state, int):
            state = [(0, state)]

        chis_ = {i: root_lorentzian(f=f_prob, f0=self.f_r + self.chis[i], kappa=self.kappa, amp=self.amp,
                                    baseline=self.baseline, ) for i in range(len(self.chis))}
        lorentzian_values = chis_

        # Find the state at different time points
        state_at_t = np.zeros(len(t_list))
        resonator_frequency_at_t = np.zeros(len(t_list))
        lorentzian_value_at_t = np.zeros(len(t_list), dtype=complex)

        # sort the state by t0
        state = sorted(state, key=lambda x: x[0])

        for t0, s in state:
            t_idx = t_list >= t0
            state_at_t[t_idx] = s
            resonator_frequency_at_t[t_idx] = self.f_r + self.chis[s]
            lorentzian_value_at_t[t_idx] = lorentzian_values[s]

        signal = lorentzian_value_at_t * envelope

        noise = np.random.normal(
            scale=noise_std, size=signal.shape
        ) + 1j * np.random.normal(scale=noise_std, size=signal.shape)
        return signal + noise

    pass


class DispersiveReadoutSimulatorRealData(DispersiveReadoutSimulator):
    """
    A simulator for dispersive readout in quantum computing.
    """

    def __init__(
            self,
            data_traces: np.ndarray,
            sampling_rate: float = 1e3,
            t1s: List[float] = ([100.0, 50.0, 100 / 3]),
    ):
        """
        Data trace is complex and in the following shape: (qubit_state, num_traces, num_samples)
        The qubit state is in the order of 0, 1, 2, 3.
        Args:
            data_traces: data traces for different qubit states,
            sampling_rate: sampling rate of the data traces,
            t1s: t1s for different qubit states,
        """
        self.data_traces = data_traces
        self.data_traces_mean = np.mean(data_traces, axis=1)
        self.data_traces_std = np.std(data_traces, axis=1)
        self.sampling_rate = sampling_rate
        self.width = self.data_traces_mean.shape[-1] / self.sampling_rate
        self.t1s = t1s

    def _simulate_trace(
            self,
            state: Union[int, List[Tuple[float, int]]],
            f_prob: float = 0,  # not used
            noise_std: float = None,
            return_states_over_time: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulates a signal trace for a given quantum state.

        Args:
            state (int): Quantum state to simulate.
            f_prob (float): The readout frequency used for probing the resonator.
            noise_std (float): Standard deviation for noise.
            return_states_over_time: whether to return the states over time.
        Returns:
            np.ndarray: Simulated signal trace with noise.

        """
        if noise_std is None:
            noise_std = self.data_traces_std.mean()

        if isinstance(state, int):
            state = [(0, state)]

        t_list = get_t_list(
            self.sampling_rate, self.width
        )  # np.zeros_like(self.data_traces[0, :, :], dtype=complex)

        # Find the state at different time points
        state_at_t = np.zeros(len(t_list))
        signal_at_t = np.zeros(len(t_list), dtype=complex)

        # sort the state by t0
        state = sorted(state, key=lambda x: x[0])

        for t0, s in state:
            t_idx = t_list >= t0
            state_at_t[t_idx] = s
            signal_at_t[t_idx] = self.data_traces_mean[s, t_idx]

        noise = np.random.normal(
            scale=noise_std, size=signal_at_t.shape
        ) + 1j * np.random.normal(scale=noise_std, size=signal_at_t.shape)
        signal_at_t = signal_at_t + noise

        if return_states_over_time:
            return signal_at_t, np.asarray(state_at_t)

        return signal_at_t


if __name__ == "__main__":
    sim = DispersiveReadoutSimulatorSyntheticData(
        f_r=9000,
        kappa=0.5,
        chis=[0, -0.25, -0.5, -0.75],
        t1s=[1.0, 5.0, 10 / 3],
        width=10,
    )
    data = sim.simulate_trace_with_decay(state=1, f_prob=9000)

    from matplotlib import pyplot as plt

    plt.plot(data.real, data.imag)
    plt.show()

    # sim.plot_iq(
    #    states=[0, [(0, 1), (2, 0)], [(0, 2), (2, 1), (4, 0)], 3],
    #    f_prob=9000,
    #    noise_std=5,
    # )
