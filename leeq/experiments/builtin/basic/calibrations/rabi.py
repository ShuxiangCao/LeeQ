from typing import Any, Dict, Optional, Union

import numpy as np
from k_agents.inspection.decorator import text_inspection, visual_inspection
from plotly import graph_objects as go

from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory import fits
from leeq.utils import setup_logging
from leeq.utils.compatibility import *

logger = setup_logging(__name__)

__all__ = ["NormalisedRabi", "MultiQubitRabi", "PowerRabi"]


class NormalisedRabi(Experiment):
    _experiment_result_analysis_instructions = """
    The Normalised Rabi experiment is a quantum mechanics experiment that involves the measurement of oscillations.
    A successful Rabi experiment will show a clear, regular oscillatory pattern with amplitude greater than 0.2.
    If less than 3 oscillations are observed, the experiment is considered failed. If more than 10 oscillations are observed, the experiment is considered failed. The new suggested driving amplitude should allow the observation of 5 oscillations, and can refer to the suggested amplitude in the analysis.
    """

    @log_and_record
    def run(self,
            dut_qubit: Any,
            amp: float = 0.2,
            start: float = 0.01,
            stop: float = 0.3,
            step: float = 0.002,
            fit: bool = True,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            pulse_discretization: bool = True,
            update=True,
            initial_lpb: Optional[Any] = None,
            drive_frequency: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Run a Rabi experiment on a given qubit for rough calibration of the driving amplitude.
        Note that this experiment is only for rough calibration, and the final calibration should be done using
        a more accurate method.

        Parameters:
        dut_qubit (Any): Device under test (DUT) qubit object.
        amp (float): Amplitude of the Rabi pulse. Default is 0.05.
        start (float): Start width for the pulse width sweep. Default is 0.01.
        stop (float): Stop width for the pulse width sweep. Default is 0.15.
        step (float): Step width for the pulse width sweep. Default is 0.001.
        fit (bool): Whether to fit the resulting data to a sinusoidal function. Default is True.
        collection_name (str): Collection name for retrieving c1. Default is 'f01'.
        mprim_index (int): Index for retrieving measurement primitive. Default is 0.
        pulse_discretization (bool): Whether to discretize the pulse. Default is False.
        update (bool): Whether to update the qubit parameters If you are tuning up the qubit set it to True. Default is False.
        initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.
        drive_frequency (Optional[float]): Override frequency for the Rabi pulse. If specified, overrides the pulse frequency. Default is None.

        Returns:
        Dict[str, Any]: Fitted parameters if fit is True, None otherwise.

        Example:
            >>> # Run an experiment to calibrate the driving amplitude of a single qubit gate
            >>> rabi_experiment = NormalisedRabi(
            >>> dut_qubit=dut, amp=0.05, start=0.01, stop=0.3, step=0.002, fit=True,
            >>> collection_name='f01', mprim_index=0, pulse_discretization=True, update=True)
        """
        # Get c1 from the DUT qubit
        c1 = dut_qubit.get_c1(collection_name)
        rabi_pulse = c1['X'].clone()

        if amp is not None:
            rabi_pulse.update_pulse_args(
                amp=amp, phase=0., shape='square', width=step)
        else:
            amp = rabi_pulse.amp

        # New frequency override logic
        if drive_frequency is not None:
            rabi_pulse.update_pulse_args(freq=drive_frequency)

        if not pulse_discretization:
            # Set up sweep parameters
            swpparams = [SweepParametersSideEffectFactory.func(
                rabi_pulse.update_pulse_args, {}, 'width'
            )]
            swp = Sweeper(
                np.arange,
                n_kwargs={'start': start, 'stop': stop, 'step': step},
                params=swpparams
            )
            pulse = rabi_pulse
        else:
            # Sometimes it is expensive to update the pulse envelope everytime, so we can keep the envelope the same
            # and just change the number of pulses
            pulse = LogicalPrimitiveBlockSweep([prims.SerialLPB(
                [rabi_pulse] * k, name='rabi_pulse') for k in range(int((stop - start) / step + 0.5))])
            swp = Sweeper.from_sweep_lpb(pulse)

        # Get the measurement primitive
        mprim = dut_qubit.get_measurement_prim_intlist(mprim_index)
        self.mp = mprim

        # Create the loopback pulse (lpb)
        lpb = pulse + mprim

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Run the basic experiment
        basic(lpb, swp, '<z>')

        # Extract the data
        self.data = np.squeeze(mprim.result())

        if not fit:
            return None

        # Fit data to a sinusoidal function and return the fit parameters
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=step)

        # Update the qubit parameters, to make one pulse width correspond to a pi pulse
        # Here we suppose all pulse envelopes give unit area when width=1,
        # amp=1
        normalised_pulse_area = c1['X'].calculate_envelope_area() / c1['X'].amp
        two_pi_area = amp * (1 / self.fit_params['Frequency'])
        new_amp = two_pi_area / 2 / normalised_pulse_area
        self.guess_amp = new_amp

        if update:
            c1.update_parameters(amp=new_amp)

    @log_and_record(overwrite_func_name='NormalisedRabi.run')
    def run_simulated(self,
                      dut_qubit: Any,
                      amp: float = 0.05,
                      start: float = 0.01,
                      stop: float = 0.15,
                      step: float = 0.001,
                      fit: bool = True,
                      collection_name: str = 'f01',
                      mprim_index: int = 0,
                      pulse_discretization: bool = True,
                      update=True,
                      initial_lpb: Optional[Any] = None,
                      drive_frequency: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Run a simulated Rabi experiment on a given qubit and analyze the results.

        Parameters:
        dut_qubit (Any): Device under test (DUT) qubit object.
        amp (float): Amplitude of the Rabi pulse. Default is 0.05.
        start (float): Start width for the pulse width sweep. Default is 0.01.
        stop (float): Stop width for the pulse width sweep. Default is 0.15.
        step (float): Step width for the pulse width sweep. Default is 0.001.
        fit (bool): Whether to fit the resulting data to a sinusoidal function. Default is True.
        collection_name (str): Collection name for retrieving c1. Default is 'f01'.
        mprim_index (int): Index for retrieving measurement primitive. Default is 0.
        pulse_discretization (bool): Whether to discretize the pulse. Default is False.
        update (bool): Whether to update the qubit parameters. Default is True.
        initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.
        drive_frequency (Optional[float]): Override frequency for the Rabi pulse. If specified, overrides the pulse frequency. Default is None.

        Returns:
        Dict[str, Any]: Fitted parameters if fit is True, None otherwise.

        """
        if initial_lpb is not None:
            logger.warning("initial_lpb is ignored in the simulated mode.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut_qubit)

        c1 = dut_qubit.get_c1(collection_name)

        # hard code a virtual dut here
        rabi_rate_per_amp = simulator_setup.get_omega_per_amp(
            c1.channel)  # MHz
        omega = rabi_rate_per_amp * amp

        # Detuning - use drive_frequency if provided, otherwise use pulse frequency
        pulse_frequency = drive_frequency if drive_frequency is not None else c1['X'].freq
        delta = virtual_transmon.qubit_frequency - pulse_frequency

        # Time array (let's consider 100 ns for demonstration)
        t = np.arange(start, stop, step)  # 1000 points from 0 to 100 ns

        # Rabi oscillation formula
        self.data = (omega ** 2) / (delta ** 2 + omega ** 2) * \
            np.sin(0.5 * np.sqrt(delta ** 2 + omega ** 2) * t) ** 2

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the
            # sampling noise
            self.data = np.random.binomial(
                shot_number, self.data) / shot_number

        quiescent_state_distribution = virtual_transmon.quiescent_state_distribution
        standard_deviation = np.sum(quiescent_state_distribution[1:])

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, self.data.shape)

        self.data = (2 * self.data - 1)

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, self.data.shape)

        random_noise_sum = np.random.normal(
            0, standard_deviation / 2, self.data.shape)

        self.data = np.clip(self.data * (0.5 - quiescent_state_distribution[0]) * 2 * random_noise_factor + random_noise_sum, -1, 1)

        # Fit data to a sinusoidal function and return the fit parameters
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=step)
        # Update the qubit parameters, to make one pulse width correspond to a pi pulse
        # Here we suppose all pulse envelopes give unit area when width=1,
        # amp=1
        normalised_pulse_area = c1['X'].calculate_envelope_area() / c1['X'].amp
        two_pi_area = amp * (1 / self.fit_params['Frequency'])
        new_amp = two_pi_area / 2 / normalised_pulse_area
        self.guess_amp = new_amp

        if update:
            c1.update_parameters(amp=new_amp)

    @register_browser_function()
    @visual_inspection("""
    Here is a plot of data from a quantum mechanics experiment involving Rabi oscillations. Can you analyze whether
        this plot shows a successful experiment or a failed one? Please consider the following aspects in your analysis:
    1. Clarity of Oscillation: Describe if the data points show a clear, regular oscillatory pattern.
    2. Amplitude and Frequency: Note any inconsistencies in the amplitude and frequency of the oscillations.
    3. Overall Pattern: Provide a general assessment of the plot based on the typical characteristics of successful
        Rabi oscillation experiments.
    """)
    def plot(self) -> go.Figure:
        """
        Plots Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features scatter points
        representing the original data and a sine fit for each qubit involved in the
        experiment.
        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(
            args['start'],
            args['stop'],
            args['step'] / 5)

        # Create subplots: each qubit's data gets its own plot
        fig = go.Figure()
        # Scatter plot of the actual data
        fig.add_trace(
            go.Scatter(
                x=t,
                y=self.data,
                mode='markers',
                marker={
                    "color": 'Blue',
                    "size": 7,
                    "opacity": 0.5,
                    "line": {"color": 'Black', "width": 2},
                },
                name='data'
            )
        )

        # Fit data
        f = self.fit_params['Frequency']
        a = self.fit_params['Amplitude']
        p = self.fit_params['Phase'] - 2.0 * np.pi * f * args['start']
        o = self.fit_params['Offset']
        fit = a * np.sin(2.0 * np.pi * f * t_interpolate + p) + o

        # Line plot of the fit
        fig.add_trace(
            go.Scatter(
                x=t_interpolate,
                y=fit,
                mode='lines',
                line={"color": 'Red'},
                name='fit',
                visible='legendonly'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Time Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font={
                "family": 'Courier New, monospace',
                "size": 12,
                "color": 'Black'
            },
            plot_bgcolor='white'
        )

        return fig

    def live_plots(self, step_no=None) -> go.Figure:
        """
        Plots Rabi oscillations live using data from an experiment run.

        Parameters:
        step_no (int): Number of steps to plot. Default is None.

        Returns:
        go.Figure: Plotly figure.

        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        data = np.squeeze(self.mp.result())

        # Create subplots: each qubit's data gets its own plot
        fig = go.Figure()
        # Scatter plot of the actual data
        fig.add_trace(
            go.Scatter(
                x=t[:step_no[0]],
                y=data[:step_no[0]],
                mode='lines',
                marker={
                    "color": 'Blue',
                    "size": 7,
                    "opacity": 0.5,
                    "line": {"color": 'Black', "width": 2}
                },
                name='data'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Time Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font={
                "family": 'Courier New, monospace',
                "size": 12,
                "color": 'Black'
            },
            plot_bgcolor='white'
        )

        return fig

    @text_inspection
    def fitting(self) -> str:
        """
        Get the prompt to analyze the result.

        Returns:
        str: The prompt to analyze the result.
        """

        oscillation_freq = self.fit_params['Frequency']
        experiment_time_duration = self._get_run_args_dict()['stop'] - self._get_run_args_dict()['start']
        oscillation_count = (experiment_time_duration * oscillation_freq)

        return (f"The fitting result of the Rabi oscillation suggest the amplitude of {self.fit_params['Amplitude']}, "
                f"the frequency of {self.fit_params['Frequency']}, the phase of {self.fit_params['Phase']}. The offset of"
                f" {self.fit_params['Offset']}. The suggested new driving amplitude is {self.guess_amp}."
                f"From the fitting results, the plot should exhibit {oscillation_count} oscillations.")


class PowerRabi(Experiment):
    @log_and_record
    def run(self,
            dut_qubit: Any,
            width: float = None,
            amp_start: float = 0.01,
            amp_stop: float = 0.4,
            amp_step: float = 0.01,
            fit: bool = True,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            update=True,
            initial_lpb: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Run a Rabi experiment on a given qubit for rough calibration of the driving amplitude.
        Note that this experiment is only for rough calibration, and the final calibration should be done using
        a more accurate method.

        Parameters:
        dut_qubit (Any): Device under test (DUT) qubit object.
        amp (float): Amplitude of the Rabi pulse. Default is 0.05.
        start (float): Start width for the pulse width sweep. Default is 0.01.
        stop (float): Stop width for the pulse width sweep. Default is 0.15.
        step (float): Step width for the pulse width sweep. Default is 0.001.
        fit (bool): Whether to fit the resulting data to a sinusoidal function. Default is True.
        collection_name (str): Collection name for retrieving c1. Default is 'f01'.
        mprim_index (int): Index for retrieving measurement primitive. Default is 0.
        pulse_discretization (bool): Whether to discretize the pulse. Default is False.
        update (bool): Whether to update the qubit parameters If you are tuning up the qubit set it to True. Default is False.
        initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.

        Returns:
        Dict[str, Any]: Fitted parameters if fit is True, None otherwise.

        Example:
            >>> # Run an experiment to calibrate the driving amplitude of a single qubit gate
            >>> rabi_experiment = PowerRabi(setup)
            >>> rabi_experiment.run(dut_qubit=dut, amp_start=0.01, amp_stop=0.4, amp_step=0.01)
        """
        # Get c1 from the DUT qubit
        c1 = dut_qubit.get_c1(collection_name)
        rabi_pulse = c1['X'].clone()

        if width is not None:
            rabi_pulse.update_pulse_args(
                width=width, phase=0., shape='square', amp=amp_start)
        else:
            pass

        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            rabi_pulse.update_pulse_args, {}, 'amp'
        )]
        swp = Sweeper(
            np.arange,
            n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
            params=swpparams
        )
        pulse = rabi_pulse

        # Get the measurement primitive
        mprim = dut_qubit.get_measurement_prim_intlist(mprim_index)
        self.mp = mprim

        # Create the loopback pulse (lpb)
        lpb = pulse + mprim

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Run the basic experiment
        basic(lpb, swp, '<z>')

        # Extract the data
        self.data = np.squeeze(mprim.result())

        if not fit:
            return None

        # Fit data to a sinusoidal function and return the fit parameters
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=amp_step)

        # The amplitude should be chosen such that the rabi rotation finish half a period.
        if update:
            self.optimal_amp = 1 / self.fit_params['Frequency'] / 2
            c1.update_parameters(amp=self.optimal_amp)

    @register_browser_function()
    def plot(self) -> go.Figure:
        """
        Plots Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features scatter points
        representing the original data and a sine fit for each qubit involved in the
        experiment.
        """

        args = self._get_run_args_dict()
        t = np.arange(args['amp_start'], args['amp_stop'], args['amp_step'])
        amp_interpolate = np.arange(
            args['amp_start'],
            args['amp_stop'],
            args['amp_step'] / 5)

        # Create subplots: each qubit's data gets its own plot
        fig = go.Figure()
        # Scatter plot of the actual data
        fig.add_trace(
            go.Scatter(
                x=t,
                y=self.data,
                mode='markers',
                marker={
                    "color": 'Blue',
                    "size": 7,
                    "opacity": 0.5,
                    "line": {"color": 'Black', "width": 2},
                },
                name='data'
            )
        )

        # Fit data
        f = self.fit_params['Frequency']
        a = self.fit_params['Amplitude']
        p = self.fit_params['Phase'] - 2.0 * np.pi * f * args['amp_start']
        o = self.fit_params['Offset']
        fit = a * np.sin(2.0 * np.pi * f * amp_interpolate + p) + o

        # Line plot of the fit
        fig.add_trace(
            go.Scatter(
                x=amp_interpolate,
                y=fit,
                mode='lines',
                line={"color": 'Red'},
                name='fit',
                visible='legendonly'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Power Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font={
                "family": 'Courier New, monospace',
                "size": 12,
                "color": 'Black'
            },
            plot_bgcolor='white'
        )

        return fig

    def live_plots(self, step_no=None) -> go.Figure:
        """
        Plots Rabi oscillations live using data from an experiment run.

        Parameters:
        step_no (int): Number of steps to plot. Default is None.

        Returns:
        go.Figure: Plotly figure.

        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        data = np.squeeze(self.mp.result())

        # Create subplots: each qubit's data gets its own plot
        fig = go.Figure()
        # Scatter plot of the actual data
        fig.add_trace(
            go.Scatter(
                x=t[:step_no[0]],
                y=data[:step_no[0]],
                mode='lines',
                marker={
                    "color": 'Blue',
                    "size": 7,
                    "opacity": 0.5,
                    "line": {"color": 'Black', "width": 2}
                },
                name='data'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Time Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font={
                "family": 'Courier New, monospace',
                "size": 12,
                "color": 'Black'
            },
            plot_bgcolor='white'
        )

        return fig

    @log_and_record(overwrite_func_name='PowerRabi.run')
    def run_simulated(self, dut_qubit, width=None, amp_start=0.01, amp_stop=0.4,
                      amp_step=0.01, fit=True, collection_name='f01', update=True):
        """
        Simulate power Rabi oscillations by sweeping amplitude.

        The Rabi frequency is: Ω = amp * omega_per_amp
        The population oscillates as: P = sin²(Ω * π_time / 2)
        """
        import numpy as np

        # Get setup and virtual qubit
        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_qubit = simulator_setup.get_virtual_qubit(dut_qubit)
        if virtual_qubit is None:
            raise ValueError(f"No virtual qubit found for {dut_qubit}")

        # Get calibration parameters
        c1 = dut_qubit.get_c1(collection_name)
        pi_pulse = c1['X']

        # Use provided width or get from existing pulse
        if width is not None:
            pi_time = width
        else:
            pi_time = pi_pulse.width

        # Get omega per amp from setup
        channel = c1.channel
        omega_per_amp = simulator_setup.get_omega_per_amp(channel)

        # Calculate expected oscillations
        amp_range = np.arange(amp_start, amp_stop, amp_step)
        rabi_frequencies = amp_range * omega_per_amp  # MHz

        # Calculate rotation angle for each amplitude
        # θ = Ω * t = (amp * omega_per_amp) * pi_time
        rotation_angles = rabi_frequencies * pi_time * 2 * np.pi  # radians

        # Calculate excited state population
        # P_e = sin²(θ/2) for starting from ground state
        populations = np.sin(rotation_angles / 2) ** 2

        # Add noise if enabled
        if setup().status().get_param('Sampling_Noise'):
            # Use a default readout fidelity of 0.95 for noise modeling
            readout_fidelity = 0.95
            noise_level = (1 - readout_fidelity) / 2
            noise = np.random.normal(0, noise_level, len(populations))
            populations = np.clip(populations + noise, 0, 1)

        # Store data for plotting
        self.data = populations

        if not fit:
            return None

        # Fit data to extract parameters
        self.fit_params = fits.fit_sinusoidal(populations, time_step=amp_step)

        # Find optimal amplitude for pi pulse
        if update:
            # The amplitude should give us a pi rotation (population = 1)
            # This happens when rotation_angle = pi, so amp * omega_per_amp * pi_time = 1
            self.optimal_amp = 1 / (omega_per_amp * pi_time)

            # Update the qubit parameters
            c1.update_parameters(amp=self.optimal_amp)

        # Don't return anything to match the regular run() behavior


class MultiQubitRabi(Experiment):
    @log_and_record
    def run(self,
            duts: list[Any],
            amps: Union[float, list[float]] = 0.05,
            start: float = 0.01,
            stop: float = 0.15,
            step: float = 0.001,
            fit: bool = True,
            collection_names: Union[str, list[str]] = 'f01',
            mprim_indexes: Union[int, list[int]] = 0,
            pulse_discretization: bool = True,
            update=False,
            initial_lpb: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Run a Rabi experiment on a given qubit and analyze the results.

        Parameters:
        duts (list[Any]): List of device under test (DUT) qubit objects.
        amps (Union[float, list(float)]): Amplitude of the Rabi pulse. Default is 0.05.
        start (float): Start width for the pulse width sweep. Default is 0.01.
        stop (float): Stop width for the pulse width sweep. Default is 0.15.
        step (float): Step width for the pulse width sweep. Default is 0.001.
        fit (bool): Whether to fit the resulting data to a sinusoidal function. Default is True.
        collection_names (Union[str, list(str)]): Collection name for retrieving c1. Default is 'f01'. If a list is
            provided, the length of the list must match the length of duts.
        mprim_indexes (Union[int, list(int)]): Index for retrieving measurement primitive. Default is 0. If a list is
            provided, the length of the list must match the length of duts.
        update (bool): Whether to update the qubit parameters. Default is False.
        initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.

        Returns:
        Dict[str, Any]: Fitted parameters if fit is True, None otherwise.
        """

        if not isinstance(amps, list):
            amps = [amps] * len(duts)
        if not isinstance(collection_names, list):
            collection_names = [collection_names] * len(duts)
        if not isinstance(mprim_indexes, list):
            mprim_indexes = [mprim_indexes] * len(duts)

        assert len(duts) == len(amps) == len(collection_names) == len(
            mprim_indexes), "Length of duts, amps, collection_names, and mprim_indexes must be the same."

        rabi_pulses = []

        for i in range(len(duts)):
            dut_qubit = duts[i]
            collection_name = collection_names[i]
            amp = amps[i]
            c1 = dut_qubit.get_c1(collection_name)
            rabi_pulse = c1['X'].clone()

            if amp is not None:
                rabi_pulse.update_pulse_args(
                    amp=amp, phase=0., shape='square', width=step)
            else:
                amps[i] = rabi_pulse.amp

            rabi_pulses.append(rabi_pulse)

        if not pulse_discretization:
            # Set up sweep parameters
            swpparams = [SweepParametersSideEffectFactory.func(
                rabi_pulse.update_pulse_args, {}, 'width'
            ) for rabi_pulse in rabi_pulses]
            swp = Sweeper(
                np.arange,
                n_kwargs={'start': start, 'stop': stop, 'step': step},
                params=swpparams
            )
            pulse = prims.ParallelLPB(rabi_pulses)
        else:
            # Sometimes it is expensive to update the pulse envelope everytime, so we can keep the envelope the same
            # and just change the number of pulses
            pulse = LogicalPrimitiveBlockSweep([
                prims.SerialLPB([prims.ParallelLPB(rabi_pulses)] * k, name='rabi_pulse') for k in
                range(int((stop - start) / step + 0.5))
            ])
            swp = Sweeper.from_sweep_lpb(pulse)

        # Get the measurement primitive
        mprims = [dut_qubit.get_measurement_prim_intlist(
            mprim_index) for dut_qubit, mprim_index in zip(duts, mprim_indexes, strict=False)]
        self.mps = mprims

        # Create the loopback pulse (lpb)
        lpb = pulse + prims.ParallelLPB(mprims)

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Run the basic experiment
        basic(lpb, swp, '<z>')

        # Extract the data
        self.data = [np.squeeze(mprim.result()) for mprim in mprims]

        if not fit:
            return None

        # Fit data to a sinusoidal function and return the fit parameters
        self.fit_params = [
            fits.fit_sinusoidal(
                data, time_step=step) for data in self.data]

        if update:
            for i in range(len(duts)):
                # Update the qubit parameters, to make one pulse width correspond to a pi pulse
                # Here we suppose all pulse envelopes give unit area when
                # width=1, amp=1
                c1 = duts[i].get_c1(collection_names[i])
                normalised_pulse_area = c1['X'].calculate_envelope_area(
                ) / c1['X'].amp
                two_pi_area = amps[i] * (1 / self.fit_params[i]['Frequency'])
                new_amp = two_pi_area / 2 / normalised_pulse_area
                c1.update_parameters(amp=new_amp)

    @register_browser_function()
    def plot_all(self):
        for i in range(len(self.data)):
            fig = self.plot(i)
            fig.show()

    def plot(self, i) -> go.Figure:
        """
        Plots Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features scatter points
        representing the original data and a sine fit for each qubit involved in the
        experiment.

        Parameters:
            i (int): Index of the qubit to plot.
        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(
            args['start'],
            args['stop'],
            args['step'] / 5)

        # Create subplots: each qubit's data gets its own plot
        fig = go.Figure()
        # Scatter plot of the actual data
        fig.add_trace(
            go.Scatter(
                x=t,
                y=self.data[i],
                mode='markers',
                marker={
                    "color": 'Blue',
                    "size": 7,
                    "opacity": 0.5,
                    "line": {"color": 'Black', "width": 2}
                },
                name='data'
            )
        )

        # Fit data
        f = self.fit_params[i]['Frequency']
        a = self.fit_params[i]['Amplitude']
        p = self.fit_params[i]['Phase'] - 2.0 * np.pi * f * args['start']
        o = self.fit_params[i]['Offset']
        fit = a * np.sin(2.0 * np.pi * f * t_interpolate + p) + o

        # Line plot of the fit
        fig.add_trace(
            go.Scatter(
                x=t_interpolate,
                y=fit,
                mode='lines',
                line={"color": 'Red'},
                name='fit'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title=f'Time Rabi: {args["duts"][i].hrid}',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font={
                "family": 'Courier New, monospace',
                "size": 12,
                "color": 'Black'
            },
            plot_bgcolor='white'
        )

        return fig

    @log_and_record(overwrite_func_name='MultiQubitRabi.run')
    def run_simulated(self,
                      duts: list[Any],
                      amps: Union[float, list[float]] = 0.05,
                      start: float = 0.01,
                      stop: float = 0.15,
                      step: float = 0.001,
                      fit: bool = True,
                      collection_names: Union[str, list[str]] = 'f01',
                      mprim_indexes: Union[int, list[int]] = 0,
                      pulse_discretization: bool = True,
                      update=False,
                      initial_lpb: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Run simulated multi-qubit Rabi experiment.

        Parameters same as run() method.
        """
        # Convert scalar parameters to lists
        if not isinstance(amps, list):
            amps = [amps] * len(duts)
        if not isinstance(collection_names, list):
            collection_names = [collection_names] * len(duts)
        if not isinstance(mprim_indexes, list):
            mprim_indexes = [mprim_indexes] * len(duts)

        assert len(duts) == len(amps) == len(collection_names) == len(mprim_indexes), \
            "Length of duts, amps, collection_names, and mprim_indexes must be the same."

        # Get simulation setup
        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()

        # Create time array
        time_points = np.arange(start, stop, step)

        # Simulate for each qubit
        self.data = []

        for i, dut in enumerate(duts):
            # Get virtual qubit
            virtual_qubit = simulator_setup.get_virtual_qubit(dut)
            if virtual_qubit is None:
                raise ValueError(f"No virtual qubit found for {dut}")

            # Get calibration info
            c1 = dut.get_c1(collection_names[i])
            channel = c1.channel

            # Calculate Rabi frequency
            omega_per_amp = simulator_setup.get_omega_per_amp(channel)
            omega = amps[i] * omega_per_amp  # MHz

            # Get detuning
            delta = virtual_qubit.qubit_frequency - c1['X'].freq

            # Calculate effective Rabi frequency
            omega_eff = np.sqrt(omega**2 + delta**2)

            # Calculate Rabi oscillations
            populations = (omega / omega_eff)**2 * np.sin(0.5 * omega_eff * time_points * 2 * np.pi)**2

            # Convert to expectation value <z> = 1 - 2*P_e
            z_expectation = 1 - 2 * populations

            # Apply T1/T2 decoherence if enabled
            if hasattr(virtual_qubit, 't1') and hasattr(virtual_qubit, 't2'):
                # Apply exponential decay from decoherence
                t1_decay = np.exp(-time_points / virtual_qubit.t1)
                t2_decay = np.exp(-time_points / virtual_qubit.t2)

                # Decoherence affects the oscillation amplitude
                z_expectation = z_expectation * t2_decay + (1 - t1_decay)

            # Add sampling noise if enabled
            if setup().status().get_param('Sampling_Noise'):
                shot_number = setup().status().get_param('Shot_Number')
                # Convert z expectation to probability
                prob_excited = (1 - z_expectation) / 2
                # Simulate binomial sampling
                counts_excited = np.random.binomial(shot_number, prob_excited)
                # Convert back to z expectation
                z_expectation = 1 - 2 * counts_excited / shot_number

            self.data.append(z_expectation)

        if not fit:
            return None

        # Fit data for each qubit
        self.fit_params = [
            fits.fit_sinusoidal(data, time_step=step) for data in self.data
        ]

        if update:
            for i in range(len(duts)):
                # Update the qubit parameters
                c1 = duts[i].get_c1(collection_names[i])
                normalised_pulse_area = c1['X'].calculate_envelope_area() / c1['X'].amp
                two_pi_area = amps[i] * (1 / self.fit_params[i]['Frequency'])
                new_amp = two_pi_area / 2 / normalised_pulse_area
                c1.update_parameters(amp=new_amp)
