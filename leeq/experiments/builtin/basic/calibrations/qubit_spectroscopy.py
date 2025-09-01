from typing import Any, Optional, Union

import numpy as np
import plotly.graph_objects as go
from k_agents.inspection.decorator import visual_inspection

from leeq import Experiment, ExperimentManager, Sweeper, SweepParametersSideEffectFactory
from leeq.chronicle import log_and_record, register_browser_function
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils.compatibility import setup

__all__ = ['QubitSpectroscopyFrequency', 'QubitSpectroscopyAmplitudeFrequency']


# Helper class for simulation mode - moved outside method for pickling
class _MockMP:
    """Mock measurement primitive for simulation compatibility."""
    def __init__(self, parent, mprim):
        self.parent = parent
        self.freq = mprim.freq
        self.channel = mprim.channel
        self.amp = mprim.amp
    
    def result(self):
        return self.parent.trace


# Worker function for parallel IQ simulation - moved outside class for pickling
def _simulate_iq_point(freq: float, 
                       effective_amp_drive: float,
                       drive_channel: int,
                       readout_channel: int, 
                       f_readout: float,
                       effective_amp_readout: float,
                       sim) -> complex:
    """
    Worker function for parallel IQ response calculation.
    
    This function runs in a separate process and must be pickle-able.
    It calculates the IQ response for a single parameter point.
    
    Parameters
    ----------
    freq : float
        Drive frequency in MHz
    effective_amp_drive : float
        Effective drive amplitude in MHz
    drive_channel : int
        Drive channel number
    readout_channel : int
        Readout channel number
    f_readout : float
        Readout frequency in MHz
    effective_amp_readout : float
        Effective readout amplitude in MHz
    sim : CWSpectroscopySimulator
        Simulator instance (will be pickled/unpickled across processes)
        
    Returns
    -------
    complex
        Complex IQ response for this parameter point
    """
    # Simulate for this frequency and amplitude using the same interface
    iq_responses = sim.simulate_spectroscopy_iq(
        drives=[(drive_channel, freq, effective_amp_drive)],
        readout_params={readout_channel: {'frequency': f_readout, 
                                        'amplitude': effective_amp_readout}}
    )
    return iq_responses[readout_channel]


class QubitSpectroscopyFrequency(Experiment):
    EPII_INFO = {
        "name": "QubitSpectroscopyFrequency",
        "description": "Frequency sweep spectroscopy to find qubit resonances",
        "purpose": "Performs a frequency sweep on a qubit while keeping the drive amplitude fixed to identify the qubit's resonant frequency. The experiment drives the qubit at various frequencies and measures the resonator response to detect when the qubit is excited.",
        "attributes": {
            "mp": {
                "type": "MeasurementPrimitive",
                "description": "The measurement primitive used for the experiment"
            },
            "trace": {
                "type": "np.ndarray[complex]",
                "description": "Raw IQ trace data from the measurement",
                "shape": "(n_frequency_points,)"
            },
            "result": {
                "type": "dict",
                "description": "Processed results containing magnitude and phase",
                "keys": {
                    "Magnitude": "np.ndarray[float] - Magnitude of IQ response",
                    "Phase": "np.ndarray[float] - Unwrapped phase of IQ response"
                }
            },
            "freq_arr": {
                "type": "np.ndarray[float]",
                "description": "Frequency array for the sweep (MHz)",
                "shape": "(n_frequency_points,)"
            },
            "frequency_guess": {
                "type": "float",
                "description": "Estimated resonant frequency based on maximum deviation from baseline (MHz)"
            }
        },
        "notes": [
            "The frequency_guess uses first 10 points as baseline reference",
            "Phase is automatically unwrapped for continuity",
            "In simulation, disable_noise=True provides deterministic results",
            "Hardware mode ignores the disable_noise parameter"
        ]
    }
    
    """
    A class used to represent the QubitSweepPlottly experiment,
    specialized for conducting frequency sweeps on qubits and visualizing the results.

    This experiment sweeps the drive frequency to find qubit resonances. In simulation mode,
    a noise-free option is available via the disable_noise parameter to generate clean
    data for validation and benchmarking.

    Attributes
    ----------
    trace : ndarray
        A numpy array holding the trace data from the measurement.
    result : dict
        A dictionary holding the processed results of the experiment, specifically the magnitude and phase.
    frequency_guess : float
        A guess of the resonant frequency based on the data acquired in the sweep.

    Methods
    -------
    run(dut_qubit, res_freq, start, stop, step, num_avs, rep_rate, mp_width, amp, disable_noise=False):
        Runs the frequency sweep experiment on the qubit.
    run_simulated(dut_qubit, res_freq, start, stop, step, num_avs, rep_rate, mp_width, amp, disable_noise=False):
        Runs the simulated frequency sweep with optional noise-free mode.
    plot_magnitude():
        Plots the magnitude component of the results from the frequency sweep.
    plot_phase():
        Plots the phase component of the results from the frequency sweep.
    
    Examples
    --------
    Basic usage with default noisy simulation:
    
    >>> exp = QubitSpectroscopyFrequency(
    ...     dut_qubit=qubit,
    ...     start=4900.0,
    ...     stop=5100.0,
    ...     step=2.0,
    ...     num_avs=1000
    ... )
    
    Clean data generation for validation (simulation only):
    
    >>> exp_clean = QubitSpectroscopyFrequency(
    ...     dut_qubit=qubit,
    ...     start=4900.0,
    ...     stop=5100.0,
    ...     step=2.0,
    ...     num_avs=1000,
    ...     disable_noise=True
    ... )
    
    The disable_noise parameter provides deterministic results without baseline
    dropout or Gaussian noise, ideal for physics validation and testing.
    """

    @log_and_record
    def run(
            self,
            dut_qubit: Any,
            res_freq: Optional[float] = None,
            start: float = 3.e3,
            stop: float = 8.e3,
            step: float = 5.,
            num_avs: int = 1000,
            rep_rate: float = 0.,
            mp_width: float = 0.5,
            amp: float = 0.01,
            disable_noise: bool = False) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (qubit object).
        res_freq : float, optional
            The resonant frequency to set for the measurement primitive (MHz). Default: None.
        start : float, optional
            Start frequency for the sweep (MHz). Default: 3000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 8000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 1000
        rep_rate : float, optional
            Repetition rate for the pulse. Default: 0.0
        mp_width : float, optional
            Width for the measurement primitive pulse. Default: 0.5
        amp : float, optional
            Amplitude of the drive pulse. Default: 0.01
        disable_noise : bool, optional
            If True, disable noise in simulation mode for clean data generation.
            Ignored in hardware mode. Default: False.

        Returns
        -------
        None
            Results are stored in instance attributes.
        
        Notes
        -----
        The disable_noise parameter is only effective when running in simulation mode.
        Hardware experiments will ignore this parameter for safety reasons.
        """

        # Get the measurement primitive and update its parameters
        mp = dut_qubit.get_default_measurement_prim_int()

        # save the measurement primitive for live plot use
        self.mp = mp

        if mp_width is None:
            mp.update_pulse_args(width=rep_rate)
        else:
            mp.update_pulse_args(width=mp_width)

        mp.set_transform_function(None)
        if res_freq is not None:
            mp.update_freq(res_freq)

        # Prepare the pulse for the experiment
        pulse = dut_qubit.get_default_c1()['X'].clone()
        pulse.update_pulse_args(amp=amp, width=mp_width, shape='square')

        # Combine the pulse and measurement primitive
        lpb = pulse * mp

        # Set up the frequency sweeper
        swp = Sweeper(
            np.arange,
            n_kwargs={
                'start': start,
                'stop': stop,
                'step': step},
            params=[
                SweepParametersSideEffectFactory.func(
                    pulse.update_freq,
                    {},
                    'freq',
                    name='freq')])

        # Conduct the experiment with the specified parameters
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp)

        # Retrieve and process the results
        self.trace = np.squeeze(mp.result())
        self.result = {
            'Magnitude': np.absolute(self.trace),
            'Phase': np.unwrap(np.angle(self.trace)),
        }
        
        # Store frequency array for later access
        self.freq_arr = np.arange(start=start, stop=stop, step=step)

        # Estimate the resonant frequency based on the results
        mean_level = np.average(self.result['Magnitude'][:10])
        self.frequency_guess = self.freq_arr[
            np.argmax(abs(self.result['Magnitude'] - mean_level))]

    @log_and_record(overwrite_func_name='QubitSpectroscopyFrequency.run')
    def run_simulated(
            self,
            dut_qubit: Any,
            res_freq: Optional[float] = None,
            start: float = 3.e3,
            stop: float = 8.e3,
            step: float = 5.,
            num_avs: int = 1000,
            rep_rate: float = 0.,
            mp_width: float = 0.5,
            amp: float = 0.01,
            disable_noise: bool = False) -> None:
        """
        Execute the experiment in simulation mode.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (qubit object).
        res_freq : float, optional
            The resonant frequency to set for the measurement primitive (MHz). Default: None.
        start : float, optional
            Start frequency for the sweep (MHz). Default: 3000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 8000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 1000
        rep_rate : float, optional
            Repetition rate for the pulse. Default: 0.0
        mp_width : float, optional
            Width for the measurement primitive pulse. Default: 0.5
        amp : float, optional
            Amplitude of the drive pulse. Default: 0.01
        disable_noise : bool, optional
            If True, skip noise addition (baseline dropout and Gaussian noise) for clean 
            simulation data. Useful for physics validation and testing. Default: False.

        Returns
        -------
        None
            Results are stored in instance attributes.
        
        Notes
        -----
        When disable_noise=True:
        - No baseline dropout (20% of points normally set to mean value)
        - No Gaussian noise (normally scaled by 100/log(num_avs)/sqrt(mp_width))
        - Results are deterministic and repeatable
        - Ideal for comparing against theoretical models
        
        Examples
        --------
        Standard noisy simulation:
        
        >>> exp = QubitSpectroscopyFrequency(
        ...     dut_qubit=qubit,
        ...     start=4900.0, stop=5100.0, step=2.0,
        ...     num_avs=1000
        ... )
        
        Clean simulation for validation:
        
        >>> exp_clean = QubitSpectroscopyFrequency(
        ...     dut_qubit=qubit,
        ...     start=4900.0, stop=5100.0, step=2.0,
        ...     num_avs=1000,
        ...     disable_noise=True
        ... )
        """

        simulator_setup = setup().get_default_setup()
        
        # Prepare parameters
        mprim = dut_qubit.get_default_measurement_prim_intlist()
        f_readout = mprim.freq if res_freq is None else res_freq
        
        # Get drive channel info
        drive_channel = dut_qubit.get_default_c1().channel
        omega_per_amp_drive = simulator_setup.get_omega_per_amp(drive_channel)
        effective_amp_drive = amp * omega_per_amp_drive
        
        # For readout, we need to check if there's a separate readout channel
        # If the measurement primitive has a different channel, use that
        # Otherwise use the drive channel (single-channel case)
        readout_channel = mprim.channel if hasattr(mprim, 'channel') else drive_channel
        omega_per_amp_readout = simulator_setup.get_omega_per_amp(readout_channel)
        effective_amp_readout = mprim.amp * omega_per_amp_readout
        
        # Use new CW spectroscopy simulator
        from leeq.theory.simulation.numpy.cw_spectroscopy import CWSpectroscopySimulator
        
        sim = CWSpectroscopySimulator(simulator_setup)
        freq_qdrive = np.arange(start, stop, step)
        response = []
        
        for freq in freq_qdrive:
            iq_responses = sim.simulate_spectroscopy_iq(
                drives=[(drive_channel, freq, effective_amp_drive)],
                readout_params={readout_channel: {'frequency': f_readout, 
                                               'amplitude': effective_amp_readout}}
            )
            response.append(iq_responses[readout_channel])
        
        response = np.array(response)
        
        if not disable_noise:
            # Add noise (same as original)
            num_elements_to_baseline = int(len(response) * 0.2)
            indices_to_baseline = np.random.choice(response.size, num_elements_to_baseline, replace=False)
            response[indices_to_baseline] = response.mean()
            
            noise_scale = 100 / np.log(num_avs) / np.sqrt(mp_width if mp_width else 0.5)
            noise = (np.random.normal(0, noise_scale, response.shape) + 
                     1j * np.random.normal(0, noise_scale, response.shape))
            
            self.trace = response + noise
        else:
            # Clean response without noise
            self.trace = response
        self.result = {
            'Magnitude': np.absolute(self.trace),
            'Phase': np.unwrap(np.angle(self.trace)),
        }
        
        # Store frequency array for later access
        self.freq_arr = freq_qdrive
        
        # Frequency guess
        mean_level = np.average(self.result['Magnitude'][:10])
        self.frequency_guess = freq_qdrive[np.argmax(abs(self.result['Magnitude'] - mean_level))]

    @register_browser_function()
    @visual_inspection(
        """
        Given a plot of the phase response of a resonator as a function of frequency, analyze the stability and
        features of the phase curve. Identify any distinct and sharp deviations from the baseline phase level,
        such as steep dips or peaks. If such features are present at specific frequencies, indicating a significant
        change in the phase response, conclude that a qubit has been detected at these frequencies. If the plot
        shows a noisy or relatively stable phase without any pronounced features across the frequency range, conclude
        that no qubit has been observed
        """
    )
    def plot_magnitude(self, step_no: tuple[int] = None) -> go.Figure:
        """
        Generates a plot for the magnitude response from the frequency sweep data.

        Parameters:
        -----------
        step_no : tuple[int]
            The number of steps to plot (default is None).

        Returns
        -------
        fig : go.Figure object
        """

        # Retrieve the arguments used in the `run` method
        args = self._get_run_args_dict()

        # Create a new plot
        fig = go.Figure()

        # Generate frequency array for the x-axis
        f = np.arange(args['start'], args['stop'], args['step'])

        # Add the magnitude data as a trace to the plot
        trace = self.trace
        y = np.absolute(trace)

        if step_no is not None:
            f = f[:step_no[0]]
            y = y[:step_no[0]]

        fig.add_trace(go.Scatter(x=f, y=y,
                                 mode='lines',
                                 name='Magnitude'))

        # Update the layout of the plot
        fig.update_layout(
            title='Qubit spectroscopy resonator response magnitude',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Magnitude',
            plot_bgcolor='white')

        return fig

    @register_browser_function()
    def plot_phase(self, step_no: tuple[int] = None) -> go.Figure:
        """
        Generates a plot for the phase response from the frequency sweep data.

        Parameters:
        -----------
        step_no : tuple[int]
            The number of steps to plot (default is None).

        Returns
        -------
        go.Figure object
        """

        # Retrieve the arguments used in the `run` method
        args = self._get_run_args_dict()

        # Create a new plot
        fig = go.Figure()

        # Generate frequency array for the x-axis
        f = np.arange(args['start'], args['stop'], args['step'])

        # Add the phase data as a trace to the plot
        trace = self.trace
        y = np.unwrap(np.angle(trace))

        if step_no is not None:
            f = f[:step_no[0]]
            y = y[:step_no[0]]

        fig.add_trace(go.Scatter(x=f, y=y,
                                 mode='lines',
                                 name='Phase'))

        # Update the layout of the plot
        fig.update_layout(title='Qubit spectroscopy resonator response phase',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Phase', plot_bgcolor='white')

        return fig

    def live_plots(self, step_no: tuple[int]):
        """
        Generates live plots for the magnitude and phase responses from the frequency sweep data.

        Parameters:
        -----------
        step_no : tuple[int]
            The number of steps to plot (default is None).

        Returns
        -------
        figure : go.Figure object
        """
        return self.plot_phase(step_no)


class QubitSpectroscopyAmplitudeFrequency(Experiment):
    EPII_INFO = {
        "name": "QubitSpectroscopyAmplitudeFrequency",
        "description": "2D spectroscopy sweeping both frequency and amplitude",
        "purpose": "Performs a 2D sweep of both frequency and amplitude to map out the qubit response across different drive conditions. This helps identify power-dependent effects, multiphoton transitions, and optimal drive parameters for qubit control.",
        "attributes": {
            "mp": {
                "type": "MeasurementPrimitive or _MockMP",
                "description": "The measurement primitive used for the experiment"
            },
            "trace": {
                "type": "np.ndarray[complex]",
                "description": "2D array of raw IQ trace data",
                "shape": "(n_amplitude_points, n_frequency_points)"
            },
            "result": {
                "type": "dict",
                "description": "Processed results containing magnitude and phase",
                "keys": {
                    "Magnitude": "np.ndarray[float] - 2D magnitude of IQ response",
                    "Phase": "np.ndarray[float] - 2D phase of IQ response"
                }
            },
            "freq_arr": {
                "type": "np.ndarray[float]",
                "description": "Frequency array for the sweep (MHz)",
                "shape": "(n_frequency_points,)"
            },
            "amp_arr": {
                "type": "np.ndarray[float]",
                "description": "Amplitude array for the sweep",
                "shape": "(n_amplitude_points,)"
            },
            "performance_metrics": {
                "type": "dict",
                "description": "Performance metrics for simulation (simulation mode only)",
                "keys": {
                    "execution_time": "float - Time taken in seconds",
                    "memory_used_mb": "float - Memory usage in MB",
                    "parallel_enabled": "bool - Whether parallel processing was used",
                    "num_workers": "int - Number of worker processes",
                    "grid_size": "tuple - (n_amps, n_freqs)",
                    "total_points": "int - Total number of points simulated"
                }
            }
        },
        "notes": [
            "2D sweep creates amplitude x frequency grid",
            "Parallel processing available in simulation for 4-8x speedup",
            "disable_noise=True provides clean 2D maps for validation",
            "Phase is not unwrapped in 2D to preserve structure"
        ]
    }
    
    """
    A class used to represent the Qubit Spectroscopy Amplitude Frequency experiment.
    
    This experiment performs a 2D sweep of both frequency and amplitude to map out
    the qubit response across different drive conditions. In simulation mode,
    a noise-free option is available via the disable_noise parameter.

    Attributes
    ----------
    mp : Type of mp
        The measurement primitive used for the experiment.
    trace : np.ndarray
        A 2D numpy array holding the complex IQ trace data from the measurement.
    result : Dict[str, np.ndarray]
        Dictionary containing 'Magnitude' and 'Phase' arrays of the measured response.

    Methods
    -------
    run(dut_qubit, start, stop, step, num_avs, rep_rate, mp_width, qubit_amp_start, qubit_amp_stop, qubit_amp_step, disable_noise=False):
        Executes the qubit spectroscopy experiment on hardware.
    run_simulated(dut_qubit, start, stop, step, num_avs, rep_rate, mp_width, qubit_amp_start, qubit_amp_stop, qubit_amp_step, disable_noise=False):
        Executes the simulated qubit spectroscopy experiment with optional noise-free mode.
    plot_magnitude():
        Plots the magnitude from the experiment results.
    plot_magnitude_logscale():
        Plots the magnitude from the experiment results in logarithmic scale.
    plot_phase():
        Plots the phase from the experiment results.
    _plot(data, name, log_scale=False):
        Helper function to create a plotly plot.
        
    Examples
    --------
    Standard 2D spectroscopy with noise:
    
    >>> exp = QubitSpectroscopyAmplitudeFrequency(
    ...     dut_qubit=qubit,
    ...     start=4900.0, stop=5100.0, step=5.0,
    ...     qubit_amp_start=0.01, qubit_amp_stop=0.05, qubit_amp_step=0.002,
    ...     num_avs=1000
    ... )
    
    Clean 2D map for validation:
    
    >>> exp_clean = QubitSpectroscopyAmplitudeFrequency(
    ...     dut_qubit=qubit,
    ...     start=4900.0, stop=5100.0, step=5.0,
    ...     qubit_amp_start=0.01, qubit_amp_stop=0.05, qubit_amp_step=0.002,
    ...     num_avs=1000,
    ...     disable_noise=True
    ... )
    """

    @log_and_record
    def run(self,
            dut_qubit: Any,
            start: float = 3.e3,
            stop: float = 8.e3,
            step: float = 5.,
            num_avs: int = 1000,
            rep_rate: float = 0.,
            mp_width: Union[int,
                            None] = 0.5,
            qubit_amp_start: float = 0.01,
            qubit_amp_stop: float = 0.03,
            qubit_amp_step: float = 0.001,
            disable_noise: bool = False,
            use_parallel: bool = True,
            num_workers: Union[int, None] = None) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (qubit object).
        start : float, optional
            Start frequency for the sweep (MHz). Default: 3000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 8000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 1000
        rep_rate : float, optional
            Repetition rate for the pulse. Default: 0.0
        mp_width : Union[int, None], optional
            Measurement primitive width, if None uses rep_rate. Default: 0.5
        qubit_amp_start : float, optional
            Start amplitude for qubit drive. Default: 0.01
        qubit_amp_stop : float, optional
            Stop amplitude for qubit drive. Default: 0.03
        qubit_amp_step : float, optional
            Amplitude increment for qubit drive. Default: 0.001
        disable_noise : bool, optional
            If True, disable noise in simulation mode for clean data generation.
            Ignored in hardware mode. Default: False.
        use_parallel : bool, optional
            If True, use CPU parallelization for steady-state calculations to achieve 
            4-8x speedup. Only effective in simulation mode. Default: True.
        num_workers : Union[int, None], optional
            Number of worker processes for parallel execution. If None, auto-detect
            CPU cores. Only used when use_parallel=True. Default: None.

        Returns
        -------
        None
            Results are stored in instance attributes.
        
        Notes
        -----
        The disable_noise and use_parallel parameters are only effective when running 
        in simulation mode. Hardware experiments will ignore these parameters for 
        safety reasons.
        """

        # Clone and update measurement primitive based on given parameters
        mp = dut_qubit.get_default_measurement_prim_int().clone()
        mp.update_pulse_args(
            width=rep_rate) if mp_width is None else mp.update_pulse_args(
            width=mp_width)
        mp.set_transform_function(None)

        self.mp = mp
        
        # Store frequency and amplitude arrays
        self.freq_arr = np.arange(start, stop, step)
        self.amp_arr = np.arange(qubit_amp_start, qubit_amp_stop, qubit_amp_step)

        # Clone the default pulse
        pulse = dut_qubit.get_default_c1()['X'].clone()

        # Linear pulse builder (assumption based on the context)
        lpb = pulse * mp

        # Configure the frequency sweeper
        swp_freq = Sweeper(
            np.arange,
            n_kwargs={
                'start': start,
                'stop': stop,
                'step': step},
            params=[
                SweepParametersSideEffectFactory.func(
                    mp.update_freq,
                    {},
                    'freq')])

        # Configure the amplitude sweeper
        swp_amp = Sweeper(
            np.arange,
            n_kwargs={
                'start': qubit_amp_start,
                'stop': qubit_amp_stop,
                'step': qubit_amp_step},
            params=[
                SweepParametersSideEffectFactory.func(
                    pulse.update_pulse_args,
                    {},
                    'amp')])

        # Execute the experiment with configured parameters
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp_amp + swp_freq)

        # Process the results
        self.trace = np.squeeze(mp.result())
        self.result = {
            'Magnitude': np.absolute(
                self.trace), 'Phase': np.angle(
                self.trace)}

    @log_and_record(overwrite_func_name='QubitSpectroscopyAmplitudeFrequency.run')
    def run_simulated(
            self,
            dut_qubit: Any,
            start: float = 3.e3,
            stop: float = 8.e3,
            step: float = 5.,
            num_avs: int = 1000,
            rep_rate: float = 0.,
            mp_width: Union[int, None] = 0.5,
            qubit_amp_start: float = 0.01,
            qubit_amp_stop: float = 0.03,
            qubit_amp_step: float = 0.001,
            disable_noise: bool = False,
            use_parallel: bool = True,
            num_workers: Union[int, None] = None) -> None:
        """
        Execute the experiment in simulation mode.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (qubit object).
        start : float, optional
            Start frequency for the sweep (MHz). Default: 3000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 8000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 1000
        rep_rate : float, optional
            Repetition rate for the pulse. Default: 0.0
        mp_width : Union[int, None], optional
            Measurement primitive width, if None uses rep_rate. Default: 0.5
        qubit_amp_start : float, optional
            Start amplitude for qubit drive. Default: 0.01
        qubit_amp_stop : float, optional
            Stop amplitude for qubit drive. Default: 0.03
        qubit_amp_step : float, optional
            Amplitude increment for qubit drive. Default: 0.001
        disable_noise : bool, optional
            If True, skip noise addition (baseline dropout and Gaussian noise) for clean 
            2D simulation data. Useful for physics validation and testing. Default: False.
        use_parallel : bool, optional
            If True, use CPU parallelization for steady-state calculations to achieve 
            4-8x speedup. Default: True.
        num_workers : Union[int, None], optional
            Number of worker processes for parallel execution. If None, auto-detect
            CPU cores. Only used when use_parallel=True. Default: None.
        
        Notes
        -----
        When disable_noise=True:
        - No baseline dropout (20% of points normally set to mean value)
        - No Gaussian noise (normally scaled by 100/log(num_avs)/sqrt(mp_width))
        - Results are deterministic and repeatable across the entire 2D map
        - Ideal for comparing against theoretical models or benchmarking
        
        Examples
        --------
        Standard 2D noisy simulation:
        
        >>> exp = QubitSpectroscopyAmplitudeFrequency(
        ...     dut_qubit=qubit,
        ...     start=4900.0, stop=5100.0, step=5.0,
        ...     qubit_amp_start=0.01, qubit_amp_stop=0.05, qubit_amp_step=0.002,
        ...     num_avs=1000
        ... )
        
        Clean 2D map for analysis:
        
        >>> exp_clean = QubitSpectroscopyAmplitudeFrequency(
        ...     dut_qubit=qubit,
        ...     start=4900.0, stop=5100.0, step=5.0,
        ...     qubit_amp_start=0.01, qubit_amp_stop=0.05, qubit_amp_step=0.002,
        ...     num_avs=1000,
        ...     disable_noise=True
        ... )
        
        Fast parallel 2D sweep (4-8x speedup):
        
        >>> exp_parallel = QubitSpectroscopyAmplitudeFrequency(
        ...     dut_qubit=qubit,
        ...     start=4900.0, stop=5100.0, step=5.0,
        ...     qubit_amp_start=0.01, qubit_amp_stop=0.05, qubit_amp_step=0.002,
        ...     num_avs=1000,
        ...     use_parallel=True,
        ...     disable_noise=True
        ... )
        """
        
        # Get simulation setup
        simulator_setup = setup().get_default_setup()
        
        # Prepare parameters
        mprim = dut_qubit.get_default_measurement_prim_intlist()
        
        # Create a mock mp object that returns our simulated data
        self.mp = _MockMP(self, mprim)  # Store for plotting compatibility
        
        f_readout = mprim.freq
        
        # Get drive channel info
        drive_channel = dut_qubit.get_default_c1().channel
        omega_per_amp_drive = simulator_setup.get_omega_per_amp(drive_channel)
        
        # For readout, check if there's a separate readout channel
        readout_channel = mprim.channel if hasattr(mprim, 'channel') else drive_channel
        omega_per_amp_readout = simulator_setup.get_omega_per_amp(readout_channel)
        effective_amp_readout = mprim.amp * omega_per_amp_readout
        
        # Use new CW spectroscopy simulator
        from leeq.theory.simulation.numpy.cw_spectroscopy import CWSpectroscopySimulator
        
        sim = CWSpectroscopySimulator(simulator_setup)
        
        # Create frequency and amplitude arrays
        freq_array = np.arange(start, stop, step)
        amp_array = np.arange(qubit_amp_start, qubit_amp_stop, qubit_amp_step)
        
        # Store arrays as attributes for later access
        self.freq_arr = freq_array
        self.amp_arr = amp_array
        
        # Create 2D array to store results
        response_2d = np.zeros((len(amp_array), len(freq_array)), dtype=complex)
        
        # Performance monitoring setup
        import time
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # Perform 2D sweep - choose parallel or sequential
        if use_parallel:
            # Parallel processing using ProcessPoolExecutor
            try:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                
                # Auto-detect workers if not specified
                if num_workers is None:
                    num_workers = multiprocessing.cpu_count()
                
                print(f"Using parallel processing with {num_workers} workers...")
                
                # Create all parameter combinations
                param_points = []
                for i, amp in enumerate(amp_array):
                    effective_amp_drive = amp * omega_per_amp_drive
                    for j, freq in enumerate(freq_array):
                        param_points.append((i, j, freq, effective_amp_drive))
                
                # Process points in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            _simulate_iq_point,
                            freq, effective_amp_drive, drive_channel, 
                            readout_channel, f_readout, effective_amp_readout, sim
                        )
                        for i, j, freq, effective_amp_drive in param_points
                    ]
                    
                    # Collect results and place in 2D array
                    for idx, future in enumerate(futures):
                        i, j = param_points[idx][0], param_points[idx][1]
                        response_2d[i, j] = future.result()
                        
                print("Parallel processing completed successfully")
                
            except Exception as e:
                print(f"Warning: Parallel processing failed ({e}), falling back to sequential")
                use_parallel = False  # Fall back to sequential
        
        if not use_parallel:
            # Sequential processing (original implementation)
            for i, amp in enumerate(amp_array):
                effective_amp_drive = amp * omega_per_amp_drive
                
                for j, freq in enumerate(freq_array):
                    # Simulate for this frequency and amplitude
                    iq_responses = sim.simulate_spectroscopy_iq(
                        drives=[(drive_channel, freq, effective_amp_drive)],
                        readout_params={readout_channel: {'frequency': f_readout, 
                                                       'amplitude': effective_amp_readout}}
                    )
                    response_2d[i, j] = iq_responses[readout_channel]
        
        # Performance monitoring results
        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        execution_time = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Store performance metrics as attributes
        self.performance_metrics = {
            'execution_time': execution_time,
            'memory_used_mb': memory_used,
            'parallel_enabled': use_parallel,
            'num_workers': num_workers if use_parallel else 1,
            'grid_size': (len(amp_array), len(freq_array)),
            'total_points': len(amp_array) * len(freq_array)
        }
        
        print(f"2D sweep completed in {execution_time:.2f} seconds "
              f"({'parallel' if use_parallel else 'sequential'})")
        print(f"Memory usage: {memory_used:.1f} MB, Grid: {len(amp_array)}x{len(freq_array)} points")
        
        # Add noise to simulate realistic measurements
        if not disable_noise:
            width = mp_width if mp_width is not None else rep_rate
            if width is None:
                width = 0.5
                
            # Add baseline dropout (similar to 1D spectroscopy)
            num_elements_to_baseline = int(response_2d.size * 0.2)
            indices_to_baseline = np.random.choice(response_2d.size, num_elements_to_baseline, replace=False)
            response_2d_flat = response_2d.flatten()
            mean_value = response_2d_flat.mean()
            response_2d_flat[indices_to_baseline] = mean_value
            response_2d = response_2d_flat.reshape(response_2d.shape)
            
            # Add Gaussian noise
            noise_scale = 100 / np.log(num_avs) / np.sqrt(width)
            noise = (np.random.normal(0, noise_scale, response_2d.shape) + 
                     1j * np.random.normal(0, noise_scale, response_2d.shape))
            
            response_2d = response_2d + noise
        
        # Store results in the same format as the regular run method
        self.trace = response_2d
        self.result = {
            'Magnitude': np.absolute(self.trace),
            'Phase': np.angle(self.trace)
        }

    @register_browser_function()
    def plot_magnitude(self) -> go.Figure:
        """
        Plots the magnitude from the experiment results.

        Returns
        -------
        go.Figure
            A plotly graph object representing the magnitude plot.
        """

        trace = np.squeeze(self.mp.result())
        data = np.abs(trace)
        return self._plot(data, name='magnitude')

    @register_browser_function()
    def plot_magnitude_logscale(self) -> go.Figure:
        """
        Plots the magnitude from the experiment results in logarithmic scale.

        Returns
        -------
        go.Figure
            A plotly graph object representing the magnitude plot in log scale.
        """

        trace = np.squeeze(self.mp.result())
        data = np.abs(trace)
        return self._plot(np.log(data), name='logscale magnitude')

    @register_browser_function()
    def plot_phase(self) -> go.Figure:
        """
        Plots the phase from the experiment results.

        Returns
        -------
        go.Figure
            A plotly graph object representing the phase plot.
        """

        trace = np.squeeze(self.mp.result())
        data = np.unwrap(np.angle(trace))
        return self._plot(data=data, name='phase')

    def _plot(self, data: np.ndarray, name: str,
              log_scale: bool = False) -> go.Figure:
        """
        Helper function to create a plotly plot.

        Parameters
        ----------
        data : np.ndarray
            Array containing data to plot.
        name : str
            Name of the plot, used in the title.
        log_scale : bool, optional
            If True, plot y-axis will be in logarithmic scale (default is False).

        Returns
        -------
        go.Figure
            A plotly graph object.
        """

        # Retrieve arguments used during the 'run' function to use for plotting
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])
        amps = np.arange(
            args['qubit_amp_start'],
            args['qubit_amp_stop'],
            args['qubit_amp_step'])

        # Create a heatmap figure
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=f,
            y=amps,
            colorscale='Viridis'))

        # Update layout with titles
        fig.update_layout(xaxis_title='Frequency [MHz]',
                          yaxis_title='Qubit driving amplitude [a.u.]',
                          title=f'Qubit spectroscopy {name} sweep')

        return fig

    def live_plots(self, step_no: tuple[int]):
        """
        Generates live plots for the magnitude and phase responses from the frequency sweep data.

        Returns
        -------
        figure : go.Figure object
        """
        return self.plot_phase()
