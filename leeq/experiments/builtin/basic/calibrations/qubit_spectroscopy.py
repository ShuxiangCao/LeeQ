import numpy as np
import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function
from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory, ExperimentManager

from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go


class QubitSpectroscopyFrequency(Experiment):
    """
    A class used to represent the QubitSweepPlottly experiment,
    specialized for conducting frequency sweeps on qubits and visualizing the results.

    ...

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
    run(dut_qubit, res_freq, start, stop, step, num_avs, rep_rate, mp_width, amp):
        Runs the frequency sweep experiment on the qubit.
    plot_magnitude():
        Plots the magnitude component of the results from the frequency sweep.
    plot_phase():
        Plots the phase component of the results from the frequency sweep.
    """

    @log_and_record
    def run(self, dut_qubit: Any, res_freq: Optional[float] = None, start: float = 3.e3, stop: float = 8.e3,
            step: float = 5., num_avs: int = 1000,
            rep_rate: float = 0., mp_width: float = 0.5, amp: float = 0.01) -> None:
        """
        Conducts a frequency sweep on the designated qubit and records the response.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (DUT), which is the qubit on which the experiment is performed.
        res_freq : float, optional
            The resonant frequency to set for the measurement primitive (default is None).
        start : float
            The start frequency for the sweep (default is 3000 MHz).
        stop : float
            The stop frequency for the sweep (default is 8000 MHz).
        step : float
            The frequency increment for the sweep (default is 5 MHz).
        num_avs : int
            The number of averages to take during the measurement (default is 500).
        rep_rate : float
            The repetition rate for the pulse (default is 0).
        mp_width : float
            The width for the measurement primitive pulse (default is 0.5).
        amp : float
            The amplitude of the pulse (default is 1).

        Returns
        -------
        None
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
        swp = Sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step}, params=[
            SweepParametersSideEffectFactory.func(pulse.update_freq, {}, 'freq', name='freq')])

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

        # Estimate the resonant frequency based on the results
        mean_level = np.average(self.result['Magnitude'][:10])
        self.frequency_guess = np.arange(start=start, stop=stop, step=step)[
            np.argmax(abs(self.result['Magnitude'] - mean_level))]

    @register_browser_function(available_after=(run,))
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
        args = self.retrieve_args(self.run)

        # Create a new plot
        fig = go.Figure()

        # Generate frequency array for the x-axis
        f = np.arange(args['start'], args['stop'], args['step'])

        # Add the magnitude data as a trace to the plot
        trace = np.squeeze(self.mp.result())
        y = np.absolute(trace)

        if step_no is not None:
            f = f[:step_no[0]]
            y = y[:step_no[0]]

        fig.add_trace(go.Scatter(x=f, y=y,
                                 mode='lines',
                                 name='Magnitude'))

        # Update the layout of the plot
        fig.update_layout(title='Qubit spectroscopy resonator response magnitude',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Magnitude', plot_bgcolor='white')

        return fig

    @register_browser_function(available_after=(run,))
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
        args = self.retrieve_args(self.run)

        # Create a new plot
        fig = go.Figure()

        # Generate frequency array for the x-axis
        f = np.arange(args['start'], args['stop'], args['step'])

        # Add the phase data as a trace to the plot
        trace = np.squeeze(self.mp.result())
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
    """
    A class used to represent the Qubit Spectroscopy Amplitude Frequency experiment.

    ...

    Attributes
    ----------
    mp : Type of mp
        Description of mp
    trace : np.ndarray
        Description of trace
    result : Dict[str, np.ndarray]
        Description of result

    Methods
    -------
    run(dut_qubit, start, stop, step, num_avs, rep_rate, mp_width, qubit_amp_start, qubit_amp_stop, qubit_amp_step):
        Executes the qubit spectroscopy experiment.
    plot_magnitude():
        Plots the magnitude from the experiment results.
    plot_magnitude_logscale():
        Plots the magnitude from the experiment results in logarithmic scale.
    plot_phase():
        Plots the phase from the experiment results.
    _plot(data, name, log_scale=False):
        Helper function to create a plotly plot.
    """

    @log_and_record
    def run(self, dut_qubit: Any, start: float = 3.e3, stop: float = 8.e3, step: float = 5.,
            num_avs: int = 1000, rep_rate: float = 0., mp_width: Union[int, None] = 0.5,
            qubit_amp_start: float = 0.01, qubit_amp_stop: float = 0.03, qubit_amp_step: float = 0.001) -> None:
        """
        Executes the qubit spectroscopy experiment by sweeping both the frequency and amplitude.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (DUT) - qubit.
        start : float
            The starting frequency in MHz.
        stop : float
            The stopping frequency in MHz.
        step : float
            The frequency step in MHz.
        num_avs : int
            Number of averages for measurement.
        rep_rate : float
            The repetition rate in some units (needs clarification).
        mp_width : Union[int, None]
            Measurement primitive width, if None, width is set to rep_rate.
        qubit_amp_start : float
            The start amplitude for qubit.
        qubit_amp_stop : float
            The stop amplitude for qubit.
        qubit_amp_step : float
            The amplitude step for qubit.
        """

        # Clone and update measurement primitive based on given parameters
        mp = dut_qubit.get_default_measurement_prim_int().clone()
        mp.update_pulse_args(width=rep_rate) if mp_width is None else mp.update_pulse_args(width=mp_width)
        mp.set_transform_function(None)

        self.mp = mp

        # Clone the default pulse
        pulse = dut_qubit.get_default_c1()['X'].clone()

        # Linear pulse builder (assumption based on the context)
        lpb = pulse * mp

        # Configure the frequency sweeper
        swp_freq = Sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step}, params=[
            SweepParametersSideEffectFactory.func(mp.update_freq, {}, 'freq')])

        # Configure the amplitude sweeper
        swp_amp = Sweeper(np.arange,
                          n_kwargs={'start': qubit_amp_start, 'stop': qubit_amp_stop, 'step': qubit_amp_step}, params=[
                SweepParametersSideEffectFactory.func(pulse.update_pulse_args, {}, 'amp')])

        # Execute the experiment with configured parameters
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp_amp + swp_freq)

        # Process the results
        self.trace = np.squeeze(mp.result())
        self.result = {'Magnitude': np.absolute(self.trace), 'Phase': np.angle(self.trace)}

    @register_browser_function(available_after=(run,))
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

    @register_browser_function(available_after=(run,))
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

    @register_browser_function(available_after=(run,))
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

    def _plot(self, data: np.ndarray, name: str, log_scale: bool = False) -> go.Figure:
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
        args = self.retrieve_args(self.run)
        f = np.arange(args['start'], args['stop'], args['step'])
        amps = np.arange(args['qubit_amp_start'], args['qubit_amp_stop'], args['qubit_amp_step'])

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
