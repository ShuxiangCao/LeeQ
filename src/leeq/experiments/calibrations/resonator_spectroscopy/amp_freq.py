from .common import *

class ResonatorSweepAmpFreqWithExtraInitialLPB(Experiment):
    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.,
            num_avs: int = 200,
            rep_rate: float = 0.,
            mp_width: Optional[float] = 8,
            initial_lpb: LogicalPrimitiveBlock = None,
            amp_start: float = 0,
            amp_stop: float = 1,
            amp_step: float = 0.05) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        start : float, optional
            Start frequency for the sweep (MHz). Default: 8000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 9000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 200
        rep_rate : float, optional
            Repetition rate. Default: 0.0
        mp_width : Optional[float], optional
            Measurement primitive width (μs). If None, uses rep_rate. Default: 8.0
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial LPB for state preparation. Default: None
        amp_start : float, optional
            Start amplitude for the sweep. Default: 0.0
        amp_stop : float, optional
            Stop amplitude for the sweep. Default: 1.0
        amp_step : float, optional
            Amplitude increment. Default: 0.05

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        # Get the original measurement primitive.
        mprim_index = '0'
        mp = dut_qubit.get_measurement_prim_intlist(mprim_index).clone()

        # Update the pulse arguments with either the provided mp_width or
        # rep_rate if mp_width is None.
        mp.update_pulse_args(
            width=mp_width if mp_width is not None else rep_rate)

        # Remove any previous transform functions.
        mp.set_transform_function(None)

        self.mp = mp

        lpb = mp

        # If initial_lpb is provided, concatenate it with delay and mp.
        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Define the frequency sweeper using np.arange and updating mp's
        # frequency.
        swp_freq = Sweeper(
            np.arange,
            n_kwargs={
                "start": start,
                "stop": stop,
                "step": step},
            params=[
                SweepParametersSideEffectFactory.func(
                    mp.update_freq,
                    {},
                    "freq")],
        )

        # Define the amplitude sweeper using np.arange and updating mp's
        # amplitude.
        swp_amp = Sweeper(
            np.arange,
            n_kwargs={
                'start': amp_start,
                'stop': amp_stop,
                'step': amp_step},
            params=[
                SweepParametersSideEffectFactory.func(
                    mp.update_pulse_args,
                    {},
                    'amp')])

        # Perform the experiment with specified setup parameters.
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp_freq + swp_amp)

        # Save the result in trace attribute, transposing for further analysis.
        self.trace = np.squeeze(mp.result()).transpose()

    def _plot_data(self, x, y, z, title):
        """
        Plot the magnitude of the resonator response.

        Parameters:
            x (np.ndarray): The x-axis data.
            y (np.ndarray): The y-axis data.
            z (np.ndarray): The z-axis data.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='Viridis')
        )

        fig.update_layout(
            title=title,
            xaxis_title="Frequency [MHz]",
            yaxis_title="Driving Amplitude [a.u.]",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_magnitude(self):
        """
        Plot the magnitude of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(
                start=args['start'],
                stop=args['stop'],
                step=args['step']),
            y=np.arange(
                start=args['amp_start'],
                stop=args['amp_stop'],
                step=args['amp_step']),
            z=np.abs(trace),
            title="Resonator response magnitude")

    @register_browser_function(available_after=(run,))
    def plot_phase(self):
        """
        Plot the phase of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """

        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(
                start=args['start'],
                stop=args['stop'],
                step=args['step']),
            y=np.arange(
                start=args['amp_start'],
                stop=args['amp_stop'],
                step=args['amp_step']),
            z=np.unwrap(
                np.angle(trace)),
            title="Resonator response phase")

    @register_browser_function(available_after=(run,))
    def plot_phase_gradient(self):
        """
        Plot the phase gradient of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(
                start=args['start'],
                stop=args['stop'],
                step=args['step']),
            y=np.arange(
                start=args['amp_start'],
                stop=args['amp_stop'],
                step=args['amp_step']),
            z=np.gradient(
                np.unwrap(
                    np.angle(trace)),
                axis=1),
            title="Resonator response phase gradient")

    @register_browser_function(available_after=(run,))
    def plot_mag_logscale(self):
        """
        Plot the magnitude of the resonator response in log scale.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()
        return self._plot_data(
            x=np.arange(
                start=args['start'], stop=args['stop'], step=args['step']), y=np.arange(
                start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step']), z=np.log(
                np.abs(trace)), title="Resonator response magnitude (log scale)")

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot_phase_gradient()
