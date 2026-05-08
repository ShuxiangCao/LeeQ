from .common import *

class StarkSingleQubitT1(experiment):
    """Perform a T1 experiment applying a Stark shifting drive instead of the delay time."""

    @log_and_record
    def run(self,
            qubit: Any,
            collection_name: str = 'f01',
            initial_lpb: Optional[Any] = None,
            mprim_index: int = 0,
            start=0, stop=3, step=0.03,
            stark_offset=50,
            amp=0.1,
            width=400,
            rise=0.01,
            trunc=1.2):
        """
        Execute Stark T1 experiment on hardware.

        Parameters
        ----------
        qubit : Any
            The qubit to perform the experiment on.
        collection_name : str, optional
            Gate collection name. Default: 'f01'.
        initial_lpb : Any, optional
            Initial pulse sequence. Default: None.
        mprim_index : int, optional
            Measurement primitive index. Default: 0.
        start : float, optional
            Start time (us). Default: 0.
        stop : float, optional
            Stop time (us). Default: 3.
        step : float, optional
            Time step (us). Default: 0.03.
        stark_offset : float, optional
            Stark frequency offset (MHz). Default: 50.
        amp : float, optional
            Stark pulse amplitude. Default: 0.1.
        width : float, optional
            Initial pulse width. Default: 400.
        rise : float, optional
            Pulse rise time. Default: 0.01.
        trunc : float, optional
            Pulse truncation. Default: 1.2.

        Returns
        -------
        None
            Results stored in instance attributes.
        """

        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.stark_offset = stark_offset

        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(mprim_index)

        self.mp = mp

        self.original_freq = c1['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        cs_pulse = c1['X'].clone()
        cs_pulse.update_pulse_args(amp=amp, freq=self.frequency, phase=0., shape='blackman_square', width=self.stop,
                                   rise=rise, trunc=trunc)

        lpb = c1['X'] + cs_pulse + mp

        if initial_lpb:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        basic(lpb, swp=swp, basis="<z>")
        self.trace = np.squeeze(mp.result())

    @register_browser_function(available_after=(run,))
    def plot_t1(self, fit=True, step_no=None) -> go.Figure:
        """
        Plot the T1 decay graph based on the trace and fit parameters using Plotly.

        Parameters:
        fit (bool): Whether to fit the trace. Defaults to True.
        step_no (Tuple[int]): Number of steps to plot.

        Returns:
        go.Figure: The Plotly figure object.
        """
        self.trace = None
        self.fit_params = {}  # Initialize as an empty dictionary or suitable default value

        args = self._get_run_args_dict()

        t = np.arange(0, args['stop'], args['step'])
        trace = np.squeeze(self.mp.result())

        if step_no is not None:
            t = t[:step_no[0]]
            trace = trace[:step_no[0]]

        # Create traces for scatter and line plot
        trace_scatter = go.Scatter(
            x=t, y=trace,
            mode='markers',
            marker={
                'symbol': 'x',
                'size': 10,
                'color': 'blue'
            },
            name='Experiment data'
        )

        title = f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}"

        data = [trace_scatter]

        if fit:
            fit_params = self.fit_exp_decay_with_cov(trace, args[
                'step'])  # Assuming fit_exp_decay_with_cov is a method of the class

            self.fit_params = fit_params

            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'][0] * np.exp(-t / fit_params['Decay'][0]) + fit_params['Offset'][0],
                mode='lines',
                line={
                    'color': 'blue'
                },
                name='Decay fit'
            )
            title = (f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}<br>"
                     f"T1={fit_params['Decay'][0]:.2f} ± {fit_params['Decay'][1]:.2f} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis={'title': 'Time (us)'},
            yaxis={'title': 'P(1)'},
            plot_bgcolor='white',
            showlegend=True
        )

        fig = go.Figure(data=data, layout=layout)

        return fig

    def fit_exp_decay_with_cov(self, trace, time_resolution):
        # Example implementation of the fit_exp_decay_with_cov function
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        t = np.arange(0, len(trace) * time_resolution, time_resolution)
        try:
            popt, pcov = curve_fit(exp_decay, t, trace, maxfev=2400)
            A, tau, C = popt
            perr = np.sqrt(np.diag(pcov))
            return {'Amplitude': (A, perr[0]), 'Decay': (tau, perr[1]), 'Offset': (C, perr[2])}
        except (OptimizeWarning, RuntimeError):
            return {'Amplitude': (np.nan, np.nan), 'Decay': (np.nan, np.nan), 'Offset': (np.nan, np.nan)}
