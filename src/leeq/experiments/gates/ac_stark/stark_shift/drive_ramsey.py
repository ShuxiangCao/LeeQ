from .common import *

class StarkDriveRamseyTwoQubits(experiment):
    """Performs Ramsey on two qubits with Stark shift on one."""

    @log_and_record
    def run(self, qubits,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            # Replace 'Any' with the actual type
            initial_lpb: Optional[Any] = None,
            start: float = 0.0,
            stop: float = 1.0,
            step: float = 0.005,
            set_offset: float = 10.0,
            stark_offset=50, amp=0.1, width=0.1, rise=0.01, trunc=1.2,
            update: bool = False) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        qubits : list
            List of two qubit objects for the experiment.
        collection_name : str, optional
            Name of the gate collection. Default: 'f01'
        mprim_index : int, optional
            Measurement primitive index. Default: 0
        initial_lpb : Any, optional
            Initial logical primitive block. Default: None
        start : float, optional
            Start time for Ramsey sweep (us). Default: 0.0
        stop : float, optional
            Stop time for Ramsey sweep (us). Default: 1.0
        step : float, optional
            Time step (us). Default: 0.005
        set_offset : float, optional
            Frequency offset for Ramsey (MHz). Default: 10.0
        stark_offset : float, optional
            Stark drive frequency offset (MHz). Default: 50
        amp : float, optional
            Stark drive amplitude. Default: 0.1
        width : float, optional
            Stark pulse width (us). Default: 0.1
        rise : float, optional
            Stark pulse rise time (us). Default: 0.01
        trunc : float, optional
            Stark pulse truncation. Default: 1.2
        update : bool, optional
            Whether to update qubit frequencies. Default: False

        Returns
        -------
        None
            Results are stored in instance attributes.
        """

        if len(qubits) != 2:
            raise ValueError("AC Stark calibration requires exactly 2 qubits.")

        self.set_offset = set_offset
        self.step = step
        self.stop = stop
        self.stark_offset = stark_offset

        c1s = [qubit.get_c1(collection_name) for qubit in qubits]  # get qubits. Ramsey on both, stark shift first qubit
        mps = [qubit.get_measurement_prim_intlist(mprim_index) for qubit in qubits]

        start_level = int(collection_name[1])
        end_level = int(collection_name[2])
        self.level_diff = end_level - start_level

        self.original_freqs = [c1['Xp'].freq for c1 in c1s]
        for c1 in c1s:
            c1.update_parameters(freq=c1['Xp'].freq + self.set_offset / self.level_diff)

        self.original_freq = c1s[0]['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        cs_pulse = c1s[0]['Xp'].clone()
        cs_pulse.update_pulse_args(amp=amp, freq=self.frequency, phase=0., shape='blackman_square', width=self.stop,
                                   rise=rise, trunc=trunc)

        lpb = prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + cs_pulse + prims.ParallelLPB(
            [c1['Xm'] for c1 in c1s]) + prims.ParallelLPB(
            list(mps))  # stark shift ramsey on first qubit, normal ramsey on second

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        # Execute the basic experiment routine
        basic(lpb, swp, '<z>')

        self.result = [np.squeeze(mp.result()) for mp in mps]
        self.traces = self.result

        for i, c1 in enumerate(c1s):
            c1.update_parameters(freq=self.original_freqs[i])

    def live_plots(self, step_no: Optional[Tuple[int]] = None) -> go.Figure:
        args = self._get_run_args_dict()
        data0 = np.squeeze(self.traces[0])
        data1 = np.squeeze(self.traces[1])
        t = np.arange(args['start'], args['stop'], args['step'])

        if step_no is not None:
            t = t[:step_no[0]]
            data0 = data0[:step_no[0]]
            data1 = data1[:step_no[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=data0, mode='lines+markers', name='trace0'))
        fig.add_trace(go.Scatter(x=t, y=data1, mode='lines+markers', name='trace1'))
        fig.update_layout(
            title=f"Ramsey {args['qubits'][0].hrid} and {args['qubits'][1].hrid} transition {args['collection_name']}",
            xaxis_title="Time (us)",
            yaxis_title="<z>",
            legend_title="Legend",
            font={'family': "Courier New, monospace", 'size': 12, 'color': "Black"},
            plot_bgcolor="white"
        )
        return fig

    def analyze_data(self) -> None:
        args = self._get_run_args_dict()
        from leeq.theory.fits import fit_1d_freq_exp_with_cov

        self.frequency_shift = [[], []]
        self.frequency_guess = [None, None]
        self.error_bar = [None, None]
        self.fitted_freq_offset = [None, None]
        self.fit_params = [None, None]  # Update to store fit_params for each trace

        for i in range(2):
            try:
                self.fit_params[i] = fit_1d_freq_exp_with_cov(self.traces[i], dt=args['step'])
                fitted_freq_offset = (self.fit_params[i]['Frequency'].n - self.set_offset) / self.level_diff
                self.fitted_freq_offset[i] = fitted_freq_offset
                self.frequency_guess[i] = self.original_freqs[i] - fitted_freq_offset
                self.error_bar[i] = self.fit_params[i]['Frequency'].s
                self.frequency_shift[i].append(self.frequency_guess[i] - self.original_freqs[i])
            except Exception:
                self.frequency_guess[i] = 0
                self.error_bar[i] = np.inf

    def dump_results_and_configuration(self) -> Tuple[
            float, float, Any, Dict[str, Union[float, str]], datetime.datetime]:
        args = copy.copy(self._get_run_args_dict())
        del args['initial_lpb']
        args['drive_freq'] = args['qubits'][0].get_c1(args['collection_name'])['X'].freq
        args['qubits'] = [qubit.hrid for qubit in args['qubits']]
        return self.frequency_guess, self.error_bar, self.traces, args, datetime.datetime.now()

    @register_browser_function(available_after=('run',))
    def plot(self) -> go.Figure:
        self.analyze_data()
        args = self._get_run_args_dict()

        time_points = np.arange(args['start'], args['stop'], args['step'])
        time_points_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=[f"{args['qubits'][i].hrid} f = {self.frequency_guess[i]} MHz" for i in
                                            range(2)])

        for i in range(2):
            fig.add_trace(go.Scatter(x=time_points, y=self.traces[i], mode='markers', name=f'Trace {i}'), row=i + 1,
                          col=1)
            if self.fit_params[i]:
                frequency = self.fit_params[i]['Frequency'].n
                amplitude = self.fit_params[i]['Amplitude'].n
                phase = self.fit_params[i]['Phase'].n - 2.0 * np.pi * frequency * args['start']
                offset = self.fit_params[i]['Offset'].n
                decay = self.fit_params[i]['Decay'].n

                fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                    np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

                fig.add_trace(go.Scatter(x=time_points_interpolate, y=fitted_curve, mode='lines', name=f'Fit {i}'),
                              row=i + 1, col=1)
                if i == 0:
                    fig.update_yaxes(title_text="<z>", row=i + 1, col=1)
                if i == 1:
                    fig.update_xaxes(title_text="Time (us)", row=i + 1, col=1)

        fig.update_layout(
            title_text=f"Stark drive Ramsey decay {args['qubits'][0].hrid} and {args['qubits'][1].hrid} transition {args['collection_name']}",
            plot_bgcolor="white",
            width=2000,  # Set your desired width here
            height=800  # Set your desired height here
        )
        return fig

    def plot_fft(self, plot_range: Tuple[float, float] = (0.05, 1)) -> go.Figure:
        self.analyze_data()
        args = self._get_run_args_dict()
        time_step = args['step']

        fig = go.Figure()
        for i in range(2):
            fft_magnitudes = np.abs(np.fft.rfft(self.traces[i]))
            frequencies = np.fft.rfftfreq(len(self.traces[i]), time_step)
            mask = (frequencies > plot_range[0]) & (frequencies < plot_range[1])
            fig.add_trace(go.Scatter(x=frequencies[mask], y=fft_magnitudes[mask], mode='lines', name=f'Trace {i}'))

        fig.update_layout(
            title='Ramsey Spectrum',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Strength',
            plot_bgcolor="white"
        )
        return fig

    @text_inspection
    def fitting(self) -> str:
        self.analyze_data()

        if all(err == np.inf for err in self.error_bar):
            return "The Ramsey experiment failed to fit the data for both traces."

        result_str = ""
        for i in range(2):
            if self.error_bar[i] != np.inf:
                result_str += (
                    f"The Ramsey experiment for trace {i} of qubit {self._get_run_args_dict()['qubits'].hrid} "
                    f"has been analyzed. The expected offset was set to {self.set_offset:.3f} MHz, "
                    f"and the measured offset is {self.fitted_freq_offset[i]:.3f}±{self.error_bar[i]:.3f} MHz.\n")
            else:
                result_str += f"The Ramsey experiment failed to fit the data for trace {i}.\n"

        return result_str
