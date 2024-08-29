Please read the example of the full class below:

```python



class NormalisedRabi(Experiment):

    @log_and_record
    def run(self,
            dut_qubit: Any,
            amp: float = 0.05,
            start: float = 0.01,
            stop: float = 0.3,
            step: float = 0.002,
            fit: bool = True,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            pulse_discretization: bool = True,
            update=True,
            initial_lpb: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Run a Rabi experiment on a given qubit and analyze the results.

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
        self.fit_params = self.fit_sinusoidal(self.data, time_step=step)

        # Update the qubit parameters, to make one pulse width correspond to a pi pulse
        # Here we suppose all pulse envelopes give unit area when width=1,
        # amp=1
        normalised_pulse_area = c1['X'].calculate_envelope_area() / c1['X'].amp
        two_pi_area = amp * (1 / self.fit_params['Frequency'])
        new_amp = two_pi_area / 2 / normalised_pulse_area
        self.guess_amp = new_amp

        if update:
            c1.update_parameters(amp=new_amp)
            print(f"Amplitude updated: {new_amp}")

    @register_browser_function()
    def plot(self) -> go.Figure:
        """
        Plots Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features scatter points
        representing the original data and a sine fit for each qubit involved in the
        experiment.
        """

        args = self.retrieve_args(self.run)
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
                marker=dict(
                    color='Blue',
                    size=7,
                    opacity=0.5,
                    line=dict(color='Black', width=2),
                ),
                name=f'data'
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
                line=dict(color='Red'),
                name=f'fit',
                visible='legendonly'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Time Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font=dict(
                family='Courier New, monospace',
                size=12,
                color='Black'
            ),
            plot_bgcolor='white'
        )

        return fig

    @staticmethod
    def fit_sinusoidal(
            data: np.ndarray,
            time_step: float,
            use_freq_bound: bool = True,
            fix_frequency: bool = False,
            start_time: float = 0,
            freq_guess: Optional[float] = None,
            **kwargs: Any
    ) -> Dict[str, float]:
        """
        Fit a sinusoidal model to 1D data.
    
        Parameters:
        data (np.ndarray): The 1D data array to fit.
        time_step (float): The time interval between data points.
        use_freq_bound (bool): Whether to bound the frequency during optimization.
        fix_frequency (bool): Whether to keep the frequency fixed during optimization.
        start_time (float): The start time of the data.
        freq_guess (Optional[float]): An initial guess for the frequency.
        **kwargs (Any): Additional keyword arguments for the optimizer.
    
        Returns:
        dict: A dictionary containing the optimized parameters and the residual of the fit.
        """

        # Ensure frequency is provided if it's fixed
        if fix_frequency:
            assert freq_guess is not None, "Initial frequency guess must be provided if frequency is fixed."

        # Estimate initial frequency if not provided
        if freq_guess is None:
            rfft = np.abs(np.fft.rfft(data))
            frequencies = np.fft.rfftfreq(len(data), time_step)
            # Skip the first element which is the zero-frequency component
            max_index = np.argmax(rfft[1:]) + 1
            dominant_freq = frequencies[max_index]

            omega = dominant_freq

            freq_resolution = frequencies[1] - frequencies[0]
            min_omega = dominant_freq - freq_resolution
            max_omega = dominant_freq + freq_resolution
        else:
            omega = freq_guess
            max_omega = omega * 1.5
            min_omega = omega * 0.5

        # Generate time data
        time = np.linspace(start_time, start_time +
                           time_step * (len(data) - 1), len(data))

        # Initial parameter guesses based on data properties
        offset = np.mean(data)
        amplitude = 0.5 * (np.max(data) - np.min(data))
        phase = np.arcsin(np.clip((data[0] - offset) / amplitude, -1, 1))
        if data[1] - data[0] < 0:
            phase = np.pi - phase

        # Objective functions for optimization
        def leastsq(params: List[float], t: np.ndarray, y: np.ndarray) -> float:
            omega, amplitude, phase, offset = params
            fit = amplitude * np.sin(2. * np.pi * omega * t + phase) + offset
            return np.mean((fit - y) ** 2) * 1e5

        def leastsq_without_omega(
                params: List[float],
                omega: float,
                t: np.ndarray,
                y: np.ndarray) -> float:
            amplitude, phase, offset = params
            fit = amplitude * np.sin(2. * np.pi * omega * t + phase) + offset
            return np.mean((fit - y) ** 2) * 1e5

        # Optimization process
        optimization_args = dict()
        optimization_args.update(kwargs)

        if not fix_frequency:
            if use_freq_bound:
                optimization_args['bounds'] = (
                    (min_omega, max_omega), (None, None), (None, None), (None, None))
            result = minimize(leastsq, np.array([omega, amplitude, phase, offset]), args=(
                time, data), **optimization_args)
            omega, amplitude, phase, offset = result.x
        else:
            result = minimize(leastsq_without_omega, np.array(
                [amplitude, phase, offset]), args=(omega, time, data), **optimization_args)
            amplitude, phase, offset = result.x

        residual = result.fun

        # Ensure amplitude is positive and phase is within [-pi, pi]
        if amplitude < 0:
            phase += np.pi
            amplitude = -amplitude

        phase = phase % (2 * np.pi)
        if phase > np.pi:
            phase -= 2 * np.pi

        return {
            'Frequency': omega,
            'Amplitude': amplitude,
            'Phase': phase,
            'Offset': offset,
            'Residual': residual}

    def live_plots(self, step_no=None) -> go.Figure:
        """
        Plots Rabi oscillations live using data from an experiment run.

        Parameters:
        step_no (int): Number of steps to plot. Default is None.

        Returns:
        go.Figure: Plotly figure.

        """

        args = self.retrieve_args(self.run)
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
                marker=dict(
                    color='Blue',
                    size=7,
                    opacity=0.5,
                    line=dict(color='Black', width=2)
                ),
                name=f'data'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Time Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font=dict(
                family='Courier New, monospace',
                size=12,
                color='Black'
            ),
            plot_bgcolor='white'
        )

        return fig

```