from .common import *

class ConditionalStarkShiftContinuousPhaseSweep(Experiment):
    """
    This class represents an experiment for tuning up a Rabi oscillation under a conditional Stark shift in a quantum
    mechanics setup. The objective is to analyze whether the plot of the experiment data shows a clear sinusoidal
    oscillatory pattern for both ground and excited states.

    Attributes:
        duts (List[Qubit]): The list of qubits involved in the experiment.
        frequency (Optional[float]): The frequency used in the experiment.
        amp_control (float): The amplitude control value.
        amp_target (float): The amplitude target value.
        phase (float): The phase value initialized to 0.
        width (float): The width value initialized to 0.
        start (float): The starting value for the pulse width sweep.
        stop (float): The stopping value for the pulse width sweep.
        step (float): The step value for the pulse width sweep.
        fitting_2D (Optional[object]): The 2D fitting result.
    """

    _v_prompt: str = """
Here is a plot of data from a quantum mechanics experiment. The data is plotted in the blue and red data points.
Please analyze whether this plot shows a successful experiment by showing show a clear, sinusoidal oscillatory pattern?
If the pattern is identified for both blue and red data points, the experiment is considered successful.
Otherwise, the experiment is considered failed.
"""

    _v_prompt: str = """
I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
(blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data.
Can you inspect the figure, analyze the oscillations, and conclude whether the experiment is valid based on the
presence of sinusoidal oscillations?
"""

    #
    # _experiment_result_analysis_instructions = """
    #     This experiment is a Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two qubits under microwave drives.
    #     Please check the results of the visual inspection of the plots and the fitting results.
    #
    #     For visual inspection, if any of the plot does not show a clear sinusoidal oscillatory pattern, the experiment is considered failed.
    #
    #     For the fitting results, check if all the fitting parameters are physical and plausible. Then look at the sampled points
    #     per period. If it is smaller than 6 then the experiment needs to increase the sweep point and the experiment is considered failed. Estimate how much
    #     you need to increase the sweep points to get approximately 8 points per period. Note that the maximum total sweep point should be 100.
    #
    #     For the fitting results, if the number of periods is less than 2, the experiment is considered failed. Estimate how much you need to increase the stop time
    #     to obtain about 3 periods.
    #
    #     If the above check passes, the experiment is considered successful.
    # """

    @log_and_record
    def run(
            self,
            qubits: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.015,
            start: float = 0,
            stop: float = 15,
            sweep_points=30,
            axis: str = 'Y',
            echo: bool = True,
            iz_rate_cancel: float = 0,
            iz_rise_drop: float = 0,
            phase_sweep_points: int = 10,
    ) -> None:
        """
        Runs the Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two
        qubits under microwave drives.

        Args:
            qubits (List[Qubit]): The list of qubits to be used in the experiment.
            amp_control (float): The amplitude control value.
            amp_target (float): The amplitude target value.
            frequency (Optional[float]): The frequency value, if not provided, it will be calculated.
            rise (float): The rise time.
            start (float): The start value for the pulse width sweep.
            stop (float): The stop value for the pulse width sweep.
            sweep_points (int): The number of sweep points.
            axis (str): The axis for the experiment.
            echo (bool): Whether to include echo sequences.
            iz_rate_cancel (float): The iz rate cancel value.
            iz_rise_drop (float): The iz rise drop value.
            phase_sweep_points (int): The number of phase sweep points.
        """
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = (stop - start) / sweep_points
        self.fitting_2D = None
        self.phase_sweep_points = phase_sweep_points

        if frequency is None:
            freq_01 = qubits[1].get_c1('f01')['X'].freq
            freq_12 = qubits[1].get_c1('f12')['X'].freq

            anharmonicity = freq_01 - freq_12
            self.frequency = freq_01 - 0.3 * anharmonicity
        else:
            self.frequency = frequency

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0],
                                                  target_q=self.duts[1],
                                                  amp_target=self.amp_target,
                                                  amp_control=self.amp_control,
                                                  frequency=self.frequency, rise=rise,
                                                  width=self.width,
                                                  phase_diff=self.phase_diff,
                                                  iz_control=0,
                                                  iz_target=0,
                                                  echo=False,
                                                  trunc=1.0, zz_interaction_positive=True)

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

        flip_both = c1_control['Y'] * c1_target['Y']

        if echo:
            lpb = cs_pulse + flip_both + cs_pulse + flip_both
        else:
            lpb = cs_pulse

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        lpb_readout = prims.SweepLPB([c1_target['Yp'], c1_target['Xm']])
        swp_readout = sweeper.from_sweep_lpb(lpb_readout)

        iz_gate = c1_target.z_omega(iz_rate_cancel * 2 * np.pi)
        iz_gate_fix = c1_target.z(-iz_rise_drop)

        lpb = c1_target[
            'Ym'] * lpb_flip_control + lpb + iz_gate + iz_gate_fix + lpb_readout + mprim_target * mprim_control

        swp_params = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'phase'),
        ]

        swp_phase = sweeper(np.linspace, n_kwargs={'start': 0, 'stop': np.pi * 2,
                                                   'num': self.phase_sweep_points},
                            params=swp_params)

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
            sparam.func(iz_gate.set_virtual_width, {}, 'width'),
        ]

        if echo:
            swp = sweeper(np.arange, n_kwargs={'start': start / 2, 'stop': stop / 2,
                                               'step': self.step / 2},
                          params=swpparams)
        else:
            swp = sweeper(np.arange,
                          n_kwargs={'start': start, 'stop': stop, 'step': self.step},
                          params=swpparams)

        basic(lpb, swp=swp_phase + swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

    def analyze_results(self):
        if self.fitting_2D is None:

            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                self.fit_result = fits.fit_2d_freq(self.complex_data, dt=self.step,
                                                   use_freq_bound=False)
                self.fitting_2D.append(self.fit_result)

            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2

            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
                self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
                self.fitting_2D[1]['Phase'])) / 2


        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def analyze_results_with_errs(self):
        if self.fitting_2D is None:
            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                fit_results = fits.fit_2d_freq_with_cov(self.complex_data, dt=self.step,
                                                        use_freq_bound=False)
                self.fitting_2D.append(fit_results)

            # Calculate iz_rate and zz_rate for Curve Fit
            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2
            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
                self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
                self.fitting_2D[1]['Phase'])) / 2


        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def plot_matplotlib(self):

        self.analyze_results_with_errs()

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, fit_params, use_imaginary_part=False):
            plt.scatter(t, data, label=label, alpha=0.5)

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params[
                'Offset_imag'].nominal_value

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate,
                     np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)

        plot_specific_axis(data=self.result[:, 1, 0], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

    def plot_specific_axis(self, fig, data, label, fit_params, use_imaginary_part=False):

        color_ground = 'mediumblue'
        color_excited = 'crimson'

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(
            go.Scatter(x=t, y=data, mode="lines+markers", name=label, opacity=0.5,
                       marker={'color': color}))

        f = fit_params['Frequency'].nominal_value
        a = fit_params['Amplitude'].nominal_value
        p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
        o_real = fit_params['Offset_real'].nominal_value
        o_imag = fit_params['Offset_imag'].nominal_value

        fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (
            o_real + 1j * o_imag)

        fig.add_trace(go.Scatter(x=t_interpolate,
                                 y=np.real(fit) if not use_imaginary_part else np.imag(
                                     fit),
                                 mode='lines', name=f'{label} Fit',
                                 line={'color': color}, visible='legendonly'))

    @register_browser_function()
    @visual_inspection(_v_prompt)
    def plot_X_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 0], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=False)
        self.plot_specific_axis(fig, data=self.result[:, 1, 0], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - X axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @register_browser_function()
    @visual_inspection(_v_prompt)
    def plot_Y_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 1], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 1], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Y axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @text_inspection
    def fitting(self) -> Union[str, None]:

        self.analyze_results_with_errs()

        prompt = f"""
        The fitting reports when the control qubit is at the ground state, the target is oscillating at a frequency of {self.fitting_2D[0]['Frequency']} MHz,
        and when the control qubit is at the excited state, the target is oscillating at a frequency of {self.fitting_2D[1]['Frequency']} MHz.
        Therefore the IZ rate is {self.iz_rate} MHz and the ZZ rate is {np.abs(self.zz_rate)} MHz. The sampling number per ZZ period is {np.abs(1 / self.zz_rate / self.step)}.
        We have observed {np.abs(self.stop * self.zz_rate)} periods of the ZZ interaction.
        Note that the sign of the ZZ rate is corresponds to the direction of the rotation and it is allowed to be negative.
        """

        return prompt



class ConditionalStarkShiftContinuous(Experiment):
    """
    This class represents an experiment for tuning up a Rabi oscillation under a conditional Stark shift in a quantum
    mechanics setup. The objective is to analyze whether the plot of the experiment data shows a clear sinusoidal
    oscillatory pattern for both ground and excited states.

    Attributes:
        duts (List[Qubit]): The list of qubits involved in the experiment.
        frequency (Optional[float]): The frequency used in the experiment.
        amp_control (float): The amplitude control value.
        amp_target (float): The amplitude target value.
        phase (float): The phase value initialized to 0.
        width (float): The width value initialized to 0.
        start (float): The starting value for the pulse width sweep.
        stop (float): The stopping value for the pulse width sweep.
        step (float): The step value for the pulse width sweep.
        fitting_2D (Optional[object]): The 2D fitting result.
        phase_diff (float): The phase difference.
    """

    _v_prompt: str = """
Here is a plot of data from a quantum mechanics experiment. The data is plotted in the blue and red data points.
Please analyze whether this plot shows a successful experiment by showing show a clear, sinusoidal oscillatory pattern?
If the pattern is identified for both blue and red data points, the experiment is considered successful.
Otherwise, the experiment is considered failed.
"""

    _v_prompt: str = """
I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
(blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data.
Can you inspect the figure, analyze the oscillations, and conclude whether the experiment is valid based on the
presence of sinusoidal oscillations?
"""

    _experiment_result_analysis_instructions = """
This experiment is a Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two qubits under microwave drives.
Please check the results of the visual inspection of the plots and the fitting results.

If the any of above check fails, the experiment is considered failed.
"""

    @log_and_record(overwrite_func_name='ConditionalStarkShiftContinuous.run')
    def run_simulated(
            self,
            duts: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.015,
            start: float = 0,
            stop: float = 15,
            sweep_points=30,
            axis: str = 'Y',
            echo: bool = True,
            iz_rate_cancel: float = 0,
            phase_diff: float = 0,
            iz_rise_drop: float = 0
    ) -> None:
        """
        Runs the Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two
        qubits under microwave drives.

        Args:
            qubits (List[Qubit]): The list of qubits to be used in the experiment.
            amp_control (float): The amplitude control value.
            amp_target (float): The amplitude target value.
            frequency (Optional[float]): The frequency value, if not provided, it will be calculated.
            rise (float): The rise time.
            start (float): The start value for the pulse width sweep.
            stop (float): The stop value for the pulse width sweep.
            sweep_points (int): The number of sweep points.
            axis (str): The axis for the experiment.
            echo (bool): Whether to include echo sequences.
            iz_rate_cancel (float): The iz rate cancel value.
            phase_diff (float): The phase difference value.
            iz_rise_drop (float): The iz rise drop value.
        """

        qubits = duts
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = (stop - start) / sweep_points
        self.rise = rise
        self.fitting_2D = None
        self.phase_diff = phase_diff

        if frequency is None:
            freq_01 = qubits[1].get_c1('f01')['X'].freq
            freq_12 = qubits[1].get_c1('f12')['X'].freq

            anharmonicity = freq_01 - freq_12
            self.frequency = freq_01 - 0.3 * anharmonicity
        else:
            self.frequency = frequency

        c1_control = self.duts[0].get_default_c1()
        self.duts[1].get_default_c1()

        simulator_setup = setup().get_default_setup()
        virtual_transmon_1 = simulator_setup.get_virtual_qubit(qubits[0])
        virtual_transmon_2 = simulator_setup.get_virtual_qubit(qubits[1])

        # Evaluate the ZZ value

        from leeq.theory.sizzel_gate.sizzel_simulation import ret_zz

        coupling = simulator_setup.get_coupling_strength_by_qubit(virtual_transmon_1,
                                                                  virtual_transmon_2)
        drive_omega = simulator_setup.get_omega_per_amp(
            channel=c1_control.channel) * amp_control

        delta = self.frequency - virtual_transmon_1.qubit_frequency
        freq_delta = virtual_transmon_2.qubit_frequency - virtual_transmon_1.qubit_frequency

        zz_value = ret_zz(
            alpha1=virtual_transmon_1.anharmonicity,
            alpha2=virtual_transmon_2.anharmonicity,
            J=coupling,
            eps=drive_omega,
            delta=delta,
            freq_delta=freq_delta
        )[0]  # Delta 11 is the ZZ value

        self.result, self.result_control = (
            _generate_zz_interaction_data_from_simulation(qubits=qubits,
                                                          start=start,
                                                          stop=stop,
                                                          sweep_points=sweep_points,
                                                          zz_value=zz_value,
                                                          drive_frequency=self.frequency,
                                                          amp_control=amp_control
                                                          ))

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.015,
            start: float = 0,
            stop: float = 15,
            sweep_points=30,
            axis: str = 'Y',
            echo: bool = True,
            iz_rate_cancel: float = 0,
            phase_diff: float = 0,
            iz_rise_drop: float = 0
    ) -> None:
        """
        Runs the Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two
        qubits under microwave drives.

        Args:
            qubits (List[Qubit]): The list of qubits to be used in the experiment.
            amp_control (float): The amplitude control value.
            amp_target (float): The amplitude target value.
            frequency (Optional[float]): The frequency value, if not provided, it will be calculated.
            rise (float): The rise time.
            start (float): The start value for the pulse width sweep.
            stop (float): The stop value for the pulse width sweep.
            sweep_points (int): The number of sweep points.
            axis (str): The axis for the experiment.
            echo (bool): Whether to include echo sequences.
            iz_rate_cancel (float): The iz rate cancel value.
            phase_diff (float): The phase difference value.
            iz_rise_drop (float): The iz rise drop value.
        """
        qubits = duts
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = (stop - start) / sweep_points
        self.rise = rise
        self.fitting_2D = None
        self.phase_diff = phase_diff

        if frequency is None:
            freq_01 = qubits[1].get_c1('f01')['X'].freq
            freq_12 = qubits[1].get_c1('f12')['X'].freq

            anharmonicity = freq_01 - freq_12
            self.frequency = freq_01 - 0.3 * anharmonicity
        else:
            self.frequency = frequency

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0],
                                                  target_q=self.duts[1],
                                                  amp_target=self.amp_target,
                                                  amp_control=self.amp_control,
                                                  frequency=self.frequency, rise=rise,
                                                  width=self.width,
                                                  phase_diff=self.phase_diff,
                                                  iz_control=0,
                                                  iz_target=0,
                                                  echo=False,
                                                  trunc=1.0, zz_interaction_positive=True)

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

        flip_both = c1_control['Y'] * c1_target['Y']

        if echo:
            lpb = cs_pulse + flip_both + cs_pulse + flip_both
        else:
            lpb = cs_pulse

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        lpb_readout = prims.SweepLPB([c1_target['Yp'], c1_target['Xm']])
        swp_readout = sweeper.from_sweep_lpb(lpb_readout)

        iz_gate = c1_target.z_omega(iz_rate_cancel * 2 * np.pi)
        iz_gate_fix = c1_target.z(-iz_rise_drop)

        lpb = c1_target[
            'Ym'] * lpb_flip_control + lpb + iz_gate + iz_gate_fix + lpb_readout + mprim_target * mprim_control

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
            sparam.func(iz_gate.set_virtual_width, {}, 'width'),
        ]

        if echo:
            swp = sweeper(np.arange, n_kwargs={'start': start / 2, 'stop': stop / 2,
                                               'step': self.step / 2},
                          params=swpparams)
        else:
            swp = sweeper(np.arange,
                          n_kwargs={'start': start, 'stop': stop, 'step': self.step},
                          params=swpparams)

        basic(lpb, swp=swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

    def analyze_results(self):

        if self.fitting_2D is None:

            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                self.fit_result = fits.fit_2d_freq(self.complex_data, dt=self.step,
                                                   use_freq_bound=False)
                self.fitting_2D.append(self.fit_result)

            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2

            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
                self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
                self.fitting_2D[1]['Phase'])) / 2


        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def analyze_results_with_errs(self):

        if self.fitting_2D is None:
            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                fit_results = fits.fit_2d_freq_with_cov(self.complex_data, dt=self.step,
                                                        use_freq_bound=False)
                self.fitting_2D.append(fit_results)

            # Calculate iz_rate and zz_rate for Curve Fit
            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2
            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
                self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
                self.fitting_2D[1]['Phase'])) / 2


        if len(self.fitting_2D) != 2:
            return {'error': 'Fitting failed'}

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def plot_matplotlib(self):

        self.analyze_results_with_errs()

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, fit_params, use_imaginary_part=False):
            plt.scatter(t, data, label=label, alpha=0.5)

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params[
                'Offset_imag'].nominal_value

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate,
                     np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)

        plot_specific_axis(data=self.result[:, 1, 0], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

    def plot_specific_axis(self, fig, data, label, fit_params=None,
                           use_imaginary_part=False):

        color_ground = 'mediumblue'
        color_excited = 'crimson'

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(
            go.Scatter(x=t, y=data, mode="lines+markers", name=label, opacity=0.5,
                       marker={'color': color}))

        if fit_params is not None:
            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o_real = fit_params['Offset_real'].nominal_value
            o_imag = fit_params['Offset_imag'].nominal_value

            fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (
                o_real + 1j * o_imag)

            fig.add_trace(go.Scatter(x=t_interpolate, y=np.real(
                fit) if not use_imaginary_part else np.imag(fit),
                mode='lines', name=f'{label} Fit',
                line={'color': color}, visible='legendonly'))

    def get_ai_inspection_results(self):
        """
        Returns the results of the AI inspection for the experiment.
        """

        inspection_results = super().get_ai_inspection_summary()

        if self.fitting_2D is None:
            try:
                self.analyze_results_with_errs()
            except Exception as e:
                inspection_results['error'] = str(e)

        try:
            zz_rate = self.zz_rate

            inspection_results['Calibrated parameters'] = {
                'amp_control': self.amp_control,
                'amp_target': self.amp_target,
                'frequency': self.frequency,
                'rise': self.rise,
                'phase_diff': self.phase_diff,
                'width': np.abs(0.125 / zz_rate.nominal_value) / 2,
                'zz_interaction_positive': self.zz_rate > 0,
                'zz_rate': self.zz_rate.n
            }
        except Exception:
            pass

        return inspection_results

    @register_browser_function()
    def plot_X_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 0], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=False)
        self.plot_specific_axis(fig, data=self.result[:, 1, 0], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - X axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @register_browser_function()
    def plot_Y_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 1], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 1], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Y axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    def plot_Z_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 2], label="Ground",
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 2], label="Excited",
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Z axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing ZZ interaction Hamiltonian tomography in the Fourier space. The X-axis represents
the frequency, and the Y-axis shows the amplitude of the fourier transformed value.
My objective is to determine if the experiment is a success.
The success of the experiment depends on observing two clear peaks in the Fourier space,
one for the ground state and one for the excited state. They should be symmetric around the center of the plot.
If the peaks are not clear, the experiment is considered failed.
If you observe more than two clear peaks, the experiment is considered failed.
Otherwise, the experiment is considered successful.

For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftContinuous.plot_fourier.png")
The following Image is a failure case for the experiment due to the presence of multiple peaks:
Image("ref_images/failure_ConditionalStarkShiftContinuous.plot_fourier.png")
""")
    def plot_fourier(self):
        fig = go.Figure()

        def plot_fourier_trace(fig, data, label, fit_params=None):
            color_ground = 'mediumblue'
            color_excited = 'crimson'

            args = {'start': self.start, 'stop': self.stop, 'step': self.step}
            t = np.arange(args['start'], args['stop'], args['step'])

            color = color_ground if label == 'Ground' else color_excited

            # Compute the Fourier Transform of the data

            data = data - np.mean(data)

            fourier_transform = np.fft.fft(data)
            frequencies = np.fft.fftfreq(t.size, d=args['step'])

            # Compute the amplitude of the Fourier Transform
            amplitude = np.abs(fourier_transform)

            # Sort frequencies in ascending order and reorder amplitude accordingly
            sorting_indices = np.argsort(frequencies)
            sorted_frequencies = frequencies[sorting_indices]
            sorted_amplitude = amplitude[sorting_indices]

            fig.add_trace(
                go.Scatter(x=sorted_frequencies, y=sorted_amplitude, mode="lines+markers",
                           name=label, opacity=0.5, marker={'color': color}))

        plot_fourier_trace(fig=fig,
                           data=self.result[:, 0, 0] + 1.j * self.result[:, 0, 1],
                           label="Ground")
        plot_fourier_trace(fig=fig,
                           data=self.result[:, 1, 0] + 1.j * self.result[:, 1, 1],
                           label="Excited")

        fig.update_layout(
            title="ZZ interaction Hamiltonian tomography - Fourier Transform",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Amplitude [a.u.]",
            plot_bgcolor='white',
            legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing status of a qubit over an experiment.
My objective is to determine if the experiment is a success. The success of the experiment should see the state
of the qubits remains stable throughout the experiment.
If you observe oscillations, or the lines crosses each other, the experiment is considered failed.
Otherwise, the experiment is considered successful.
For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftContinuous.plot_control_population.png")
The following Image is a failure case for the experiment:
Image("ref_images/failure_ConditionalStarkShiftContinuous.plot_control_population.png")
    """)
    def plot_control_population(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result_control[:, 0, 1], label="Ground",
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result_control[:, 1, 1], label="Excited",
                                use_imaginary_part=True)

        fig.update_layout(title="Control qubit state - Z axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @text_inspection
    def fitting(self) -> Union[str, None]:

        result = self.analyze_results_with_errs()

        error = result.get('error', None)

        if error is not None:
            return "The experiment failed due to fitting error."

        prompt_1 = f"""
The fitting reports when the control qubit is at the ground state, the target is oscillating at a frequency of {self.fitting_2D[0]['Frequency']} MHz,
and when the control qubit is at the excited state, the target is oscillating at a frequency of {self.fitting_2D[1]['Frequency']} MHz.
Therefore the IZ rate is {self.iz_rate} MHz and the ZZ rate is {self.zz_rate} MHz. The sampling number per ZZ period is {np.abs(1 / self.zz_rate / self.step)}.
We have observed {np.abs(self.stop * self.zz_rate)} periods of the ZZ interaction.
Note that the sign of the ZZ rate is corresponds to the direction of the rotation and it is allowed to be negative.
"""

        z_control_diff = self.result_control[:, 0, 1] - self.result_control[:, 1, 1]
        z_control_diff_max = np.max(np.abs(z_control_diff))
        z_control_diff_min = np.min(np.abs(z_control_diff))

        prompt_2 = f"""
The expectation value of the control qubit along the Z axis is stable through the whole experiment.
The maximum difference between the ground and excited state is {z_control_diff_max} and the minimum difference is {z_control_diff_min}.
The experiment should be considered successful if the minimum difference is greater than 50% of the maximum difference.
If the experiment is failed because of the population of the control qubit does not meet the criteria, do not retry and directly report the failure.
"""

        analyze_prompt = """
For the fitting results, if the absolute value of oscillation frequency is significantly different between the ground and excited states (more than 50%),
the experiment is considered failed due to fitting error.
For the fitting results, if the oscillation amplitude is less than 0.2, the experiment is considered failed due to noise data.
The experiment is otherwise considered success.
"""

        prompt = f"""
{prompt_1}
{prompt_2}
{analyze_prompt}
<requirement>
Output your analysis of the success/failure of the experiment by a JSON with the following keys:
"fitting_analysis" (string): The analysis about whether the experiment succeeded or failed.
"success" (bool): Whether the experiment succeeded.
</requirement>
"""
        res = Chat(prompt).complete(parse="dict", cache=True, expensive=True)
        return res
