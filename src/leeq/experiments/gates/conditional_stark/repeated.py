from .common import *

class ConditionalStarkShiftRepeatedGate(Experiment):
    _v_prompt = """
    I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
    lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
    the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
    (blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
    The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data.
    The amplitude of the oscillations should be significant (more than 0.3), otherwise the experiment is invalid.
    You should observe multiple oscillation periods in the data, otherwise the experiment is invalid.
    Can you inspect the figure, analyze the oscillations, and conclude whether the experiment is valid based on the
    presence of sinusoidal oscillations?
    """

    _experiment_result_analysis_instructions = """
This experiment is a Repeated Gate Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two qubits under microwave drives.
Please check the results of the visual inspection of the plots and the fitting results.

For visual inspection, if the inspection plot_fourier indicates a failure, the experiment is considered failed and Suggested parameter updates to None.

If the experiment is failed because of the population of the control qubit doesnot meet the criteria, do not retry and directly report the failure.

If the above check passes, the experiment is considered successful.
"""

    @log_and_record(overwrite_func_name='ConditionalStarkShiftRepeatedGate.run')
    def run_simulated(self, duts, amp_control, amp_target, frequency, phase_diff=0,
                      rise=0.03,
                      echo=True, iz_control=0, iz_target=0, width=0, start_gate_number=0,
                      gate_count=40, zz_rate=None):
        """
        Runs the Repeated Gate Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions
        between two qubits under microwave drives.

        Parameters:
        duts (List[Qubit]): The list of qubits to be used in the experiment.
        amp_control (float): The amplitude applied to the control qubit (the first qubit in the list).
        amp_target (float): The amplitude applied to the target qubit (the second qubit in the list).
        frequency (float): The frequency of the stark drive pulse.
        phase_diff (float): The phase difference between the stark drive pulse send to the control and target qubits.
        rise (float): The rising and dropping edge time of the stark drive pulses.
        echo (bool): Whether to include echo sequences.
        iz_control (float): The active cancellation of the IZ rate applied to the control qubit.
        iz_target (float): The active cancellation of the IZ rate applied to the target qubit.
        width (float): The width of the stark drive pulses.
        start_gate_number (int): The number of gates to start with.
        gate_count (int): The maximum number of gates to apply in the sweep.
        zz_rate (float): The ZZ rate of the interation measured from the conditional stark shift continious experiment.
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase_diff = phase_diff
        self.width = width
        self.iz_control = iz_control
        self.iz_target = iz_target
        self.rise = rise
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        if zz_rate is None:
            zz_rate = 0.125 / 2 / self.width

        self.zz_rate_continous = zz_rate

        c1_control = self.duts[0].get_default_c1()
        self.duts[1].get_default_c1()

        simulator_setup = setup().get_default_setup()
        virtual_transmon_1 = simulator_setup.get_virtual_qubit(self.duts[0])
        virtual_transmon_2 = simulator_setup.get_virtual_qubit(self.duts[1])

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

        # assume we under shoot by 10%

        zz_value = zz_value * 0.9 * width

        self.result, self.result_control = (
            _generate_zz_interaction_data_from_simulation(qubits=self.duts,
                                                          start=start_gate_number,
                                                          stop=start_gate_number + gate_count,
                                                          sweep_points=gate_count,
                                                          zz_value=zz_value,
                                                          drive_frequency=self.frequency,
                                                          amp_control=amp_control
                                                          ))

    @log_and_record
    def run(self, duts, amp_control, amp_target, frequency, phase_diff=0, rise=0.03,
            echo=True, iz_control=0, iz_target=0, width=0, start_gate_number=0,
            gate_count=40, zz_rate=None):
        """
        Runs the Repeated Gate Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions
        between two qubits under microwave drives.

        Parameters:
        duts (List[Qubit]): The list of qubits to be used in the experiment.
        amp_control (float): The amplitude applied to the control qubit (the first qubit in the list).
        amp_target (float): The amplitude applied to the target qubit (the second qubit in the list).
        frequency (float): The frequency of the stark drive pulse.
        phase_diff (float): The phase difference between the stark drive pulse send to the control and target qubits.
        rise (float): The rising and dropping edge time of the stark drive pulses.
        echo (bool): Whether to include echo sequences.
        iz_control (float): The active cancellation of the IZ rate applied to the control qubit.
        iz_target (float): The active cancellation of the IZ rate applied to the target qubit.
        width (float): The width of the stark drive pulses.
        start_gate_number (int): The number of gates to start with.
        gate_count (int): The maximum number of gates to apply in the sweep.
        zz_rate (float): The ZZ rate of the interation measured from the conditional stark shift continious experiment.
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase_diff = phase_diff
        self.width = width
        self.iz_control = iz_control
        self.iz_target = iz_target
        self.rise = rise
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        if zz_rate is None:
            zz_rate = 0.125 / 2 / self.width

        self.zz_rate_continous = zz_rate

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

        c2 = prims.build_CZ_stark_from_parameters(
            control_q=self.duts[0],
            target_q=self.duts[1],
            amp_target=self.amp_target,
            amp_control=self.amp_control,
            frequency=self.frequency,
            rise=self.rise,
            width=self.width,
            phase_diff=self.phase_diff,
            iz_control=self.iz_control,
            iz_target=self.iz_target,
            echo=echo,
            trunc=1.05,
            zz_interaction_positive=True
            # It doesn't matter what to use here, after one tomography we will find out.
        )

        cs_pulse = c2.get_z_canceled_cs_pulse()

        lpb = cs_pulse

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        lpb_readout = prims.SweepLPB([c1_target['Yp'], c1_target['Xm']])
        swp_readout = sweeper.from_sweep_lpb(lpb_readout)

        self.pulse_train, self.result = self.run_repeated_gate_experiment(
            initial_lpb=c1_target['Ym'],
            initial_gate=lpb_flip_control,
            repeated_block=lpb,
            final_gate=lpb_readout,
            pulse_count=range(start_gate_number, start_gate_number + gate_count),
            swp_initial=swp_flip,
            swp_posterior=swp_readout,
            fit=False
        )

        self.analyze_results()

    def run_repeated_gate_experiment(self, initial_lpb, initial_gate, repeated_block,
                                     final_gate, pulse_count,
                                     swp_initial, swp_posterior, fit=True):
        """
        Function to run the repeated gate experiment based on inatial lpb, gate and pulse count.
        """

        int_target = initial_lpb
        ini_control = initial_gate
        rep = repeated_block
        fin = final_gate

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        sequence_lpb = []

        for n in pulse_count:
            sequence = LogicalPrimitiveBlockSerial(
                [int_target * ini_control] + [rep] * (n) + [
                    fin + mprim_target * mprim_control])
            sequence_lpb.append(sequence)

        lpb = LogicalPrimitiveBlockSweep(sequence_lpb)
        swp = sweeper.from_sweep_lpb(lpb)

        swp_flip = swp_initial
        swp_readout = swp_posterior

        basic(lpb, swp=swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

        self.N = 1
        self.pulse_count = pulse_count

        if fit:
            self.fit()

        return lpb, self.result

    def analyze_results(self):

        t_start = self.start_gate_number
        t_stop = self.start_gate_number + self.gate_count
        t_step = 1

        np.arange(t_start, t_stop, t_step)

        self.fitting_2D = []
        for i in range(2):
            self.real_part = self.result[:, i, 0]
            self.imag_part = self.result[:, i, 1]

            self.complex_data = self.real_part + 1j * self.imag_part

            self.fit_result = fit_2d_freq_with_cov(self.complex_data, dt=t_step,
                                                   freq_guess=0.125, use_freq_bound=True)
            self.fitting_2D.append(self.fit_result)

        self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
            'Frequency']) / 2
        self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
            'Frequency']) / 2


        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
        }

    def plot_specific_axis(self, fig, t, data, label, fit_params=None, t_interpolate=None,
                           use_imaginary_part=False):
        """
        Helper function to plot specific axis using Plotly based on the real or imaginary part of the fit.
        """
        color_ground = 'mediumblue'
        color_excited = 'crimson'
        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(
            go.Scatter(x=t, y=data, mode='lines+markers', name=label, opacity=0.5,
                       marker={'color': color}))

        if fit_params is not None:
            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * t[0]
            o_real = fit_params['Offset_real'].nominal_value
            o_imag = fit_params['Offset_imag'].nominal_value

            fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (
                o_real + 1j * o_imag)
            fit_values = np.real(fit) if not use_imaginary_part else np.imag(fit)

            fig.add_trace(
                go.Scatter(x=t_interpolate, y=fit_values, mode='lines',
                           name=f'{label} Fit', line={'color': color},
                           visible='legendonly'))

    @register_browser_function()
    # @visual_analyze_prompt(_v_prompt)
    def plot_x_axis(self):
        """
        Plot the results for the X axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number,
                                  self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 0, 0], label="Ground",
                                fit_params=self.fitting_2D[0], use_imaginary_part=False)
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 1, 0], label="Excited",
                                fit_params=self.fitting_2D[1], use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - X axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @register_browser_function()
    # @visual_analyze_prompt(_v_prompt)
    def plot_y_axis(self):
        """
        Plot the results for the Y axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number,
                                  self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 0, 1], label="Ground",
                                fit_params=self.fitting_2D[0], use_imaginary_part=False)
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 1, 1], label="Excited",
                                fit_params=self.fitting_2D[1], use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - Y axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    # @register_browser_function()
    def plot_z_axis(self):
        """
        Plot the results for the Y axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number,
                                  self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 0, 2], label="Ground",
                                use_imaginary_part=False)
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 1, 2], label="Excited",
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - Z axis",
                          xaxis_title="Pulse count",
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
Image("ref_images/success_ConditionalStarkShiftRepeatedGate.plot_fourier.png")
The following Image is a failure case for the experiment due to the presence of multiple peaks:
Image("ref_images/failure_3_ConditionalStarkShiftRepeatedGate.plot_fourier.png")
                """)
    def plot_fourier(self):
        fig = go.Figure()

        def plot_fourier_trace(fig, data, label, fit_params=None):
            color_ground = 'mediumblue'
            color_excited = 'crimson'

            t = np.arange(self.start_gate_number,
                          self.start_gate_number + self.gate_count, 1)

            color = color_ground if label == 'Ground' else color_excited

            # Compute the Fourier Transform of the data

            data = data - np.mean(data)

            fourier_transform = np.fft.fft(data)
            frequencies = np.fft.fftfreq(t.size, d=1)

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
            title="ZZ interaction repeated gate tomography - Fourier Transform",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Amplitude [a.u.]",
            plot_bgcolor='white',
            legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing status of a qubit over an experiment.
My objective is to determine if the experiment is a success. The success of the experiment should see the state
of the qubits remains stable in the Y axis throughout the experiment.
If you observe oscillations in the Y axis, or the lines crosses each other, the experiment is considered failed.
Otherwise, the experiment is considered successful.
For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftRepeatedGate.plot_control_population.png")
The following Image is a failure case for the experiment due to the presence of multiple peaks:
Image("ref_images/failure_ConditionalStarkShiftRepeatedGate.plot_control_population.png")
""")
    def plot_control_population(self):

        fig = go.Figure()
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)

        self.plot_specific_axis(fig, t=t, data=self.result_control[:, 0, 1],
                                label="Ground",
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, t=t, data=self.result_control[:, 1, 1],
                                label="Excited",
                                use_imaginary_part=True)

        fig.update_layout(title="Control qubit state - Z axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend={'x': 0, 'y': 1, 'traceorder': 'normal'})

        return fig

    def plot(self):
        """
        Plot the results.
        """
        args = self._get_run_args_dict()

        t = np.arange(args['start_gate_number'],
                      args['start_gate_number'] + args['gate_count'], 1)
        t_interpolate = np.arange(args['start_gate_number'],
                                  args['start_gate_number'] + args['gate_count'], 1 / 10)

        def plot_specific_axis(data, label, fit_params, use_imaginary_part):
            data = data.squeeze()

            plt.scatter(t, data, label=label, alpha=0.5)

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args[
                'start_gate_number']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params[
                'Offset_imag'].nominal_value

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate,
                     np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(6, 5))

        desired_num_ticks = 10  # Desired number of ticks
        step = max(1, len(t) // desired_num_ticks)
        xticks_subset = t[::step]
        plt.title("ZZ interaction repeated gate tomography - X axis")

        plot_specific_axis(data=self.result[:, 0, 0], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse count")
        plt.ylabel("<X>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction repeated gate tomography - Y axis")

        plot_specific_axis(data=self.result[:, 0, 1], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse count")
        plt.ylabel("<Y>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.show()

    def get_ai_inspection_results(self):
        """
        Returns the results of the AI inspection for the experiment.
        """

        inspection_results = super().get_ai_inspection_summary()
        zz_pgc = self.zz_rate
        zz_rate = self.zz_rate_continous
        zz_rate = np.sign(zz_pgc) * np.abs(zz_rate)

        target_zz = np.sign(zz_pgc) * 0.125
        zz_diff = target_zz - zz_pgc
        width_diff = np.sign(zz_diff / zz_rate) * min(
            np.abs(zz_diff / zz_rate / 2), 0.05 * self.width
        )
        width = self.width

        inspection_results['Calibrated parameters'] = {
            'amp_control': self.amp_control,
            'amp_target': self.amp_target,
            'frequency': self.frequency,
            'rise': self.rise,
            'phase_diff': self.phase_diff,
            'width': width + width_diff,
            'zz_interaction_positive': self.zz_rate > 0
        }

        return inspection_results

    @text_inspection
    def fitting(self) -> Union[str, None]:
        """
        Returns the analyzed result prompt for the experiment.
        """

        z_control_diff = self.result_control[:, 0, 1] - self.result_control[:, 1, 1]
        z_control_diff_max = np.max(np.abs(z_control_diff))
        z_control_diff_min = np.min(np.abs(z_control_diff))

        if z_control_diff_min < 0.25 * z_control_diff_max:
            extra_prompt = "The experiment failed because the population of the control qubit does not meet the criteria"
        else:
            extra_prompt = "The experiment is successful."

        prompt = f"""
        The expectation value of the control qubit along the Z axis is stable through the whole experiment.
        The maximum difference between the ground and excited state is {z_control_diff_max} and the minimum difference is {z_control_diff_min}.
        The experiment should be considered successful if the minimum difference is greater than 25% of the maximum difference.
        """ + extra_prompt

        return prompt
