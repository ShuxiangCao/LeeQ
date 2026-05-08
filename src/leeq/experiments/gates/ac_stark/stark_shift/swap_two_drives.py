from .common import *

class StarkTwoQubitsSWAPTwoDrives(experiment):
    """Perform Stark Shifted T1 with independent drives on both qubits."""

    @log_and_record
    def run(self, qubits, amp_control, amp_target, rise=0.01, start=0, stop=3, step=0.03,
            phase_diff=0, stark_offset=50, initial_lpb=None):
        """
        Execute Stark T1 with two drives on hardware.

        Parameters
        ----------
        qubits : list
            List of two qubits [control, target].
        amp_control : float
            Stark drive amplitude on control qubit.
        amp_target : float
            Stark drive amplitude on target qubit.
        rise : float, optional
            Pulse rise time. Default: 0.01.
        start : float, optional
            Start time (us). Default: 0.
        stop : float, optional
            Stop time (us). Default: 3.
        step : float, optional
            Time step (us). Default: 0.03.
        phase_diff : float, optional
            Phase difference between drives. Default: 0.
        stark_offset : float, optional
            Stark frequency offset (MHz). Default: 50.
        initial_lpb : Any, optional
            Initial pulse sequence. Default: None.

        Returns
        -------
        None
            Results stored in instance attributes.
        """

        self.duts = qubits
        self.stark_offset = stark_offset
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.fitting_2D = None
        self.phase_diff = phase_diff

        c1_control = self.duts[0].get_default_c1()  # the qubit to be stark shifted and T1 performed on
        self.duts[1].get_default_c1()  # the qubit which will just be measured at the same time

        self.original_freq = c1_control['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                  amp_target=self.amp_target, amp_control=self.amp_control,
                                                  frequency=self.frequency, rise=rise, width=self.width,
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

        lpb = c1_control['X'] + cs_pulse + mprim_control * mprim_target

        if initial_lpb:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=swpparams)

        basic(lpb, swp, 'p(1)')

        self.result_control = np.asarray(mprim_control.result(), dtype=float).flatten()
        self.result_target = np.asarray(mprim_target.result(), dtype=float).flatten()

    @register_browser_function(available_after=(run,))
    def plot_t1(self):
        args = self._get_run_args_dict()

        dark_navy = '#000080'
        dark_purple = '#800080'

        fit_control = self.fit_exp_decay_with_cov(self.result_control, args['step'])
        fit_target = self.fit_exp_decay_with_cov(self.result_target, args['step'])

        t = np.arange(0, args['stop'], args['step'])

        fig, axs = plt.subplots(1, 3, figsize=[25, 5])

        # Plot for result_control
        axs[0].set_title(f"Control T1 decay\nT1={fit_control['Decay'][0]:.2f} ± {fit_control['Decay'][1]:.2f} us")
        axs[0].set_xlabel("Time (us)")
        axs[0].set_ylabel("P(1)")
        axs[0].scatter(t, self.result_control, marker='o', color=dark_navy)
        axs[0].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
                    color=dark_navy)

        # Plot for result_target
        axs[1].set_title(f"Target T1 decay\nT1={fit_target['Decay'][0]:.2f} ± {fit_target['Decay'][1]:.2f} us")
        axs[1].set_xlabel("Time (us)")
        axs[1].set_ylabel("P(1)")
        axs[1].scatter(t, self.result_target, marker='o', color=dark_purple)
        axs[1].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
                    color=dark_purple)

        # Combined plot
        axs[2].set_title("Control and Target T1 decay")
        axs[2].set_xlabel("Time (us)")
        axs[2].set_ylabel("P(1)")
        axs[2].scatter(t, self.result_control, marker='o', color=dark_navy, label='Control')
        axs[2].scatter(t, self.result_target, marker='o', color=dark_purple, label='Target')
        axs[2].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
                    color=dark_navy, label=f'Control Fit T1={fit_control["Decay"][0]:.2f} us')
        axs[2].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
                    color=dark_purple, label=f'Target Fit T1={fit_target["Decay"][0]:.2f} us')
        axs[2].legend()

        fig.tight_layout()
        plt.show()

    def fit_exp_decay_with_cov(self, trace, time_resolution):
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
