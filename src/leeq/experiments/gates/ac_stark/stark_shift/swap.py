from .common import *

class StarkTwoQubitsSWAP(experiment):
    """Perform a Stark Shifted T1 experiment on one qubit while measuring another."""

    @log_and_record
    def run(self, qubits, amp, rise=0.01, start=0, stop=3, step=0.03,
            stark_offset=50, initial_lpb=None,
            trunc=1.2):
        """
        Execute Stark T1 experiment on two qubits on hardware.

        Parameters
        ----------
        qubits : list
            List of two qubits [control, target].
        amp : float
            Stark drive amplitude.
        rise : float, optional
            Pulse rise time. Default: 0.01.
        start : float, optional
            Start time (us). Default: 0.
        stop : float, optional
            Stop time (us). Default: 3.
        step : float, optional
            Time step (us). Default: 0.03.
        stark_offset : float, optional
            Stark frequency offset (MHz). Default: 50.
        initial_lpb : Any, optional
            Initial pulse sequence. Default: None.
        trunc : float, optional
            Pulse truncation. Default: 1.2.

        Returns
        -------
        None
            Results stored in instance attributes.
        """

        self.duts = qubits
        self.stark_offset = stark_offset
        self.amp_control = amp
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step

        c1_control = self.duts[0].get_default_c1()  # the qubit to be stark shifted and T1 performed on
        self.duts[1].get_default_c1()  # the qubit which will just be measured at the same time

        self.original_freq = c1_control['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c1_control['X'].clone()
        cs_pulse.update_pulse_args(amp=self.amp_control, freq=self.frequency, phase=0., shape='blackman_square',
                                   width=self.stop, rise=rise, trunc=trunc)

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        lpb = c1_control['X'] + cs_pulse + mprim_control * mprim_target

        if initial_lpb:
            lpb = initial_lpb + lpb

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
        # Plot Control data with both line and scatter markers
        axs[2].scatter(t, self.result_control, marker='o', color=dark_navy, label='Control', zorder=3)
        axs[2].plot(t, self.result_control, linestyle='-', marker='_', color=dark_navy, zorder=2)

        # Plot Target data with both line and scatter markers
        axs[2].scatter(t, self.result_target, marker='o', color=dark_purple, label='Target', zorder=3)
        axs[2].plot(t, self.result_target, linestyle='-', marker='_', color=dark_purple, zorder=2)

        # axs[2].scatter(t, self.result_control, marker='o', color=dark_navy, label='Control')
        # axs[2].scatter(t, self.result_target, marker='o', color=dark_purple, label='Target')
        # axs[2].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
        #             color=dark_navy, label=f'Control Fit T1={fit_control["Decay"][0]:.2f} us')
        # axs[2].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
        #             color=dark_purple, label=f'Target Fit T1={fit_target["Decay"][0]:.2f} us')
        axs[2].legend()

        fig.tight_layout()
        return fig
        # plt.show()

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
