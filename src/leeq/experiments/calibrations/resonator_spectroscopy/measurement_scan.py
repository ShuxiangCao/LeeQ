from .common import *

class MeasurementScanParams(Experiment):
    """
    Class for managing and executing measurement scan parameters
    in an experimental setup.
    """

    @log_and_record
    def run(self, dut, sweep_lpb_list, mprim_index: int,
            amp_scan: dict = None, freq_scan: dict = None,
            accumulate_snr_for_all_distinguishable_state: bool = True,
            disable_sub_plot: bool = True):
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut : Any
            Device under test.
        sweep_lpb_list : list
            List of sweep parameters.
        mprim_index : int
            Measurement primitive index.
        amp_scan : dict, optional
            Parameters for amplitude scan. Default: None
        freq_scan : dict, optional
            Parameters for frequency scan. Default: None
        accumulate_snr_for_all_distinguishable_state : bool, optional
            Flag to accumulate SNR for all distinguishable states. Default: True
        disable_sub_plot : bool, optional
            Flag to disable subplot. Default: True

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        # Initialize lists to store scan results
        self.snrs = []
        self.scanned_freqs = []
        self.scanned_amps = []
        self.measurement_scan_result = []

        # Get measurement primitives
        mprim = dut.get_measurement_prim_intlist(mprim_index)

        # Set scanned frequencies and amplitudes
        self.scanned_freqs = [
            mprim.freq] if freq_scan is None else np.arange(
            **freq_scan)
        self.scanned_amps = [
            mprim.primary_kwargs()['amp']] if amp_scan is None else np.arange(
            **amp_scan)

        # Check for plot settings in Jupyter
        plot_result_in_jupyter = setup().status().get_param("Plot_Result_In_Jupyter")
        if disable_sub_plot:
            setup().status().set_param("Plot_Result_In_Jupyter", False)

        from leeq.experiments.calibrations.state_discrimination.gaussian_mixture import MeasurementCalibrationMultilevelGMM

        # Perform measurement scan
        for freq in self.scanned_freqs:
            for amp in self.scanned_amps:
                result = MeasurementCalibrationMultilevelGMM(
                    dut=dut,
                    sweep_lpb_list=sweep_lpb_list,
                    mprim_index=mprim_index,
                    freq=freq,
                    amp=amp)

                snr = 1 / np.sum([1 / (x + 1e-6) for x in result.snr.values()]) \
                    if accumulate_snr_for_all_distinguishable_state else result.SNR[(mprim_index, mprim_index + 1)]

                self.snrs.append(snr)
                self.measurement_scan_result.append(result)

        # Restore plot settings
        setup().status().set_param("Plot_Result_In_Jupyter", plot_result_in_jupyter)
        self.snrs = np.asarray(self.snrs).reshape(
            [len(self.scanned_freqs), len(self.scanned_amps)])

    # Additional methods follow the same pattern of revision.
    # ...
    @register_browser_function(available_after=(run,))
    def plot_snr_vs_freq(self):
        """
        Plots Signal-to-Noise Ratio (SNR) versus frequency.
        """
        if len(self.scanned_freqs) == 1:
            return
        if len(self.scanned_amps) > 1:
            return
        plt.figure()
        plt.title("SNR vs Frequency")
        plt.xlabel('Resonator Frequency')
        plt.ylabel('SNR')
        plt.plot(self.scanned_freqs, np.asarray(self.snrs).flatten())
        plt.grid()
        plt.show()

    @register_browser_function(available_after=(run,))
    def plot_snr_vs_amp(self):
        """
        Plots Signal-to-Noise Ratio (SNR) versus amplitude.
        """
        if len(self.scanned_amps) == 1:
            return
        if len(self.scanned_freqs) > 1:
            return

        plt.figure()
        plt.title("SNR vs Amplitude")
        plt.xlabel('Driving Amplitude')
        plt.ylabel('SNR')
        plt.plot(self.scanned_amps, np.asarray(self.snrs).flatten())
        plt.grid()
        plt.show()

    @register_browser_function(available_after=(run,))
    def plot_snr_vs_amp_freq(self, fig_size=(10, 10)):
        """
        Plots Signal-to-Noise Ratio (SNR) versus both amplitude and frequency.

        Args:
            fig_size (tuple, optional): Figure size. Defaults to (10, 10).
        """

        if len(self.scanned_freqs) == 1 or len(self.scanned_amps) == 1:
            return

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title("SNR vs Amplitude / Frequency")
        cax = ax.imshow(
            np.asarray(
                self.snrs),
            aspect='auto',
            interpolation='nearest')

        # Adding text annotation inside the cells
        for i in range(len(self.scanned_freqs)):
            for j in range(len(self.scanned_amps)):
                ax.text(j, i, f"{self.snrs[i, j]:.2f}",
                               ha="center", va="center", color="w")

        # set ticks
        ax.set_xticks(ticks=np.arange(len(self.scanned_amps)),
                      labels=[f"{x:.2f}" for x in self.scanned_amps])
        ax.set_yticks(ticks=np.arange(len(self.scanned_freqs)),
                      labels=[f"{x:.2f}" for x in self.scanned_freqs])

        plt.xlabel('Amplitude [a.u.]')
        plt.ylabel('Frequency [MHz]')

        # Creating color bar
        fig.colorbar(cax, ax=ax)
        plt.show()

    def dump_data(self):
        """
        Dumps the scan data to a pickle file.
        """
        path = 'dump.pickle'
        data = {
            "freqs": self.scanned_freqs,
            "amps": self.scanned_amps,
            "shot_data": [x.result for x in self.measurement_scan_result],
            'clfs': [x.clf for x in self.measurement_scan_result]
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)
