from .common import *
from .transmission import ResonatorSweepTransmissionWithExtraInitialLPB

class ResonatorSweepTransmissionXiComparison(Experiment):
    """
    Class for comparing resonator sweep transmission with extra initial logical primitive block (LPB).
    It includes methods to run the experiment, and to plot magnitude and phase using both
    matplotlib and plotly.
    """

    @log_and_record
    def run(self,
            dut_qubit: Any,
            lpb_scan: Union[List, Tuple, Dict],
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.,
            num_avs: int = 5000,
            rep_rate: float = 0.,
            mp_width: Optional[float] = 8,
            amp: Optional[float] = None) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (qubit object).
        lpb_scan : Union[List, Tuple, Dict]
            LPBs for different state preparations to compare.
        start : float, optional
            Start frequency for the sweep (MHz). Default: 8000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 9000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 5000
        rep_rate : float, optional
            Repetition rate. Default: 0.0
        mp_width : Optional[float], optional
            Measurement pulse width (μs). Default: 8.0
        amp : Optional[float], optional
            Drive amplitude. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        if isinstance(lpb_scan, (tuple, list)):
            lpb_scan = dict(enumerate(lpb_scan))

        self.result_dict = {
            key: ResonatorSweepTransmissionWithExtraInitialLPB(
                dut_qubit=dut_qubit, start=start, stop=stop, step=step,
                num_avs=num_avs, rep_rate=rep_rate,
                mp_width=mp_width, initial_lpb=lpb, amp=amp
            ) for key, lpb in lpb_scan.items()
        }

    @register_browser_function(available_after=(run,))
    def plot_magnitude_plotly(self) -> None:
        """
        Plots the magnitude of the resonator spectroscopy using Plotly.
        """
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])

        fig = go.Figure()

        for key, sweep in self.result_dict.items():
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=sweep.result['Magnitude'],
                    mode='lines',
                    name=key))

        fig.update_layout(
            title='Resonator spectroscopy magnitude',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Magnitude',
            plot_bgcolor='white')

        fig.show()

    @register_browser_function(available_after=(run,))
    def plot_phase_plotly(self) -> None:
        """
        Plots the phase of the resonator spectroscopy using Plotly.
        """
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])

        fig = go.Figure()

        for key, sweep in self.result_dict.items():
            phase_trace = sweep.result['Phase']
            phase_trace_mod = sweep.UnwrapPhase(phase_trace)
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=phase_trace_mod,
                    mode='lines',
                    name=key))

        fig.update_layout(
            title='Resonator spectroscopy phase',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Phase',
            plot_bgcolor='white')

        fig.show()

    @register_browser_function(available_after=(run,))
    def plot_phase_diff_fit_plotly(self) -> None:
        """
        Plots the differentiated phase and its Lorentzian fit using Plotly.
        """
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])
        f_interpolate = np.arange(
            args['start'],
            args['stop'],
            args['step'] / 5)

        fig = go.Figure()

        for key, sweep in self.result_dict.items():
            z, f0, Q, amp, baseline, direction = sweep.fit_phase_diff()
            lorentzian_fit = sweep.root_lorentzian(
                f_interpolate, f0, Q, amp, baseline) * direction

            fig.add_trace(
                go.Scatter(
                    x=f_interpolate,
                    y=lorentzian_fit,
                    mode='lines',
                    name=f'{key} Lorentzian fit'))
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=z,
                    mode='markers',
                    name=f'{key} Phase derivative'))


        fig.update_layout(
            title='Resonator spectroscopy phase fitting',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Phase',
            plot_bgcolor='white')

        fig.show()


# New Kerr-enabled experiments for high-power regime characterization
