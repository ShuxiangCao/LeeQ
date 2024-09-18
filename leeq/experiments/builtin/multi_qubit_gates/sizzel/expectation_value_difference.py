import numpy as np
from plotly.subplots import make_subplots

from labchronicle import log_and_record, register_browser_function
from leeq import Sweeper, Experiment
from typing import List, Any
from plotly import graph_objects as go

from leeq.utils.compatibility import *


class ConditionalStarkSpectroscopyDiffAmpFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and amplitude parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Get the measurement primitives from each DUT.
        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        # Get the default control pulse for each DUT.
        c1s = [dut.get_default_c1() for dut in duts]

        # Flip both`
        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='blackman_square', amp=0, phase=0, width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Create amplitude sweeper.
        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        # Set up additional pulse sequences and sweep.
        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = prims.ParallelLPB(cs_pulses)
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        # lpb = flip_sweep_lpb + c1s[1]['Xp'] + lpb_zz + c1s[1]['Ym'] + prims.ParallelLPB(mprims)

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp_amp + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude (a.u)",
        )

        return fig

    # @register_browser_function(available_after=(run,))
    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Create a subplot figure with 1 row and 2 columns
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        # Generate the heatmap for the main result
        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),  # Adjust x to position the colorbar
        )

        # Generate the heatmap for the control result
        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),  # Adjust x to position the colorbar
        )

        # Add heatmaps to the figure
        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        # Update layout for the figure
        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),  # Add margins to make space for titles
            width=1600,  # Increase the width for better resolution
            height=600,  # Increase the height for better resolution
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        # Customize x-axis and y-axis for both subplots
        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot()


class ConditionalStarkSpectroscopyDiffAmpTargetFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and amplitude parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            amp_control_fixed=0.2,
            amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """

        self.amp_control_fixed = amp_control_fixed

        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Get the measurement primitives from each DUT.
        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        # Get the default control pulse for each DUT.
        c1s = [dut.get_default_c1() for dut in duts]

        # Flip both
        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='blackman_square', amp=self.amp_control_fixed, phase=0,
                                       width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # # Create amplitude sweeper.
        # swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
        #                   params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        # Create amplitude sweeper and apply only to the second cs_pulse only
        swp_amp = sweeper(
            np.arange,
            n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
            params=[sparam.func(cs_pulses[1].update_pulse_args, {}, 'amp')]  # Target only the second pulse
        )

        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        # Set up additional pulse sequences and sweep.
        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = prims.ParallelLPB(cs_pulses)
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp_amp + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit with Control Fixed Amplitude",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Target Driving amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control with Control Fixed Amplitude",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Target Driving amplitude (a.u)",
        )

        return fig

    # @register_browser_function(available_after=(run,))
    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Create a subplot figure with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Spectroscopy Difference in Y axis on Target Qubit with Control Fixed Amplitude",
            "Spectroscopy Difference in Y axis on Control Qubit with Control Fixed Amplitude"))

        # Generate the heatmap for the main result
        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),  # Adjust x to position the colorbar
        )

        # Generate the heatmap for the control result
        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),  # Adjust x to position the colorbar
        )

        # Add heatmaps to the figure
        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        # Update layout for the figure
        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Target Driving Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),  # Add margins to make space for titles
            width=1600,  # Increase the width for better resolution
            height=600,  # Increase the height for better resolution
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        # Customize x-axis and y-axis for both subplots
        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot()


class ConditionalStarkSpectroscopyDiffPhaseFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and phase parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            phase_diff_start: float = 0, phase_diff_stop: float = np.pi, phase_diff_step: float = np.pi / 10,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, amp=0.2, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the phase and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            phase_diff_start (float): Starting phase difference for the sweep.
            phase_diff_stop (float): Stopping phase difference for the sweep.
            phase_diff_step (float): Step size for the phase difference sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            amp (float): Amplitude for the control and target pulses.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        self.duts = duts
        self.frequency = freq_start
        self.amp_control = amp
        self.amp_target = amp
        self.width = width
        self.phase_diff = phase_diff_stop

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]
        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='blackman_square', amp=0, phase=phase_diff_start,
                                       width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Create amplitude sweeper and apply only to the second cs_pulse.
        swp_phase = sweeper(
            np.arange,
            n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
            params=[sparam.func(cs_pulses[1].update_pulse_args, {}, 'phase')]  # Target only the second pulse
        )

        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        basic(lpb, swp=swp_phase + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)

        return fig


class ConditionalStarkSpectroscopyDiffAmpPhase(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the amplitude and phase parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            phase_diff_start: float = 0, phase_diff_stop: float = np.pi, phase_diff_step: float = np.pi / 10,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, frequency=4700, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and phase and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            phase_diff_start (float): Starting phase difference for the sweep.
            phase_diff_stop (float): Stopping phase difference for the sweep.
            phase_diff_step (float): Step size for the phase difference sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            frequency (float): Frequency for the control and target pulses (MHz).
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_start
        self.amp_target = amp_start
        self.width = width
        self.phase_diff = phase_diff_step

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]
        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='blackman_square', amp=0, phase=phase_diff_start,
                                       width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Create amp sweeper.
        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        # Create phase sweeper.
        # swp_phase = sweeper(np.arange, n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
        #                     params=[sparam.func(cs_pulse.update_pulse_args, {}, 'phase') for cs_pulse in cs_pulses])

        # Create phase sweeper and apply only to the second cs_pulse.
        swp_phase = sweeper(
            np.arange,
            n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
            params=[sparam.func(cs_pulses[1].update_pulse_args, {}, 'phase')]  # Target only the second pulse
        )

        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        basic(lpb, swp=swp_amp + swp_phase + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Phase Difference (radians)', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig


class ConsidtionalStarkSpectroscopyDifferenceBase(Experiment):
    """
    A class to execute conditional Stark spectroscopy and look at the difference between the excited and non excited
    state of the expectation values.
    """

    @log_and_record
    def run(self, duts: List[Any], swp: Sweeper, rise: float = 0.01, trunc: float = 1.2, width: float = 0.7,
            echo=True) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Get the measurement primitives from each DUT.
        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        # Get the default control pulse for each DUT.
        c1s = [dut.get_default_c1() for dut in duts]

        # Flip both`
        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='blackman_square', amp=0, phase=0, width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Set up additional pulse sequences and sweep.
        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = prims.ParallelLPB(cs_pulses)

        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self._get_run_args_dict()

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Phase Difference (radians)', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig
