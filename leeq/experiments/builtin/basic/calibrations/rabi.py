from typing import Optional, Dict, Any
import numpy as np

from labchronicle import register_browser_function, log_and_record
from leeq import Experiment, SweepParametersSideEffectFactory, Sweeper
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils.compatibility import *
from leeq.theory import fits
from plotly import graph_objects as go

from leeq.utils import setup_logging

logger = setup_logging(__name__)


class NormalisedRabi(Experiment):
    @log_and_record
    def run(self,
            dut_qubit: Any,
            amp: float = 0.05,
            start: float = 0.01,
            stop: float = 0.15,
            step: float = 0.001,
            fit: bool = True,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            pulse_discretization: bool = True,
            update=False,
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
        update (bool): Whether to update the qubit parameters. Default is False.
        initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.

        Returns:
        Dict[str, Any]: Fitted parameters if fit is True, None otherwise.
        """
        # Get c1 from the DUT qubit
        c1 = dut_qubit.get_c1(collection_name)
        rabi_pulse = c1['X'].clone()

        if amp is not None:
            rabi_pulse.update_pulse_args(amp=amp, phase=0., shape='square', width=step)
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
            pulse = LogicalPrimitiveBlockSweep([
                prims.SerialLPB([rabi_pulse] * k, name='rabi_pulse') for k in range(int((stop - start) / step + 0.5))
            ])
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
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=step)

        if update:
            # Update the qubit parameters, to make one pulse width correspond to a pi pulse
            # Here we suppose all pulse envelopes give unit area when width=1, amp=1
            normalised_pulse_area = c1['X'].calculate_envelope_area() / c1['X'].amp
            two_pi_area = amp * (1 / self.fit_params['Frequency'])
            new_amp = two_pi_area / 2 / normalised_pulse_area
            c1.update_parameters(amp=new_amp)
            print(f"Amplitude updated: {new_amp}")

    @log_and_record(overwrite_func_name='NormalisedRabi.run')
    def run_simulated(self,
                      dut_qubit: Any,
                      amp: float = 0.05,
                      start: float = 0.01,
                      stop: float = 0.15,
                      step: float = 0.001,
                      fit: bool = True,
                      collection_name: str = 'f01',
                      mprim_index: int = 0,
                      pulse_discretization: bool = True,
                      update=False,
                      initial_lpb: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Run a simulated Rabi experiment on a given qubit and analyze the results.

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
        update (bool): Whether to update the qubit parameters. Default is False.
        initial_lpb (Any): Initial lpb to add to the created lpb. Default is None.

        Returns:
        Dict[str, Any]: Fitted parameters if fit is True, None otherwise.
        """
        if initial_lpb is not None:
            logger.warning("initial_lpb is ignored in the simulated mode.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut_qubit)

        c1 = dut_qubit.get_c1(collection_name)

        # hard code a virtual dut here
        rabi_rate_per_amp = simulator_setup.get_omega_per_amp(c1.channel)  # MHz
        omega = rabi_rate_per_amp * amp

        # Detuning
        delta = virtual_transmon.qubit_frequency - c1['X'].freq

        # Time array (let's consider 100 ns for demonstration)
        t = np.arange(start, stop, step)  # 1000 points from 0 to 100 ns

        # Rabi oscillation formula
        self.data = (omega ** 2) / (delta ** 2 + omega ** 2) * np.sin(0.5 * np.sqrt(delta ** 2 + omega ** 2) * t) ** 2

        # Fit data to a sinusoidal function and return the fit parameters
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=step)

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
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

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
                    line=dict(color='Black', width=2)
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
                name=f'fit'
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
