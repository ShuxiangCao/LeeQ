# This file setup the high-level simulation and provides a 2Q virtual device.
import numpy as np

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon

def simulation_setup(qubit_frequency=5040.4,readout_frequency=9645.4,quiescent_state_distribution=None):
    from labchronicle import Chronicle
    Chronicle(config={'handler':'memory'}).start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # print('readout_frequency',readout_frequency)

    if quiescent_state_distribution is None:
        quiescent_state_distribution=np.asarray(
                [
                    0.8,
                    0.15,
                    0.04,
                    0.01])
        
    virtual_transmon_a = VirtualTransmon(
        name="VQubitA",
        qubit_frequency=qubit_frequency,
        anharmonicity=-198,
        t1=70,
        t2=35,
        readout_frequency=readout_frequency,quiescent_state_distribution=quiescent_state_distribution)
    

    virtual_transmon_b = VirtualTransmon(
        name="VQubitB",
        qubit_frequency=4855.3,
        anharmonicity=-197,
        t1=60,
        t2=30,
        readout_frequency=readout_frequency,
        quiescent_state_distribution=quiescent_state_distribution)

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon_a,
                        4: virtual_transmon_b}
    )
    manager.register_setup(setup)
    return manager


configuration_a = {
    'hrid':'QA',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.1 ,
            'phase': 0.,
            'width': 0.05,
            'alpha': 500,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4-198,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.1 / np.sqrt(2),
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9645.5,
            'channel': 1,
            'shape': 'square',
            'amp': 0.15,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

configuration_b = {
    'hrid':'QB',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4855.3,
            'channel': 4,
            'shape': 'blackman_drag',
            'amp': 0.1 ,
            'phase': 0.,
            'width': 0.05,
            'alpha': 500,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4-197,
            'channel': 4,
            'shape': 'blackman_drag',
            'amp': 0.1 / np.sqrt(2),
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9025.5,
            'channel': 3,
            'shape': 'square',
            'amp': 0.15,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

def extract_results_from_experiment(exp):
    analyze_results = {
        'full':exp.get_ai_inspection_results(inspection_method='full',ignore_cache=True),
        'fitting_only':exp.get_ai_inspection_results(inspection_method='fitting_only',ignore_cache=True),
        'visual_only':exp.get_ai_inspection_results(inspection_method='visual_only',ignore_cache=True),
    }
    exp._execute_browsable_plot_function(build_static_image=True)
    return analyze_results


from typing import Optional, Dict, Any, Union
import numpy as np

from labchronicle import register_browser_function, log_and_record
from leeq import Experiment, SweepParametersSideEffectFactory, Sweeper
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils.compatibility import *
from leeq.utils.ai.vlms import visual_analyze_prompt
from leeq.theory import fits
from plotly import graph_objects as go

from leeq.utils import setup_logging


class NormalisedRabiDataValidityCheck(Experiment):
    _experiment_result_analysis_instructions = """
    The Normalised Rabi experiment is a quantum mechanics experiment that involves the measurement of oscillations.
    A successful Rabi experiment will show a clear, regular oscillatory pattern. 
    """

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
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=step)

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

    @log_and_record(overwrite_func_name='NormalisedRabiDataValidityCheck.run')
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
                      update=True,
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
        update (bool): Whether to update the qubit parameters. Default is True.
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
        rabi_rate_per_amp = simulator_setup.get_omega_per_amp(
            c1.channel)  # MHz
        omega = rabi_rate_per_amp * amp

        # Detuning
        delta = virtual_transmon.qubit_frequency - c1['X'].freq

        # Time array (let's consider 100 ns for demonstration)
        t = np.arange(start, stop, step)  # 1000 points from 0 to 100 ns

        # Rabi oscillation formula
        self.data = (omega ** 2) / (delta ** 2 + omega ** 2) * \
                    np.sin(0.5 * np.sqrt(delta ** 2 + omega ** 2) * t) ** 2

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the
            # sampling noise
            self.data = np.random.binomial(
                shot_number, self.data) / shot_number

        quiescent_state_distribution = virtual_transmon.quiescent_state_distribution
        standard_deviation = np.sum(quiescent_state_distribution[1:])

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, self.data.shape)

        self.data = (2 * self.data - 1)

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, self.data.shape)

        random_noise_sum = np.random.normal(
            0, standard_deviation/2, self.data.shape)

        self.data = np.clip(self.data * (0.5 - quiescent_state_distribution[0]) * 2 * random_noise_factor + random_noise_sum, -1, 1)

        # Fit data to a sinusoidal function and return the fit parameters
        self.fit_params = fits.fit_sinusoidal(self.data, time_step=step)
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
    @visual_analyze_prompt("""
Analyze this quantum mechanics Rabi oscillation experiment plot. Determine if it shows a successful or failed experiment by evaluating:
    1. Oscillation behaviour in the figure. It may not be perfect, but it needs to distinguish from random noise data. 
    2. Amplitude and frequency consistency. inconsistent oscillation is considered a failure.
    """)
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

    def get_analyzed_result_prompt(self) -> str:
        """
        Get the prompt to analyze the result.

        Returns:
        str: The prompt to analyze the result.
        """

        oscillation_freq = self.fit_params['Frequency']
        experiment_time_duration = self.retrieve_args(self.run)['stop'] - self.retrieve_args(self.run)['start']
        oscillation_count = (experiment_time_duration * oscillation_freq)

        return (f"The fitting result of the Rabi oscillation suggest the amplitude of {self.fit_params['Amplitude']}, "
                f"the frequency of {self.fit_params['Frequency']}, the phase of {self.fit_params['Phase']}. The offset of"
                f" {self.fit_params['Offset']}. The suggested new driving amplitude is {self.guess_amp}."
                f"From the fitting results, the plot should exhibit {oscillation_count} oscillations.")

