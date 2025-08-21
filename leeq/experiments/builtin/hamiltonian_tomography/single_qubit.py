from typing import List, Optional, Union

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory, basic_run
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.built_in.common import Delay
from leeq.core.primitives.logical_primitives import (
    LogicalPrimitive,
    LogicalPrimitiveBlock,
    LogicalPrimitiveBlockParallel,
    LogicalPrimitiveBlockSweep
)
from leeq.theory.fits import fit_1d_freq_exp_with_cov, fit_2d_freq_with_cov
from leeq.utils import setup_logging

logger = setup_logging(__name__)


class HamiltonianTomographySingleQubitBase(Experiment):
    """
    Base class for implementing Hamiltonian tomography.

    """

    def _run(self,
             duts: List[TransmonElement],
             lpb: LogicalPrimitiveBlock,
             swp: Sweeper,
             tomography_axis: Union[str, List[str]],
             initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]] = None,
             collection_name='f01',
             mprim_index='0'
             ):
        """
        This method runs the tomography experiment.

        Prameters:
        ----------
        duts: List[TransmonElement]
            List of transmon elements to be characterized.
        lpb: LogicalPrimitiveBlock
            Logical primitive block to be used for the experiment, describe the pulse that generates the
             Hamiltonian to be characterized.
        swp: Sweeper
            Sweeper object to be used for the experiment.
        tomography_axis: Union[str, List[str]]
            Axis along which the Hamiltonian tomography is to be performed. Must be one of 'X', 'Y', 'Z'.
        initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]]
            Initial logical primitive block to be used for the experiment.
        collection_name: str
            Name of the collection (transition frequency) to be used for the experiment.
        """
        self.results = None

        if isinstance(tomography_axis, str):
            tomography_axis = [tomography_axis]

        if not isinstance(tomography_axis, list):
            msg = 'tomography_axis must be a list of strings or a string'
            logger.error(msg)
            raise ValueError(msg)

        def get_tomography_lpb(dut, axis):
            """Get the LPB that rotates the qubit measurement axis to Z axis."""
            assert axis in ['X', 'Y', 'Z']
            c1 = dut.get_c1(collection_name)
            if axis == 'X':
                return c1['Yp']
            elif axis == 'Y':
                return c1['Xm']
            elif axis == 'Z':
                return c1['I']

        tomography_lpb = []

        for axis in tomography_axis:
            if axis not in ['X', 'Y', 'Z']:
                msg = 'Axis must be one of X, Y, Z. Got: {}'.format(axis)
                logger.error(msg)
                raise ValueError(msg)
            tomography_lpb.append(LogicalPrimitiveBlockParallel([get_tomography_lpb(dut, axis) for dut in duts]))

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        tomography_lpb = LogicalPrimitiveBlockSweep(tomography_lpb)
        swp_tomo = Sweeper.from_sweep_lpb(tomography_lpb)

        self.tomography_axis = tomography_axis
        self.duts = duts

        lpb_run = lpb + tomography_lpb + LogicalPrimitiveBlockParallel(mprims)

        if initial_lpb is not None:
            lpb_run = initial_lpb + lpb_run

        basic_run(lpb_run, swp + swp_tomo, "<z>")

        # Each result has the axis: [time index, tomography axis index]
        self.results = [np.squeeze(mprim.result()) for mprim in mprims]

    def analyze_data(self, step_t):
        """
        This method analyses the data obtained from the experiment.

        Parameters:
        -----------
        step_t: float
            Step time of the Hamiltonian under characterization.
        """
        if self.results is None:
            msg = 'No data to analyse. Run the experiment first.'
            logger.error(msg)
            raise ValueError(msg)

        self.analyzed_results = []

        def _analyze_single_qubit_result(result):
            # Each result has the axis: [time index, tomography axis index]. If there is only one axis, the
            # result is a 1D array.
            if len(result.shape) == 1:
                return fit_1d_freq_exp_with_cov(z=result, dt=step_t)
            elif len(result.shape) == 2:
                z = result[:, 0] + 1j * result[:, 1]
                return fit_2d_freq_with_cov(z=z, dt=step_t)
            else:
                raise NotImplementedError()

        for result in self.results:  # For each qubit
            self.analyzed_results.append(_analyze_single_qubit_result(result))


class HamiltonianTomographySingleQubitXYBase(HamiltonianTomographySingleQubitBase):
    """
    Hamiltonian tomography equivalent to the ramsey experiment.
    """

    def _run(self,
             duts: List[TransmonElement],
             lpb: LogicalPrimitiveBlock,
             swp: Sweeper,
             initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]] = None,
             collection_name='f01',
             mprim_index='0'
             ):
        """
        This method runs the stark shift tomography experiment.

        Parameters:
        -----------
        duts: List[TransmonElement]
            List of transmon elements to be characterized.
        lpb: LogicalPrimitiveBlock
            Logical primitive block to be used for the experiment, describe the pulse that generates the
                Hamiltonian to be characterized.
        swp: Sweeper
            Sweeper object to be used for the experiment.
        initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]]
            Initial logical primitive block to be used for the experiment.
        collection_name: str
            Name of the collection (transition frequency) to be used for the experiment.
        mprim_index: str
            Index of the measurement primitive to be used for the experiment.
        """

        return super()._run(duts=duts,
                            lpb=lpb,
                            swp=swp,
                            tomography_axis=['X', 'Y'],
                            initial_lpb=initial_lpb,
                            collection_name=collection_name,
                            mprim_index=mprim_index
                            )

    @register_browser_function()
    def plot_all(self):
        self.analyze_data()
        for i in range(len(self.analyzed_results)):
            fig = self.plot(i)
            fig.show()

    def plot(self, i: int):
        """
                Plots the fitted curve using data from the experiment.

                This method uses Plotly for generating the plot. It analyzes the data, performs
                curve fitting, and then plots the actual data along with the fitted curve.

                Parameters:
                    i: The index of the qubit for which to plot the data.
                """
        args = self._get_run_args_dict()
        fit_params = self.analyzed_results[i]

        # Generate time points based on the experiment arguments
        time_points = np.arange(args['start_time'], args['stop_time'], args['step_time'])
        time_points_interpolate = np.arange(
            args['start_time'], args['stop_time'], args['step_time'] / 10)

        # Extract fitting parameters
        frequency = fit_params['Frequency']
        amplitude = fit_params['Amplitude']
        phase = fit_params['Phase'] - 2.0 * \
            np.pi * frequency * args['start_time']
        offset = fit_params['Offset']

        # Generate the fitted curve
        fitted_curve = amplitude * np.exp(1.j * (
            2.0 * np.pi * frequency * time_points_interpolate + phase)) + offset

        # Create a plot using Plotly
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=self.results[i][:, 0],
                mode='markers',
                name='Data X'),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=self.results[i][:, 1],
                mode='markers',
                name='Data Y'),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=time_points_interpolate,
                y=fitted_curve.real,
                mode='lines',
                name='Fit X'),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=time_points_interpolate,
                y=fitted_curve.imag,
                mode='lines',
                name=f'Fit Y'),
            row=1,
            col=1)

        # Set plot layout details
        title_text = f"Tomography fit {self.duts[i].hrid} transition {args['collection_name']}: <br>"

        fig.update_layout(
            title_text=title_text,
            xaxis_title=f"Time (us) <br> Frequency: {frequency}",
            yaxis_title="<z>",
            plot_bgcolor="white")
        return fig


class HamiltonianTomographySingleQubitStarkShift(HamiltonianTomographySingleQubitXYBase):
    """
    Hamiltonian tomography reveals the Z term under the stark shift drive.
    """

    @log_and_record
    def run(self,
            dut: TransmonElement,
            stark_amp: float,
            stark_freq: float,
            start_time: float,
            stop_time: float,
            step_time: float,
            initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]] = None,
            collection_name: str = 'f01',
            mprim_index: str = '0'
            ):
        """
        This method runs the stark shift tomography experiment.

        Parameters:
        -----------
        dut: TransmonElement
            Transmon element to be characterized.
        start_time: float
            Start time of the stark shift drive.
        stop_time: float
            Stop time of the stark shift drive.
        step_time: float
            Step time of the stark shift drive.
        stark_amp: float
            Amplitude of the stark shift drive.
        stark_freq: float
            Frequency of the stark shift drive.
        initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]]
            Initial logical primitive block to be used for the experiment.
        collection_name: str
            Name of the collection (transition frequency) to be used for the experiment.
        mprim_index: str
            Index of the measurement primitive to be used for the experiment.
        """

        # Get c1 from the DUT qubit
        c1 = dut.get_c1(collection_name)
        stark_shift_drive = c1['X'].clone()

        parameters = {
            'freq': stark_freq,
            'shape': 'square',
            'phase': 0.,
            'amp': stark_amp
        }

        stark_shift_drive.update_pulse_args(**parameters)

        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            stark_shift_drive.update_pulse_args, {}, 'width'
        )]

        swp = Sweeper(
            np.arange,
            n_kwargs={'start': start_time, 'stop': stop_time, 'step': step_time},
            params=swpparams
        )

        initial_preparation_lpb = dut.get_c1(collection_name)['Ym']  # Start by moving the bloch vector to X axis

        if initial_lpb is not None:
            initial_preparation_lpb = initial_lpb + initial_preparation_lpb

        self._run(duts=[dut],
                  lpb=stark_shift_drive,
                  swp=swp,
                  initial_lpb=initial_preparation_lpb,
                  collection_name=collection_name,
                  mprim_index=mprim_index
                  )

    def analyze_data(self):
        """
        This method analyses the data obtained from the stark shift tomography experiment.

        Parameters:
        -----------
        start_time: float
            Start time of the stark shift drive.
        stop_time: float
            Stop time of the stark shift drive.
        step_time: float
            Step time of the stark shift drive.
        """
        kwargs = self._get_run_args_dict()
        super().analyze_data(kwargs['step_time'])


class HamiltonianTomographySingleQubitOffresonanceDrive(HamiltonianTomographySingleQubitXYBase):
    """
    Hamiltonian tomography equivalent to the ramsey experiment.
    """

    @log_and_record
    def run(self,
            dut: TransmonElement,
            offset_freq: float,
            start_time: float,
            stop_time: float,
            step_time: float,
            initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]] = None,
            collection_name: str = 'f01',
            mprim_index: str = '0'
            ):
        """
        This method runs the stark shift tomography experiment.

        Parameters:
        -----------
        dut: TransmonElement
            Transmon element to be characterized.
        start_time: float
            Start time of the stark shift drive.
        stop_time: float
            Stop time of the stark shift drive.
        step_time: float
            Step time of the stark shift drive.
        offset_freq: float
            Frequency of the stark shift drive.
        initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]]
            Initial logical primitive block to be used for the experiment.
        collection_name: str
            Name of the collection (transition frequency) to be used for the experiment.
        mprim_index: str
            Index of the measurement primitive to be used for the experiment.
        """

        # Get c1 from the DUT qubit
        c1 = dut.get_c1(collection_name)

        original_frequency = c1.freq

        c1.update_parameters(freq=offset_freq + original_frequency)

        delay = Delay(0)

        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            delay.update_parameters, {}, 'delay_time'
        )]

        swp = Sweeper(
            np.arange,
            n_kwargs={'start': start_time, 'stop': stop_time, 'step': step_time},
            params=swpparams
        )

        initial_preparation_lpb = dut.get_c1(collection_name)['Ym']  # Start by moving the bloch vector to X axis

        if initial_lpb is not None:
            initial_preparation_lpb = initial_lpb + initial_preparation_lpb

        self._run(duts=[dut],
                  lpb=delay,
                  swp=swp,
                  initial_lpb=initial_preparation_lpb,
                  collection_name=collection_name,
                  mprim_index=mprim_index
                  )

        c1.update_parameters(freq=original_frequency)

    def analyze_data(self):
        """
        This method analyses the data obtained from the stark shift tomography experiment.

        Parameters:
        -----------
        start_time: float
            Start time of the stark shift drive.
        stop_time: float
            Stop time of the stark shift drive.
        step_time: float
            Step time of the stark shift drive.
        """
        kwargs = self._get_run_args_dict()
        super().analyze_data(kwargs['step_time'])
