from labchronicle import log_and_record, register_browser_function
from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory, basic_run
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.utils.compatibility import prims

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class CrossAllXYDragMultiSingleQubitMultilevel(Experiment):
    """
    Class for running a single AllXY drag experiment on a single qubit with a multilevel system.
    """

    @log_and_record
    def run(self,
            dut,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            initial_lpb=None,
            N: int = 1,
            inv_alpha_start: float = None,
            inv_alpha_stop: float = None,
            num: int = 21
            ) -> None:
        """
        Runs the AllXY drag experiment.

        Parameters:
            dut (Any): The device under test.
            collection_name (str): The name of the collection.
            mprim_index (int): The index of the measurement primitive.
            initial_lpb (LogicalPrimitiveBlock): The initial pulse sequence.
            N (int): The number of repetitions for the All XY value.
            inv_alpha_start (float): The start value of the 1/alpha parameter.
            inv_alpha_stop (float): The stop value of the 1/alpha parameter.
            num (int): The number of points in the sweep.
        """

        c1 = dut.get_c1(collection_name)
        parameters = c1['X'].get_parameters()

        if 'alpha' not in parameters:
            raise RuntimeError(
                f'The pulse shape {parameters["shape"]} does not support DRAG')

        alpha = parameters['alpha']

        if inv_alpha_start is None:
            inv_alpha_start = 1 / alpha - 0.006
        if inv_alpha_stop is None:
            inv_alpha_stop = 1 / alpha + 0.006

        # Define the pulse sequence for the experiment.
        vz_pi = c1.z(np.pi)

        pulse_train_block = c1['X'] + vz_pi + \
            c1['Y'] + vz_pi + c1['X'] + c1['Y']

        # Add additional pulses based on the value of N.
        pulse_train = prims.SerialLPB([pulse_train_block] * N)

        lpb_tail = prims.SweepLPB([c1['Xp'], c1['Xm']])

        # Define a function to update alpha in the pulse sequence.
        def update_alpha(n): return c1.update_parameters(alpha=1 / n)

        # Create a sweeper for the alpha parameter.
        self.sweep_values = np.linspace(inv_alpha_start, inv_alpha_stop, num)
        swp = Sweeper(
            self.sweep_values,
            params=[
                SweepParametersSideEffectFactory.func(
                    update_alpha,
                    argument_name='n',
                    kwargs={})])

        swp = swp + Sweeper.from_sweep_lpb(lpb_tail)

        # meas = qubit.get_default_measurement_prim_int()  #
        # prims.ParallelLPB(m_prims)
        mp = dut.get_measurement_prim_intlist(mprim_index)

        lpb = pulse_train + lpb_tail + mp

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        self.mp = mp

        basic_run(lpb, swp, '<z>')

        self.result = np.squeeze(mp.result())

    def linear_fit(self):
        self.fit_xp = np.polyfit(self.sweep_values, self.result[:, 0], deg=1)
        self.fit_xm = np.polyfit(self.sweep_values, self.result[:, 1], deg=1)
        self.optimum = (self.fit_xp[0] - self.fit_xm[0]) / \
            (self.fit_xm[1] - self.fit_xp[1])

    @register_browser_function()
    def plot(self):
        self.linear_fit()
        plt.figure()
        plt.plot(self.sweep_values, self.result[:, 0], 'ro', alpha=0.5)
        plt.plot(
            self.sweep_values,
            self.fit_xp[0] *
            self.sweep_values +
            self.fit_xp[1],
            'r-')
        plt.plot(self.sweep_values, self.result[:, 1], 'bo', alpha=0.5)
        plt.plot(
            self.sweep_values,
            self.fit_xm[0] *
            self.sweep_values +
            self.fit_xm[1],
            'b-')
        plt.xlabel(u"DRAG coefficient")
        plt.ylabel(u"<z>")
        # plt.legend()
        plt.show()


class CrossAllXYDragMultiRunSingleQubitMultilevel(Experiment):
    """
    Class for running experiments to calibrate the alpha parameter (DRAG coefficient)
    by performing an all XY drag on a single qubit with a multilevel system.
    """

    @log_and_record
    def run(
            self,
            dut,
            collection_name: str = 'f01',
            initial_lpb: LogicalPrimitiveBlock = None,
            mprim_index: int = 0,
            update: bool = True,
            reset_alpha_before_calibration=True) -> None:
        """
        Calibrates the alpha parameter by performing an all XY drag.

        Parameters:
            dut (Any): The device under test.
            collection_name (str): The name of the collection.
            initial_lpb (LogicalPrimitiveBlock): The initial pulse sequence.
            mprim_index (int): The index of the measurement primitive.
            update (bool): Whether to update the alpha parameter in the DUT.
            reset_alpha_before_calibration (bool): Whether to reset the alpha parameter before calibration.
        """
        print(f'Calibrating alpha by all XY {collection_name}')

        self.alpha_calibration_results = {}

        # Retrieve the current alpha value from the device under test (DUT).
        alpha: float = dut.get_c1(collection_name)[
            'Xp'].primary_kwargs()['alpha']

        # Reset alpha before calibration if the flag is set.
        if reset_alpha_before_calibration:
            alpha = 1e9

        # Initialize bounds for alpha.
        alpha_lower_bound: Optional[float] = None
        alpha_higher_bound: Optional[float] = None

        # Perform up to 10 iterations to find a trustable value for alpha.
        for i in range(10):
            # Define the search interval around the current 1/alpha estimate.
            start_point = 1 / alpha - 0.006
            stop_point = 1 / alpha + 0.006

            # Create an AllXY experiment instance.
            allxy = CrossAllXYDragMultiSingleQubitMultilevel(
                collection_name=collection_name,
                mprim_index=mprim_index,
                initial_lpb=initial_lpb,
                N=1,
                inv_alpha_start=start_point,
                inv_alpha_stop=stop_point,
                num=21,
                dut=dut)

            key = (collection_name, mprim_index)

            # Store the calibration result.
            if key in self.alpha_calibration_results:
                self.alpha_calibration_results[key].append(allxy)
            else:
                self.alpha_calibration_results[key] = [allxy]

            # Update alpha estimate based on the inverse of the obtained value
            # from the experiment.
            alpha_0 = allxy.optimum

            print('Guessed alpha:', alpha_0)

            # Check if the new alpha value is within the trustable range.
            middle_point = (start_point + stop_point) / 2
            width = stop_point - start_point
            if np.abs(1 / alpha_0 - middle_point) < width / 3:
                # We have a trustable value.
                break

            # Update the bounds for alpha using binary search technique.
            alpha_current_guess = alpha_0
            if alpha_current_guess > alpha:
                if alpha_lower_bound is None or alpha > alpha_lower_bound:
                    alpha_lower_bound = alpha
            else:
                if alpha_higher_bound is None or alpha < alpha_higher_bound:
                    alpha_higher_bound = alpha

            print(
                'alpha_lower:',
                alpha_lower_bound,
                'alpha_higher:',
                alpha_higher_bound)

            # Update the alpha value for the next iteration.
            if alpha_higher_bound is None or alpha_lower_bound is None:
                alpha = alpha_0
            else:
                alpha = (alpha_lower_bound + alpha_higher_bound) / 2

        # Update the alpha parameter in the DUT if the update flag is set.
        if update:
            dut.get_c1(collection_name).update_parameters(alpha=alpha_0)


class DragPhaseCalibrationMultiQubitsMultilevel(Experiment):
    @log_and_record
    def run(self,
            duts,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            initial_lpb=None,
            N: int = 1,
            inv_alpha_start: float = None,
            inv_alpha_stop: float = None,
            num: int = 21
            ) -> None:
        """
        Runs the AllXY drag experiment.

        Parameters:
            dut (Any): The device under test.
            collection_name (str): The name of the collection.
            mprim_index (int): The index of the measurement primitive.
            initial_lpb (LogicalPrimitiveBlock): The initial pulse sequence.
            N (int): The number of repetitions for the All XY value.
            inv_alpha_start (float): The start value of the 1/alpha parameter.
            inv_alpha_stop (float): The stop value of the 1/alpha parameter.
            num (int): The number of points in the sweep.
        """
        pass
