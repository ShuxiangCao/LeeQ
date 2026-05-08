from .common import *
from .ramsey import StarkRamseyMultilevel

class StarkZZShiftTwoQubitMultilevel(Experiment):
    """Class to compute ZZ Shift for Two Qubit Multilevel system."""

    @log_and_record
    def run(self,
            duts: List[TransmonElement],
            collection_name: str = 'f01',
            mprim_index: int = 0,
            start: float = 0.0,
            stop: float = 1,
            step: float = 0.005,
            set_offset: int = 10,
            stark_offset=50,
            disable_sub_plot: bool = False) -> None:
        """Run the ZZ Shift experiment.

        Parameters:
            duts: The DUTs (Device Under Test).
            collection_name: The name of the frequency collection (e.g., 'f01').
            mprim_index: The index of the measurement primitive.
            start: The start frequency for the sweep.
            stop: The stop frequency for the sweep.
            step: The step size for the frequency sweep.
            set_offset: The frequency offset.
            disable_sub_plot: Whether to disable subplots.
            hardware_stall: Whether to use hardware stall.

        Returns:
            None
        """
        # Ensure there are exactly 2 DUTs (Device Under Test)
        if len(duts) != 2:
            raise ValueError("AC Stark calibration requires exactly 2 DUTs.")

        plot_result_in_jupyter = setup().status().get_param("Plot_Result_In_Jupyter")

        if disable_sub_plot:
            setup().status().set_param("Plot_Result_In_Jupyter", False)

        c1q1 = duts[0].get_c1(collection_name)
        c1q2 = duts[1].get_c1(collection_name)

        # Q1 ramsey Q2 steady
        self.q1_ramsey_q2_ground = StarkRamseyMultilevel(
            duts[0],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=None,
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        self.q1_ramsey_q2_excited = StarkRamseyMultilevel(
            duts[0],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=c1q2['X'],
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        # Q2 ramsey Q1 steady
        self.q2_ramsey_q1_ground = StarkRamseyMultilevel(
            duts[1],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=None,
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        self.q2_ramsey_q1_excited = StarkRamseyMultilevel(
            duts[1],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=c1q1['X'],
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        self.zz = [
            self.q1_ramsey_q2_excited.frequency_guess
            - self.q1_ramsey_q2_ground.frequency_guess,
            self.q2_ramsey_q1_excited.frequency_guess
            - self.q2_ramsey_q1_ground.frequency_guess,
        ]

        self.zz_error = [
            self.q1_ramsey_q2_excited.error_bar
            - self.q1_ramsey_q2_ground.error_bar,
            self.q2_ramsey_q1_excited.error_bar
            - self.q2_ramsey_q1_ground.error_bar,
        ]

        setup().status().set_param("Plot_Result_In_Jupyter", plot_result_in_jupyter)
