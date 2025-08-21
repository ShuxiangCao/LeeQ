from leeq.core.primitives import *
from leeq.core.primitives.built_in.common import Delay
from leeq.core.primitives.built_in.sizzel_gate import SiZZelTwoQubitGateCollection
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockParallel as ParallelLPB
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial as SerialLPB
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep

SeriesLPB = SerialLPB


class SweepLPB:
    """
    For compatibility reasons
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and (
            isinstance(
                args[0],
                list) or isinstance(
                args[0],
                tuple)):
            return LogicalPrimitiveBlockSweep(children=args[0])
        else:
            return LogicalPrimitiveBlockSweep(children=args)


def build_CZ_stark_from_parameters(
        control_q,
        target_q,
        width,
        amp_control,
        amp_target,
        frequency,
        rise,
        zz_interaction_positive,
        iz_control=0,
        iz_target=0,
        phase_diff=0,
        echo=False,
        trunc=1.05,
):
    """
    Build a CZ Stark shift gate from parameters.

    Args:
        control_q (TransmonElement): The control qubit.
        target_q (TransmonElement): The target qubit.
        width (float): The width of the gate.
        amp_control (float): The amplitude of the control qubit.
        amp_target (float): The amplitude of the target qubit.
        frequency (float): The frequency of the gate.
        rise (float): The rise time of the gate.
        zz_interaction_positive (bool): Whether the interaction is positive. Determines the direction of the gate.
        iz_control (float): The z component of the control qubit.
        iz_target (float): The z component of the target qubit.
        phase_diff (float): The phase difference between the control and target qubits.
        echo (bool): Whether to use echo.
        trunc (float): The truncation factor.
    """
    lpb = SiZZelTwoQubitGateCollection(
        name='zz',
        dut_control=control_q,
        dut_target=target_q,
        parameters={
            'width': width,
            'amp_control': amp_control,
            'amp_target': amp_target,
            'freq': frequency,
            'rise': rise,
            'iz_control': iz_control,
            'iz_target': iz_target,
            'phase_diff': phase_diff,
            'echo': echo,
            'trunc': trunc,
            'zz_interaction_positive': zz_interaction_positive
        }
    )
    return lpb
