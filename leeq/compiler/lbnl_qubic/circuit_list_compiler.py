import functools
import inspect
from typing import Union

import numpy as np

from leeq import setup
from leeq.compiler.compiler_base import LPBCompiler, MeasurementSequence
from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.compiler.utils.time_base import get_t_list
from leeq.core.context import ExperimentContext
from leeq.core.primitives.built_in.common import DelayPrimitive, PhaseShift
from leeq.core.primitives.logical_primitives import (
    LogicalPrimitiveCombinable,
    LogicalPrimitiveBlockParallel,
    LogicalPrimitiveBlockSerial,
    LogicalPrimitiveBlockSweep,
    MeasurementPrimitive,
    LogicalPrimitive,
)
from leeq.experiments.experiments import ExperimentManager
from leeq.utils import setup_logging

logger = setup_logging(__name__)


def compare_dicts(dict1, dict2, rtol=1e-05, atol=1e-08):
    """
    Compare two dictionaries element by element.

    Parameters:
    - dict1, dict2: dictionaries to compare
    - rtol, atol: relative and absolute tolerances for np.allclose

    Returns:
    True if the dictionaries are considered equal, otherwise False.
    """
    # Check if both inputs are dictionaries
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        raise ValueError("Both inputs should be dictionaries.")

    # Check if both dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Compare element by element
    for key, value1 in dict1.items():
        value2 = dict2[key]

        # If values are both numbers, use np.allclose for comparison
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            if not np.allclose(value1, value2, rtol=rtol, atol=atol):
                return False

        # If values are both dictionaries, compare them recursively
        elif isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dicts(value1, value2, rtol=rtol, atol=atol):
                return False

        # For other types, use simple equality comparison
        else:
            if value1 != value2:
                return False

    # If loop completes without returning False, dictionaries are equal
    return True


class QubiCCircuitListLPBCompiler(LPBCompiler):
    """
    The QubiCCircuitListLPBCompiler class defines a compiler that is used to compile the logical primitive block to
    instructions that going to be passed to the compiler. It aims to translate the LPB tree structure into the QubiC
    circuit list structure.

    QubiC is an open source quantum control system developed by AQT at LBNL and UC Berkeley. Please refer to the
    QubiC project for more information: https://gitlab.com/LBL-QubiC
    """

    def __init__(self, leeq_channel_to_qubic_channel: dict[str, str]):
        """
        Initialize the QubiCCircuitListLPBCompiler class.

        Parameters:
            leeq_channel_to_qubic_channel (Dict[str, str]): The mapping from the LeeQ channel name to the QubiC
                                                            channel name (the `dest`).
        """
        super().__init__("QubiCCircuitListLPBCompiler")
        self._circuit_list = []
        self._phase_shift = {}
        self._current_context = None
        self._leeq_channel_to_qubic_channel = leeq_channel_to_qubic_channel
        self._qubic_channel_to_lpb_uuid = {}

        # Indicates if the waveform shape has been changed. If not we skip uploading
        # the waveform to board to save time. Here we only look at sweep LPB. If the selected
        # lpb has changed, the waveform is dirty. Otherwise, it kept clean.
        self._envelope_dirty = False
        # Indicates if the waveform shape has been changed. If not we can skip uploading
        # the frequency to board to save time. It is done by checking the parameters for each lpbs.
        self._frequency_dirty = False
        # Indicates if the command (lpb or phase etc) has been changed.
        # If not we can omit uploading the waveform to board to save time.
        self._command_dirty = False

        self._lpb_uuid_to_parameter_last_compiled = {}
        self._sweep_lpb_uuid_to_last_selection = {}
        self._compiled_lpb_uuid = []

    def clear(self):
        """
        Reset the compiler
        """
        self._envelope_dirty = False
        self._frequency_dirty = False
        self._command_dirty = False
        self._circuit_list = []
        self._phase_shift = {}
        self._current_context = None
        self._qubic_channel_to_lpb_uuid = {}

        self._lpb_uuid_to_parameter_last_compiled = {}
        self._sweep_lpb_uuid_to_last_selection = {}
        self._compiled_lpb_uuid = []

        super().clear()

    @staticmethod
    def _get_envelope_function(pulse_shape_name: str):
        """
        Get the pulse shape function with the given name.

        Parameters:
            pulse_shape_name (str): The name of the pulse shape.

        Returns:
            callable: evaluated pulse shape function.

        """

        env_func = PulseShapeFactory().get_pulse_shape_function(pulse_shape_name)

        @functools.wraps(env_func)
        def func(dt, **kwargs):
            """
            Evaluate the pulse shape function with the given parameters.
            Adpat the interface between qubic defined functions and leek defined functions.

            Parameters:
                dt (float): The sampling rate of the pulse shape. In Msps unit.

            Returns:
                Tuple[np.ndarray, np.ndarray]: The time list and the pulse shape envelope.
            """

            # amp will be modified by the qubic system, so we always pass amp = 1
            kwargs = kwargs.copy()
            kwargs['amp'] = 1

            sampling_rate = 1 / dt / 1e6  # In Msps unit
            t = get_t_list(sampling_rate=sampling_rate, width=kwargs['width'])
            env = env_func(sampling_rate=sampling_rate, **kwargs)

            return t, env

        return func

    def compile_lpb(self, context: ExperimentContext,
                    lpb: LogicalPrimitiveCombinable):
        """
        Compile the logical primitive block to instructions that going to be passed to the setup.
        The compiled instructions are a list of dictionary, follows the QubiC circuit definition.

        Three different components are dealt with in this compiler. See the dispatched methods for details.

        1. The atomic logical primitive (Drive pulses, Delay etc.): They have direct mapping to the QubiC circuit
            definition, basically rearranging the dictionary content.

        2. The measurement logical primitive: QubiC requires to define the demodulation of the pulse, therefore we need
            to compile the measurement primitive into a drive pulse, a demodulation pulse.

        3. Virtual Z gate from phase shifting: QubiC naturally supports phase shifting, however to make LeeQ gets more
            control, we track the phase shifting within the compiler, and compile phase shifted pulses to QubiC.

        4. The scheduling: While LeeQ uses LPB to define a tree structure for scheduling, QubiC uses a list of circuit
            and specifying barriers for scheduling. Therefore, we need to compile the LPB tree structure into a list of
            circuit, and insert barriers at Serial and Parallel LPBs, for all channels. Currently the parallel blocks
            do not support running pulses on the same channels, as it may be difficult to find the right scheduling.
            TODO: Implement the parallel blocks on the same channel.

        5. The delay primitive: QubiC needs to provide delay scope, while LeeQ uses the tree structure to infer the
            behavior. Here a compromise is made: The delay primitive is only allowed to be in a Serial LPB, and the
            delay scope is the union of the former child. When the delay primitive is in a Parallel LPB, an error will
            be raised. In principle we can calculate the width of the whole sequence and implement delay primitive, but
            it is very rarely used anyway.
            TODO: Implement the delay primitive in Parallel LPB.

        6. TODO: The branching: Not implemented yet, need to figure how what is the best way to implement this.

        The compiled circuits will be stored in the context.instructions and passes to the setup and engine.
        """
        self._current_context = context
        self._qubic_channel_to_lpb_uuid = {}
        self._circuit_list = []
        self._phase_shift = {}

        # Keep track of the compiled UUID, and remove untouched UUID from the dirty tracking data structures
        self._compiled_lpb_uuid = []
        self._command_dirty = False
        self._frequency_dirty = False
        self._envelope_dirty = False

        circuit_list, scope = self._compile_lpb(lpb)
        context.instructions = {
            'circuits': circuit_list,
            "qubic_channel_to_lpb_uuid": self._qubic_channel_to_lpb_uuid,
            "dirtiness": {
                'envelope': self._envelope_dirty,
                'command': self._command_dirty,
                'frequency': self._frequency_dirty,
            }
        }

        # Now look at the compiled UUID, and remove unseen UUID from the dirty tracking data structures
        for lpb_id in list(self._sweep_lpb_uuid_to_last_selection.keys()):
            if lpb_id not in self._compiled_lpb_uuid:
                del self._sweep_lpb_uuid_to_last_selection[lpb_id]

        for lpb_id in list(self._lpb_uuid_to_parameter_last_compiled.keys()):
            if lpb_id not in self._compiled_lpb_uuid:
                del self._lpb_uuid_to_parameter_last_compiled[lpb_id]

        self._current_context = None
        return context

    @functools.singledispatchmethod
    def _compile_lpb(self, lpb):
        """
        Compile the logical primitive block to instructions.
        """
        msg = "The compiler does not support the logical primitive block type " + \
              str(type(lpb))
        logger.error(msg)
        raise NotImplementedError(msg)

    def _check_parameter_if_dirty(self, lpb: LogicalPrimitive):
        """
        Check if the parameter has been updated and mark if its dirty to the class

        Parameters:
            lpb (LogicalPrimitive): a lpb to be checked.
        """

        # Track the UUID of the lpbs, if they are not seen we remove them from the tracking data structure
        self._compiled_lpb_uuid.append(lpb.uuid)

        _parameter_diff = lambda x: x in parameters and \
                                    parameters[x] != self._lpb_uuid_to_parameter_last_compiled[lpb.uuid][x]

        parameters = lpb.get_parameters()
        lpb_id = lpb.uuid

        if lpb_id not in self._lpb_uuid_to_parameter_last_compiled:
            # New primitive, make it dirty
            self._command_dirty = True
            self._envelope_dirty = True
            self._frequency_dirty = True
        else:
            if parameters == self._lpb_uuid_to_parameter_last_compiled[lpb_id]:
                # Passed check
                return

            last_compiled_value = self._lpb_uuid_to_parameter_last_compiled[lpb_id].copy()

            if _parameter_diff('freq'):
                self._frequency_dirty = True
                del parameters['freq']
                del last_compiled_value['freq']

            if _parameter_diff('shape'):
                self._envelope_dirty = True
                del parameters['shape']
                del last_compiled_value['shape']

            if _parameter_diff('amp'):
                self._command_dirty = True
                del parameters['amp']
                del last_compiled_value['amp']

            if _parameter_diff('phase'):
                self._command_dirty = True
                del parameters['phase']
                del last_compiled_value['phase']

            if parameters != last_compiled_value:
                # Something else has changed as well, not sure if its shape or command, then mark them both dirty
                self._command_dirty = True
                self._envelope_dirty = True

        self._lpb_uuid_to_parameter_last_compiled[lpb_id] = lpb.get_parameters()

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitive):
        """
        Compile the atomic logical primitives to QubiC definitions.

        The atomic logical primitives are the drive pulses.
        The QubiC circuit definition is a dictionary, with the following keys:

        Example:
        ```
        {
            'name': 'pulse',
            'phase': 0,
            'freq': 4944383311, # In Hz
            'amp': 0.334704954261188,
            'twidth': 2.4e-08, # In seconds
            'env': {
                'env_func': 'cos_edge_square',
                'paradict': {'ramp_fraction': 0.25}},
            'dest': 'Q0.qdrv' # The channel name
        }
        ```
        """

        self._check_parameter_if_dirty(lpb)

        primitive_scope = self._leeq_channel_to_qubic_channel[lpb.channel]
        qubic_dest = primitive_scope + ".qdrv"

        # If there is no setup set, we don't update the parameters
        exp_manager = ExperimentManager()
        if exp_manager.get_default_setup() is None:
            parameters = lpb.get_parameters()
        else:
            parameters = exp_manager.status().get_modified_lpb_parameters_from_channel_callback(
                channel=lpb.channel, parameters=lpb.get_parameters()
            )

        phase_shift = 0

        if (
                "phase" in parameters
                and lpb.channel in self._phase_shift
                and "transition_name" in parameters
                and parameters["transition_name"] in self._phase_shift[lpb.channel]
        ):
            phase_shift += self._phase_shift[lpb.channel][lpb.transition_name]

        # Here we can't put all the parameters through, as they will be used as hashes to determine if the
        # waveform has changed. We only put the parameters that are used in the waveform generation.
        env_func = self._get_envelope_function(parameters['shape'])
        required_parameters = inspect.signature(env_func).parameters
        parameters_keeping = {key: parameters[key] for key in required_parameters if (key in parameters) and \
                              key not in ['amp', 'freq', 'phase']}

        env = {"env_func": env_func, "paradict": parameters_keeping}

        qubic_pulse_dict = {
            "name": "pulse",
            "phase": phase_shift + parameters["phase"],
            "freq": int(parameters['freq'] * 1e6),  # In Hz
            "amp": parameters['amp'],
            "twidth": parameters['width'] / 1e6,  # In seconds
            "env": env,
            "dest": qubic_dest,  # The channel name
        }

        return [qubic_pulse_dict], {qubic_dest}

    @_compile_lpb.register
    def _(self, lpb: MeasurementPrimitive):
        """
        Compile the logical primitive block to instructions.

        For measurement primitive we need to compile the measurement pulse and the demodulation pulse.

        Example from QubiC:
        ```
        [
            {
                "freq": "Q1.readfreq",
                "phase": 0.0,
                "dest": "Q1.rdrv",
                "twidth": 2e-06,
                "t0": 0.0,
                "amp": 0.018964980535141,
                "env": [
                    {
                        "env_func": "cos_edge_square",
                        "paradict": {
                            "ramp_fraction": 0.25,
                            "twidth": 2e-06
                        }
                    }
                ]
            },
            {
                "freq": "Q1.readfreq",
                "phase": 0,
                "dest": "Q1.rdlo",
                "twidth": 2e-06,
                "t0": 6e-07,
                "amp": 1.0,
                "env": [
                    {
                        "env_func": "square",
                        "paradict": {
                            "phase": 0.0,
                            "amplitude": 1.0,
                            "twidth": 2e-06
                        }
                    }
                ]
            }
        ]
        ```

        Note that frequency can accept numbers as well as strings, and the string is the name of the frequency variable.
        """

        assert isinstance(lpb, MeasurementPrimitive)

        self._check_parameter_if_dirty(lpb)

        primitive_scope = self._leeq_channel_to_qubic_channel[lpb.channel]
        # Compile the measurement pulse

        # If there is no setup set, we don't update the parameters
        exp_manager = ExperimentManager()
        if exp_manager.get_default_setup() is None:
            modified_parameters = lpb.get_parameters()
        else:
            modified_parameters = exp_manager.status().get_modified_lpb_parameters_from_channel_callback(
                channel=lpb.channel, parameters=lpb.get_parameters()
            )

        # Here we can't put all the parameters through, as they will be used as hashes to determine if the
        # waveform has changed. We only put the parameters that are used in the waveform generation.
        env_func = self._get_envelope_function(modified_parameters['shape'])
        required_parameters = inspect.signature(env_func).parameters
        parameters_keeping = {key: modified_parameters[key] for key in required_parameters
                              if (key in modified_parameters) and \
                              key not in ['amp', 'freq', 'phase']}

        env = {"env_func": env_func, "paradict": parameters_keeping}

        drive_pulse = {
            "name": 'pulse',
            "freq": (modified_parameters['freq'] * 1e6),  # In Hz,
            "phase": modified_parameters['phase'],
            "dest": primitive_scope + ".rdrv",
            "twidth": modified_parameters['width'] / 1e6,  # In seconds
            "amp": modified_parameters['amp'],
            "env": [env],
        }

        # A delay is introduced between the start of the drive and start of the measurement.
        # First it takes roughly 100ns for the signal to arrive the resonator in the fridge and
        # comback. Also the delay prevents the on-board crosstalk of the qubit drive signal being picked up
        delay_between_drive_and_measure = {
            'name': 'delay',
            't': 100e-9,
            'scope': [primitive_scope]
        }

        # TODO: Give attribute t, in FPGA clocks (FPGAConfig), take attribute to quantize time stamps
        # go to the run_compiler_stage function, reimplement it with a different pass (remove schedule)
        # specifiy parameter t, then do not require delay or barrier anymore.

        demodulate_pulse = {
            "name": 'pulse',
            "freq": (modified_parameters['freq'] * 1e6),  # In Hz,
            "phase": modified_parameters['phase'],
            "dest": primitive_scope + ".rdlo",
            "twidth": modified_parameters['width'] / 1e6,  # In seconds
            "amp": 1,  # Always use a full amp for demodulation
            "env": [
                {
                    "env_func": "square",
                    # Here we use square pulse for demodulation, which means no
                    # window function is applied
                    "paradict": {"phase": 0.0, "amplitude": 1.0, "twidth": modified_parameters['width'] / 1e6},
                }
            ],
        }

        if primitive_scope in self._qubic_channel_to_lpb_uuid and self._qubic_channel_to_lpb_uuid[
            primitive_scope] != lpb.uuid:
            msg = "Two measurement primitives exists for a same channel, which is not supported."
            logger.error(msg)
            raise NotImplementedError(msg)

        self._qubic_channel_to_lpb_uuid[primitive_scope] = lpb.uuid

        return [drive_pulse, delay_between_drive_and_measure, demodulate_pulse], {primitive_scope + ".rdrv"}

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSerial):
        """
        Compile the logical primitive block to instructions.

        First compile the dict for each child, then find the scope of each child. Insert barriers between the children,
        where the barrier scope should be the union of the former child and the latter child.

        Note that we do not need barriers at the beginning and the end of the block, as the other block will take care
        of it.
        """

        # If it's an empty block, return empty list
        if lpb.children is None or not lpb.children:
            return [], set()

        # Compile the children
        child_circuits = []
        child_scopes = []
        for child in lpb.children:
            child_circuit, child_scope = self._compile_lpb(child)
            if len(child_circuit) == 0:
                continue
            child_circuits.append(child_circuit)
            child_scopes.append(child_scope)

        # Insert barriers
        compiled_circuit = []
        block_scope = set.union(*child_scopes)

        # The first child

        # If the first child is a delay primitive, we need to modify
        # the delay primitive to have the entire scope of the block.
        if len(
                child_circuits[0]) == 1 and child_circuits[0][0]["name"] == "delay":
            child_circuits[0][0]["scope"] = list(set(x.split('.')[0] for x in block_scope))
            child_scopes[0] = block_scope

        compiled_circuit.extend(child_circuits[0])

        # The middle children
        for i in range(1, len(child_circuits)):
            # If the child is a delay primitive, we need to modify its scope to be the previous child's scope.
            # For this operation we do not need a barrier.
            if len(
                    child_circuits[i]) == 1 and child_circuits[i][0]["name"] == "delay":
                child_circuits[i][0]["scope"] = list(set(x.split('.')[0] for x in child_scopes[i - 1]))
            else:
                # If we are only dealing with one channel, and the previous and current child are on the same channel,
                # we do not need a barrier.
                union_scope = child_scopes[i - 1].union(child_scopes[i])
                if len(block_scope) > 1 and len(union_scope) > 1:
                    compiled_circuit.append(
                        {"name": "barrier", "scope": list(set(x.split('.')[0] for x in union_scope))}
                    )

            compiled_circuit.extend(child_circuits[i])

        return compiled_circuit, block_scope

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockParallel):
        """
        Compile the logical primitive block to instructions.

        For parallel blocks, each child is compiled separately, and the scope of the parallel block is the union of
        the scope of the children. Yet we do not support running pulses on the same channel in parallel blocks.
        """

        # If it's an empty block, return empty list
        if lpb.children is None or not lpb.children:
            return [], set()

        # Compile the children
        child_circuits = []
        child_scopes = []
        for child in lpb.children:
            child_circuit, child_scope = self._compile_lpb(child)
            if len(child_circuit) == 0:
                continue
            child_circuits.append(child_circuit)
            child_scopes.append(child_scope)

        # Make sure the children are not running on the same channel
        block_scope = []
        for i in range(len(child_scopes)):
            # Check if the child is a lonely delay primitive. If so we raise an
            # error.
            if len(
                    child_circuits[i]) == 1 and child_circuits[i][0]["name"] == "delay":
                msg = (
                    "Parallel blocks do not support delay primitives (yet). Try to attach the delay primitive to a "
                    "serial block. ")
                logger.error(msg)
                raise NotImplementedError(msg)

            block_scope.extend(child_scopes[i])

        assert len(block_scope) == len(set(block_scope)), (
            "Parallel blocks do not support "
            "running pulses on the same channel (yet)."
        )

        # Assemble the circuit
        compiled_circuit = sum(child_circuits, [])
        block_scope = set(block_scope)
        if len(block_scope) > 1:
            compiled_circuit.append(
                {"name": "barrier", "scope": list(set(x.split('.')[0] for x in block_scope))})

        return compiled_circuit, block_scope

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSweep):
        """
        Compile the logical primitive block to instructions.
        """

        if lpb.uuid not in self._sweep_lpb_uuid_to_last_selection:
            self._command_dirty = True
        else:
            if lpb.selected != self._sweep_lpb_uuid_to_last_selection[lpb.uuid]:
                self._command_dirty = True

        self._sweep_lpb_uuid_to_last_selection[lpb.uuid] = lpb.selected

        return self._compile_lpb(lpb.current_lpb)

    @_compile_lpb.register
    def _(self, lpb: DelayPrimitive):
        """
        Compile the logical primitive block to instructions.

        Here we only return a delay dict with no scope, and the scope needs to be modified by the parent block compling
        process.
        """

        self._check_parameter_if_dirty(lpb)

        return [{"name": "delay", "t": lpb.get_delay_time() /
                                       1e6}], set()  # In seconds

    @_compile_lpb.register
    def _(self, lpb: PhaseShift):

        self._check_parameter_if_dirty(lpb)

        # If there is no setup set, we don't update the parameters
        exp_manager = ExperimentManager()
        if exp_manager.get_default_setup() is None:
            parameters = lpb.get_parameters()
        else:
            parameters = exp_manager.status().get_modified_lpb_parameters_from_channel_callback(
                channel=lpb.channel, parameters=lpb.get_parameters()
            )

        if lpb.channel not in self._phase_shift:
            self._phase_shift[lpb.channel] = {}

        for k, m in parameters["transition_multiplier"].items():
            if k not in self._phase_shift[lpb.channel]:
                self._phase_shift[lpb.channel][k] = m * \
                                                    parameters["phase_shift"]
            else:
                self._phase_shift[lpb.channel][k] += m * \
                                                     parameters["phase_shift"]

        return [], set()
