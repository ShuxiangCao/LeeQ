import functools
from typing import Union

from leeq.compiler.compiler_base import LPBCompiler
from leeq.core.context import ExperimentContext
from leeq.core.primitives.built_in.common import DelayPrimitive, PhaseShift
from leeq.core.primitives.logical_primitives import LogicalPrimitiveCombinable, LogicalPrimitiveBlockParallel, \
    LogicalPrimitiveBlockSerial, LogicalPrimitiveBlockSweep, MeasurementPrimitive, LogicalPrimitive
from leeq.utils import setup_logging

logger = setup_logging(__name__)


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

    def compile_lpb(self, context: ExperimentContext, lpb: LogicalPrimitiveCombinable):
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
        circuit_list, scope = self._compile_lpb(lpb)
        context.instructions = circuit_list
        self._current_context = None
        return context

    @functools.singledispatchmethod
    def _compile_lpb(self, lpb):
        """
        Compile the logical primitive block to instructions.
        """
        msg = "The compiler does not support the logical primitive block type " + str(type(lpb))
        logger.error(msg)
        raise NotImplementedError(msg)

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

        primitive_scope = self._leeq_channel_to_qubic_channel[lpb.channel]
        qubic_dest = primitive_scope + '.qdrv'

        parameters = lpb.get_parameters()

        phase_shift = 0

        if 'phase' in parameters and lpb.channel in self._phase_shift and \
                parameters['transition_name'] in self._phase_shift[lpb.channel]:
            phase_shift += self._phase_shift[lpb.channel][lpb.transition_name]

        env = {
            'env_func': lpb.shape,
            'paradict': lpb.get_parameters()
        }

        qubic_pulse_dict = {
            'name': 'pulse',
            'phase': phase_shift + parameters['phase'],
            'freq': int(lpb.freq * 1e6),  # In Hz
            'amp': lpb.amp,
            'twidth': lpb.width / 1e6,  # In seconds
            'env': env,
            'dest': qubic_dest  # The channel name
        }

        return [qubic_pulse_dict], {primitive_scope}

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

        primitive_scope = self._leeq_channel_to_qubic_channel[lpb.channel]
        # Compile the measurement pulse

        drive_pulse = {
            "freq": lpb.freq,
            "phase": lpb.phase,
            "dest": primitive_scope + ".rdrv",
            "twidth": lpb.width / 1e6,
            "t0": 0.0,  # TODO: make it a parameter
            "amp": lpb.amp,
            "env": [
                {
                    "env_func": lpb.shape,
                    "paradict": lpb.get_parameters()
                }
            ]
        }

        demodulate_pulse = {
            "freq": lpb.freq,
            "phase": lpb.phase,
            "dest": primitive_scope + ".rdlo",
            "twidth": lpb.width / 1e6,
            "t0": 6e-07,  # TODO: make it a parameter
            "amp": 1,  # Always use full amp for demodulation
            "env": [
                {
                    "env_func": "square",
                    # Here we use square pulse for demodulation, which means no window function is applied
                    "paradict": {
                        "phase": 0.0,
                        "amplitude": 1.0,
                        "twidth": lpb.width
                    }
                }
            ]
        }

        return [drive_pulse, demodulate_pulse], set(primitive_scope, )

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
        if len(child_circuits[0]) == 1 and child_circuits[0][0]['name'] == 'delay':
            child_circuits[0][0]['scope'] = list(block_scope)
            child_scopes[0] = block_scope

        compiled_circuit.extend(child_circuits[0])

        # The middle children
        for i in range(1, len(child_circuits)):

            # If the child is a delay primitive, we need to modify its scope to be the previous child's scope.
            # For this operation we do not need a barrier.
            if len(child_circuits[i]) == 1 and child_circuits[i][0]['name'] == 'delay':
                child_circuits[i][0]['scope'] = list(child_scopes[i - 1])
            else:
                # If we are only dealing with one channel, and the previous and current child are on the same channel,
                # we do not need a barrier.
                union_scope = child_scopes[i - 1].union(child_scopes[i])
                if len(block_scope) > 1 and len(union_scope) > 1:
                    compiled_circuit.append({
                        'name': 'barrier',
                        'scope': list(union_scope)
                    })

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

            # Check if the child is a lonely delay primitive. If so we raise an error.
            if len(child_circuits[i]) == 1 and child_circuits[i][0]['name'] == 'delay':
                msg = "Parallel blocks do not support delay primitives (yet). Try to attach the delay primitive to a " \
                      "serial block. "
                logger.error(msg)
                raise NotImplementedError(msg)

            block_scope.extend(child_scopes[i])

        assert len(block_scope) == len(set(block_scope)), ("Parallel blocks do not support "
                                                           "running pulses on the same channel (yet).")

        # Assemble the circuit
        compiled_circuit = sum(child_circuits, [])
        block_scope = set(block_scope)
        compiled_circuit.append(
            {
                'name': 'barrier',
                'scope': list(block_scope)
            }
        )

        return compiled_circuit, block_scope

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSweep):
        """
        Compile the logical primitive block to instructions.
        """
        return self._compile_lpb(lpb.current_lpb)

    @_compile_lpb.register
    def _(self, lpb: DelayPrimitive):
        """
        Compile the logical primitive block to instructions.

        Here we only return a delay dict with no scope, and the scope needs to be modified by the parent block compling
        process.
        """

        return {
            'name': 'delay',
            't': lpb.get_delay_time() / 1e6  # In seconds
        }, set()

    @_compile_lpb.register
    def _(self, lpb: PhaseShift):
        parameters = lpb.get_parameters()
        if lpb.channel not in self._phase_shift:
            self._phase_shift[lpb.channel] = {}

        for k, m in parameters['transition_multiplier'].items():
            if k not in self._phase_shift[lpb.channel]:
                self._phase_shift[lpb.channel][k] = m * parameters['phase_shift']
            else:
                self._phase_shift[lpb.channel][k] += m * parameters['phase_shift']

        return [], set()
