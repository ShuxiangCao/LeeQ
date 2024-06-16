import uuid
import numpy as np

from leeq import LogicalPrimitiveFactory, LogicalPrimitive
from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.core.primitives.built_in.compatibility import PulseArgsUpdatable
from leeq.core.primitives.collections import LogicalPrimitiveCollection

from leeq.core.primitives.built_in.simple_drive import SimpleDrive


class SiZZelTwoQubitGateCollection(LogicalPrimitiveCollection):
    _default_parameters = {
        'drive': 'SiZZelDrive',
        'iz_control': 0,
        'iz_target': 0,
        'echo': False,
        'phase_diff': 0,
        'shape': 'soft_square',
        'rise': 0.01,
        'trunc': 1.0,
        'width': 0.1,
    }

    def validate_parameters(self):
        """
        Validate the parameters of the logical primitive collection.
        """

        assert 'freq' in self._parameters, "The frequency is not found in the parameters."
        assert 'amp_control' in self._parameters, "The control amplitude is not found in the parameters."
        assert 'amp_target' in self._parameters, "The target amplitude is not found in the parameters."
        assert 'iz_control' in self._parameters, "The control IZ strength is not found in the parameters."
        assert 'iz_target' in self._parameters, "The target IZ strength is not found in the parameters."
        assert 'phase_diff' in self._parameters, "The phase difference is not found in the parameters."
        assert 'echo' in self._parameters, "The echo configuration is not found in the parameters."
        assert 'width' in self._parameters, "The width is not found in the parameters."

    def update_parameters(self, **kwargs):
        """
        Update the parameters of the logical primitive collection.
        """
        super().update_parameters(**kwargs)

        self['stark_drive_control'].update_parameters(**{
            "freq": self._parameters['freq'],
            "amp": self._parameters['amp_control'],
            "shape": self._parameters['shape'],
            "width": self._parameters['width'],
            "rise": self._parameters['rise'],
            "trunc": self._parameters['trunc'],
            "phase": 0,
        }
        )
        self['stark_drive_target'].update_parameters(**{
            "freq": self._parameters['freq'],
            "amp": self._parameters['amp_target'],
            "shape": self._parameters['shape'],
            "phase": self._parameters['phase_diff'],
            "width": self._parameters['width'],
            "rise": self._parameters['rise'],
            "trunc": self._parameters['trunc'],
        })

    def __init__(self, name, parameters, dut_control, dut_target):
        """
        Initialize the SiZZel gate collection.

        Parameters:
            name (str): The name of the SiZZel gate collection.
            parameters (dict): The parameters of the SiZZel gate collection.
            dut_control (TransmonElement): The
            dut_target (TransmonElement): The
        """

        parameters_default = {**self._default_parameters}
        parameters_default.update(parameters)
        parameters = parameters_default

        super().__init__(name, parameters)

        self.dut_control = dut_control
        self.dut_target = dut_target
        self.c1_control = dut_control.get_default_c1()
        self.c1_target = dut_target.get_default_c1()

        self.channels = (
            dut_control.get_default_c1()['X'].primary_channel(),
            dut_target.get_default_c1()['X'].primary_channel()
        )

        factory = LogicalPrimitiveFactory()
        factory.register_collection_template(SimpleDrive)

        primitives_params = {
            'stark_drive_control': {
                "freq": self._parameters['freq'],
                "channel": self.dut_control.get_default_c1()['X'].primary_channel(),
                "shape": self._parameters['shape'],
                "amp": self._parameters['amp_control'],
                "phase": 0,
                "width": self._parameters['width'],
                "rise": self._parameters['rise'],
            },
            'stark_drive_target': {
                "freq": self._parameters['freq'],
                "channel": self.dut_target.get_default_c1()['X'].primary_channel(),
                "shape": self._parameters['shape'],
                "amp": self._parameters['amp_target'],
                "phase": self._parameters['phase_diff'],
                "width": self._parameters['width'],
                "rise": self._parameters['rise'],
            },
        }

        factory = LogicalPrimitiveFactory()
        for primitive_name, primitive_parameters in primitives_params.items():
            self._primitives[primitive_name] = factory(
                name=self._name + "." + primitive_name,
                class_name='SimpleDrive',
                parameters=primitive_parameters,
            )

    def get_stark_drive_pulses(self):
        """
        Get the stark drive pulses.
        """
        return self['stark_drive_control'] * self['stark_drive_target']

    def get_z_cancellation_pulse(self):
        """
        Get the z cancellation pulse. The z cancellation pulse is a pulse that cancels the iz rotation of both qubits.
        """
        width = self.get_parameters()['width']

        z_control_cancel = self.iz_control
        z_target_cancel = self.iz_target

        return self.c1_control.z(z_control_cancel) * \
            self.c1_target.z(z_target_cancel)

    def get_z_canceled_cs_pulse(self):
        """
        Get the z canceled SiZZel pulse. The z canceled SiZZel pulse is a lpb that first do stark shift gate and then
        apply virtual Z gate to cancel the iz rotation of both qubits.
        """

        full_pulse = self.get_stark_drive_pulses()

        if self.echo:

            flip_both = self.c1_control['X'] * self.c1_target['X']
            return full_pulse + flip_both + full_pulse+ flip_both + self.get_z_cancellation_pulse()

        return full_pulse + self.get_z_cancellation_pulse()

    def get_zzm_pi_over_4(self):
        """
        Get the zzm gate with pi/4 rotation.
        """

        if self.echo:
            flip_both = self.c1_control['X'] * self.c1_target['X']
            full_pulse = self.get_stark_drive_pulses()
            lpb = full_pulse + flip_both + full_pulse + flip_both + \
                self.get_z_cancellation_pulse() + self.get_z_cancellation_pulse()
        else:
            full_pulse = self.get_z_canceled_cs_pulse()
            lpb = full_pulse + full_pulse

        return lpb

    def get_zzm(self):
        """
        Get the zzm gate with pi/2 rotation.
        """

        # Each pulse do a np.pi/8 rotation , each gate requires pi/2 rotation

        if self.echo:
            flip_both = self.c1_control['X'] * self.c1_target['X']
            full_pulse = self.get_stark_drive_pulses()
            lpb = full_pulse + full_pulse + flip_both + full_pulse + full_pulse + flip_both + self.get_z_cancellation_pulse() \
                + self.get_z_cancellation_pulse() + self.get_z_cancellation_pulse() + self.get_z_cancellation_pulse()

        else:

            full_pulse = self.get_z_canceled_cs_pulse()
            lpb = full_pulse + full_pulse + full_pulse + full_pulse

        return lpb

    def get_zzp(self):
        """
        Get the zzp gate with pi/2 rotation.
        """
        return self.get_zzm() + self.c1_control.z(np.pi) * self.c1_target.z(np.pi)

    def get_cz(self):
        """
        Get the CZ gate.
        """
        return self.get_zzm() + self.c1_control.z(-np.pi / 2) * \
            self.c1_target.z(-np.pi / 2)

    def get_zxp(self, additional_echo=None, empty_pulse=False):
        """
        Get the ZXP gate.
        """

        return self.c1_target.hadamard() + self.get_zzp() + self.c1_target.hadamard()

    def get_zxm(self, additional_echo=None, empty_pulse=False):
        """
        Get the ZXM gate.
        """
        return self.c1_target.hadamard() + self.get_zzm() + self.c1_target.hadamard()

    def get_cnot(self, additional_echo=None):
        """
        Get the CNOT gate.
        """
        return self.c1_target.hadamard() + self.get_cz() + self.c1_target.hadamard()

    def get_cnot_like(self, additional_echo=None):
        """
        Get the CNOT-like gate. This is for randomized benchmarking.
        """

        return self.get_zxm(additional_echo)

    def get_iswap_like(self, additional_echo=None):
        """
        Get the iSWAP-like gate. This is for randomized benchmarking.
        """
        control_c1, target_c1 = self.c1_control, self.c1_target

        return self.get_zxm(additional_echo) + (
            control_c1['Ym'] * target_c1['Ym']) + self.get_zxm(additional_echo)

    def get_swap_like(self, additional_echo=None):
        """
        Get the SWAP-like gate. This is for randomized benchmarking.
        """
        control_c1, target_c1 = self.c1_control, self.c1_target

        return self.get_zxm(additional_echo) + (
            control_c1['Ym'] * target_c1['Ym']) + self.get_zxm(additional_echo) + (
            control_c1['Xp'] * (target_c1['Xp'] + target_c1['Ym'])) + self.get_zxm(additional_echo)

    def get_clifford(
            self,
            i,
            control_c1=None,
            target_c1=None,
            ignore_identity=False,
            additional_echo=None):
        """
        Get the Clifford gate by index, for randomized benchmarking.
        """

        control_c1, target_c1 = self.c1_control, self.c1_target

        from leeq.theory.cliffords.two_qubit_cliffords import get_c2_info

        info = get_c2_info(i)
        if info[0] == 0:
            return control_c1.get_clifford(
                info[1], ignore_identity) * target_c1.get_clifford(info[2], ignore_identity)
        elif info[0] == 1:
            return control_c1.get_clifford(
                info[1],
                ignore_identity) * target_c1.get_clifford(
                info[2],
                ignore_identity) + self.get_cnot_like(additional_echo) + control_c1.get_clifford(
                info[3],
                ignore_identity) * target_c1.get_clifford(
                info[4],
                ignore_identity)
        elif info[0] == 2:
            return control_c1.get_clifford(
                info[1],
                ignore_identity) * target_c1.get_clifford(
                info[2],
                ignore_identity) + self.get_iswap_like(additional_echo) + control_c1.get_clifford(
                info[3],
                ignore_identity) * target_c1.get_clifford(
                info[4],
                ignore_identity)
        else:
            return control_c1.get_clifford(
                info[1], ignore_identity) * target_c1.get_clifford(
                info[2], ignore_identity) + self.get_swap_like(additional_echo)

    def get_cphase(self):
        """
        Get the CPhase gate.
        """
        return self.get_zzp() + self.c1_control.z(np.pi / 2) * self.c1_target.z(np.pi)

    def random_clifford(self, control_c1=None, target_c1=None):
        """
        Get a random Clifford gate. This is for randomized benchmarking.
        """
        return self.get_clifford(
            np.random.randint(11520),
            control_c1,
            target_c1)
