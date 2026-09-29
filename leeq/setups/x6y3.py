"""Read-only translation of the X6Y3 qcal calibration into native LeeQ LPBs.

No RPC, pickle loading, configuration writes or calibration updates occur here.
qcal is an optional dependency, used only for its original envelope functions.
"""

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.core.primitives.built_in.common import Delay, PhaseShift
from leeq.core.primitives.built_in.simple_drive import SimpleDispersiveMeasurement, SimpleDrive
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial


@lru_cache(maxsize=256)
def _qcal_samples(sampling_rate, width, source_env, kwargs_json):
    from qcal.sequence.pulse_envelopes import pulse_envelopes
    samples = pulse_envelopes[source_env](
        length=width / 1e6, sample_rate=sampling_rate * 1e6,
        **json.loads(kwargs_json))
    samples.setflags(write=False)
    return samples


def qcal_envelope(sampling_rate, width, source_env, envelope_kwargs, amp=1.0, phase=0.0):
    """Evaluate qcal's envelope with LeeQ units (Msps/us); preserve its samples."""
    samples = _qcal_samples(sampling_rate, width, source_env,
                           json.dumps(envelope_kwargs, sort_keys=True, allow_nan=False))
    return samples * amp * np.exp(1j * phase)


def _finite(value, name, *, positive=False, nonnegative=False):
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'Invalid {name}: {value}')
    return value


class X6Y3Calibration:
    """Independent X/X90 definitions and calibrated readout for Q0/Q1.

    The YAML is the source of truth. Frequencies become MHz, times become us;
    phases pass through unchanged, exactly as in qcal's QubiC transpiler.
    In particular, a numerically large demod phase is NOT converted from degrees.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self._source = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self._source).hexdigest()
        self._config = yaml.safe_load(self._source)
        hardware = self._config['hardware']
        if hardware.get('qubit_LO') is not None or hardware.get('readout_LO') is not None:
            raise ValueError('X6Y3 importer requires direct RF frequencies (no external LO)')
        if self._config.get('initialize'):
            raise ValueError('DC initialization is outside the X6Y3 subset')
        reset = self._config['reset']
        if reset['active']['enable'] or reset['unconditional']['enable']:
            raise ValueError('Active/unconditional reset is outside the X6Y3 subset')
        if self._config['readout'].get('esp', {}).get('enable'):
            raise ValueError('Excited-state promotion is outside the GE subset')
        self.passive_delay_us = (_finite(reset['passive']['delay'], 'passive delay', nonnegative=True)
                                 * 1e6 if reset['passive']['enable'] else 0.0)
        self.source_herald = bool(self._config['readout'].get('herald', False))
        factory = PulseShapeFactory()
        if 'x6y3_qcal' not in factory.get_available_pulse_shapes():
            factory.register_pulse_shape('x6y3_qcal', qcal_envelope)
        # Validate selected qubits eagerly, without generating any waveforms.
        for qubit in (0, 1):
            self.gate(qubit, 'X90')
            self.gate(qubit, 'X')
            self.measurement(qubit)

    @property
    def snapshot(self):
        return copy.deepcopy(self._config)

    def _check_qubit(self, qubit):
        if type(qubit) is not int or qubit not in (0, 1):
            raise ValueError('The Huracan subset supports Q0 and Q1 only')

    def verify_channel_metadata(self, metadata):
        """Reject waveform sample-rate drift between calibration and firmware."""
        for q in (0, 1):
            for element, converter in (('qdrv', 'DAC'), ('rdrv', 'DAC'), ('rdlo', 'ADC')):
                hw = self._config['hardware']
                source_rate = hw['sample_rate'][converter] / hw['interpolation_ratio'][element]
                channel = metadata[f'Q{q}.{element}']['elem_params']
                target_rate = metadata['fpga_clk_freq'] * channel['samples_per_clk'] / channel['interp_ratio']
                if not np.isclose(source_rate, target_rate, rtol=1e-12, atol=0):
                    raise ValueError(f'Q{q}.{element} waveform sample rate differs from calibration')

    def phase_shift(self, qubit, radians):
        self._check_qubit(qubit)
        return PhaseShift(name=f'Q{qubit}.GE.virtual_z', parameters={
            'channel': 2 * qubit, 'phase_shift': _finite(radians, 'phase'),
            'transition_multiplier': {'f01': 1}})

    def gate(self, qubit, name='X90', *, amplitude_scale=1.0,
             frequency_offset_mhz=0.0, phase_offset_rad=0.0):
        self._check_qubit(qubit)
        if name not in ('X', 'X90'):
            raise ValueError('Only the independently calibrated GE X and X90 gates are supported')
        scale = _finite(amplitude_scale, 'amplitude scale', nonnegative=True)
        ge = self._config['single_qubit'][qubit]['GE']
        frequency = _finite(ge['freq'] / 1e6 + _finite(frequency_offset_mhz, 'frequency offset'),
                            'drive frequency', positive=True)
        phase_offset = _finite(phase_offset_rad, 'phase offset')
        children = []
        for pulse in ge[name]['pulse']:
            if pulse['channel'] != f'Q{qubit}.qdrv':
                raise ValueError('Gate targets a different drive channel')
            kwargs = copy.deepcopy(pulse['kwargs'])
            if pulse['env'] == 'virtualz':
                children.append(self.phase_shift(qubit, kwargs['phase']))
                continue
            if pulse['env'] != 'FAST_DRAG':
                raise ValueError('This importer validates the supplied FAST_DRAG gate family only')
            amplitude = _finite(kwargs.pop('amp'), 'drive amplitude', nonnegative=True) * scale
            if amplitude > 1:
                raise ValueError('Drive amplitude exceeds full scale')
            phase = _finite(kwargs.pop('phase', 0.0), 'drive phase')
            children.append(SimpleDrive(name=f'Q{qubit}.GE.{name}', parameters={
                'channel': 2 * qubit, 'transition_name': 'f01',
                'freq': frequency,
                'width': _finite(pulse['time'], 'drive duration', positive=True) * 1e6,
                'amp': amplitude, 'phase': phase + phase_offset, 'shape': 'x6y3_qcal',
                'source_env': pulse['env'], 'envelope_kwargs': kwargs}))
        if not any(isinstance(child, SimpleDrive) for child in children):
            raise ValueError('Gate contains no drive pulse')
        return LogicalPrimitiveBlockSerial(children)

    def measurement(self, qubit):
        self._check_qubit(qubit)
        readout = self._config['readout'][qubit]
        if readout['channel'] != qubit:
            raise ValueError('Unexpected readout channel mapping')
        demod = readout['demod']
        for pulse in (readout, demod):
            if pulse['env'] != 'cosine_square':
                raise ValueError('This importer validates cosine_square readout windows only')
        amplitude = _finite(readout['amp'], 'readout amplitude', nonnegative=True)
        if amplitude > 1:
            raise ValueError('Readout amplitude exceeds full scale')
        freq = _finite(readout['freq'], 'readout frequency', positive=True) / 1e6
        return SimpleDispersiveMeasurement(name=f'Q{qubit}.readout', parameters={
            'channel': 2 * qubit + 1, 'freq': freq, 'phase': 0.0, 'amp': amplitude,
            'width': _finite(readout['time'], 'readout duration', positive=True) * 1e6,
            'shape': 'x6y3_qcal', 'source_env': readout['env'],
            'envelope_kwargs': copy.deepcopy(readout['kwargs']),
            'distinguishable_states': [0, 1],
            'demodulation': {
                'delay': _finite(demod['delay'], 'demod delay', nonnegative=True) * 1e6,
                'width': _finite(demod['time'], 'demod duration', positive=True) * 1e6,
                'phase': _finite(demod['phase'], 'demod phase'), 'freq': freq, 'amp': 1.0,
                'shape': 'x6y3_qcal', 'source_env': demod['env'],
                'envelope_kwargs': copy.deepcopy(demod['kwargs'])}})

    def sequence(self, qubit, operations):
        """Build native LeeQ gates/delays, with no QubiC instruction injection."""
        self._check_qubit(qubit)
        children = [Delay(self.passive_delay_us)]
        for operation in operations:
            if operation['gate'] in ('X90', 'X'):
                children.append(self.gate(qubit, operation['gate'],
                                          amplitude_scale=operation.get('amplitude_scale', 1.0),
                                          frequency_offset_mhz=operation.get('frequency_offset_mhz', 0.0),
                                          phase_offset_rad=operation.get('phase_offset_rad', 0.0)))
            elif operation['gate'] == 'Idle':
                children.append(Delay(_finite(operation['time_us'], 'idle duration', nonnegative=True)))
            elif operation['gate'] == 'Rz':
                children.append(self.phase_shift(qubit, operation['phase_rad']))
            else:
                raise ValueError(f'Unsupported operation: {operation}')
        return LogicalPrimitiveBlockSerial(children)
