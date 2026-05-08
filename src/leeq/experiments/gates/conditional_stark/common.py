# Conditional AC stark shift induced CZ gate
import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from uncertainties import ufloat

from leeq import Experiment
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlockSweep
from leeq.theory import fits
from leeq.theory.estimator.kalman import KalmanFilter1D
from leeq.theory.fits.fit_exp import fit_2d_freq_with_cov
from leeq.utils import setup_logging
from leeq.utils.optional_dependencies import (
    Chat,
    Singleton,
    dict_to_html,
    display_chat,
    execute_experiment_from_instruction,
    get_exp_from_var_table,
    text_inspection,
    visual_inspection,
)
from leeq.utils.compatibility import *
from leeq.utils.compatibility import prims
from leeq.utils.high_level_simulations.noise import apply_noise_to_data

logger = setup_logging(__name__)



def _qubit_z_expectation_value_off_resonance_drive(f_qubit, f_drive, t_start, t_stop,
                                                   t_step, drive_rate):
    # Convert frequencies from MHz to Hz
    f_qubit = f_qubit * 1e6
    f_drive = f_drive * 1e6
    Omega_R = drive_rate * 1e6

    # Time array in seconds
    t = np.arange(t_start, t_stop, t_step) * 1e-6

    # Calculate detuning
    Delta = 2 * np.pi * (f_drive - f_qubit)

    # Effective Rabi frequency
    Omega_eff = np.sqrt(Omega_R ** 2 + Delta ** 2)

    # Population in the excited state as a function of time
    P_excited = (Omega_R ** 2 / Omega_eff ** 2) * np.sin(Omega_eff * t / 2) ** 2

    # Z expectation value (difference between ground and excited state populations)
    Z_expectation = 1 - 2 * P_excited

    return Z_expectation



def _generate_zz_interaction_data_from_simulation(qubits,
                                                  start,
                                                  stop,
                                                  sweep_points,
                                                  zz_value,
                                                  drive_frequency,
                                                  amp_control
                                                  ):
    simulator_setup = setup().get_default_setup()
    virtual_transmon_1 = simulator_setup.get_virtual_qubit(qubits[0])
    virtual_transmon_2 = simulator_setup.get_virtual_qubit(qubits[1])

    duts = qubits

    c1_control = duts[0].get_default_c1()
    duts[1].get_default_c1()

    # Evaluate the ZZ value

    simulator_setup.get_coupling_strength_by_qubit(virtual_transmon_1,
                                                              virtual_transmon_2)
    drive_omega = simulator_setup.get_omega_per_amp(
        channel=c1_control.channel) * amp_control

    drive_frequency - virtual_transmon_1.qubit_frequency
    virtual_transmon_2.qubit_frequency - virtual_transmon_1.qubit_frequency

    step = (stop - start) / sweep_points
    zz_oscillation_t = np.arange(start, stop, step)

    zz_oscillation_ground_x = np.cos(2 * np.pi * zz_oscillation_t * zz_value)
    zz_oscillation_excited_x = np.cos(2 * np.pi * zz_oscillation_t * -zz_value)

    zz_oscillation_ground_y = np.sin(2 * np.pi * zz_oscillation_t * zz_value)
    zz_oscillation_excited_y = np.sin(2 * np.pi * zz_oscillation_t * -zz_value)

    # Evaluate the oscillation of the control qubit
    control_qubit_oscillation = _qubit_z_expectation_value_off_resonance_drive(
        f_qubit=virtual_transmon_1.qubit_frequency,
        f_drive=drive_frequency,
        t_start=start,
        t_stop=stop,
        t_step=(stop - start) / sweep_points,
        drive_rate=drive_omega
    )

    # Evaluate the oscillation of the target qubit
    target_qubit_oscillation = _qubit_z_expectation_value_off_resonance_drive(
        f_qubit=virtual_transmon_2.qubit_frequency,
        f_drive=drive_frequency,
        t_start=start,
        t_stop=stop,
        t_step=(stop - start) / sweep_points,
        drive_rate=drive_omega
    )

    zz_oscillation_ground_x *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)
    zz_oscillation_excited_x *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)
    zz_oscillation_ground_y *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)
    zz_oscillation_excited_y *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)

    #
    # Add noise to the data
    #

    zz_oscillation_ground_x = apply_noise_to_data(virtual_transmon_2,
                                                  zz_oscillation_ground_x)
    zz_oscillation_excited_x = apply_noise_to_data(virtual_transmon_2,
                                                   zz_oscillation_excited_x)
    zz_oscillation_ground_y = apply_noise_to_data(virtual_transmon_2,
                                                  zz_oscillation_ground_y)
    zz_oscillation_excited_y = apply_noise_to_data(virtual_transmon_2,
                                                   zz_oscillation_excited_y)

    control_qubit_oscillation_ground = apply_noise_to_data(virtual_transmon_1,
                                                           -control_qubit_oscillation)
    control_qubit_oscillation_excited = apply_noise_to_data(virtual_transmon_1,
                                                            control_qubit_oscillation)

    target_qubit_oscillation = apply_noise_to_data(virtual_transmon_2,
                                                   target_qubit_oscillation)

    result = np.array([[zz_oscillation_ground_x, zz_oscillation_excited_x],
                       [zz_oscillation_ground_y, zz_oscillation_excited_y]]).transpose(
        [2, 1, 0])

    result = np.zeros_like(result)
    # Time index, ground/excited state, X/Y axis

    result[:, 0, 0] = zz_oscillation_ground_x
    result[:, 1, 0] = zz_oscillation_excited_x
    result[:, 0, 1] = zz_oscillation_ground_y
    result[:, 1, 1] = zz_oscillation_excited_y

    result_control = np.array([
        [control_qubit_oscillation_ground,
         control_qubit_oscillation_excited],
        [control_qubit_oscillation_ground,
         control_qubit_oscillation_excited]
    ]).transpose([2, 1, 0])

    result_control = np.zeros_like(result_control)
    result_control[:, 0, 0] = control_qubit_oscillation_ground
    result_control[:, 1, 0] = control_qubit_oscillation_excited
    result_control[:, 0, 1] = control_qubit_oscillation_ground
    result_control[:, 1, 1] = control_qubit_oscillation_excited

    return result, result_control




__all__ = [name for name in globals() if not name.startswith("_")] + [
    "_qubit_z_expectation_value_off_resonance_drive",
    "_generate_zz_interaction_data_from_simulation",
]
