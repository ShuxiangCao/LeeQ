import leeq
import json
from simulated_setup import *  # Change to your customized setup file
import numpy as np
from scipy import optimize as so
from leeq.experiments.builtin import *
import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function

from leeq.utils.compatibility import *
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.builtin import *
from pprint import pprint

dut_dict = {
    'Q1': {'Active': True, 'Tuneup': False, 'FromLog': False, 'Params': configuration_a},
    'Q2': {'Active': True, 'Tuneup': False, 'FromLog': False, 'Params': configuration_b}
}

duts_dict = {}
for hrid, dd in dut_dict.items():
    if (dd['Active']):
        if (dd['FromLog']):
            dut = TransmonElement.load_from_calibration_log(dd['Params']['hrid'])
        else:
            dut = TransmonElement(name=dd['Params']['hrid'], parameters=dd['Params'])

        dut.print_config_info()
        duts_dict[hrid] = dut

dut = None

from mllm.config import default_models
from mllm.provider_switch import *

from mllm.cache.cache_service import caching

caching.cache_kv.inactive = True


### Resonator spec
def run_benchmark_res_spec(scan_params):
    dut = duts_dict['Q1']
    res = ResonatorSweepTransmissionWithExtraInitialLPB(dut, **scan_params)
    return extract_results_from_experiment(res)


def run_all_benchmarks_resonator_spec(save_path_prefix, num_samples=10):
    success_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        readout_frequency = 500 * np.random.random(1) + 9000
        readout_sweep_width = 10 + 1000 * np.random.random(1)
        readout_scan_offset = readout_sweep_width / 3 * np.random.random(1)

        center_frequency = readout_frequency + readout_scan_offset

        simulation_setup(readout_frequency=readout_frequency)
        setup().status().set_param("AIAutoInspectPlots", False)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", False)

        scan_params = {
            'start': center_frequency - readout_sweep_width / 2,
            'stop': center_frequency + readout_sweep_width / 2,
            'step': readout_sweep_width / 200,
            'num_avs': 100,
            'rep_rate': 0.0,
            'mp_width': 1,
            'amp': 0.001
        }

        try:
            result = run_benchmark_res_spec(scan_params)
        except:
            continue

        success_inspections[i] = result
        with open(save_path_prefix + 'resonator_spec_success_cases.json', 'w') as f:
            json.dump(success_inspections, f)

    failed_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        offset = 1000 + 10 * np.random.random(1)[0] + 2

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", True)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", True)

        scan_params = {
            'amp': 0.1 + 0.05 * np.random.random(1)[0],
            'freq': 9645.5 - 10 + 1 * (0.5 - np.random.random(1)[0]),
        }

        try:
            result = run_benchmark_res_spec(scan_params)
        except:
            continue

        failed_inspections[i] = result
        with open(save_path_prefix + 'resonator_spec_failed_cases.json', 'w') as f:
            json.dump(failed_inspections, f)


### GMM

def run_single_benchmark_gmm(scan_params):
    dut = duts_dict['Q1']
    res = MeasurementCalibrationMultilevelGMM(dut=dut, **scan_params)
    return extract_results_from_experiment(res)


def run_all_benchmarks_gmm(save_path_prefix, num_samples=10):
    success_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", False)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", False)

        scan_params = {
            'amp': 0.1 + 0.05 * np.random.random(1)[0],
            'freq': 9645.5 + 1 * (0.5 - np.random.random(1)[0]),
        }

        try:
            result = run_single_benchmark_gmm(scan_params)
        except:
            continue

        success_inspections[i] = result
        with open(save_path_prefix + 'gmm_success_cases.json', 'w') as f:
            json.dump(success_inspections, f)

    failed_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        offset = 1000 + 10 * np.random.random(1)[0] + 2

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", False)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", False)

        scan_params = {
            'amp': 0.1 + 0.05 * np.random.random(1)[0],
            'freq': 9645.5 - 10 + 1 * (0.5 - np.random.random(1)[0]),
        }

        try:
            result = run_single_benchmark_gmm(scan_params)
        except:
            continue

        failed_inspections[i] = result
        with open(save_path_prefix + 'gmm_failed_cases.json', 'w') as f:
            json.dump(failed_inspections, f)


### RABI

def run_single_benchmark_rabi(scan_params):
    dut = duts_dict['Q1']
    res = NormalisedRabi(dut_qubit=dut, **scan_params)
    return extract_results_from_experiment(res)


def run_all_benchmarks_rabi(save_path_prefix, num_samples=10):
    success_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        offset = (10 * np.random.random(1) + 2)[0]

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", False)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", False)

        scan_params = {
            'amp': 0.1 + 0.2 * np.abs(np.random.random(1)),
            'stop': 0.5,
            'step': 0.01
        }

        try:
            result = run_single_benchmark_rabi(scan_params)
        except:
            continue

        success_inspections[i] = result
        with open(save_path_prefix + 'rabi_success_cases.json', 'w') as f:
            json.dump(success_inspections, f)

    failed_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        offset = 1000 + 10 * np.random.random(1)[0] + 2

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", False)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", False)

        scan_params = {
            'amp': 0.0001 + 0.0002 * np.abs(np.random.random(1)),
            'stop': 0.5,
            'step': 0.01
        }

        try:
            result = run_single_benchmark_rabi(scan_params)
        except:
            continue

        failed_inspections[i] = result
        with open(save_path_prefix + 'rabi_failed_cases.json', 'w') as f:
            json.dump(failed_inspections, f)


### DRAG

def run_single_benchmark_drag(scan_params):
    dut = duts_dict['Q1']
    res = DragCalibrationSingleQubitMultilevel(dut=dut, **scan_params)
    return extract_results_from_experiment(res)


def run_all_benchmarks_drag(save_path_prefix, num_samples=10):
    # Success cases
    success_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", False)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", False)

        span_width = 0.006 + 0.002 * np.random.random(1)[0]
        center = -0.005 + span_width * 0.2 * np.random.random(1)[0]

        scan_params = {
            'inv_alpha_start': center - span_width / 2,
            'inv_alpha_stop': center + span_width / 2
        }

        try:
            result = run_single_benchmark_drag(scan_params)
        except:
            continue

        success_inspections[i] = result
        with open(save_path_prefix + 'drag_success_cases.json', 'w') as f:
            json.dump(success_inspections, f)

    failed_inspections = {}

    for i in range(num_samples):
        np.random.seed(i)

        simulation_setup()
        setup().status().set_param("AIAutoInspectPlots", True)
        ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", True)

        sign = np.random.random(1)[0] > 0.5

        span_width = 0.006 + 0.1 * np.random.random(1)[0]
        center = -0.005 + (-1) ** sign * span_width * 0.6 * (1 + np.random.random(1)[0])

        scan_params = {
            'inv_alpha_start': center - span_width / 2,
            'inv_alpha_stop': center + span_width / 2
        }

        try:
            result = run_single_benchmark_drag(scan_params)
        except:
            continue

        failed_inspections[i] = result
        with open(save_path_prefix + 'drag_failed_cases.json', 'w') as f:
            json.dump(failed_inspections, f)
