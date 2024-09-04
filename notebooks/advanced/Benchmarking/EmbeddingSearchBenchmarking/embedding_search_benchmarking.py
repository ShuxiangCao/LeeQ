import ast

from mllm import display_chats
from mllm.utils import p_map

import leeq
import argparse
from leeq.experiments.builtin import *
import os
import json

from mllm.cache.cache_service import caching

experiment_prompt = {
    'GMM': (MeasurementCalibrationMultilevelGMM, [
        'Run GMM measurement calibration',
        'Implement GMM model calibration for measurement',
        'Calibrate the measurement GMM model',
        'Calibrate the measurement GMM model with amplitude=0.2 and drive frequency = 9876 MHz',
        'Run measurement calibration with amp=0.1 and frequency = 8790 MHz',
        'Implement calibration for measurement',
        'Calibrate measurement parameters',
        'Run measurement calibration',
        'Implement measurement calibration for state discrimination',
        'Do measurement calibration'
    ]),
    'Ramsey': (SimpleRamseyMultilevel, [
        'Run ramsey experiment with default parameters',
        'Run ramsey experiment to calibrate qubit frequency with default parameters',
        'Implement ramsey experiment for qubit frequency calibration',
        'Calibrate qubit frequency with ramsey experiment',
        'Qubit frequency calibration with ramsey experiment',
        'Implement ramsey experiments to estimate qubit frequency.',
        'Qubit frequency estimation with ramsey experiment',
        'Do ramsey experiment to calibrate qubit frequency',
        'Do Ramsey interferometry experiment',
        'Calibrate qubit frequency using the ramsey experiment'
    ]),
    'Rabi': (NormalisedRabi, [
        'Run rabi experiment to calibrate single qubit gate driving amplitudes',
        'Measure Rabi oscillations to determine single qubit gate driving amplitudes',
        'Implement Rabi experiment to find pi pulse duration',
        'Calibrate the driving amplitudes for single qubit gate by Rabi',
        'Determine single qubit gate parameter using Rabi experiment',
        'Run Rabi experiment with default parameters',
        'Single qubit gate amplitudes estimation using Rabi experiments',
        'Do Rabi experiment to measure qubit drive amplitudes',
        'Run Rabi experiment',
        'Calibrate qubit drive amplitudes using Rabi experiment'
    ]),
    'Pingpong': (AmpPingpongCalibrationSingleQubitMultilevel, [
        'Fine-tune single qubit gate driving amplitude settings using pingpong method',
        'Adjust single qubit gate driving amplitude for optimal fidelity',
        'Calibrate signal amplitude with pingpong feedback loop',
        'Run single qubit gate amplitude tuning using pingpong experiment',
        'Implement pingpong feedback calibration for fine tuning single qubit gate amplitude',
        'Optimize driving parameters with the pingpong method',
        'Pingpong tuning of amplitudes settings',
        'Amplitude fine-tuning with pingpong approach',
        'Iterative tuning of amplitude using pingpong experiment',
        'Calibrate amplitude settings using iterative pingpong method'
    ]),
    'Resonator spectroscopy': (ResonatorSweepTransmissionWithExtraInitialLPB, [
        'Run resonator spectroscopy to determine resonant frequencies',
        'Implement spectroscopy on resonator',
        'Calibrate resonator using spectroscopic techniques',
        'Measure quality factor of resonators with spectroscopy',
        'Determine resonator location via resonator spectroscopy',
        'Spectroscopic analysis of resonator bandwidth using resonator spectroscopy',
        'Run full spectroscopic scan on resonator',
        'Discover resonators with spectroscopy',
        'Resonator frequency mapping using spectroscopy',
        'Measure resonator frequency response using spectroscopy'
    ]),
    'T1': (SimpleT1, [
        'Run T1 experiment to measure relaxation time',
        'Implement T1 relaxation time measurement',
        'Determine qubit T1 relaxation times',
        'Measure T1 relaxation time of qubit',
        'Do T1 experiment',
        'Measure T1 Relaxation time',
        'Do T1 experiment for qubit decay',
        'T1 relaxation profiling of qubits',
        'Experimentally determine qubit T1 times',
        'Measure decay characteristics with T1 experiment'
    ]),
    'T2': (SpinEchoMultiLevel, [
        'Run experiment to measure dephasing time',
        'Implement T2 echo experiment',
        'Determine qubit T2 echo experiment with free evolution time equals 200',
        'Measure coherence time using T2 echo',
        'Calibrate for T2 dephasing time measurements',
        'Quantify qubit coherence with echo experiments',
        'Do T2 echo to observe qubit dephasing',
        'T2 coherence profiling of qubits',
        'Experimentally determine qubit T2 times',
        'Measure qubit coherence decay using T2 experiment'
    ]),
    'RB1Q': (SingleQubitRandomizedBenchmarking, [
        'Run randomized benchmarking to measure single-qubit gate fidelity',
        'Implement one qubit randomized benchmarking',
        'Measure single qubit gate errors using randomized benchmarking',
        'Characterize single qubit gates performance using randomized benchmarking',
        'Determine fidelity of single qubit operations with randomized benchmarking',
        'Randomized benchmarking for error characterization on single qubit gate',
        'Perform single-qubit benchmarking for error rates',
        'Implement randomized benchmarking for single qubits',
        'Calibrate and measure single-qubit errors with randomized benchmarking',
        'Quantify one qubit performance with randomized benchmarking'
    ]),
    'Drag': (DragCalibrationSingleQubitMultilevel, [
        'Implement DRAG calibration to reduce gate errors',
        'Calibrate DRAG parameters to optimize qubit control',
        'Run DRAG calibration for improved qubit gate fidelity',
        'Optimize DRAG coefficients for quantum gates',
        'Measure and adjust DRAG parameters for qubit gates',
        'Calibrate pulse shaping using DRAG technique',
        'DRAG parameter tuning for gate error reduction',
        'Implement DRAG calibration routines',
        'Optimize qubit gate performance with DRAG calibration',
        'DRAG calibration to minimize leakage errors'
    ])
}


def check_code(codes, exp_class):
    tree = ast.parse(codes)
    call_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # check whether the value of the variable is an instance of the class
            if isinstance(node.value,
                          ast.Call) and node.value.func.id == exp_class.__name__:
                call_node = node.value
            else:
                return False

    # check whether the call node is a valid class instantiation by checking the parameters
    class_argspec = inspect.getfullargspec(exp_class.run)
    n_optional_args = len(class_argspec.defaults)
    class_must_args = class_argspec.args[:-n_optional_args]
    class_optional_args = class_argspec.args[-n_optional_args:]
    class_must_args = class_must_args[1:]
    n_call_args = len(call_node.args)
    call_keywords = [kw.arg for kw in call_node.keywords]
    needed_n_args = len(class_must_args)
    for kw in call_keywords:
        if kw not in class_optional_args:
            if kw not in class_must_args:
                return False
            else:
                needed_n_args -= 1
    if n_call_args < needed_n_args:
        return False
    return True


from leeq.utils.ai.code_indexer import build_leeq_code_ltm
from leeq.utils.ai.staging.stage_execution import get_codegen_wm, CodegenModel
from leeq.utils.ai.variable_table import VariableTable


def benchmark_single(key, exp_class, description, code_cog_model, codegen=True):
    input_var_table = VariableTable()
    print("Description:", description)
    codegen_wm = get_codegen_wm(description, input_var_table)
    recall_res = code_cog_model.recall(codegen_wm)

    additional_info = []
    if codegen:
        codes = code_cog_model.codegen(codegen_wm, recall_res)
        success = check_code(codes, exp_class)
        additional_info.append(codes)
    else:
        obtained_exp_cls_names = [x.idea.exp_cls.__name__ for x in
                                  recall_res.idea_results]
        additional_info.append(obtained_exp_cls_names)
        success = exp_class.__name__ in obtained_exp_cls_names

    return success, additional_info


def benchmark_all():
    leeq_code_ltm, exps_var_table = build_leeq_code_ltm(add_document_procedures=False)
    code_cog_model = CodegenModel()
    code_cog_model.n_recall_items = 3
    for idea in leeq_code_ltm.ideas:
        code_cog_model.lt_memory.add_idea(idea)

    results_list = {}

    def benchmark_one_experiment(_exp_name):
        results = []
        exp_class = experiment_prompt[_exp_name][0]
        exp_prompts = experiment_prompt[_exp_name][1]
        def benchmark_one_prompt(_prompt):
            try:
                success, additional_info = benchmark_single(_exp_name, exp_class, _prompt, code_cog_model, codegen=True)
            except Exception as e:
                success = False
                additional_info = str(e)
            return success, additional_info

        for prompt, (success, additional_info) in p_map(benchmark_one_prompt, exp_prompts):
            print(success, additional_info)
            results.append((prompt, success, additional_info))
        return results

    for exp_name, res in p_map(benchmark_one_experiment, list(experiment_prompt.keys()), n_workers=1):
        results_list[exp_name] = res

    return results_list


def main(model, shots):
    """
    Main function that runs benchmarks based on command line arguments.
    """
    # Set up command line argument parsing

    from mllm.config import default_models
    default_models["normal"] = model
    default_models["expensive"] = model
    print("Running benchmarks with model:", model)

    model_path_str = model.replace('/', '_')

    file_path = f'./recall_benchmark_{model_path_str}.json'

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    filtered_results = [v for v in results.values() if v['status'] == 'success']

    results = {i: v for i, v in enumerate(filtered_results)}

    for i in range(len(results), shots):
        try:
            result = {
                'status': 'success',
                'results': benchmark_all()
            }
            results[i] = result
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e)
            }
            results[i] = result

        with open(file_path, 'w') as f:
            json.dump(results, f)


def entry():
    parser = argparse.ArgumentParser(description="Run benchmarks with specified models.")
    parser.add_argument('model', type=str, nargs='?', help="Name of the model to use.",
                        default="chatgpt-4o-latest")
    parser.add_argument('shots', type=int, nargs='?', default=3,
                        help="Number of shots (iterations) to run (default: 3).")

    args = parser.parse_args()
    main(args.model, args.shots)


if __name__ == '__main__':
    with display_chats(1):
        entry()
