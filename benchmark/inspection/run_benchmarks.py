import argparse
from typing import Callable

from mllm.utils.parser import parse_options

# Importing necessary modules and functions from other files
from benchmarks import *  # Make sure to import the new unified function
from mllm.provider_switch import default_models

# Mapping of benchmark names to their respective function calls and the default configuration to run
benchmark_name_to_config: dict[str, tuple[Callable, str]] = {
    'resonator_spec': (run_benchmark_res_spec, 'success'),
    'gmm': (run_single_benchmark_gmm, 'success'),
    'rabi': (run_single_benchmark_rabi, 'success'),
    'drag': (run_single_benchmark_drag, 'success'),
}

def set_default_to_chatgpt_4o_latest() -> None:
    """
    Set the default model to chatgpt-4o-latest.
    """
    default_models["normal"] = "chatgpt-4o-latest"
    default_models["expensive"] = "chatgpt-4o-latest"
    default_models["vision"] = "chatgpt-4o-latest"
    default_models["embedding"] = "text-embedding-3-large"
def set_default_to_gemini_1_5():
    default_models["normal"] = "gemini/gemini-1.5-pro"
    default_models["expensive"] = "gemini/gemini-1.5-pro"
    default_models["vision"] = "gemini/gemini-1.5-pro"
    default_models["embedding"] = "text-embedding-3-large"

def set_default_to_anthropic_opus():
    default_models["normal"] = "claude-3-opus-20240229"
    default_models["expensive"] = "claude-3-opus-20240229"
    default_models["vision"] = "claude-3-opus-20240229"
    default_models["embedding"] = "text-embedding-3-large"

def set_default_to_llama_():
    benchmark_config["enable_few_shot"] = False
    default_models["normal"] = "replicate/meta/meta-llama-3-70b-instruct"
    default_models["expensive"] = "replicate/meta/meta-llama-3-70b-instruct"
    default_models["vision"] = "replicate/yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174"
    default_models["embedding"] = "text-embedding-3-large"

# Mapping of model names to functions that set the model as default
model_name_to_func: dict[str, Callable] = {
    'openai': set_default_to_chatgpt_4o_latest,
    'anthropic': set_default_to_anthropic_opus,
    'gemini': set_default_to_gemini_1_5,
    'llama': set_default_to_llama_,
}

def main() -> None:
    """
    Main function that runs benchmarks based on command line arguments.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run benchmarks with specified models.")
    parser.add_argument('benchmark', type=str, help="Name of the benchmark to run.")
    parser.add_argument('model', type=str, help="Name of the model to use.")
    parser.add_argument('config', type=str, help="Configuration to use ('success', 'failure', etc.)")
    parser.add_argument('shots', type=int, nargs='?', default=20,
                        help="Number of shots (iterations) to run (default: 20).")

    args = parser.parse_args()

    # Validate the benchmark name
    if args.benchmark not in benchmark_name_to_config:
        print(f"Unknown benchmark name: {args.benchmark}")
        return

    # Validate the model name
    if args.model not in model_name_to_func:
        print(f"Unknown model name: {args.model}")
        return

    # Validate the config
    benchmark_func, default_config = benchmark_name_to_config[args.benchmark]
    config = args.config if args.config in EXPERIMENT_CONFIGS[benchmark_func.__name__] else default_config

    # Set the model and run the benchmark
    model_name_to_func[args.model]()

    # You have to enable this option before using the `correct_json_by_model` rule
    parse_options.correct_json_by_model = True

    run_benchmarks(f'./{args.model}_{args.benchmark}_{config}_cases', args.shots, benchmark_func, config)

if __name__ == '__main__':
    main()
