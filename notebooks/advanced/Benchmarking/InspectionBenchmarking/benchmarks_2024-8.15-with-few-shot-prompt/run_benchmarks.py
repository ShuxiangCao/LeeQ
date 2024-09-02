import sys
import argparse
from typing import Callable

# Importing necessary modules and functions from other files
from benchmarks import *  # Make sure to import the new unified function
from mllm.provider_switch import *

# Mapping of benchmark names to their respective function calls and the default configuration to run
benchmark_name_to_config: dict[str, tuple[Callable, str]] = {
    'resonator_spec': (run_benchmark_res_spec, 'success'),
    'gmm': (run_single_benchmark_gmm, 'success'),
    'rabi': (run_single_benchmark_rabi, 'success'),
    'drag': (run_single_benchmark_drag, 'success'),
}

# Mapping of model names to functions that set the model as default
model_name_to_func: dict[str, Callable] = {
    'openai': lambda: None,
    'anthropic': set_default_to_anthropic,
    'gemini': set_default_to_gemini,
    'llama': set_default_to_llama,
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

    from mllm.config import parse_options
    # You have to enable this option before using the `correct_json_by_model` rule
    parse_options.correct_json_by_model = True

    run_benchmarks(f'./{args.model}_{args.benchmark}_{config}_cases', args.shots, benchmark_func, config)

if __name__ == '__main__':
    main()
