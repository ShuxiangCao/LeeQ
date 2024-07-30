import sys
from typing import Callable
import argparse

# Importing necessary modules and functions from other files
from benchmarks import *
from mllm.provider_switch import *

# Mapping of benchmark names to their respective function calls
benchmark_name_to_func: dict[str, Callable] = {
    'resonator_spec': run_all_benchmarks_resonator_spec,
    'gmm': run_all_benchmarks_gmm,
    'rabi': run_all_benchmarks_rabi,
    'drag': run_all_benchmarks_drag,
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
    parser.add_argument('shots', type=int, nargs='?', default=10,
                        help="Number of shots (iterations) to run (default: 10).")

    args = parser.parse_args()

    # Validate the benchmark name
    if args.benchmark not in benchmark_name_to_func:
        print(f"Unknown benchmark name: {args.benchmark}")
        return

    # Validate the model name
    if args.model not in model_name_to_func:
        print(f"Unknown model name: {args.model}")
        return

    # Set the model and run the benchmark
    model_name_to_func[args.model]()
    benchmark_name_to_func[args.benchmark](save_path_prefix=f'./{args.model}_', num_samples=args.shots)

if __name__ == '__main__':
    main()
