#!/bin/bash

# Array of benchmark names
benchmarks=('resonator_spec' 'gmm' 'rabi' 'drag')

# Array of model names
models=('openai' 'anthropic' 'gemini' 'llama')

# Number of shots, you can adjust this value as needed
shots=100

# Create a directory for logs if it does not exist
mkdir -p benchmark_logs

# Run experiments in parallel
for benchmark in "${benchmarks[@]}"
do
    for model in "${models[@]}"
    do
        echo "Running $benchmark with model $model"
        python run_benchmarks.py $benchmark $model $shots > "benchmark_logs/${benchmark}_${model}.out" 2> "benchmark_logs/${benchmark}_${model}.err" &
    done
done

# Wait for all processes to finish
wait
echo "All experiments completed."
