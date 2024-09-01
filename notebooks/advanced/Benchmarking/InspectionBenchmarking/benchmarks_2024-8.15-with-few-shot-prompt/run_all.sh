#!/bin/bash

# Array of benchmark names
benchmarks=('resonator_spec' 'gmm' 'rabi' 'drag')
# benchmarks=('rabi')
configs=('success' 'failure')
# Array of model names
#models=('openai')
# models=('openai')
#models=('llama')
#models=('gemini')
models=('gemini' 'llama' 'openai')
# models=('openai' 'anthropic' 'gemini' 'llama ')

# Number of shots, you can adjust this value as needed
shots=20

# Create a directory for logs if it does not exist
mkdir -p benchmark_logs

# Run experiments in parallel
for benchmark in "${benchmarks[@]}"
do
    for model in "${models[@]}"
    do
        for config in "${configs[@]}"
        do
            echo "Running $benchmark $config with model $model"
            python run_benchmarks.py $benchmark $model $config $shots > "benchmark_logs/${benchmark}_${model}_${config}.out" 2> "benchmark_logs/${benchmark}_${model}_${config}.err" &
        done
    done
done

# Wait for all processes to finish
wait
echo "All experiments completed."
