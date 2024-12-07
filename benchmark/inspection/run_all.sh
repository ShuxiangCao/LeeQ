#!/bin/bash

# Array of benchmark names
benchmarks=('resonator_spec' 'gmm' 'rabi' 'drag')
#benchmarks=('gmm')
#benchmarks=('rabi')
#benchmarks=('resonator_spec')
#benchmark=('drag')

# Array of config names
#configs=('success')
#configs=('failure')
configs=('success' 'failure')

# Array of model names
#models=('openai')
#models=('llama')
#models=('anthropic')
#models=('gemini')
models=('openai' 'anthropic' 'gemini' 'llama')
# Number of shots, you can adjust this value as needed
shots=20
max_jobs=8
overwrite="false"

# Create a directory for logs if it does not exist
mkdir -p benchmark_logs

# Run experiments in parallel
for benchmark in "${benchmarks[@]}"
do
    for model in "${models[@]}"
    do
        for config in "${configs[@]}"
        do
            while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
              sleep 1  # Wait until there are fewer than max_jobs running
            done
            echo "Running $benchmark $config with model $model"
            python run_benchmarks.py $benchmark $model $config $shots $overwrite > "benchmark_logs/${benchmark}_${model}_${config}.out" 2> "benchmark_logs/${benchmark}_${model}_${config}.err" &
        done
    done
done

# Wait for all processes to finish
wait
echo "All experiments completed."