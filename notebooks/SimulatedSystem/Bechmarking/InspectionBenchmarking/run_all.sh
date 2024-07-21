#!/bin/bash

# Array of benchmark names
benchmarks=('resonator_spec' 'gmm' 'rabi' 'drag')

# Array of model names
models=('openai' 'anthropic' 'gemini' 'llama')

# Number of shots, you can adjust this value as needed
shots=10

# Create a directory for logs if it does not exist
mkdir -p logs

# Run experiments in parallel
for benchmark in "${benchmarks[@]}"
do
    for model in "${models[@]}"
    do
        echo "Running $benchmark with model $model"
        python your_script_name.py $benchmark $model $shots > "logs/${benchmark}_${model}.out" 2> "logs/${benchmark}_${model}.err" &
    done
done

# Wait for all processes to finish
wait
echo "All experiments completed."
