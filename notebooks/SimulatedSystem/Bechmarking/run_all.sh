#!/bin/bash

# Array of model names
models=('gpt-4o')

# Number of shots, you can adjust this value as needed
shots=100

# Create a directory for logs if it does not exist
mkdir -p benchmark_logs

# Run experiments in parallel
for model in "${models[@]}"
do
    echo "Running $benchmark with model $model"
    python embedding_search_benchmarking.py $model $shots > "benchmark_logs/embedding_search_${model}.out" 2> "benchmark_logs/embedding_search__${model}.err" &
done

# Wait for all processes to finish
wait
echo "All experiments completed."
