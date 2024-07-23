#!/bin/bash

# Array of model names
models=('gpt-4o' 'gpt-4o-mini' 'gpt-4-turbo' 'gpt-4')
models=('claude-3-5-sonnet-20240620' 'claude-3-haiku-20240307' 'claude-3-opus-20240229')
models=('gemini-1.5-pro' 'gemini-1.5-flash')
models=('replicate/meta/meta-llama-3-70b-instruct' 'replicate/meta/meta-llama-3-8b-instruct')
#models=('gpt-4o' 'gpt-4o-mini' 'gpt-4-turbo' 'gpt-4' 'claude-3-5-sonnet-20240620' 'claude-3-haiku-20240307' 'claude-3-opus-20240229' 'gemini-1.5-pro' 'gemini-1.5-flash' 'replicate/meta/meta-llama-3-70b-instruct' 'replicate/meta/meta-llama-3-8b-instruct')
#models=('gpt-4o' 'gpt-4o-mini' 'gpt-4-turbo' 'gpt-4' 'claude-3-5-sonnet-20240620' 'claude-3-haiku-20240307' 'claude-3-opus-20240229' 'gemini-1.5-pro' 'gemini-1.5-flash' 'replicate/meta/meta-llama-3-70b-instruct' 'replicate/meta/meta-llama-3-8b-instruct')

# Number of shots, you can adjust this value as needed
shots=100

# Create a directory for logs if it does not exist
mkdir -p benchmark_logs

model_path = $(echo "$model" | sed 's/\//_/g')

# Run experiments in parallel
for model in "${models[@]}"
do
    echo "Running $benchmark with model $model"
    python embedding_search_benchmarking.py $model $shots > "benchmark_logs/embedding_search_${model_path}.out" 2> "benchmark_logs/embedding_search__${model_path}.err" &
done

# Wait for all processes to finish
wait
echo "All experiments completed."
