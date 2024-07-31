#!/bin/bash

# Array of model names

# models=(
#   'mistral/open-mistral-nemo'
#   'mistral/codestral-latest'
#   'mistral/open-mistral-7b'
#   'mistral/open-mixtral-8x7b'
#   'mistral/open-mixtral-8x22b'
#   'mistral/open-codestral-mamba'
#   )


models=(
   'mistral/open-mixtral-8x7b'
   'mistral/open-mixtral-8x22b'
   'mistral/open-codestral-mamba'
  )
# Number of shots, you can adjust this value as needed
shots=5

# Create a directory for logs if it does not exist
mkdir -p embedding_search_benchmark_logs

# Run experiments in parallel
for model in "${models[@]}"
do
    model_path=$(echo "$model" | sed 's/\//_/g')
    echo "Running $benchmark with model $model"
    python embedding_search_benchmarking.py "$model" $shots > "embedding_search_benchmark_logs/embedding_search_${model_path}.out" 2> "embedding_search_benchmark_logs/embedding_search_${model_path}.err"
done

# Wait for all processes to finish
wait
echo "All experiments completed."
