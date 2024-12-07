#!/bin/bash

# Define arrays of parameter values
seeds=(0 1 2 3 4 5 7 8 9 10)
algs=("PPO" "A2C")
envs=("CartPole" "Acrobot")
ts=(0.125 0.25 0.5 1 2 4 8)
alphs=(0.1 0.01 0.001 0.0001 0.00001 0.000001)

# Create or overwrite the parameters file
output_file="parameters.txt"
> "$output_file"

# Generate parameter combinations for each seed
for seed in "${seeds[@]}"; do
  for alg in "${algs[@]}"; do
    for env in "${envs[@]}"; do
      for t in "${ts[@]}"; do
        for alph in "${alphs[@]}"; do
          path="/home/saarhin/scratch/rlprj/experiments/results13"
          # Append the parameter combination to the file
          echo "--seed $seed --alg $alg --env $env --t_multip $t --alph $alph --path $path " >> "$output_file"
        done
      done
    done
  done
done
