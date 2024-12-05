#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --output=results/%a.zip     # Output log file for each array task
#SBATCH --error=logs/error_%A_%a.log       # Error log file  for each task
#SBATCH --time=02:30:00               # Time for each task
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --mem=16G                     # Memory per task
#SBATCH --array=0-150                   # Array index range (adjust based on parameter file size)

module load python                 # Load necessary modules

# Extract the parameter set for this array job
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" parameters.txt)

# Print which parameters are being used (for debugging)
# Print which parameters are being used (for debugging)
echo "Running with parameters: $PARAMS"

# Execute the script with the extracted parameters
python experiments.py  $PARAMS --task_ID $SLURM_ARRAY_TASK_ID