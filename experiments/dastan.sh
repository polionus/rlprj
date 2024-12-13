#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/saarhin/scratch/rlprj/experiments/logs12/error_%A_%a.log       # Error log file  for each task
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --mem=4G                     # Memory per task
#SBATCH --array=0-3935                   # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3


cd $SLURM_TMPDIR
module load python                 # Load necessary modules
virtualenv --no-download timeRL
source timeRL/bin/activate

pip install numpy gymnasium torch stable-baselines3


# Extract the parameter set for this array job
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /home/saarhin/scratch/rlprj/experiments/parameters.txt)

cp /home/saarhin/scratch/rlprj/experiments/experiments.py $SLURM_TMPDIR
cp /home/saarhin/scratch/rlprj/experiments/custom_cartpole.py $SLURM_TMPDIR
cp /home/saarhin/scratch/rlprj/experiments/custom_acrobot.py $SLURM_TMPDIR
python $SLURM_TMPDIR/experiments.py $PARAMS --task_ID $SLURM_ARRAY_TASK_ID 
# Print which parameters are being used (for debugging)
# Print which parameters are being used (for debugging)
# echo "Running with parameters: $PARAMS"

# Execute the script with the extracted parameters
# python experiments.py  $PARAMS --task_ID $SLURM_ARRAY_TASK_ID
