#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/$USER/scratch/rlprj/Experiments/logs12/error_%A_%a.log       # Error log file  for each task
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --mem=4G                     # Memory per task
#SBATCH --array=0-6239                   # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3


cd $SLURM_TMPDIR
module load python                 # Load necessary modules
virtualenv --no-download timeRL
source timeRL/bin/activate

pip install numpy gymnasium torch stable-baselines3


# Extract the parameter set for this array job
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /home/$USER/scratch/rlprj/CC/parameters.txt)

cp /home/$USER/scratch/rlprj/Experiments/main.py $SLURM_TMPDIR
cp /home/$USER/scratch/rlprj/Environmnts/custom_cartpole.py $SLURM_TMPDIR
cp /home/$USER/scratch/rlprj/Environments/custom_acrobot.py $SLURM_TMPDIR
python $SLURM_TMPDIR/main.py $PARAMS --task_ID $SLURM_ARRAY_TASK_ID 
# Print which parameters are being used (for debugging)
# Print which parameters are being used (for debugging)
# echo "Running with parameters: $PARAMS"

# Execute the script with the extracted parameters
# python experiments.py  $PARAMS --task_ID $SLURM_ARRAY_TASK_ID
