#!/bin/bash

#SBATCH --job-name=SA_exercise_model        # Job name
#SBATCH --output=logs/sa_exercise_%j.out    # Standard output and error log (%j expands to jobID)
#SBATCH --error=logs/sa_exercise_%j.err     # Separate error log
#SBATCH --nodes=1                           # Using single node for multiprocessing
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=64                  # Number of CPUs allocated to this task (for mp.Pool)
#SBATCH --mem=1M                            # Total memory for the job
#SBATCH --time=6:30:00                      # Wall time limit (HH:MM:SS)
#SBATCH --partition=defq,milanq,fpgaq       # Partitions to run the job on


echo "Activating environment..."

# Load miniforge3 (conda)
echo "Initializing miniforge2"
echo "Attempting to initialize miniforge3 and activate environment..."
MINIFORGE_BASE_PATH="/home/ingvildsd/miniforge3"
source "${MINIFORGE_BASE_PATH}/etc/profile.d/conda.sh"
echo "conda.sh sourced from ${MINIFORGE_BASE_PATH}"

# Activate Conda environment
YOUR_ENV_NAME="musclex"
echo "Activating conda environment: ${YOUR_ENV_NAME}..."
conda activate "${YOUR_ENV_NAME}"
echo "Conda environment activated."

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

echo " Running sensitivity analysis script..."
# Run the Python script
# The script should be in the same directory or provide a full path
python3 "sensitivity_analysis.py"

echo "Job finished at $(date)"
