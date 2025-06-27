#!/bin/bash
#SBATCH --job-name=NewJobName
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/NewJobName/%x-%j.out
#SBATCH --error=logs/NewJobName/%x-%j.err

config_path= ''
temp_path= ''


module load gcc python openmpi
source ~/new_nequip/bin/activate

export PYTHONUNBUFFERED=1
# export WANDB_DIR=./wandb_logs/trial_${SLURM_ARRAY_TASK_ID}
echo "Running Nequip training with config path: /home/hassanza/final_test/TunEOS/${config_path} and temp path: ${temp_path}"
srun nequip-train -cp "/home/hassanza/final_test/TunEOS/${config_path}" -cn "${temp_path}"