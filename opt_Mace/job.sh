#!/bin/bash
#SBATCH --job-name=NewJobName
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/NewJobName/%x-%j.out
#SBATCH --error=logs/NewJobName/%x-%j.err

config_path= ''
temp_path= ''

module load gcc python openmpi
source ~/mace/bin/activate

srun mace_run_train --config "/home/hassanza/final_test/TunEOS/${config_path}/${temp_path}"
# srun python ./test.py
# srun python /home/hassanza/mace/lib64/python3.9/site-packages/mace/cli/preprocess_data.py --config ./mace_0.3.13_preproc.yaml