#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --output=task1.out
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=5:00:00

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate vivo_dataset

# Print the job has started
echo "Job Started"

srun python src/train.py --config configs/task1.yaml