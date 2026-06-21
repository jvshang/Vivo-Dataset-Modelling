#!/bin/bash
#SBATCH --job-name=vivo_experiments
#SBATCH --output=vivo_experiments_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=2:00:00

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate vivo_dataset
export WANDB_MODE="offline"

# Print the job has started
echo "Job Started with configs: $@"

# Default config files to run if none are provided as arguments
DEFAULT_CONFIGS=(
    "configs/task1/standard.yaml"
    "configs/task1/under_sample.yaml"
    "configs/task1/use_class_weights.yaml"

    "configs/task2/standard.yaml"
    "configs/task2/under_sample.yaml"
    "configs/task2/use_class_weights.yaml"

)

if [ $# -eq 0 ]; then
    echo "No config files provided as arguments. Using default configs defined in script."
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
else
    CONFIGS=("$@")
fi

for config_file in "${CONFIGS[@]}"; do
    echo "Starting experiment with config: $config_file"
    # Run the script directly in the background instead of creating a constrained SLURM step
    # This prevents SLURM from allocating only 1 CPU core per script and starving the GridSearchCV
    python src/train.py --config "$config_file" &
done

# Wait for all parallel background jobs to finish
wait

echo "All experiments finished."