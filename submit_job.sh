#!/bin/bash
#SBATCH --job-name=bart-large-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Load required modules (adjust as needed for your cluster)
module purge
module load anaconda/3

# Activate conda environment
source activate appraise-plm

# Run your Python command
python src/multi_task_regression.py \
	--model_name "facebook/bart-large" \
	--train_path "data/crowd-enVent-train.tsv" \
	--val_path "data/crowd-enVent-val.tsv" \
	--test_path "data/crowd-enVent-test.tsv" \
	--output_dir "models" \
	--save_best_model

# Rename the best model directory
mv models/best_model models/bart-large

echo "Job completed."
