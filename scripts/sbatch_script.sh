#!/bin/bash
#SBATCH --job-name=IEMOCAP
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

input="./data/raw/IEMOCAP_full_release/iemocap_simplified.csv"
output="./data/appraised/IEMOCAP.csv"
col1=""
col2=""
col="utterance"

# Load required modules (adjust as needed for your cluster)
module purge
module load anaconda/3

# Activate conda environment
source activate appraise-plm

# Run your Python command
python3 src/predict_dialogue_appraisals.py \
	--input_path $input \
	--output_path $output \
	--model_path models/deberta-large/ \
	--text_col $col

echo "Job completed."
