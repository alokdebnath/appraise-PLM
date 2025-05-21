#!/bin/bash

#SBATCH -p compute    # which partition to run on ('compute' is default)
#SBATCH -J appraiseRoBERTa    # arbitrary name for the job (you choose)
#SBATCH --output=logs/appraiseRoBERTa_%j.out
#SBATCH --error=logs/appraiseRoBERTa_%j.err
#SBATCH --cpus-per-task=4    # tell Slurm how many CPU cores you need, if different from default; your job won't be able to use more than this
#SBATCH --mem=50000    # how much RAM you need (50GB in this case), if different from default; your job won't be able to use more than this
#SBATCH -t 12:30:00    # maximum execution time: in this case twelve hours and thirty minutes (optional)



# Uncomment the following to get a log of memory usage; NOTE don't use this if you plan to run multiple processes in your job and you are placing "wait" at the end of the job file, else Slurm won't be able to tell when your job is completed!

vmstat -S M {interval_secs} >> memory_usage_$SLURM_JOBID.log &

module load anaconda3-2024.02-1
source /home/adebnath/anaconda3/bin/activate
eval "${conda shell.bash hook}"

# Your commands here
conda activate base
python3 ~/phd-work/appraisal-theoretic-evaluation/appraise-BERT/notebooks/roberta-train.py
conda deactivate
