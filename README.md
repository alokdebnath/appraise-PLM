# Dialogue Appraisal Estimation with Pre-trained Language Models

This repository contains a complete pipeline for training appraisal dimension prediction models and analyzing dialogue datasets. It combines model training, inference, and comprehensive data analysis tools.

## Overview

The repository provides:
1. **Model Training**: Multi-task regression models for predicting 21 appraisal dimensions from text
2. **Inference**: Tools for predicting appraisals on dialogue datasets
3. **Analysis**: Exploratory analysis and emotion-appraisal correlation studies

## Repository Structure

```
appraise-PLM/
├── src/                          # Source code
│   ├── multi_task_regression.py  # Main training script
│   ├── emotion_classification.py # Emotion classification models
│   └── predict_dialogue_appraisals.py  # Inference script
├── analysis/                     # Analysis scripts
│   ├── step2_exploratory_analysis.py    # Dataset exploration
│   ├── step2_analyze_prompt_response.py # Prompt-response analysis
│   └── step3_emotion_analysis.py         # Emotion-appraisal correlation
├── data/                         # Data directory
│   ├── crowd-enVent-*.tsv        # Training data (crowd-enVent dataset)
│   ├── raw/                      # Raw dialogue datasets
│   └── appraised/                # Datasets with predicted appraisals
├── models/                       # Saved model checkpoints (gitignored)
├── configs/                      # Configuration files
│   └── model_configs.yaml        # Model training configuration
├── scripts/                      # Shell scripts for batch processing
│   ├── submit_job.sh             # SLURM job submission for training
│   ├── sbatch_script.sh          # Example SLURM script for inference
│   ├── step2_annotate.sh         # Run exploratory analysis
│   └── step3_analyze.sh          # Run emotion-appraisal analysis
└── slurm/                        # SLURM job outputs (gitignored)
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For cluster environments, activate your conda environment:
```bash
source activate appraise-plm
```

## Workflow

### Step 1: Model Training

Train a multi-task regression model to predict 21 appraisal dimensions from text.

1. Configure the model in `configs/model_configs.yaml`:
   - `model_name`: HuggingFace model name (e.g., "roberta-base", "deberta-large")
   - `num_dimensions`: Number of appraisal dimensions (default: 21)
   - `hidden_size`: Hidden size of the model
   - `dropout`: Dropout rate
   - `learning_rate`: Learning rate for training
   - `batch_size`: Batch size for training
   - `num_epochs`: Number of training epochs
   - `warmup_steps`: Number of warmup steps for learning rate scheduler
   - `max_length`: Maximum sequence length for tokenization

2. Run training:
```bash
python src/multi_task_regression.py \
    --model_name "roberta-base" \
    --train_path "data/crowd-enVent-train.tsv" \
    --val_path "data/crowd-enVent-val.tsv" \
    --test_path "data/crowd-enVent-test.tsv" \
    --output_dir "models" \
    --save_best_model
```

Or submit a SLURM job:
```bash
sbatch scripts/submit_job.sh
```

The model will:
- Train on the provided data
- Save the best model based on validation loss
- Evaluate on the test set if provided
- Output metrics including MSE, MAE, and R² scores

### Step 2: Predicting Appraisals on Dialogue Datasets

Use a trained model to predict appraisal dimensions on dialogue datasets.

```bash
python src/predict_dialogue_appraisals.py \
    --input_path data/raw/IEMOCAP_full_release/iemocap_simplified.csv \
    --output_path data/appraised/IEMOCAP.csv \
    --model_path models/deberta-large/ \
    --text_col utterance
```

Or use the SLURM script:
```bash
sbatch scripts/sbatch_script.sh
```

### Step 3: Exploratory Analysis

Analyze the appraised datasets to understand appraisal distributions and patterns.

```bash
python analysis/step2_exploratory_analysis.py \
    --datasets "EmW=data/appraised/EmoWOZ-multiwoz.csv" \
    --dims "suddenness,familiarity,predict_event,pleasantness,unpleasantness,goal_relevance,chance_responsblt,self_responsblt,other_responsblt,predict_conseq,goal_support,urgency,self_control,other_control,chance_control,accept_conseq,standards,social_norms,attention,not_consider,effort" \
    --outdir analysis/step2/EmoWOZ \
    --topk 6 --bins 30
```

Or use the provided script:
```bash
bash scripts/step2_annotate.sh
```

This generates:
- Summary statistics for each appraisal dimension
- Correlation matrices and heatmaps
- Distribution plots
- Top-k analysis

### Step 4: Prompt-Response Analysis

For datasets with prompt-response pairs, analyze differences between prompts and responses:

```bash
python analysis/step2_analyze_prompt_response.py \
    data/appraised/pairs.csv \
    analysis/step2/pairs \
    --prompt-prefix prompt_ \
    --response-prefix response_ \
    --dims "suddenness,familiarity,..."
```

This generates:
- Prompt-only / Response-only / Average / Delta summaries
- Correlation matrices for each view
- Overlay histograms for Prompt vs Response
- Effect sizes (Cohen's d)
- Distribution shifts (KL and Jensen-Shannon divergence)
- t-SNE projections

### Step 5: Emotion-Appraisal Correlation Analysis

Analyze correlations between appraisal dimensions and emotion labels:

```bash
python analysis/step3_emotion_analysis.py \
    --dataset data/appraised/ESConv.csv \
    --dims "suddenness,familiarity,predict_event,pleasantness,unpleasantness,goal_relevance,chance_responsblt,self_responsblt,other_responsblt,predict_conseq,goal_support,urgency,self_control,other_control,chance_control,accept_conseq,standards,social_norms,attention,not_consider,effort" \
    --label-cols survey_score_seeker_initial_emotion_intensity \
    --outdir analysis/step3/ESConv
```

Or use the provided script:
```bash
bash scripts/step3_analyze.sh
```

This generates:
- Correlation matrices between appraisals and labels
- ANOVA/effect-size statistics
- Box plots of appraisal distributions per label category
- Heatmaps of appraisal-label correlations
- Heatmaps of appraisal means grouped by label

## Data Format

### Training Data (TSV format)
- Text column containing the input text
- 21 columns for appraisal dimensions (values should be numeric)

### Dialogue Datasets (CSV format)
- Should contain a text column (specified via `--text_col`)
- Output will include all original columns plus 21 appraisal dimension columns

## Model Configuration

The model can be configured through `configs/model_configs.yaml` or command-line arguments:

- `model_name`: HuggingFace model name (e.g., "roberta-base", "bert-base-uncased", "deberta-large")
- `num_dimensions`: Number of appraisal dimensions (default: 21)
- `hidden_size`: Hidden size of the model
- `dropout`: Dropout rate
- `learning_rate`: Learning rate for training
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `warmup_steps`: Number of warmup steps for learning rate scheduler
- `max_length`: Maximum sequence length for tokenization

## Supported Models

The code supports any HuggingFace transformer model. Tested models include:
- RoBERTa (base, large)
- BERT (base, large)
- DeBERTa (large)
- T5 (base)
- MPNet (base)

## Output

### Training Output
- Best model saved to `models/best_model/` or specified output directory
- Training metrics logged to Weights & Biases (if configured)
- Validation and test metrics (MSE, MAE, R²)

### Inference Output
- CSV file with original data plus 21 appraisal dimension columns

### Analysis Output
- CSV files with summary statistics
- PNG files with visualizations (heatmaps, distributions, etc.)
- Organized in `analysis/step2/` and `analysis/step3/` directories

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- torch>=1.9.0
- transformers>=4.15.0
- pandas>=1.3.0
- numpy>=1.19.0
- scikit-learn>=0.24.0
- matplotlib (for analysis scripts)
- wandb>=0.12.0 (optional, for experiment tracking)

## Notes

- Model files, logs, and large data files are gitignored (see `.gitignore`)
- Analysis outputs are regenerated and can be excluded from version control
- Raw data files are excluded but directory structure is preserved
- SLURM job outputs are stored in `slurm/` directory

## Citation

```
@inproceedings{debnath-etal-2025-appraisal,
    title = "An Appraisal Theoretic Approach to Modelling Affect Flow in Conversation Corpora",
    author = "Debnath, Alok  and
      Graham, Yvette  and
      Conlan, Owen",
    editor = "Boleda, Gemma  and
      Roth, Michael",
    booktitle = "Proceedings of the 29th Conference on Computational Natural Language Learning",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.conll-1.16/",
    doi = "10.18653/v1/2025.conll-1.16",
    pages = "233--250",
    ISBN = "979-8-89176-271-8",
    abstract = "This paper presents a model of affect in conversations by leveraging Appraisal Theory as a generalizable framework. We propose that the multidimensional cognitive model of Appraisal Theory offers significant advantages for analyzing emotions in conversational contexts, addressing the current challenges of inconsistent annotation methodologies across corpora. To demonstrate this, we present AppraisePLM, a regression and classification model trained on the crowd-EnVent corpus that outperforms existing models in predicting 21 appraisal dimensions including \textit{pleasantness}, \textit{self-control}, and \textit{alignment with social norms}. We apply AppraisePLM to diverse conversation datasets spanning task-oriented dialogues, general-domain chit-chat, affect-specific conversations, and domain-specific affect analysis. Our analysis reveals that AppraisePLM successfully extrapolates emotion labels across datasets, while capturing domain-specific patterns in affect flow {--} change in conversational emotion over the conversation. This work highlights the entangled nature of affective phenomena in conversation and positions affect flow as a promising model for holistic emotion analysis, offering a standardized approach to evaluate and benchmark affective capabilities in conversational agents."
}
```