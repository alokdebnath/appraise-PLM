import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TextValueDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row['generated_text']
        label = row['suddenness'] - 1  # Convert value to 0-4 classes

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Create DataLoader
def create_dataloader(df, tokenizer, max_length, batch_size):
    dataset = TextValueDataset(df, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

train_df = load_dataset('data/train.csv')
val_df = load_dataset('data/val.csv')
test_df = load_dataset('data/test.csv')

# Define model initialization function
def model_init():
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Base training arguments (will be modified by Optuna)
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=3000,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"  # Used for early stopping
)

# Define the Optuna objective function
def objective(trial):
    # Define hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    max_length = trial.suggest_int("max_length", 32, 128, step=32)
    num_train_epochs = trial.suggest_int("num_train_epochs", 5, 10)

    # Update the training arguments with trial parameters
    training_args.learning_rate = learning_rate
    training_args.weight_decay = weight_decay
    training_args.per_device_train_batch_size = batch_size
    training_args.per_device_eval_batch_size = batch_size
    training_args.num_train_epochs = num_train_epochs

    # Create a dataloader dynamically for the different max_length
    train_loader = create_dataloader(train_df, tokenizer, max_length, batch_size)
    val_loader = create_dataloader(val_df, tokenizer, max_length, batch_size)
    test_loader = create_dataloader(test_df, tokenizer, max_length, batch_size)

    # Initialize Trainer with dynamic parameters
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model and return validation loss for Optuna to minimize
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

# Run the hyperparameter search with Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Adjust n_trials based on resources

# Output the best parameters found by Optuna
print("Best hyperparameters: ", study.best_params)
