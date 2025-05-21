import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import os
import itertools
import numpy as np

# For mixed precision training
from torch.amp import autocast, GradScaler

# Custom dataset class
class TextValueDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, column):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column = column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row['generated_text']
        label = (row[self.column] - 1)/4 # Convert value to 0-4 classes

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
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load dataset
def load_dataset(file_path):
    if '.tsv' in file_path:
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)
    return df

# Create DataLoader
def create_dataloader(df, tokenizer, max_length, batch_size, column):
    dataset = TextValueDataset(df, tokenizer, max_length, column)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup with gradient checkpointing enabled
def create_model(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    return model

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    eval_loss = 0
    eval_loop = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in eval_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            loss = outputs.loss
            eval_loss += loss.item()
            eval_loop.set_postfix(loss=loss.item())

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            # rmse = np.sqrt(((np.array(predictions) - np.array(true_labels)) ** 2).mean())

    # accuracy = accuracy_score(true_labels, predictions)
    # precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    return eval_loss, np.sqrt(mse), mse, mae


# Training function with mixed precision support
def train(model, train_loader, val_loader, test_loader, epochs, device, lr, save_path,
          weight_decay=0.1,
          # max_grad_norm=1.0,
          patience=3):
    optimizer = AdamW(model.parameters(), lr=lr)
    model = model.to(device)

    epoch_counter = 0
    min_val_loss = 1e10
    total_steps = 100 * epochs
    warmup_steps = int(0.06 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, desc="Training")

        for batch in train_loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision training with autocast
            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Backpropagation with scaler
            scaler.scale(loss).backward()

            # Gradient clipping and optimizer step with scaler
            # scaler.unscale_(optimizer)
            # clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f'Average Training Loss: {avg_train_loss:.4f}')

        # Validation after each epoch
        # val_loss, val_rmse, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        val_loss, val_rmse, val_mse, val_mae = evaluate(model, val_loader, device)
        # print(f'Validation Accuracy: {val_accuracy:.4f}, Validation RMSE: {val_rmse: .4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        print(f'Validation RMSE: {val_rmse:.4f}, Validation MSE: {val_mse:.4f}, Validation MAE: {val_mae:.4f}')

        # Save model after each epoch
        if val_loss <= min_val_loss:
            # model_save_path = os.path.join(save_path, )
            model_save_path = save_path
            # torch.save(model.state_dict(), model_save_path)
            model.save_pretrained(model_save_path)
            print(f'Model saved to {model_save_path}')
            min_val_loss = val_loss
            epoch_counter = 0
        else:
            epoch_counter +=1
            print(f'Model not saved due to higher validation loss {val_loss:.4f} compared to {min_val_loss:.4f}')

        if epoch_counter >= patience:
            print(f'Model did not improve after {epoch - patience} epochs for {patience} epochs. Training halted')
            break
    # Evaluation
    # _, test_accuracy, test_rmse, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)

    _, test_rmse, test_mse, test_mae = evaluate(model, test_loader, device)

    # print(f'Test Accuracy: {test_accuracy:.4f}, Test RMSE: {test_rmse: .4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    print(f'Test RMSE: {test_rmse:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}')


# Main function
def main():
    # Configurations
    max_length = 256
    batch_size = 16
    epochs = 10
    num_labels = 1
    lr = 3e-5
    save_path = './models'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load datasets
    model_name =  'FacebookAI/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Changed to roberta-large
    train_df = load_dataset('../data/crowd-enVent-train.tsv')
    test_df = load_dataset('../data/crowd-enVent-test.tsv')
    val_df = load_dataset('../data/crowd-enVent-val.tsv')

    # Create DataLoaders
    for column in list(train_df.columns[list(train_df.columns).index('suddenness') : list(train_df.columns).index('effort') + 1]):
        print('='*80)
        print(model_name, column)

        save_path = './models/' + model_name.split('/')[0] + '/' + column + '_' + model_name.split('/')[0]  # Directory to save models
        os.makedirs(save_path, exist_ok=True)
        train_loader = create_dataloader(train_df, tokenizer, max_length, batch_size, column)
        val_loader = create_dataloader(val_df, tokenizer, max_length, batch_size, column)
        test_loader = create_dataloader(test_df, tokenizer, max_length, batch_size, column)

        # Initialize model
        model = create_model(model_name, num_labels)

        # Train and evaluate the model
        train(model, train_loader, val_loader, test_loader, epochs, device, lr, save_path)

if __name__ == '__main__':
    main()
