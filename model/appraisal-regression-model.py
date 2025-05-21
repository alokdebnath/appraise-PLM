import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.nn import Linear, MSELoss, DataParallel
from transformers.optimization import AdamW
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from tqdm import tqdm
from time import sleep
import argparse
import random

# Function to train the regression model
def train_regression_model(model, tokenizer, dim, train_dataloader,
                           val_dataloader, num_epochs=10, learning_rate=3e-5,
                           device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    # Define loss function and optimizer
    criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        print(f"{'='*80} \n Epoch {epoch + 1}/{num_epochs} \n {'='*80} ")
        train_batch_ix = 0
        val_batch_ix = 0

        # for batch in train_dataloader:
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                train_batch_ix += 1
                model.train()

                # Move input and target to the GPU if available
                inputs = tokenizer(batch['sentence'], return_tensors='pt', truncation=True, padding=True)
                inputs.to(device)
                targets = batch['label'].unsqueeze(1).to(inputs['input_ids'].device)
                optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    self_attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=12)
                    attention_output, _ = self_attention_layer(last_hidden_states, last_hidden_states, last_hidden_states)
                    linear_regression_layer = nn.Linear(768, 1)
                    flattened_output = attention_output.mean(dim=1)
                    logits = linear_regression_layer(flattened_output)
                    # Compute the loss
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Print the loss for each batch
                tepoch.set_postfix(dim=dim, train_loss=loss.item())
        print('-'*80)

        with tqdm(val_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                val_batch_ix += 1
                model.eval()
                with torch.no_grad():
                    inputs = tokenizer(batch['sentence'], return_tensors='pt', truncation=True, padding=True)
                    inputs.to(device)
                    targets = batch['label'].unsqueeze(1).to(inputs['input_ids'].device)

                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Compute the loss
                    loss = criterion(logits, targets)
                    logits = logits.detach().cpu().tolist()
                    targets = targets.detach().cpu().tolist()

                    mse = mean_squared_error(targets, logits)
                    mae = mean_absolute_error(targets, logits)
                    evs = explained_variance_score(targets, logits)
                    r2 = r2_score(targets, logits)

                tepoch.set_postfix(dim=dim, val_loss=loss.item(), mse=mse, mae=mae, evs=evs, r2=r2)

        # Clear CUDA Cache
        del mse, mae, evs, r2
        torch.cuda.empty_cache()
        return model