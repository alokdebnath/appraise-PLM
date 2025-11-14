import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import logging
import os
from dataclasses import dataclass
import yaml
import wandb
import argparse
import random
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the multi-task regression model."""
    model_name: str  # HuggingFace model name
    num_dimensions: int = 21  # Number of appraisal dimensions
    hidden_size: int = 768  # Hidden size of the model
    dropout: float = 0.1
    learning_rate: float = 3e-5
    batch_size: int = 4
    num_epochs: int = 1
    warmup_steps: int = 100
    patience: int = 4  # Number of epochs to wait for improvement before early stopping
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    project_name: str = "appraisal-dimension-prediction"
    run_name: Optional[str] = None
    train_path: str = "data/crowd-enVent-train.tsv"
    val_path: str = "data/crowd-enVent-val.tsv"
    test_path: Optional[str] = "data/crowd-enVent-test.tsv"
    output_dir: str = "models"
    save_best_model: bool = True
    log_interval: int = 100
    
    def __post_init__(self):
        """Initialize appraisal dimensions after instance creation."""
        self.appraisal_dimensions = [
            'suddenness', 'familiarity', 'predict_event', 'pleasantness',
            'unpleasantness', 'goal_relevance', 'chance_responsblt',
            'self_responsblt', 'other_responsblt', 'predict_conseq',
            'goal_support', 'urgency', 'self_control', 'other_control',
            'chance_control', 'accept_conseq', 'standards', 'social_norms',
            'attention', 'not_consider', 'effort'
        ]
        # Verify that num_dimensions matches the number of appraisal dimensions
        if self.num_dimensions != len(self.appraisal_dimensions):
            raise ValueError(
                f"num_dimensions ({self.num_dimensions}) does not match "
                f"number of appraisal dimensions ({len(self.appraisal_dimensions)})"
            )

class AppraisalDataset(Dataset):
    """Dataset class for appraisal dimension regression."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: Dict[str, List[float]], 
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dimension_names = list(labels.keys())
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels for all dimensions
        labels = torch.tensor([
            self.labels[dim][idx] for dim in self.dimension_names
        ], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

class MultiTaskRegressionModel(nn.Module):
    """Multi-task regression model for appraisal dimensions."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load the base model
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        # Get the actual hidden size from the model
        self.hidden_size = self.encoder.config.hidden_size
        logger.info(f"Using model hidden size: {self.hidden_size}")
        
        # Check if model is encoder-decoder
        self.is_encoder_decoder = hasattr(self.encoder, 'decoder')
        logger.info(f"Model type: {'Encoder-Decoder' if self.is_encoder_decoder else 'Encoder-only'}")
        
        # Classification heads for each appraisal dimension
        self.regression_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(self.hidden_size, 1)
            ) for dim in config.appraisal_dimensions
        })
        
        # Initialize weights
        self._init_weights()
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None) -> 'MultiTaskRegressionModel':
        """Load a pretrained model from a directory."""
        if config is None:
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = ModelConfig(**config_dict)
            else:
                raise ValueError(f"No config file found at {config_path}")
        
        model = cls(config)
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"No model weights found at {state_dict_path}")
        
        return model
    
    def save_pretrained(self, model_path: str):
        """Save the model in a format compatible with HuggingFace's from_pretrained."""
        os.makedirs(model_path, exist_ok=True)
        
        # Save config
        config_dict = vars(self.config)
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model weights
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        torch.save(self.state_dict(), state_dict_path)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def _init_weights(self):
        """Initialize the weights of the regression heads."""
        for head in self.regression_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Get base model outputs
        if self.is_encoder_decoder:
            # For encoder-decoder models, we only use the encoder part
            # and create a dummy decoder input
            batch_size = input_ids.shape[0]
            decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            # Use encoder's last hidden state
            pooled_output = outputs.encoder_last_hidden_state[:, 0, :]
        else:
            # For encoder-only models
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Use [CLS] token representation
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get predictions for each dimension
        predictions = {}
        for dim, head in self.regression_heads.items():
            predictions[dim] = head(pooled_output).squeeze(-1)
        
        return predictions

class MultiTaskTrainer:
    """Trainer class for the multi-task regression model."""
    
    def __init__(
        self,
        model: MultiTaskRegressionModel,
        config: ModelConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Move model to device
        self.model.to(config.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize scheduler
        num_training_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Initialize wandb
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=vars(config)
        )
        
    def train(self) -> Dict[str, List[float]]:
        """Train the model."""
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs}'):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss for each dimension
                loss = 0
                for i, (dim, pred) in enumerate(outputs.items()):
                    loss += self.criterion(pred.squeeze(), labels[:, i])
                loss = loss / len(outputs)  # Average loss across dimensions
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % self.config.log_interval == 0:
                    logger.info(f'Batch {num_batches}: Loss = {loss.item():.4f}')
            
            avg_train_loss = total_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate(self.val_dataloader)
            history['val_loss'].append(val_metrics['loss'])
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                if self.config.save_best_model:
                    self.save_model(os.path.join(self.config.output_dir, 'best_model'))
                    wandb.save(os.path.join(self.config.output_dir, 'best_model'))
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        return history
    
    def evaluate(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss for each dimension
                loss = 0
                batch_predictions = []
                for i, (dim, pred) in enumerate(outputs.items()):
                    loss += self.criterion(pred.squeeze(), labels[:, i])
                    batch_predictions.append(pred.squeeze().cpu().numpy())
                loss = loss / len(outputs)
                
                total_loss += loss.item()
                all_predictions.append(np.column_stack(batch_predictions))
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics
        mse = mean_squared_error(all_labels, all_predictions)
        mae = mean_absolute_error(all_labels, all_predictions)
        r2 = r2_score(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, path: str):
        """Save the model using HuggingFace's save_pretrained format."""
        # Remove .pt extension if present
        if path.endswith('.pt'):
            path = path[:-3]
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Log the save
        logger.info(f"Model saved to {path}")
        
        # Save to wandb if available
        if wandb.run is not None:
            wandb.save(path)
    
    @classmethod
    def load_model(cls, path: str, config: Optional[ModelConfig] = None) -> 'MultiTaskTrainer':
        """Load a saved model."""
        model = MultiTaskRegressionModel.from_pretrained(path, config)
        return cls(model, model.config, None, None)  # Dataloaders need to be set separately

def load_data(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load the data from TSV files."""
    try:
        # Read only the first few rows to get column names
        train_df_sample = pd.read_csv(train_path, sep='\t', nrows=5)
        logger.info(f"Columns in dataset: {train_df_sample.columns.tolist()}")
        
        # Read full datasets
        train_df = pd.read_csv(train_path, sep='\t')
        val_df = pd.read_csv(val_path, sep='\t')
        test_df = pd.read_csv(test_path, sep='\t') if test_path else None
        
        # Define appraisal dimension columns
        appraisal_cols = [
            'suddenness', 'familiarity', 'predict_event', 'pleasantness',
            'unpleasantness', 'goal_relevance', 'chance_responsblt',
            'self_responsblt', 'other_responsblt', 'predict_conseq',
            'goal_support', 'urgency', 'self_control', 'other_control',
            'chance_control', 'accept_conseq', 'standards', 'social_norms',
            'attention', 'not_consider', 'effort'
        ]
        
        # Verify required columns exist
        required_cols = ['generated_text'] + appraisal_cols
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        logger.info(f"Loaded {len(train_df)} training examples")
        logger.info(f"Loaded {len(val_df)} validation examples")
        if test_df is not None:
            logger.info(f"Loaded {len(test_df)} test examples")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    config: ModelConfig
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Prepare dataloaders for training."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        logger.info(f"Loaded tokenizer for {config.model_name}")
        
        # Define appraisal dimension columns
        appraisal_cols = [
            'suddenness', 'familiarity', 'predict_event', 'pleasantness',
            'unpleasantness', 'goal_relevance', 'chance_responsblt',
            'self_responsblt', 'other_responsblt', 'predict_conseq',
            'goal_support', 'urgency', 'self_control', 'other_control',
            'chance_control', 'accept_conseq', 'standards', 'social_norms',
            'attention', 'not_consider', 'effort'
        ]
        
        # Verify all required columns exist
        for df, name in [(train_df, 'train'), (val_df, 'validation')]:
            missing_cols = [col for col in appraisal_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {name} dataset: {missing_cols}")
        
        # Create datasets
        train_dataset = AppraisalDataset(
            texts=train_df['generated_text'].tolist(),
            labels={col: train_df[col].tolist() for col in appraisal_cols},
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        val_dataset = AppraisalDataset(
            texts=val_df['generated_text'].tolist(),
            labels={col: val_df[col].tolist() for col in appraisal_cols},
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        test_dataset = None
        if test_df is not None:
            test_dataset = AppraisalDataset(
                texts=test_df['generated_text'].tolist(),
                labels={col: test_df[col].tolist() for col in appraisal_cols},
                tokenizer=tokenizer,
                max_length=config.max_length
            )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        logger.info(f"Created dataloaders with batch size {config.batch_size}")
        return train_dataloader, val_dataloader, test_dataloader
        
    except Exception as e:
        logger.error(f"Error preparing dataloaders: {str(e)}")
        raise

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-task regression model for appraisal dimensions")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--num_dimensions", type=int, default=21,
                        help="Number of appraisal dimensions")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size of the model")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=6e-6,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--patience", type=int, default=4,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--project_name", type=str, default="appraisal-dimension-prediction",
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--train_path", type=str, default="../data/crowd-enVent-train.tsv",
                        help="Path to training data")
    parser.add_argument("--val_path", type=str, default="../data/crowd-enVent-val.tsv",
                        help="Path to validation data")
    parser.add_argument("--test_path", type=str, default="../data/crowd-enVent-test.tsv",
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Save best model based on validation loss")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Logging interval")
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = get_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config from arguments
    config = ModelConfig(**vars(args))
    
    # Load data
    train_df, val_df, test_df = load_data(
        train_path=config.train_path,
        val_path=config.val_path,
        test_path=config.test_path
    )
    
    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        train_df, val_df, test_df, config
    )
    
    # Initialize model
    model = MultiTaskRegressionModel(config)
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    
    # Train model
    history = trainer.train()
    
    # Evaluate on test set if available
    if test_dataloader is not None:
        test_metrics = trainer.evaluate(test_dataloader)
        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            wandb.log({f"test/{metric}": value})
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 