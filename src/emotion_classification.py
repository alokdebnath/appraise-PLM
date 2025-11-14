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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
    """Configuration for the emotion classification models."""
    model_name: str  # HuggingFace model name
    num_emotions: int = 6  # Number of emotion classes
    hidden_size: int = 768  # Hidden size of the model
    dropout: float = 0.1
    learning_rate: float = 3e-5
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 100
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    project_name: str = "emotion-classification"
    run_name: Optional[str] = None
    train_path: str = "data/crowd-enVent-train.tsv"
    val_path: str = "data/crowd-enVent-val.tsv"
    test_path: Optional[str] = "data/crowd-enVent-test.tsv"
    output_dir: str = "models"
    save_best_model: bool = True
    log_interval: int = 100
    use_appraisal: bool = False  # Whether to use appraisal dimensions as input

class EmotionDataset(Dataset):
    """Dataset class for emotion classification."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        appraisal_values: Optional[Dict[str, List[float]]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.appraisal_values = appraisal_values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        
        if self.tokenizer is not None:
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['input_ids'] = encoding['input_ids'].squeeze()
            item['attention_mask'] = encoding['attention_mask'].squeeze()
        
        if self.appraisal_values is not None:
            appraisal_tensor = torch.tensor([
                self.appraisal_values[dim][idx] for dim in self.appraisal_values.keys()
            ], dtype=torch.float)
            item['appraisal_values'] = appraisal_tensor
        
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class DirectEmotionClassifier(nn.Module):
    """Direct emotion classification model using transformer encoder."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load the base model
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        # Get the actual hidden size from the model
        self.hidden_size = self.encoder.config.hidden_size
        logger.info(f"Using model hidden size: {self.hidden_size}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, config.num_emotions)
        )
        
        # Initialize weights
        self._init_weights()
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None) -> 'DirectEmotionClassifier':
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
        """Initialize the weights of the classifier."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Get base model outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get predictions
        logits = self.classifier(pooled_output)
        
        return logits

class AppraisalEmotionClassifier(nn.Module):
    """Emotion classification model using appraisal dimensions as input."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # MLP for processing appraisal dimensions
        self.appraisal_encoder = nn.Sequential(
            nn.Linear(21, 128),  # 21 appraisal dimensions
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_emotions)
        )
        
        # Initialize weights
        self._init_weights()
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None) -> 'AppraisalEmotionClassifier':
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
        
        logger.info(f"Model saved to {model_path}")
    
    def _init_weights(self):
        """Initialize the weights of the MLP."""
        for module in self.appraisal_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, appraisal_values: torch.Tensor) -> torch.Tensor:
        """Forward pass using appraisal dimensions."""
        return self.appraisal_encoder(appraisal_values)

class EmotionTrainer:
    """Trainer class for emotion classification models."""
    
    def __init__(
        self,
        model: Union[DirectEmotionClassifier, AppraisalEmotionClassifier],
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
        self.criterion = nn.CrossEntropyLoss()
        
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
            'val_loss': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="Training")):
                # Move batch to device
                if isinstance(self.model, DirectEmotionClassifier):
                    input_ids = batch['input_ids'].to(self.config.device)
                    attention_mask = batch['attention_mask'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                else:
                    appraisal_values = batch['appraisal_values'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    # Forward pass
                    logits = self.model(appraisal_values)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                # Log training metrics
                if batch_idx % self.config.log_interval == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/step": epoch * len(self.train_dataloader) + batch_idx
                    })
            
            avg_train_loss = train_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate(self.val_dataloader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)
            
            # Log validation metrics
            wandb.log({
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/f1": val_metrics['f1'],
                "epoch": epoch + 1
            })
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss and self.config.save_best_model:
                best_val_loss = val_metrics['loss']
                self.save_model(os.path.join(self.config.output_dir, 'best_model.pt'))
        
        return history
    
    def evaluate(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if isinstance(self.model, DirectEmotionClassifier):
                    input_ids = batch['input_ids'].to(self.config.device)
                    attention_mask = batch['attention_mask'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                else:
                    appraisal_values = batch['appraisal_values'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    # Forward pass
                    logits = self.model(appraisal_values)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_predictions, 
            average='weighted'
        )
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Save the model using HuggingFace's save_pretrained format."""
        self.model.save_pretrained(path)
        if wandb.run is not None:
            wandb.save(path)
    
    @classmethod
    def load_model(cls, path: str, config: Optional[ModelConfig] = None) -> 'EmotionTrainer':
        """Load a saved model."""
        if config is None:
            config_path = os.path.join(path, 'config.json')
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        
        if config.use_appraisal:
            model = AppraisalEmotionClassifier.from_pretrained(path, config)
        else:
            model = DirectEmotionClassifier.from_pretrained(path, config)
        
        return cls(model, config, None, None)  # Dataloaders need to be set separately

def load_data(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    use_appraisal: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load the data from TSV files."""
    try:
        # Read datasets
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
        required_cols = ['generated_text', 'emotion']
        if use_appraisal:
            required_cols.extend(appraisal_cols)
        
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
        # Create emotion label mapping
        emotion_labels = sorted(train_df['emotion'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(emotion_labels)}
        logger.info(f"Emotion labels: {emotion_labels}")
        
        # Convert emotion labels to indices
        train_labels = [label_to_idx[label] for label in train_df['emotion']]
        val_labels = [label_to_idx[label] for label in val_df['emotion']]
        test_labels = [label_to_idx[label] for label in test_df['emotion']] if test_df is not None else None
        
        # Create datasets
        if config.use_appraisal:
            # Define appraisal dimension columns
            appraisal_cols = [
                'suddenness', 'familiarity', 'predict_event', 'pleasantness',
                'unpleasantness', 'goal_relevance', 'chance_responsblt',
                'self_responsblt', 'other_responsblt', 'predict_conseq',
                'goal_support', 'urgency', 'self_control', 'other_control',
                'chance_control', 'accept_conseq', 'standards', 'social_norms',
                'attention', 'not_consider', 'effort'
            ]
            
            train_dataset = EmotionDataset(
                texts=train_df['generated_text'].tolist(),
                labels=train_labels,
                appraisal_values={col: train_df[col].tolist() for col in appraisal_cols},
                max_length=config.max_length
            )
            
            val_dataset = EmotionDataset(
                texts=val_df['generated_text'].tolist(),
                labels=val_labels,
                appraisal_values={col: val_df[col].tolist() for col in appraisal_cols},
                max_length=config.max_length
            )
            
            test_dataset = None
            if test_df is not None:
                test_dataset = EmotionDataset(
                    texts=test_df['generated_text'].tolist(),
                    labels=test_labels,
                    appraisal_values={col: test_df[col].tolist() for col in appraisal_cols},
                    max_length=config.max_length
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            logger.info(f"Loaded tokenizer for {config.model_name}")
            
            train_dataset = EmotionDataset(
                texts=train_df['generated_text'].tolist(),
                labels=train_labels,
                tokenizer=tokenizer,
                max_length=config.max_length
            )
            
            val_dataset = EmotionDataset(
                texts=val_df['generated_text'].tolist(),
                labels=val_labels,
                tokenizer=tokenizer,
                max_length=config.max_length
            )
            
            test_dataset = None
            if test_df is not None:
                test_dataset = EmotionDataset(
                    texts=test_df['generated_text'].tolist(),
                    labels=test_labels,
                    tokenizer=tokenizer,
                    max_length=config.max_length
                )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        logger.info(f"Created dataloaders with batch size {config.batch_size}")
        return train_dataloader, val_dataloader, test_dataloader
        
    except Exception as e:
        logger.error(f"Error preparing dataloaders: {str(e)}")
        raise

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train emotion classification models")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large",
                      help="HuggingFace model name")
    parser.add_argument("--num_emotions", type=int, default=13,
                      help="Number of emotion classes")
    parser.add_argument("--hidden_size", type=int, default=768,
                      help="Hidden size of the model")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                      help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--project_name", type=str, default="emotion-classification",
                      help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                      help="Wandb run name")
    parser.add_argument("--train_path", type=str, default="data/crowd-enVent-train.tsv",
                      help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/crowd-enVent-val.tsv",
                      help="Path to validation data")
    parser.add_argument("--test_path", type=str, default="data/crowd-enVent-test.tsv",
                      help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="models",
                      help="Directory to save models")
    parser.add_argument("--save_best_model", action="store_true",
                      help="Save best model based on validation loss")
    parser.add_argument("--log_interval", type=int, default=100,
                      help="Logging interval")
    parser.add_argument("--use_appraisal", action="store_true",
                      help="Use appraisal dimensions as input")
    
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
        test_path=config.test_path,
        use_appraisal=config.use_appraisal
    )
    
    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        train_df, val_df, test_df, config
    )
    
    # Initialize model
    if config.use_appraisal:
        model = AppraisalEmotionClassifier(config)
    else:
        model = DirectEmotionClassifier(config)
    
    # Initialize trainer
    trainer = EmotionTrainer(
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