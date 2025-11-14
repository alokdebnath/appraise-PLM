import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from typing import Dict, List, Optional, Union, Tuple
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogueDataset(Dataset):
    """Dataset class for dialogue turns."""
    
    def __init__(
        self, 
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        if not isinstance(text, str):
            raise ValueError(f"Index {idx} has non-string text: {repr(text)} (type: {type(text)})")

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class MultiTaskRegressionModel(nn.Module):
    """Multi-task regression model for appraisal dimensions."""
    
    def __init__(self, config_path: str):
        super().__init__()
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.config = config
        self.appraisal_dimensions = config['appraisal_dimensions']
        
        # Load the base model
        self.encoder = AutoModel.from_pretrained(config['model_name'])
        
        # Get the actual hidden size from the model
        self.hidden_size = self.encoder.config.hidden_size
        
        # Check if model is encoder-decoder
        self.is_encoder_decoder = hasattr(self.encoder, 'decoder')
        
        # Classification heads for each appraisal dimension
        self.regression_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Dropout(config['dropout']),
                nn.Linear(self.hidden_size, 1)
            ) for dim in self.appraisal_dimensions
        })
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Get base model outputs
        if self.is_encoder_decoder:
            batch_size = input_ids.shape[0]
            decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            pooled_output = outputs.encoder_last_hidden_state[:, 0, :]
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get predictions for each dimension
        predictions = {}
        for dim, head in self.regression_heads.items():
            predictions[dim] = head(pooled_output).squeeze(-1)
        
        return predictions

def load_model(model_path: str) -> Tuple[MultiTaskRegressionModel, AutoTokenizer]:
    """Load the trained model and tokenizer."""
    config_path = os.path.join(model_path, 'config.json')
    model = MultiTaskRegressionModel(config_path)
    
    # Load model weights
    state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"No model weights found at {state_dict_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config['model_name'])
    
    return model, tokenizer

def predict_appraisals(
    model: MultiTaskRegressionModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """Predict appraisal dimensions for a list of texts."""
    # Create dataset and dataloader
    dataset = DialogueDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Store predictions
    predictions = {dim: [] for dim in model.appraisal_dimensions}
    
    # Make predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting appraisals"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Store predictions
            for dim, pred in outputs.items():
                predictions[dim].extend(pred.cpu().numpy().tolist())
    
    return predictions

def process_dialogue_dataset(
    input_path: str,
    output_path: str,
    model_path: str,
    text_col: Optional[str] = None,
    text_col1: Optional[str] = None,
    text_col2: Optional[str] = None,
    speaker_col: Optional[str] = None,
    turn_col: Optional[str] = None,
    dialogue_col: Optional[str] = None,
    emotion_col: Optional[str] = None,
    batch_size: int = 32
):
    """
    Process a dialogue dataset and add appraisal predictions.
    Supports:
        - Single text column datasets
        - Two text column datasets (prompt/response style)
    """

    # Validate column setup
    if text_col is None and (text_col1 is None or text_col2 is None):
        raise ValueError("Must specify either --text_col OR both --text_col1 and --text_col2")

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=[text_col1, text_col2] if text_col1 and text_col2 else [text_col])
    logger.info(f"Loaded {len(df)} rows")

    # Check columns exist
    required_cols = []
    if text_col:
        required_cols.append(text_col)
    if text_col1 and text_col2:
        required_cols.extend([text_col1, text_col2])
    if speaker_col:
        required_cols.append(speaker_col)
    if turn_col:
        required_cols.append(turn_col)
    if dialogue_col:
        required_cols.append(dialogue_col)
    if emotion_col:
        required_cols.append(emotion_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path)

    # CASE A: Single text column
    if text_col:
        logger.info(f"Predicting appraisals for single column: {text_col}")
        # df = df.filter(lambda x: isinstance(x['text_column'], str))
        texts = df[text_col].tolist()
        preds = predict_appraisals(model, tokenizer, texts, batch_size)

        # Insert prediction columns right after text_col
        insert_at = df.columns.get_loc(text_col) + 1
        for dim, values in preds.items():
            df.insert(insert_at, dim, values)
            insert_at += 1

    # CASE B: Two text columns (prompt/response)
    elif text_col1 and text_col2:
        logger.info(f"Predicting appraisals for first column: {text_col1}")
        # df = df.filter(lambda x: isinstance(x['text_column'], str))
        texts1 = df[text_col1].tolist()
        preds1 = predict_appraisals(model, tokenizer, texts1, batch_size)

        logger.info(f"Predicting appraisals for second column: {text_col2}")
        texts2 = df[text_col2].tolist()
        preds2 = predict_appraisals(model, tokenizer, texts2, batch_size)

        # Insert predictions for col1 right after it
        insert_at = df.columns.get_loc(text_col1) + 1
        for dim, values in preds1.items():
            df.insert(insert_at, f"{text_col1}_{dim}", values)
            insert_at += 1

        # Insert predictions for col2 right after it
        insert_at = df.columns.get_loc(text_col2) + 1 # shift by added cols after col1
        for dim, values in preds2.items():
            df.insert(insert_at, f"{text_col2}_{dim}", values)
            insert_at += 1

    # Save output
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Done!")

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict appraisal dimensions for dialogue turns")
    
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input dialogue dataset (CSV format)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save output dataset with predictions")
    parser.add_argument("--model_path", type=str, default="src/model",
                        help="Path to trained model directory")
    parser.add_argument("--text_col", type=str, default=None,
                        help="Column containing text for single-turn datasets")
    parser.add_argument("--text_col1", type=str, default=None,
                        help="First text column for two-turn datasets (e.g., seeker)")
    parser.add_argument("--text_col2", type=str, default=None,
    help="Second text column for two-turn datasets (e.g., responder)")
    parser.add_argument("--speaker_col", type=str, default=None,
                        help="Name of column containing speaker index")
    parser.add_argument("--turn_col", type=str, default=None,
                        help="Name of column containing turn index")
    parser.add_argument("--dialogue_col", type=str, default=None,
                        help="Name of column containing dialogue index")
    parser.add_argument("--emotion_col", type=str, default=None,
                        help="Name of column containing emotion category")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for prediction")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Ensure input file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    # Ensure model directory exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    process_dialogue_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_path,
        text_col=args.text_col,
        text_col1=args.text_col1,
        text_col2=args.text_col2,
        speaker_col=args.speaker_col,
        turn_col=args.turn_col,
        dialogue_col=args.dialogue_col,
        emotion_col=args.emotion_col,
        batch_size=args.batch_size
    ) 
