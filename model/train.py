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

import importlib  
regressor = importlib.import_module("appraisal-regression-model")
classifier = importlib.import_module("emotion-classification-model")

def dataset_creator(filepath):
    df = pd.read_csv(filepath, sep='\t', header=0)
    columns = df.columns.tolist()
    sentences = df['generated_text'].tolist()
    
    emotion_categorical_labels = df['emotion'].tolist()
    cat_labels_list = list(set(emotion_categorical_labels))
    cat_labels_dict = {element: index for index, element in enumerate(cat_labels_list)}
    indices_list = [cat_labels_dict[emotion] for emotion in cat_labels_list]

    emotion_dimensional_labels = [df[col].tolist() for col in columns[6:18]]
    event_metadata = [df[col].tolist() for col in columns[19:23]]
    appraisal_dimension_labels = [df[col].tolist() for col in columns[24:45]]
    return columns, sentences, emotion_categorical_labels, emotion_dimensional_labels, event_metadata, appraisal_dimension_labels

class Dataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {'sentence': self.sentences[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.float)}


if __name__ == '__main__':
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    print("Using: " + str(device))
    
    
    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='roberta-large')
    parser.add_argument('-t', '--train_path', default='./data/train.csv')
    parser.add_argument('-e', '--eval_path', default='./data/test.csv')
    parser.add_argument('-v', '--val_path', default='./data/val.csv')
    parser.add_argument('-s', '--save_path', default='./models/large_')
    parser.add_argument('-b', '--batch_size', default=32)
    parser.add_argument('--seed', default=5186312)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    trainpath = args.train_path
    columns, train_sentences, train_emotion_categorical_labels, train_emotion_dimensional_labels, train_event_metadata, train_appraisal_dimension_labels = dataset_creator(trainpath)

    testpath = args.eval_path
    _, test_sentences, test_emotion_categorical_labels, test_emotion_dimensional_labels, test_event_metadata, test_appraisal_dimension_labels = dataset_creator(testpath)

    valpath = args.val_path
    _, val_sentences, val_emotion_categorical_labels, val_emotion_dimensional_labels, val_event_metadata, val_appraisal_dimension_labels = dataset_creator(valpath)

    # Create a RegressionDataset and DataLoader
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model_name = args.model_name
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)

        # print(model)
    if torch.cuda.device_count()  >  1:
        model = DataParallel(model)
    for i in range(21):
        dim = columns[24 + i]
        print(f"{'-'*20}> Training: {dim} <{'-'*20}")
        train_dataset = Dataset(train_sentences, train_appraisal_dimension_labels[i])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = Dataset(val_sentences, val_appraisal_dimension_labels[i])
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        
        model = regressor.train_regression_model(model, tokenizer, dim, train_dataloader, val_dataloader)
    
    dim = 'emotion'
    print(f"{'-'*20}> Training: {dim} <{'-'*20}")
    train_dataset = Dataset(train_sentences, train_emotion_categorical_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = Dataset(val_sentences, val_emotion_categorical_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    model = regressor.train_regression_model(model, tokenizer, dim, train_dataloader, val_dataloader)
    
