import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class TextValueDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, dial_col):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dial_col = dial_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row[self.dial_col]
        
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
        }

def create_dataloader(df, tokenizer, dial_col, max_length=256, batch_size=16):
    dataset = TextValueDataset(df, tokenizer, max_length, dial_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def appraise_dataset(df, modeldir, tokenizer, device, dial_col='utterances'):
    dataloader = create_dataloader(df, tokenizer, dial_col)
    
    for model_name in os.listdir(modeldir):
        # model_name='./models/google-t5/suddenness_google-t5/'
        if '.ipy' in model_name:
            continue
        column = str.join('_', model_name.split('_')[:-1])
        print(column)
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(modeldir, model_name))
        model = model.to(device)
        app_store = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).logits
            logits = list(itertools.chain.from_iterable(outputs.cpu().numpy().tolist()))
            logits = [(l*4)+1 for l in logits]
            app_store.extend(logits)
        df[column] = app_store
        
        app_store = []
        # Cleanup
        model.to('cpu')
        del model
        torch.cuda.empty_cache()

    return df

def split_merge(df, sp_col):
    cols = list(df.columns)
    cols = cols[:3] + [col + '_' + sp_col for col in cols[3:]]
    col_map = {list(df.columns)[i]:cols[i] for i in range(len(cols))}
    df = df.rename(columns=col_map)
    return df

def remake_df(df, appdf, sp1col, sp2col):
    sp1_df = appdf[appdf['indices'] == 1]
    sp2_df = appdf[appdf['indices'] == 0]
    sp1_df.rename(columns={"utterances": sp1col}, inplace=True)
    sp2_df.rename(columns={"utterances": sp2col}, inplace=True)
    sp1_df.reset_index(inplace=True)
    sp2_df.reset_index(inplace=True)
    sp1_df = split_merge(sp1_df, sp1col)
    sp2_df = split_merge(sp2_df, sp2col)
    df = pd.concat([df, sp1_df], axis=1)
    df = pd.concat([df, sp2_df], axis=1)
    return df

def main(datapath, sp1col, sp2col, savepath, modeldir):
    if '.csv' in datapath:
        df = pd.read_csv(datapath, low_memory=False)
    else:
        df = pd.read_json(datapath, low_memory=False)

    if 'empdial' in datapath:
        df = df.dropna()
    
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sp1_list = list(df[sp1col])
    sp2_list = list(df[sp2col])
    indexes = [(i + 1) % 2 for i in range(len(df[sp1col]) * 2)]
    utterance_list = []
    for i in range(len(sp1_list)):
        utterance_list.append(sp1_list[i])
        utterance_list.append(sp2_list[i])
    appdf = {
             'indices':indexes, 
             'utterances':utterance_list
            }
    appdf = pd.DataFrame(appdf)
    appdf = appraise_dataset(appdf, modeldir, tokenizer, device)
    df = remake_df(df, appdf, sp1col, sp2col)
    df.to_csv(savepath)

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for running the model.")

    # Adding arguments
    parser.add_argument(
        "--datapath", 
        type=str, 
        required=True, 
        help="Path to the input dataset."
    )
    parser.add_argument(
        "--sp1", 
        type=str, 
        required=True, 
        help="Columns name for speaker 1"
    )
    parser.add_argument(
        "--sp2", 
        type=str, 
        required=True, 
        help="Colunn name for speaker 2."
    )
    parser.add_argument(
        "--data_savepath", 
        type=str, 
        required=True, 
        help="Path to save processed data."
    )
    parser.add_argument(
        "--modeldir", 
        type=str, 
        required=True, 
        help="Directory to save or load the model."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # Accessing the arguments
    print("Datapath:", args.datapath)
    print("SP1:", args.sp1)
    print("SP2:", args.sp2)
    print("Data save path:", args.data_savepath)
    print("Model directory:", args.modeldir)
    
    # datapath = './dialogues/epitome/train.csv'
    # sp1 = 'seeker_post'
    # sp2 = 'response_post'
    # modeldir = './models/google-t5/'
    # data_savepath = './appraised_dialogues/google-t5_epitome_train.csv'

    
    main(args.datapath, args.sp1, args.sp2, args.data_savepath, args.modeldir)