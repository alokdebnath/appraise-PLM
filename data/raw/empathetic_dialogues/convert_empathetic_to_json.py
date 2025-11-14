#!/usr/bin/env python3
"""
Convert EmpatheticDialogues CSV to JSON Format

This script converts the EmpatheticDialogues CSV files to a structured JSON format where:
- Dialogues are grouped by conversation ID
- Each dialogue has conv_id, context, prompt, and utterances
- Each utterance has utterance_idx, utterance, speaker_idx, and split selfeval
- Selfeval is split into self_eval1 and self_eval2 by the underscore
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove quotes and extra whitespace
    text = text.strip().strip('"')
    text = re.sub(r'\s+', ' ', text)
    return text

def split_selfeval(selfeval_str):
    """Split selfeval into self_eval1 and self_eval2."""
    if not selfeval_str:
        return "", ""
    
    # Remove quotes
    selfeval_str = selfeval_str.strip().strip('"')
    
    # Split by underscore
    parts = selfeval_str.split('_')
    if len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return parts[0], ""
    else:
        return "", ""

def convert_csv_to_json(input_file, output_file):
    """Convert a CSV file to JSON format."""
    print(f"Converting {input_file} to {output_file}")
    
    # Dictionary to group utterances by conversation
    dialogues = defaultdict(lambda: {
        "conv_id": "",
        "context": "",
        "prompt": "",
        "utterances": []
    })
    
    total_rows = 0
    total_dialogues = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_rows += 1
                
                # Clean the data
                conv_id = clean_text(row['conv_id'])
                utterance_idx = clean_text(row['utterance_idx'])
                context = clean_text(row['context'])
                prompt = clean_text(row['prompt'])
                speaker_idx = clean_text(row['speaker_idx'])
                utterance = clean_text(row['utterance'])
                selfeval = clean_text(row['selfeval'])
                tags = clean_text(row['tags'])
                
                # Split selfeval
                self_eval1, self_eval2 = split_selfeval(selfeval)
                
                # Create utterance object
                utterance_obj = {
                    "utterance_idx": utterance_idx,
                    "speaker_idx": speaker_idx,
                    "utterance": utterance,
                    "self_eval1": self_eval1,
                    "self_eval2": self_eval2,
                    "tags": tags
                }
                
                # Add to dialogue
                if conv_id not in dialogues:
                    total_dialogues += 1
                    dialogues[conv_id]["conv_id"] = conv_id
                    dialogues[conv_id]["context"] = context
                    dialogues[conv_id]["prompt"] = prompt
                
                dialogues[conv_id]["utterances"].append(utterance_obj)
                
                # Show progress
                if total_rows % 10000 == 0:
                    print(f"Processed {total_rows} rows...")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
    
    # Convert to list and sort by conversation ID
    dialogue_list = list(dialogues.values())
    dialogue_list.sort(key=lambda x: x['conv_id'])
    
    # Add dialogue_id for easier reference
    for i, dialogue in enumerate(dialogue_list):
        dialogue['dialogue_id'] = i
    
    print(f"Total rows processed: {total_rows}")
    print(f"Total dialogues created: {len(dialogue_list)}")
    
    # Save to JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogue_list, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved to {output_file}")
        return dialogue_list
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def analyze_dialogues(dialogues):
    """Analyze the converted dialogues."""
    print("\n=== Analysis ===")
    
    total_utterances = sum(len(d['utterances']) for d in dialogues)
    total_dialogues = len(dialogues)
    
    # Count contexts
    context_counts = {}
    speaker_counts = {}
    utterance_lengths = []
    
    for dialogue in dialogues:
        context_counts[dialogue['context']] = context_counts.get(dialogue['context'], 0) + 1
        utterance_lengths.append(len(dialogue['utterances']))
        
        for utterance in dialogue['utterances']:
            speaker_counts[utterance['speaker_idx']] = speaker_counts.get(utterance['speaker_idx'], 0) + 1
    
    print(f"Total dialogues: {total_dialogues}")
    print(f"Total utterances: {total_utterances}")
    print(f"Average utterances per dialogue: {total_utterances / total_dialogues:.2f}")
    print(f"Min utterances per dialogue: {min(utterance_lengths)}")
    print(f"Max utterances per dialogue: {max(utterance_lengths)}")
    
    print(f"\nContext distribution:")
    for context, count in sorted(context_counts.items()):
        print(f"  {context}: {count}")
    
    print(f"\nSpeaker distribution:")
    for speaker, count in sorted(speaker_counts.items()):
        print(f"  speaker_{speaker}: {count}")

def main():
    """Convert all EmpatheticDialogues CSV files to JSON."""
    print("Converting EmpatheticDialogues CSV files to JSON format...")
    
    # File mappings
    file_mappings = [
        ("train_new.csv", "train.json"),
        ("valid_new.csv", "valid.json"),
        ("test_new.csv", "test.json")
    ]
    
    all_dialogues = []
    
    for input_file, output_file in file_mappings:
        if Path(input_file).exists():
            dialogues = convert_csv_to_json(input_file, output_file)
            if dialogues:
                analyze_dialogues(dialogues)
                all_dialogues.extend(dialogues)
                
                # Show sample
                print(f"\n=== Sample Dialogue from {output_file} ===")
                if dialogues:
                    sample = dialogues[0]
                    print(json.dumps(sample, indent=2))
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    print(f"\nConversion complete!")
    print(f"Total dialogues across all files: {len(all_dialogues)}")

if __name__ == "__main__":
    main()
