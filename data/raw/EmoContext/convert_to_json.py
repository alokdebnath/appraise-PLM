#!/usr/bin/env python3
"""
Convert EmoContext Corpus to JSON Format

This script converts the EmoContext text files to a structured JSON format where:
- Each dialogue has utterances with speaker information
- Turn 1 and Turn 3 are Speaker 1
- Turn 2 is Speaker 2
- Each dialogue has an emotion label
"""

import json
import re
from pathlib import Path

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def convert_file_to_json(input_file, output_file, has_labels=True):
    """Convert a text file to JSON format."""
    print(f"Converting {input_file} to {output_file}")
    
    dialogues = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header line
        for line_num, line in enumerate(lines[1:], 2):
            line = line.strip()
            if not line:
                continue
            
            # Split by tabs
            parts = line.split('\t')
            
            if has_labels and len(parts) != 5:
                print(f"Warning: Line {line_num} has {len(parts)} parts, expected 5. Skipping.")
                continue
            elif not has_labels and len(parts) != 4:
                print(f"Warning: Line {line_num} has {len(parts)} parts, expected 4. Skipping.")
                continue
            
            try:
                dialogue_id = int(parts[0])
                turn1 = clean_text(parts[1])
                turn2 = clean_text(parts[2])
                turn3 = clean_text(parts[3])
                
                # Create dialogue structure
                dialogue = {
                    "dialogue_id": dialogue_id,
                    "utterances": [
                        {
                            "text": turn1,
                            "speaker": "speaker_1"
                        },
                        {
                            "text": turn2,
                            "speaker": "speaker_2"
                        },
                        {
                            "text": turn3,
                            "speaker": "speaker_1"
                        }
                    ]
                }
                
                # Add emotion label if available
                if has_labels:
                    emotion = clean_text(parts[4])
                    dialogue["emotion"] = emotion
                
                dialogues.append(dialogue)
                
            except (ValueError, IndexError) as e:
                print(f"Error processing line {line_num}: {e}")
                continue
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(dialogues)} dialogues to {output_file}")
        return len(dialogues)
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return 0

def main():
    """Convert all EmoContext files to JSON format."""
    print("Converting EmoContext corpus to JSON format...")
    
    # Define file mappings
    file_mappings = [
        ("train (1).txt", "train.json", True),
        ("dev.txt", "dev.json", True),
        ("devwithoutlabels.txt", "dev_without_labels.json", False),
        ("testwithoutlabels.txt", "test_without_labels.json", False)
    ]
    
    total_dialogues = 0
    
    for input_file, output_file, has_labels in file_mappings:
        if Path(input_file).exists():
            count = convert_file_to_json(input_file, output_file, has_labels)
            total_dialogues += count
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    print(f"\nConversion complete! Total dialogues processed: {total_dialogues}")
    
    # Show sample output
    if Path("train.json").exists():
        print("\nSample JSON structure:")
        with open("train.json", 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            if sample_data:
                print(json.dumps(sample_data[0], indent=2))

if __name__ == "__main__":
    main()
