#!/usr/bin/env python3
"""
Convert EmoContext Corpus to CSV Format

Each row will have:
- dialogue_id
- utterance_id
- text
- emotion (blank if missing)
"""

import csv
import re
from pathlib import Path

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def convert_file_to_csv(input_file, output_file, has_labels=True):
    """Convert a text file to CSV format."""
    print(f"Converting {input_file} to {output_file}")
    
    rows = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        for line_num, line in enumerate(lines[1:], 2):
            line = line.strip()
            if not line:
                continue
            
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
                emotion = clean_text(parts[4]) if has_labels else ""
                
                rows.append({
                    "dialogue_id": dialogue_id,
                    "utterance_id": 1,
                    "text": turn1,
                    "emotion": emotion
                })
                rows.append({
                    "dialogue_id": dialogue_id,
                    "utterance_id": 2,
                    "text": turn2,
                    "emotion": emotion
                })
                rows.append({
                    "dialogue_id": dialogue_id,
                    "utterance_id": 3,
                    "text": turn3,
                    "emotion": emotion
                })
                
            except (ValueError, IndexError) as e:
                print(f"Error processing line {line_num}: {e}")
                continue
        
        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["dialogue_id", "utterance_id", "text", "emotion"])
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Successfully wrote {len(rows)} rows to {output_file}")
        return len(rows)
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return 0

def main():
    """Convert all EmoContext files to CSV format."""
    print("Converting EmoContext corpus to CSV format...")
    
    file_mappings = [
        ("train (1).txt", "train.csv", True),
        ("dev.txt", "dev.csv", True),
        ("devwithoutlabels.txt", "dev_without_labels.csv", False),
        ("testwithoutlabels.txt", "test_without_labels.csv", False)
    ]
    
    total_rows = 0
    for input_file, output_file, has_labels in file_mappings:
        if Path(input_file).exists():
            total_rows += convert_file_to_csv(input_file, output_file, has_labels)
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    print(f"\nConversion complete! Total rows written: {total_rows}")

if __name__ == "__main__":
    main()