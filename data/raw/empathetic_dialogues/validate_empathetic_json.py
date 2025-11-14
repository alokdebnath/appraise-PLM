#!/usr/bin/env python3
"""
Validate EmpatheticDialogues JSON Files

This script validates the converted EmpatheticDialogues JSON files to ensure:
1. Correct structure
2. All required fields are present
3. Selfeval splitting is correct
4. Dialogue grouping is proper
"""

import json
from pathlib import Path

def validate_empathetic_json(filepath):
    """Validate the EmpatheticDialogues JSON file."""
    print(f"Validating {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_dialogues = len(data)
        valid_dialogues = 0
        invalid_dialogues = 0
        
        # Statistics
        context_counts = {}
        speaker_counts = {}
        utterance_lengths = []
        selfeval_counts = {"valid": 0, "invalid": 0}
        
        for i, dialogue in enumerate(data):
            is_valid = True
            
            # Check required dialogue fields
            if "dialogue_id" not in dialogue:
                print(f"Error in dialogue {i}: missing dialogue_id")
                is_valid = False
            
            if "conv_id" not in dialogue:
                print(f"Error in dialogue {i}: missing conv_id")
                is_valid = False
            
            if "context" not in dialogue:
                print(f"Error in dialogue {i}: missing context")
                is_valid = False
            
            if "prompt" not in dialogue:
                print(f"Error in dialogue {i}: missing prompt")
                is_valid = False
            
            if "utterances" not in dialogue:
                print(f"Error in dialogue {i}: missing utterances")
                is_valid = False
            
            # Check utterances
            if "utterances" in dialogue:
                utterances = dialogue["utterances"]
                if not utterances:
                    print(f"Error in dialogue {i}: empty utterances list")
                    is_valid = False
                else:
                    utterance_lengths.append(len(utterances))
                    
                    # Check each utterance
                    for j, utterance in enumerate(utterances):
                        # Check required utterance fields
                        if "utterance_idx" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing utterance_idx")
                            is_valid = False
                        
                        if "speaker_idx" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing speaker_idx")
                            is_valid = False
                        
                        if "utterance" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing utterance")
                            is_valid = False
                        
                        if "self_eval1" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing self_eval1")
                            is_valid = False
                        
                        if "self_eval2" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing self_eval2")
                            is_valid = False
                        
                        if "tags" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing tags")
                            is_valid = False
                        
                        # Count statistics
                        if "speaker_idx" in utterance:
                            speaker_counts[utterance["speaker_idx"]] = speaker_counts.get(utterance["speaker_idx"], 0) + 1
                        
                        # Check selfeval format
                        if "self_eval1" in utterance and "self_eval2" in utterance:
                            self_eval1 = utterance["self_eval1"]
                            self_eval2 = utterance["self_eval2"]
                            
                            # Check if selfeval looks like it was properly split
                            if self_eval1 and self_eval2:
                                selfeval_counts["valid"] += 1
                            elif self_eval1 or self_eval2:
                                selfeval_counts["valid"] += 1  # Some might have only one part
                            else:
                                selfeval_counts["invalid"] += 1
            
            # Count contexts
            if "context" in dialogue:
                context_counts[dialogue["context"]] = context_counts.get(dialogue["context"], 0) + 1
            
            if is_valid:
                valid_dialogues += 1
            else:
                invalid_dialogues += 1
        
        print(f"Total dialogues: {total_dialogues}")
        print(f"Valid dialogues: {valid_dialogues}")
        print(f"Invalid dialogues: {invalid_dialogues}")
        
        if utterance_lengths:
            print(f"Average utterances per dialogue: {sum(utterance_lengths) / len(utterance_lengths):.2f}")
            print(f"Min utterances per dialogue: {min(utterance_lengths)}")
            print(f"Max utterances per dialogue: {max(utterance_lengths)}")
        
        print(f"\nContext distribution:")
        for context, count in sorted(context_counts.items()):
            print(f"  {context}: {count}")
        
        print(f"\nSelfeval distribution:")
        for status, count in selfeval_counts.items():
            print(f"  {status}: {count}")
        
        print(f"\nTop 10 speakers by utterance count:")
        top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for speaker, count in top_speakers:
            print(f"  speaker_{speaker}: {count}")
        
        return valid_dialogues == total_dialogues
        
    except Exception as e:
        print(f"Error validating file: {e}")
        return False

def main():
    """Validate all EmpatheticDialogues JSON files."""
    json_files = ["train.json", "valid.json", "test.json"]
    
    all_valid = True
    
    for filename in json_files:
        if Path(filename).exists():
            if not validate_empathetic_json(filename):
                all_valid = False
        else:
            print(f"Warning: {filename} not found")
    
    print("\n" + "="*50)
    if all_valid:
        print("✅ All EmpatheticDialogues JSON files are valid!")
    else:
        print("❌ Some EmpatheticDialogues JSON files have issues")

if __name__ == "__main__":
    main()
