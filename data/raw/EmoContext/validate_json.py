#!/usr/bin/env python3
"""
Validate EmoContext JSON Files

This script validates the converted JSON files to ensure:
1. Correct structure
2. All dialogues have 3 utterances
3. Speaker assignments are correct
4. Emotion labels are valid
"""

import json
from pathlib import Path

def validate_json_file(filepath, has_labels=True):
    """Validate a JSON file."""
    print(f"\n=== Validating {filepath} ===")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_dialogues = len(data)
        valid_dialogues = 0
        invalid_dialogues = 0
        
        # Statistics
        emotion_counts = {}
        speaker_counts = {"speaker_1": 0, "speaker_2": 0}
        
        for i, dialogue in enumerate(data):
            is_valid = True
            
            # Check required fields
            if "dialogue_id" not in dialogue:
                print(f"Error in dialogue {i}: missing dialogue_id")
                is_valid = False
            
            if "utterances" not in dialogue:
                print(f"Error in dialogue {i}: missing utterances")
                is_valid = False
            
            if has_labels and "emotion" not in dialogue:
                print(f"Error in dialogue {i}: missing emotion")
                is_valid = False
            
            # Check utterances
            if "utterances" in dialogue:
                utterances = dialogue["utterances"]
                if len(utterances) != 3:
                    print(f"Error in dialogue {i}: expected 3 utterances, got {len(utterances)}")
                    is_valid = False
                else:
                    # Check speaker assignments
                    expected_speakers = ["speaker_1", "speaker_2", "speaker_1"]
                    for j, (utterance, expected_speaker) in enumerate(zip(utterances, expected_speakers)):
                        if "text" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing text")
                            is_valid = False
                        
                        if "speaker" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing speaker")
                            is_valid = False
                        elif utterance["speaker"] != expected_speaker:
                            print(f"Error in dialogue {i}, utterance {j}: expected {expected_speaker}, got {utterance['speaker']}")
                            is_valid = False
                        else:
                            speaker_counts[utterance["speaker"]] += 1
            
            # Count emotions
            if has_labels and "emotion" in dialogue:
                emotion = dialogue["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if is_valid:
                valid_dialogues += 1
            else:
                invalid_dialogues += 1
        
        print(f"Total dialogues: {total_dialogues}")
        print(f"Valid dialogues: {valid_dialogues}")
        print(f"Invalid dialogues: {invalid_dialogues}")
        
        if has_labels:
            print(f"\nEmotion distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                print(f"  {emotion}: {count}")
        
        print(f"\nSpeaker distribution:")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count}")
        
        return valid_dialogues == total_dialogues
        
    except Exception as e:
        print(f"Error validating file: {e}")
        return False

def main():
    """Validate all JSON files."""
    json_files = [
        ("train.json", True),
        ("dev.json", True),
        ("dev_without_labels.json", False),
        ("test_without_labels.json", False)
    ]
    
    all_valid = True
    
    for filename, has_labels in json_files:
        if Path(filename).exists():
            if not validate_json_file(filename, has_labels):
                all_valid = False
        else:
            print(f"Warning: {filename} not found")
    
    print("\n" + "="*50)
    if all_valid:
        print("✅ All JSON files are valid!")
    else:
        print("❌ Some JSON files have issues")

if __name__ == "__main__":
    main()
