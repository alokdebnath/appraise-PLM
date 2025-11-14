#!/usr/bin/env python3
"""
Validate DailyDialog JSON File

This script validates the converted DailyDialog JSON file to ensure:
1. Correct structure
2. All required fields are present
3. Speaker alternation is correct
4. Emotion and dialogue act mappings are valid
"""

import json
from pathlib import Path

def validate_dailydialog_json(filepath):
    """Validate the DailyDialog JSON file."""
    print(f"Validating {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_dialogues = len(data)
        valid_dialogues = 0
        invalid_dialogues = 0
        
        # Statistics
        emotion_counts = {}
        act_counts = {}
        topic_counts = {}
        speaker_counts = {"speaker_1": 0, "speaker_2": 0}
        utterance_lengths = []
        
        for i, dialogue in enumerate(data):
            is_valid = True
            
            # Check required fields
            if "dialogue_id" not in dialogue:
                print(f"Error in dialogue {i}: missing dialogue_id")
                is_valid = False
            
            if "topic" not in dialogue:
                print(f"Error in dialogue {i}: missing topic")
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
                        if "text" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing text")
                            is_valid = False
                        
                        if "speaker" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing speaker")
                            is_valid = False
                        
                        if "emotion" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing emotion")
                            is_valid = False
                        
                        if "dialogue_act" not in utterance:
                            print(f"Error in dialogue {i}, utterance {j}: missing dialogue_act")
                            is_valid = False
                        
                        # Check speaker alternation
                        expected_speaker = f"speaker_{1 if j % 2 == 0 else 2}"
                        if utterance.get("speaker") != expected_speaker:
                            print(f"Error in dialogue {i}, utterance {j}: expected {expected_speaker}, got {utterance.get('speaker')}")
                            is_valid = False
                        
                        # Count statistics
                        if "speaker" in utterance:
                            speaker_counts[utterance["speaker"]] += 1
                        
                        if "emotion" in utterance:
                            emotion_counts[utterance["emotion"]] = emotion_counts.get(utterance["emotion"], 0) + 1
                        
                        if "dialogue_act" in utterance:
                            act_counts[utterance["dialogue_act"]] = act_counts.get(utterance["dialogue_act"], 0) + 1
            
            # Count topics
            if "topic" in dialogue:
                topic_counts[dialogue["topic"]] = topic_counts.get(dialogue["topic"], 0) + 1
            
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
        
        print(f"\nTopic distribution:")
        for topic, count in sorted(topic_counts.items()):
            print(f"  {topic}: {count}")
        
        print(f"\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}")
        
        print(f"\nDialogue act distribution:")
        for act, count in sorted(act_counts.items()):
            print(f"  {act}: {count}")
        
        print(f"\nSpeaker distribution:")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count}")
        
        return valid_dialogues == total_dialogues
        
    except Exception as e:
        print(f"Error validating file: {e}")
        return False

def main():
    """Main validation function."""
    json_file = "dailydialog.json"
    
    if not Path(json_file).exists():
        print(f"Error: {json_file} not found")
        return
    
    if validate_dailydialog_json(json_file):
        print("\n✅ DailyDialog JSON file is valid!")
    else:
        print("\n❌ DailyDialog JSON file has issues")

if __name__ == "__main__":
    main()
