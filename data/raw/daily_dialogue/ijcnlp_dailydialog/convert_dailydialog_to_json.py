#!/usr/bin/env python3
"""
Convert DailyDialog Corpus to JSON Format

This script converts the DailyDialog corpus to a structured JSON format where:
- Each dialogue is a list of utterances
- Each utterance has text, emotion, and dialogue act
- Each dialogue has a topic
- Speakers alternate (speaker_1, speaker_2, speaker_1, etc.)
"""

import json
import re
from pathlib import Path

# Emotion mapping
EMOTION_MAP = {
    0: "no_emotion",
    1: "anger", 
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

# Dialogue act mapping
ACT_MAP = {
    1: "inform",
    2: "question", 
    3: "directive",
    4: "commissive"
}

# Topic mapping
TOPIC_MAP = {
    1: "ordinary_life",
    2: "school_life", 
    3: "culture_education",
    4: "attitude_emotion",
    5: "relationship",
    6: "tourism",
    7: "health",
    8: "work",
    9: "politics",
    10: "finance"
}

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def convert_dailydialog_to_json():
    """Convert DailyDialog corpus to JSON format."""
    print("Converting DailyDialog corpus to JSON format...")
    
    # File paths
    text_file = "dialogues_text.txt"
    emotion_file = "dialogues_emotion.txt"
    act_file = "dialogues_act.txt"
    topic_file = "dialogues_topic.txt"
    
    dialogues = []
    
    try:
        # Read all files
        with open(text_file, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
        
        with open(emotion_file, 'r', encoding='utf-8') as f:
            emotion_lines = f.readlines()
        
        with open(act_file, 'r', encoding='utf-8') as f:
            act_lines = f.readlines()
        
        with open(topic_file, 'r', encoding='utf-8') as f:
            topic_lines = f.readlines()
        
        # Process each dialogue
        for dialogue_id, (text_line, emotion_line, act_line, topic_line) in enumerate(zip(text_lines, emotion_lines, act_lines, topic_lines)):
            
            # Clean lines
            text_line = text_line.strip()
            emotion_line = emotion_line.strip()
            act_line = act_line.strip()
            topic_line = topic_line.strip()
            
            if not text_line:
                continue
            
            # Split text into utterances
            utterances_text = text_line.split('__eou__')
            utterances_text = [clean_text(ut) for ut in utterances_text if clean_text(ut)]
            
            # Parse emotion and act numbers
            emotions = [int(x) for x in emotion_line.split()]
            acts = [int(x) for x in act_line.split()]
            topic = int(topic_line)
            
            # Validate that we have the right number of annotations
            if len(emotions) != len(utterances_text) or len(acts) != len(utterances_text):
                print(f"Warning: Dialogue {dialogue_id} has mismatched lengths. Text: {len(utterances_text)}, Emotions: {len(emotions)}, Acts: {len(acts)}")
                continue
            
            # Create utterances
            utterances = []
            for i, (text, emotion_num, act_num) in enumerate(zip(utterances_text, emotions, acts)):
                speaker = f"speaker_{1 if i % 2 == 0 else 2}"
                
                utterance = {
                    "text": text,
                    "speaker": speaker,
                    "emotion": EMOTION_MAP.get(emotion_num, f"unknown_{emotion_num}"),
                    "dialogue_act": ACT_MAP.get(act_num, f"unknown_{act_num}")
                }
                utterances.append(utterance)
            
            # Create dialogue
            dialogue = {
                "dialogue_id": dialogue_id,
                "topic": TOPIC_MAP.get(topic, f"unknown_{topic}"),
                "utterances": utterances
            }
            
            dialogues.append(dialogue)
            
            # Show progress
            if dialogue_id % 1000 == 0:
                print(f"Processed {dialogue_id} dialogues...")
    
    except Exception as e:
        print(f"Error processing files: {e}")
        return None
    
    print(f"Successfully processed {len(dialogues)} dialogues")
    return dialogues

def save_to_json(dialogues, output_file):
    """Save dialogues to JSON file."""
    print(f"Saving to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved {len(dialogues)} dialogues to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def analyze_dialogues(dialogues):
    """Analyze the converted dialogues."""
    print("\n=== Analysis ===")
    
    total_utterances = sum(len(d['utterances']) for d in dialogues)
    total_dialogues = len(dialogues)
    
    # Count emotions
    emotion_counts = {}
    act_counts = {}
    topic_counts = {}
    speaker_counts = {"speaker_1": 0, "speaker_2": 0}
    
    for dialogue in dialogues:
        topic_counts[dialogue['topic']] = topic_counts.get(dialogue['topic'], 0) + 1
        
        for utterance in dialogue['utterances']:
            emotion_counts[utterance['emotion']] = emotion_counts.get(utterance['emotion'], 0) + 1
            act_counts[utterance['dialogue_act']] = act_counts.get(utterance['dialogue_act'], 0) + 1
            speaker_counts[utterance['speaker']] = speaker_counts[utterance['speaker']] + 1
    
    print(f"Total dialogues: {total_dialogues}")
    print(f"Total utterances: {total_utterances}")
    print(f"Average utterances per dialogue: {total_utterances / total_dialogues:.2f}")
    
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

def main():
    """Main function."""
    # Convert to JSON
    dialogues = convert_dailydialog_to_json()
    
    if dialogues is None:
        print("Failed to convert dialogues")
        return
    
    # Save to JSON file
    output_file = "dailydialog.json"
    if save_to_json(dialogues, output_file):
        # Analyze the data
        analyze_dialogues(dialogues)
        
        # Show sample
        print(f"\n=== Sample Dialogue ===")
        if dialogues:
            sample = dialogues[0]
            print(json.dumps(sample, indent=2))
    
    print(f"\nConversion complete! File saved as {output_file}")

if __name__ == "__main__":
    main()
