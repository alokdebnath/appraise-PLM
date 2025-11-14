import csv
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
    return re.sub(r'\s+', ' ', text.strip())

def convert_dailydialog_to_csv(output_csv):
    """Convert DailyDialog corpus to CSV format."""
    print("Converting DailyDialog corpus to CSV format...")
    
    text_file = "ijcnlp_dailydialog/dialogues_text.txt"
    emotion_file = "ijcnlp_dailydialog/dialogues_emotion.txt"
    act_file = "ijcnlp_dailydialog/dialogues_act.txt"
    topic_file = "ijcnlp_dailydialog/dialogues_topic.txt"
    
    rows = []
    
    with open(text_file, 'r', encoding='utf-8') as f_text, \
         open(emotion_file, 'r', encoding='utf-8') as f_emotion, \
         open(act_file, 'r', encoding='utf-8') as f_act, \
         open(topic_file, 'r', encoding='utf-8') as f_topic:
        
        for dialogue_id, (text_line, emotion_line, act_line, topic_line) in enumerate(zip(f_text, f_emotion, f_act, f_topic)):
            text_line = text_line.strip()
            if not text_line:
                continue
            
            utterances_text = [clean_text(ut) for ut in text_line.split('__eou__') if clean_text(ut)]
            emotions = [int(x) for x in emotion_line.strip().split()]
            acts = [int(x) for x in act_line.strip().split()]
            topic_num = int(topic_line.strip())
            topic_str = TOPIC_MAP.get(topic_num, f"unknown_{topic_num}")
            
            if len(emotions) != len(utterances_text) or len(acts) != len(utterances_text):
                print(f"Warning: Dialogue {dialogue_id} length mismatch")
                continue
            
            for turn_id, (text, emo_num, act_num) in enumerate(zip(utterances_text, emotions, acts), start=1):
                rows.append({
                    'dialogue_id': dialogue_id,
                    'turn_id': turn_id,
                    'utterance': text,
                    'emotion_label': EMOTION_MAP.get(emo_num, f"unknown_{emo_num}"),
                    'dialogue_act': ACT_MAP.get(act_num, f"unknown_{act_num}"),
                    'topic': topic_str
                })
    
    print(f"Saving {len(rows)} rows to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['dialogue_id', 'turn_id', 'utterance', 'emotion_label', 'dialogue_act', 'topic']
        )
        writer.writeheader()
        writer.writerows(rows)
    
    print("CSV conversion complete!")

if __name__ == "__main__":
    convert_dailydialog_to_csv("dailydialog.csv")