import json
import csv

# Input and output paths
json_file = "emowoz-dialmage.json"
csv_file = "emowoz-dialmage.csv"

# Emotion mapping (from EmoWOZ documentation)
emotion_map = {
    0: "Neutral",
    1: "Feearful",
    2: "Dissatisfied",
    3: "Apologetic",
    4: "Abusive",
    5: "Excited",
    6: "Satisfied"
}

# 0: neutral, 1: fearful, 2: dissatisfied, 3: apologetic,  4: abusive, 5: excited, 6: satisfied.

# Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define CSV fields
fields = ["dialogue_id", "turn_id", "speaker", "utterance", "emotion_id", "emotion_label"]

# Write CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    
    for ix, dialog in enumerate(data.values()):
        dialogue_id = list(data.keys())[ix]
        logs = dialog.get("log", {})
        texts = []
        emotions = []
        for log in logs:
            # print(type(log))
            texts.append(log.get("text", []))
            emotions.append(log.get("emotion", []))

        for turn_id, text in enumerate(texts):
            speaker = "user" if turn_id % 2 == 0 else "system"

            if speaker == "user" and turn_id < len(emotions):
                emotion_id = emotions[turn_id][-1]['emotion']
                emotion_label = emotion_map.get(emotion_id, None)
            else:
                emotion_id = None
                emotion_label = None

            writer.writerow({
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "utterance": text,
                "emotion_id": emotion_id,
                "emotion_label": emotion_label
            })

print(f"✅ Converted {len(data)} dialogues to {csv_file}")