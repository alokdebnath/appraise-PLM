import json
import csv
from pathlib import Path

# ---- CONFIG ----
input_file = Path("ESConv.json")   # your JSON file
output_file = Path("ESConv.csv")  # where you want the CSV

# ---- LOAD JSON ----
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Function to flatten a single record ----
def flatten_record(record):
    flattened_rows = []

    # Top-level fields
    base = {
        "experience_type": record.get("experience_type", ""),
        "emotion_type": record.get("emotion_type", ""),
        "problem_type": record.get("problem_type", ""),
        "situation": record.get("situation", ""),
        "seeker_question1": record.get("seeker_question1", ""),
        "seeker_question2": record.get("seeker_question2", ""),
        "supporter_question1": record.get("supporter_question1", ""),
        "supporter_question2": record.get("supporter_question2", ""),
    }

    # Survey score flattening
    survey = record.get("survey_score", {})
    seeker_survey = survey.get("seeker", {})
    supporter_survey = survey.get("supporter", {})

    base.update({
        "survey_score_seeker_initial_emotion_intensity": seeker_survey.get("initial_emotion_intensity", ""),
        "survey_score_seeker_empathy": seeker_survey.get("empathy", ""),
        "survey_score_seeker_relevance": seeker_survey.get("relevance", ""),
        "survey_score_seeker_final_emotion_intensity": seeker_survey.get("final_emotion_intensity", ""),
        "survey_score_supporter_relevance": supporter_survey.get("relevance", "")
    })

    # Dialog flattening
    for idx, turn in enumerate(record.get("dialog", [])):
        row = base.copy()
        row["dialog_index"] = idx
        row["dialog_speaker"] = turn.get("speaker", "")
        row["dialog_content"] = turn.get("content", "").strip()

        # Annotation flattening
        annotation = turn.get("annotation", {})
        row["dialog_annotation_strategy"] = annotation.get("strategy", "")
        row["dialog_annotation_feedback"] = annotation.get("feedback", "")

        flattened_rows.append(row)

    return flattened_rows

# ---- Flatten and write to CSV ----
all_rows = []

# If your JSON file is a list of records, loop through it
# If it's a single record, wrap it in a list
if isinstance(data, list):
    for record in data:
        all_rows.extend(flatten_record(record))
else:
    all_rows.extend(flatten_record(data))

# Write CSV
if all_rows:
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

print(f"Flattened CSV written to {output_file}")