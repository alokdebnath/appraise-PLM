#!/bin/bash

dataset="./data/appraised/ESConv.csv"
appraisals="suddenness,familiarity,predict_event,pleasantness,unpleasantness,goal_relevance,chance_responsblt,self_responsblt,other_responsblt,predict_conseq,goal_support,urgency,self_control,other_control,chance_control,accept_conseq,standards,social_norms,attention,not_consider,effort"
emotion="survey_score_seeker_initial_emotion_intensity"
# survey_score_seeker_empathy,survey_score_seeker_relevance,survey_score_seeker_final_emotion_intensity,survey_score_supporter_relevance,dialog_speaker"

python3 analysis/step3_emotion_analysis.py \
    --dataset $dataset \
    --dims $appraisals \
    --label-cols $emotion \
    --outdir ./analysis/step3/ESConv