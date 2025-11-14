#!/bin/bash

# train="./appraised/EmoWOZ.csv"
# test="./appraised/EmoryNLP_test.csv"
# dev="./appraised/EmoryNLP_dev.csv"
corpus="./data/appraised/EmoWOZ-multiwoz.csv"
appraisals="suddenness,familiarity,predict_event,pleasantness,unpleasantness,goal_relevance,chance_responsblt,self_responsblt,other_responsblt,predict_conseq,goal_support,urgency,self_control,other_control,chance_control,accept_conseq,standards,social_norms,attention,not_consider,effort"

python3 analysis/step2_exploratory_analysis.py \
	--datasets "EmW=$corpus" \
	--dims $appraisals\
	--outdir analysis/step2/EmoWOZ \
	--topk 6 --bins 30

