#!/bin/bash

'''
# Define arrays for trial IDs and IC values
TRIAL_IDS=(1 2 3 4 5)
IC_VALUES=("" "_6_ic")

# Loop through each trial ID and IC value
for id in "${TRIAL_IDS[@]}"; do
    for ic in "${IC_VALUES[@]}"; do
        echo "Running retrieval analytics for trial ${id} with ${ic}"
        python scripts/retrieval_analytics.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic}/ --output analysis_${id}${ic}
    done
done
'''

python scripts/compare_runs.py . --runs analysis_1 analysis_2 analysis_3 analysis_4 analysis_5 --output compare_3ic
python scripts/compare_runs.py . --runs analysis_1_6_ic analysis_2_6_ic analysis_3_6_ic analysis_4_6_ic analysis_5_6_ic --output compare_6ic