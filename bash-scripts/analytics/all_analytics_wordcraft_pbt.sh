#!/bin/bash

# Define arrays for trial IDs and IC values
TRIAL_IDS=(1 2 3 4 5)
IC_VALUES=("_6_ic")

# Loop through each trial ID and IC value
for id in "${TRIAL_IDS[@]}"; do
    for ic in "${IC_VALUES[@]}"; do
        echo "Running retrieval analytics for trial ${id} with ${ic}"
        python scripts/retrieval_analytics.py logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_10ic_seg10_stitched/trial_${id}/ --output analysis/wordcraft_${id}_pbt_10ic_seg10_stitched
    done
done

python scripts/compare_runs.py . --runs analysis/wordcraft_1_pbt_10ic_seg10_stitched analysis/wordcraft_2_pbt_10ic_seg10_stitched analysis/wordcraft_3_pbt_10ic_seg10_stitched analysis/wordcraft_4_pbt_10ic_seg10_stitched analysis/wordcraft_5_pbt_10ic_seg10_stitched --output compare_10_ic --max-task-id 4000 --db_paths data/wordcraft_pbt_10ic_seg10_trial_1/learning.db data/wordcraft_pbt_10ic_seg10_trial_2/learning.db data/wordcraft_pbt_10ic_seg10_trial_3/learning.db data/wordcraft_pbt_10ic_seg10_trial_4/learning.db data/wordcraft_pbt_10ic_seg10_trial_5/learning.db