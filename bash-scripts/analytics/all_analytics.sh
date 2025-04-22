#!/bin/bash

# Define arrays for trial IDs and IC values
TRIAL_IDS=(1 2 3 4 5)
IC_VALUES=("_6_ic")

# Loop through each trial ID and IC value
for id in "${TRIAL_IDS[@]}"; do
    for ic in "${IC_VALUES[@]}"; do
        echo "Running retrieval analytics for trial ${id} with ${ic}"
        python scripts/retrieval_analytics.py logs/episodes/alfworld/train/rap_flex/openai/pbt_6ic_seg20_stitched/trial_${id}/ --output analysis/alfworld_${id}_pbt_6ic_seg20_stitched
    done
done

python scripts/compare_runs.py . --runs analysis/alfworld_1_pbt_6ic_seg20_stitched analysis/alfworld_2_pbt_6ic_seg20_stitched analysis/alfworld_3_pbt_6ic_seg20_stitched analysis/alfworld_4_pbt_6ic_seg20_stitched analysis/alfworld_5_pbt_6ic_seg20_stitched --output compare_6_ic --max-task-id 3500 --db_paths data/alfworld_pbt_6ic_seg20_trial_1/learning.db data/alfworld_pbt_6ic_seg20_trial_2/learning.db data/alfworld_pbt_6ic_seg20_trial_3/learning.db data/alfworld_pbt_6ic_seg20_trial_4/learning.db data/alfworld_pbt_6ic_seg20_trial_5/learning.db