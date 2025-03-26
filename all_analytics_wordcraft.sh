#!/bin/bash

# Define arrays for trial IDs and IC values
TRIAL_IDS=(1 2 3 4 5)

# Loop through each trial ID and IC value
for id in "${TRIAL_IDS[@]}"; do
    echo "Running retrieval analytics for trial ${id}"
    python scripts/retrieval_analytics.py logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/wordcraft_depth_2_humanic_train_${id}/ --output analysis_wordcraft_${id}
done

python scripts/compare_runs.py . --runs analysis_wordcraft_1 analysis_wordcraft_2 analysis_wordcraft_3 analysis_wordcraft_4 analysis_wordcraft_5 --output compare_wordcraft --max-task-id 4000 --db_paths data/wordcraft/depth2_humanic_train_1/learning.db data/wordcraft/depth2_humanic_train_2/learning.db data/wordcraft/depth2_humanic_train_3/learning.db data/wordcraft/depth2_humanic_train_4/learning.db data/wordcraft/depth2_humanic_train_5/learning.db
