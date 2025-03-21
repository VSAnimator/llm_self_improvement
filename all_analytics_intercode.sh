#!/bin/bash

# Define arrays for trial IDs and IC values
TRIAL_IDS=(1 2 3 4 5)

# Loop through each trial ID and IC value
for id in "${TRIAL_IDS[@]}"; do
    echo "Running retrieval analytics for trial ${id}"
    python scripts/retrieval_analytics.py logs/episodes/intercode_sql/test/rap_noplan/openai/gpt-4o-mini/bird_gold_10_cont_spider_trial_${id}/ --output analysis_intercode_sql_${id}
done

python scripts/compare_runs.py . --runs analysis_intercode_sql_1 analysis_intercode_sql_2 analysis_intercode_sql_3 analysis_intercode_sql_4 analysis_intercode_sql_5 --output compare_intercode_sql --max-task-id 800
