#!/bin/bash

# Get current directory
CURRENT_DIR=$(pwd)
# Run test script for baseline (no examples)
for i in {1..5}
do
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o \
        --agent_type rap_noplan \
        --db_path "$CURRENT_DIR/data/wordcraft/depth2_humanic/learning.db" \
        --log_name wordcraft_baseline_${i} \
        --num_passes 1 \
        --env wordcraft_test \
        --num_ic 10 \
        --parallel 10 \
        --num_tasks 500
    sleep 5
done
