# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for ic in 3 6; do
    for trial in 1 2 3 4 5; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_flex \
            --db_path "$CURRENT_DIR/data/alfworld_expel/learning.db" \
            --log_name alfworld_baseline_trial_${trial}_${ic}ic \
            --num_passes 1 \
            --env alfworld_test \
            --num_ic $ic \
            --parallel 10 \
            --num_tasks 134
        sleep 5
    done
    wait
done
