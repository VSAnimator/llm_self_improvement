# Get current directory
CURRENT_DIR=$(pwd)

# Get best or worst examples variable
'''
MODE=$1

for db_size in 10 20 40 60 100 200 400 600 800; do
    # Run test script
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_path "$CURRENT_DIR/data/alfworld_filtered/alfworld_${MODE}_examples_backups/${db_size}/learning.db/learning.db" \
        --log_name expel_${MODE}_${db_size}_examples \
        --num_ic 3 \
        --num_passes 1 \
        --env alfworld_test \
        --parallel 20 \
        --num_tasks 134
done
'''

for db_size in 20 40; do
    for trial in 1 2 3 4 5; do
        # Run test script
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_flex \
            --db_path "$CURRENT_DIR/data/alfworld_filtered/alfworld_trial_${trial}_examples_backups/${db_size}/learning.db/learning.db" \
            --log_name expel_trial_${trial}_${db_size}_examples \
            --num_ic 3 \
            --num_passes 1 \
            --env alfworld_test \
            --parallel 2 \
            --num_tasks 2 &
        sleep 3
    done
    wait
done