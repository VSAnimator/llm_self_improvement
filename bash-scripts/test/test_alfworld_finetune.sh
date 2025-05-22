# Get current directory
CURRENT_DIR=$(pwd)

# Run the test script
for trial in 1 2 3 4 5; do
    python scripts/run_agent_v2.py \
        --llm openai/ft:gpt-4o-mini-2024-07-18:stanford-graphics-lab:alfworld-worst-1747434801:BXyV1esN \
        --agent_type finetune \
        --db_path "$CURRENT_DIR/data/null/learning.db" \
        --log_name finetune_trial_${trial} \
        --num_passes 1 \
        --env alfworld_test \
        --num_ic 6 \
        --parallel 10 \
        --num_tasks 134
done


