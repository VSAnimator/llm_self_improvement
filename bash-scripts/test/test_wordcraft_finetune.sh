# Get current directory
CURRENT_DIR=$(pwd)

# Run the test script for wordcraft finetune
for i in 1 2 3 4 5; do
    python scripts/run_agent_v2.py \
        --llm openai/ft:gpt-4o-mini-2024-07-18:stanford-graphics-lab:wordcraft-worst-1747434732:BXyNRbSX \
        --agent_type finetune \
        --db_path "$CURRENT_DIR/data/null/learning.db" \
        --log_name finetune_trial_${i} \
        --num_passes 1 \
        --env wordcraft_test \
        --num_tasks 500 \
        --parallel 10 \
        --num_ic 10 \
        --start_task 0
done

