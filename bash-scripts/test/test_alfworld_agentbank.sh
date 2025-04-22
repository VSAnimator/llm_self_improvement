# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for ckpt in 2400; do
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_path "$CURRENT_DIR/data/agentbank_alfworld_backups/${ckpt}/learning.db/learning.db" \
        --db_name agentbank_db_size_${ckpt} \
        --num_passes 1 \
        --env alfworld_test \
        --parallel 10 \
        --num_ic 3 \
        --num_tasks 134 &
done
wait