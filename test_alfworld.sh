# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for train_ic in 3 6 10; do
    for test_ic in 3 6 10; do
        for ckpt in 200 800 1600; do
            python scripts/run_agent_v2.py \
                --llm openai/gpt-4o-mini \
                --agent_type rap_flex \
                --db_path "$CURRENT_DIR/data/alfworld_expel_${train_ic}ic_backups/${ckpt}/learning.db/learning.db" \
                --db_name expel_test_${train_ic}ic_${test_ic}ic \
                --num_passes 1 \
                --env alfworld_test \
                --num_ic $test_ic \
                --num_tasks 134 &
        done
        wait
    done
done